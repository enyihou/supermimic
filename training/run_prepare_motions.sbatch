#!/bin/bash
#SBATCH --job-name=supermimic-prepare-motions
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --tmp=160G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

# CPU-only. Submit from gpu01 with lab-submit --kind cpu and a private,
# job-visible --chdir and scheduler log paths. Check project quota and
# stage-out headroom on gpu01 before submitting.
# Usage: run_prepare_motions.sbatch PACKAGE_DIR BONES_G1_TGZ BONES_SHA256
#   PRIVATE_MOTIONS_ROOT COHORT_BUILD_ID MODE SELECTION_FILE SELECTION_SHA256
# MODE: authors-manifest (reviewed keep list) or --accept-approximate
#       (with the pinned SONIC source; the external script is never executed).
# shellcheck disable=SC2317
set -euo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV BASH_ENV ENV LD_PRELOAD LD_LIBRARY_PATH
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH WANDB_API_KEY
for variable in ${!CONDA_@} ${!_CONDA_@}; do unset "$variable"; done
unset _CE_CONDA _CE_M
export PYTHONNOUSERSITE=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
file_hash() { sha256sum -- "$1" | cut -d ' ' -f 1; }
free_bytes() { df -B1 --output=avail -- "$1" | awk 'NR == 2 { print $1 }'; }

[[ $# == 8 ]] || fail 'Expected eight arguments; see script usage.'
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ && "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]] ||
  fail 'Expected a single-node Slurm batch allocation.'
[[ -z "${SLURM_JOB_GPUS:-}" && "${SLURM_GPUS_ON_NODE:-0}" == 0 ]] ||
  fail 'Motion preparation must request zero GPUs.'
[[ "${LOCAL_SCRATCH:-}" == "/scratch/jobs/$SLURM_JOB_ID" &&
   -d "$LOCAL_SCRATCH" && -w "$LOCAL_SCRATCH" && ! -L "$LOCAL_SCRATCH" ]] ||
  fail 'Allocated scratch unavailable.'
[[ "$3" =~ ^[0-9a-f]{64}$ && "$8" =~ ^[0-9a-f]{64}$ ]] || fail 'Expected two input SHA-256 digests.'
[[ "$5" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$ ]] || fail 'Invalid cohort build ID.'
case "$6" in
  authors-manifest) selection_id=supermimic-v3-authors-manifest ;;
  --accept-approximate) selection_id=supermimic-v3-appendix-e-approx-v1 ;;
  *) fail 'Supply authors-manifest or an explicitly approved --accept-approximate.' ;;
esac
if [[ "$6" == --accept-approximate && "$8" != b6b50d441a4876a382eb3ff33ba940c487d1647230b5ecacaa37cfcf521258da ]]; then
  fail 'Approximate selection requires the pinned SONIC source SHA-256.'
fi
export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable

package=$(realpath -e -- "$1")
bones=$(realpath -e -- "$2")
publish_root=$(realpath -e -- "$4")
selection=$(realpath -e -- "$7")
[[ -d "$package" && -f "$bones" && -f "$selection" ]] || fail 'Missing package, BONES archive, or selection input.'
[[ -d "$publish_root" && -O "$publish_root" && -w "$publish_root" ]] ||
  fail 'PRIVATE_MOTIONS_ROOT must be an approved, owned, writable, job-visible directory.'
permissions=$(stat -c '%a' -- "$publish_root")
(( (8#$permissions & 0777) == 0700 )) || fail 'PRIVATE_MOTIONS_ROOT must be owner-only.'
dest="$publish_root/$5"
[[ ! -e "$dest" && ! -L "$dest" ]] || fail 'Cohort destination already exists.'
selection_bytes=$(stat -c '%s' -- "$selection")
(( selection_bytes > 0 && selection_bytes <= 64000000 )) || fail 'Invalid selection input size.'

workers=${SLURM_CPUS_PER_TASK:-1}
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid allocated CPU count.'
if (( workers > 1 )); then workers=$((workers - 1)); fi  # leave one core for the archive reader
if (( workers > 32 )); then workers=32; fi                 # packer and selector limit
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

work="$LOCAL_SCRATCH/motion-preparation"
mkdir -- "$work"
mkdir -- "$work"/{archives,code,env,tmp,cache,publication}
export TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache" HF_HOME="$work/cache/hf"
export PYTHONPYCACHEPREFIX="$work/cache/pycache" MPLCONFIGDIR="$work/cache/matplotlib"
export TORCH_HOME="$work/cache/torch" WANDB_DIR="$work/tmp" WANDB_CACHE_DIR="$work/cache/wandb"
cd -- "$work"
ulimit -c 0

partial=''
cohort_build_id=$5
phase=stage_inputs
finish() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ -n "$partial" && -d "$partial" ]]; then rm -rf -- "$partial"; fi
  if (( rc != 0 )); then
    printf 'Preparation failed: job_id=%s exit_code=%s phase=%s\n' "$SLURM_JOB_ID" "$rc" "$phase" >&2
    # Publish only small control metadata on failure, never mutable work or a
    # partly copied archive. Scheduler logs and terminal state remain required.
    local failure="$publish_root/.$cohort_build_id.job-$SLURM_JOB_ID.failed.txt"
    if partial=$(mktemp -- "$publish_root/.failure.$SLURM_JOB_ID.XXXXXXXX"); then
      if printf 'job_id=%s\nexit_code=%s\nphase=%s\n' "$SLURM_JOB_ID" "$rc" "$phase" > "$partial" &&
         sync -d "$partial"; then
        mv -nT -- "$partial" "$failure"
      fi
      if [[ -f "$partial" ]]; then rm -f -- "$partial"; fi
    fi
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
printf 'Preparing cohort %s with %s worker(s) (job %s).\n' "$5" "$workers" "$SLURM_JOB_ID"

# Physical free space and the job project scratch limit can both constrain
# staging. Reserve a conservative allowance for the unpacked environment,
# packed arrays and compression working set; recheck before compression.
raw_bytes=$(stat -c '%s' -- "$bones")
env_bytes=$(stat -c '%s' -- "$package/environment.tar.gz")
code_bytes=$(stat -c '%s' -- "$package/code.tar.gz")
available=$(free_bytes "$work")
(( available >= raw_bytes + env_bytes + code_bytes + 70 * 1024**3 )) ||
  fail 'Insufficient physical scratch headroom for full motion preparation.'

cp -- "$package/environment.tar.gz" "$package/code.tar.gz" "$package/SHA256SUMS" "$work/archives/"
package_hashes=$(
  cd -- "$work/archives"
  sha256sum environment.tar.gz code.tar.gz | LC_ALL=C sort > actual
  LC_ALL=C sort SHA256SUMS > expected
  cmp -s actual expected || fail 'Environment or source archive checksum mismatch on scratch.'
  cat -- actual
)
environment_sha=$(awk '$2 == "environment.tar.gz" {print $1}' <<< "$package_hashes")
code_sha=$(awk '$2 == "code.tar.gz" {print $1}' <<< "$package_hashes")
tar --no-same-owner --no-same-permissions -xzf "$work/archives/environment.tar.gz" -C "$work/env"
tar --no-same-owner --no-same-permissions -xzf "$work/archives/code.tar.gz" -C "$work/code"
rm -- "$work/archives/environment.tar.gz" "$work/archives/code.tar.gz"
printf '%s\n' 'supermimic=5fb92297ab4fe3d86629d9f0c635334791bb0d5f' 'mjlab_research=379c55e88b1dec73f93943e524a879b2ebbf45c5' |
  cmp -s "$work/code/PINS.txt" - || fail 'Unexpected source pins.'
revision=$(<"$work/code/REPRODUCTION_COMMIT.txt")
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || fail 'Invalid reproduction source revision.'
python="$work/env/bin/python"
[[ -x "$python" ]] || fail 'Staged Python executable unavailable.'
if [[ -x "$work/env/bin/conda-unpack" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$work/env/bin/activate"
  set -u
  "$python" "$work/env/bin/conda-unpack"
fi

cp -- "$selection" "$work/archives/selection.input"
[[ $(file_hash "$work/archives/selection.input") == "$8" ]] || fail 'Selection input SHA-256 mismatch on scratch.'
cp -- "$bones" "$work/archives/bones-g1.tar.gz"
chmod 400 -- "$work/archives/selection.input" "$work/archives/bones-g1.tar.gz"
raw="$work/archives/bones-g1.tar.gz"
# The selected reader checks the complete raw SHA-256 on the same open file
# descriptor it uses for scanning. Do not make a separate 23.5 GB checksum pass.

phase=choose_motions
g1="$work/code/mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
keep="$work/archives/selection.input"
keep_sha=$8
if [[ "$6" == --accept-approximate ]]; then
  "$python" "$work/code/training/select_bones_clips.py" "$raw" "$work/selection" \
    --g1-xml "$g1" --sonic-source "$keep" --sha256 "$3" --workers "$workers" --accept-approximate
  keep="$work/selection/keep.txt"
  keep_sha=$(file_hash "$keep")
fi

phase=pack_and_verify
packed="$work/packed"
"$python" "$work/code/training/pack_bones.py" "$raw" "$packed" \
  --g1-xml "$g1" --sha256 "$3" --workers "$workers" --keep-list "$keep" \
  --keep-list-sha256 "$keep_sha" --selection-id "$selection_id"
rm -- "$raw"

PYTHONPATH="$work/code/training" "$python" - "$work" "$3" "$8" "$revision" "$code_sha" "$environment_sha" "$keep_sha" "$5" <<'PY'
import json
import os
import sys
from pathlib import Path

from air_launch import REQUIRED_MOTIONS, read_motion_manifest, verify_motion_arrays

root = Path(sys.argv[1])
packed = root / "packed"
if {entry.name for entry in packed.iterdir()} != REQUIRED_MOTIONS or not all(
    (packed / name).is_file() and not (packed / name).is_symlink() for name in REQUIRED_MOTIONS
):
    raise RuntimeError("Packed output file set does not match the pinned training loader")
manifest = read_motion_manifest(packed, "full", sys.argv[2])
verify_motion_arrays(packed, manifest)
if manifest["allowlist_sha256"] != sys.argv[7]:
    raise RuntimeError("Packed output is not bound to the selected allowlist")
selection_summary = root / "selection/summary.json"
selector = None
if selection_summary.exists():
    selector = json.loads(selection_summary.read_bytes())
    if not (
        selector["approximate"] is True and selector["id"] == manifest["selection_id"]
        and selector["dataset_sha256"] == sys.argv[2]
        and selector["allowlist_sha256"] == manifest["allowlist_sha256"]
        and selector["counts"]["selected"] == manifest["clips"]
    ):
        raise RuntimeError("Approximate selection summary and packed motion manifest disagree")
result = {
    "schema_version": 1,
    "job_id": os.environ["SLURM_JOB_ID"],
    "cohort_build_id": sys.argv[8],
    "dataset_sha256": sys.argv[2],
    "selection_input_sha256": sys.argv[3],
    "reproduction_commit": sys.argv[4],
    "code_sha256": sys.argv[5],
    "environment_sha256": sys.argv[6],
    "motion_manifest": manifest,
    "approximate_selector": selector,
}
(root / "publication/preparation.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
PY

phase=archive
available=$(free_bytes "$work")
packed_bytes=$(du --apparent-size -sb -- "$packed" | cut -f 1)
(( available >= packed_bytes + packed_bytes / 50 + 1024**3 )) ||
  fail 'Insufficient physical scratch space for a second packed copy during compression.'
tar --sort=name -C "$packed" -I 'gzip -1' -cf "$work/publication/motions.tar.gz" .
gzip -t -- "$work/publication/motions.tar.gz"
cp -- "$packed/motion-manifest.json" "$work/publication/"
printf 'job_id=%s\nexit_code=0\nphase=completed\ncohort_build_id=%s\n' \
  "$SLURM_JOB_ID" "$5" > "$work/publication/job-status.txt"
(
  cd -- "$work/publication"
  sha256sum motions.tar.gz motion-manifest.json preparation.json job-status.txt > SHA256SUMS
)

phase=stage_out
archive_bytes=$(stat -c '%s' -- "$work/publication/motions.tar.gz")
available=$(free_bytes "$publish_root")
(( available >= archive_bytes + 128 * 1024**2 )) ||
  fail 'Insufficient free space at the approved publication destination.'
partial=$(mktemp -d -- "$publish_root/.$5.$SLURM_JOB_ID.partial.XXXXXXXX")
cp -- "$work/publication/"* "$partial/"
(cd -- "$partial"; sha256sum -c --status SHA256SUMS)
printf 'Packed motions and checksums verified. Check Slurm terminal state separately.\n' > "$partial/SUCCESS"
sync -d "$partial/"*
sync -- "$partial"
mv -nT -- "$partial" "$dest"
[[ ! -e "$partial" ]] || fail 'Cohort destination already exists; refusing to overwrite it.'
partial=''
sync -- "$publish_root"
printf 'Published motion cohort: %s\nPacked archive SHA-256: %s\n' \
  "$dest" "$(awk '$2 == "motions.tar.gz" { print $1 }' "$work/publication/SHA256SUMS")"
