#!/usr/bin/env bash
# Build a private smoke input on gpu01. This is CPU packaging, not a Slurm submission.
set -euo pipefail
umask 077
[[ $# == 6 ]] || {
  echo "Usage: bash $0 RUNTIME DIR_G1_TAR BONES_SHA CODE_TAR CODE_SHA NEW_OUTPUT_DIRECTORY" >&2
  exit 2
}
runtime=$(realpath -e -- "$1")
bones=$(realpath -e -- "$2")
code=$(realpath -e -- "$4")
dest=$(realpath -m -- "$6")
[[ -z "${SLURM_JOB_ID:-}" && $(uname -m) == x86_64 ]]
[[ "$3" =~ ^[a-f0-9]{64}$ && "$5" =~ ^[a-f0-9]{64}$ ]]
[[ -x "$runtime/bin/python" && -f "$bones" && -f "$code" && ! -e "$dest" ]]
[[ "$dest" == "/home/$USER/"* || "$dest" == "/public/$USER/"* ]]
parent=$(dirname -- "$dest")
[[ "$(findmnt -n -o FSTYPE -T "$parent")" == xfs && -O "$parent" ]]
mode=$(stat -c %a "$parent")
(( (8#$mode & 077) == 0 ))
mount_root=/home
[[ "$dest" != /public/* ]] || mount_root=/public
used_kib=$(xfs_quota -c 'quota -ub' "$mount_root" | awk '
  $1 ~ /^\/dev\// && $2 ~ /^[0-9]+$/ { used = $2 }
  $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { used = $1 }
  END { print used }')
available=$(df -B1 --output=avail "$parent" | awk 'NR==2 {print $1}')
[[ "$used_kib" =~ ^[0-9]+$ && "$available" =~ ^[0-9]+$ ]]
(( available >= 3000000000 && used_kib * 1024 + 3000000000 <= 200000000000 ))

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH WANDB_API_KEY PYTHONHOME
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf '%s  %s\n' "$5" "$code" | sha256sum -c --status
build=$(mktemp -d -- "$parent/.smoke-preflight.XXXXXXXX")
trap 'rm -rf -- "$build"' EXIT
mkdir -- "$build"/{code,cache,published}
export XDG_CACHE_HOME="$build/cache" TMPDIR="$build"
tar -xzf "$code" -C "$build/code" --no-same-owner --no-same-permissions
cd -- "$build"
g1_xml=code/mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml
"$runtime/bin/python" code/training/pack_bones.py "$bones" "$build/motions" \
  --g1-xml "$g1_xml" --sha256 "$3" --smoke-clips 64 --workers 4
PYTHONPATH="$build/code/mjlab_research/src:$build/code/supermimic/src" "$runtime/bin/python" - <<'PY'
import json
from pathlib import Path
from types import SimpleNamespace

from mjlab.tasks.tracking.mdp.commands import MotionLoader

joints = json.loads(Path("motions/joint_names.json").read_text())
motion = MotionLoader("motions", SimpleNamespace(joint_names=joints), device="cpu")
assert len(motion.motion_names) == 64 and motion.fps == 50
assert motion.joint_pos.shape[1] == 29 and bool(motion.joint_pos.isfinite().all())
print(f"CPU loader verified 64 motions, {motion.time_step_total} frames, 29 joints.")
PY
tar -C motions -czf published/smoke-motions.tar.gz .
install -m 600 -- motions/motion-manifest.json published/motion-manifest.json
install -m 600 -- code/REPRODUCTION_COMMIT.txt published/REPRODUCTION_COMMIT.txt
(cd published; sha256sum smoke-motions.tar.gz motion-manifest.json REPRODUCTION_COMMIT.txt > SHA256SUMS)
sync -d published/*
mv -nT -- published "$dest"
[[ ! -e published ]]
printf 'Verified smoke input package: %s\n' "$dest"
