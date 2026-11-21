#!/bin/bash
#SBATCH --job-name=supermimic-eval
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=45G
#SBATCH --time=00:25:00
#SBATCH --no-requeue

# Run on gpu01 with lab-submit; set absolute private output/error paths.
# Usage: run_eval.sbatch PACKAGE_DIR CHECKPOINT_DIR LAFAN1_TGZ LAFAN1_SHA256
#                       BONES_SHA256 PRIVATE_EVALUATIONS_ROOT
set -euo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV BASH_ENV ENV LD_PRELOAD
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 SLURM_EXPORT_ENV=ALL
for variable in ${!CONDA_@} ${!_CONDA_@}; do unset "$variable"; done
unset _CE_CONDA _CE_M

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# == 6 ]] || fail 'Expected six arguments; see script usage.'
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ && "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]] ||
  fail 'Expected a single-node Slurm batch allocation.'
[[ "${LOCAL_SCRATCH:-}" == "/scratch/jobs/$SLURM_JOB_ID" && -d "$LOCAL_SCRATCH" && ! -L "$LOCAL_SCRATCH" ]] ||
  fail 'Allocated scratch unavailable.'
[[ "$4" =~ ^[0-9a-f]{64}$ && "$5" =~ ^[0-9a-f]{64}$ ]] || fail 'Expected two SHA-256 digests.'
package_dir=$(realpath -e -- "$1")
checkpoint_dir=$(realpath -e -- "$2")
suite=$(realpath -e -- "$3")
results=$(realpath -e -- "$6")
[[ -d "$package_dir" && -d "$checkpoint_dir" && -f "$suite" ]] || fail 'Missing package or input.'
[[ -d "$results" && $(stat -c '%a:%u' -- "$results") == "700:$UID" ]] ||
  fail 'Evaluation destination must be approved and owner-only.'
attempt="$results/$SLURM_JOB_ID"
[[ ! -e "$attempt" && ! -L "$attempt" ]] || fail 'Evaluation allocation was already published.'
work="$LOCAL_SCRATCH/evaluation"
mkdir -p -- "$work"/{archives,outputs,checkpoint,env,code,tmp,cache}
export TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache" HF_HOME="$work/cache/hf"
export TORCH_HOME="$work/cache/torch" TORCH_EXTENSIONS_DIR="$work/cache/torch-extensions"
export TRITON_CACHE_DIR="$work/cache/triton" TORCHINDUCTOR_CACHE_DIR="$work/cache/torchinductor"
export CUDA_CACHE_PATH="$work/cache/cuda" MPLCONFIGDIR="$work/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$work/cache/pycache" WANDB_DIR="$work/outputs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4
cd -- "$work"
ulimit -c 0

finish() {
  local rc=$? reason=failed partial
  trap - EXIT
  [[ $rc != 0 || -s "$work/outputs/metrics.json" ]] || rc=1
  [[ $rc != 0 ]] || reason=completed
  printf 'job_id=%s\nexit_code=%s\nreason=%s\n' "$SLURM_JOB_ID" "$rc" "$reason" > "$work/outputs/job-status.txt"
  tar -C "$work/outputs" -czf "$work/outputs.tar.gz" .
  partial=$(mktemp -d "$results/.$SLURM_JOB_ID.XXXXXXXX")
  trap 'rm -rf -- "$partial"' EXIT
  cp -- "$work/outputs.tar.gz" "$work/outputs/job-status.txt" "$partial/"
  [[ $(sha256sum -- "$work/outputs.tar.gz" | cut -d ' ' -f 1) == $(sha256sum -- "$partial/outputs.tar.gz" | cut -d ' ' -f 1) ]] ||
    fail 'Evaluation stage-out checksum mismatch.'
  (cd -- "$partial"; sha256sum outputs.tar.gz job-status.txt > SHA256SUMS)
  if [[ $rc == 0 ]]; then printf 'Evaluation metrics published.\n' > "$partial/SUCCESS"; fi
  sync -f "$partial"
  mv -nT -- "$partial" "$attempt"
  [[ ! -e "$partial" ]] || fail 'Evaluation publication destination already exists.'
  sync -f "$results"
  printf 'Published evaluation %s (exit_code=%s)\n' "$SLURM_JOB_ID" "$rc"
  exit "$rc"
}
trap finish EXIT

for member in environment.tar.gz code.tar.gz SHA256SUMS; do
  cp -- "$package_dir/$member" "$work/archives/$member"
done
(
  cd -- "$work/archives"
  sha256sum environment.tar.gz code.tar.gz | LC_ALL=C sort > actual
  LC_ALL=C sort SHA256SUMS > expected
  cmp -s actual expected || fail 'Environment or code package checksum mismatch.'
)
code_sha256=$(sha256sum -- "$work/archives/code.tar.gz" | cut -d ' ' -f 1)
environment_sha256=$(sha256sum -- "$work/archives/environment.tar.gz" | cut -d ' ' -f 1)
tar --no-same-owner --no-same-permissions -xzf "$work/archives/environment.tar.gz" -C "$work/env"
tar --no-same-owner --no-same-permissions -xzf "$work/archives/code.tar.gz" -C "$work/code"
rm -- "$work/archives/environment.tar.gz" "$work/archives/code.tar.gz"
printf '%s\n' 'supermimic=5fb92297ab4fe3d86629d9f0c635334791bb0d5f' 'mjlab_research=379c55e88b1dec73f93943e524a879b2ebbf45c5' |
  cmp -s "$work/code/PINS.txt" - || fail 'Unexpected evaluation source pins.'
for member in checkpoint.tar.gz checkpoint.json SHA256SUMS; do
  cp -- "$checkpoint_dir/$member" "$work/checkpoint/$member"
done
cp -- "$suite" "$work/archives/lafan1.tar.gz"
printf '%s  %s\n' "$4" "$work/archives/lafan1.tar.gz" | sha256sum -c --status ||
  fail 'Held-out suite checksum mismatch on scratch.'
if [[ -x "$work/env/bin/conda-unpack" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$work/env/bin/activate"
  set -u
  "$work/env/bin/conda-unpack"
fi
[[ -x "$work/env/bin/python" ]] || fail 'Staged Python executable unavailable.'
export PATH="$work/env/bin:$PATH"
export PYTHONPATH="$work/code/supermimic/src:$work/code/mjlab_research/src"
/usr/bin/srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 "$work/env/bin/python" -u \
  "$work/code/training/evaluate_heldout.py" run \
  --checkpoint "$work/checkpoint" --suite "$work/archives/lafan1.tar.gz" --suite-sha256 "$4" \
  --training-dataset-sha256 "$5" --code-sha256 "$code_sha256" --environment-sha256 "$environment_sha256" \
  --scratch "$work/tmp" --output "$work/outputs/metrics.json" > "$work/outputs/evaluation.log" 2>&1
