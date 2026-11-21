#!/bin/bash
#SBATCH --job-name=supermimic-full
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --tmp=160G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@900

# Submit on gpu01 via lab-submit. Supply absolute shared Slurm log paths.
# Usage: run_full.sbatch PACKAGE_DIR PACKED_MOTIONS_TGZ MOTION_SHA256
#                        BONES_SHA256 PRIVATE_RUNS_ROOT RUN_ID
#                        [full|benchmark|smoke] [fresh|resume]
# SC2317: these functions are called by traps; embedded Bash expands at runtime.
# shellcheck disable=SC2317,SC2016
set -euo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV BASH_ENV ENV LD_PRELOAD
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
export WANDB_MODE=disabled PYTHONNOUSERSITE=1 SLURM_EXPORT_ENV=ALL
for variable in ${!CONDA_@} ${!_CONDA_@}; do unset "$variable"; done
unset _CE_CONDA _CE_M

clean_lib_path=
IFS=: read -r -a inherited_lib_paths <<< "${LD_LIBRARY_PATH:-}"
for lib_path in "${inherited_lib_paths[@]}"; do
  case "$lib_path" in
    /usr/lib/*|/usr/local/cuda*/lib64|/lib/*)
      [[ ! -d "$lib_path" ]] || clean_lib_path="${clean_lib_path:+$clean_lib_path:}$lib_path" ;;
  esac
done
if [[ -n "$clean_lib_path" ]]; then export LD_LIBRARY_PATH=$clean_lib_path; else unset LD_LIBRARY_PATH; fi
if [[ -r /etc/profile.d/nccl-cluster.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/nccl-cluster.sh
fi

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# -ge 6 && $# -le 8 ]] || fail 'Expected 6–8 positional arguments; see script usage.'
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail 'Run this script inside a Slurm batch allocation.'
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]] || fail 'Expected one allocated node.'
[[ "${LOCAL_SCRATCH:-}" == "/scratch/jobs/$SLURM_JOB_ID" ]] || fail 'Unexpected LOCAL_SCRATCH.'
[[ -d "$LOCAL_SCRATCH" && -w "$LOCAL_SCRATCH" && ! -L "$LOCAL_SCRATCH" ]] || fail 'Allocated scratch unavailable.'
[[ "$3" =~ ^[0-9a-f]{64}$ && "$4" =~ ^[0-9a-f]{64}$ ]] || fail 'Expected two SHA-256 digests.'
[[ "$6" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$ ]] || fail 'Invalid run ID.'
mode=${7:-full}
resume=${8:-fresh}
case "$mode" in full|benchmark|smoke) ;; *) fail 'Invalid recipe.' ;; esac
case "$resume" in fresh|resume) ;; *) fail 'Invalid continuation mode.' ;; esac

runs_root=$(realpath -e -- "$5")
[[ -d "$runs_root" && $(stat -c '%a:%u' -- "$runs_root") == "700:$UID" ]] ||
  fail 'PRIVATE_RUNS_ROOT must be an approved owner-only job-visible directory.'
run_dir="$runs_root/$6"
if [[ "$resume" == fresh ]]; then
  mkdir -- "$run_dir" || fail 'Run ID already exists; choose a new ID or use resume.'
else
  [[ -d "$run_dir/checkpoints" && ! -L "$run_dir" ]] || fail 'No private checkpoint directory to resume.'
fi
[[ $(stat -c '%a:%u' -- "$run_dir") == "700:$UID" ]] || fail 'Run directory must be owner-only.'
exec {lock_fd}>>"$run_dir/lock"
flock -n "$lock_fd" || fail 'Another allocation owns the run lock.'
mkdir -p -- "$run_dir/attempts"
attempt_dir="$run_dir/attempts/$SLURM_JOB_ID"
[[ ! -e "$attempt_dir" && ! -L "$attempt_dir" ]] || fail 'Attempt was already published; do not requeue in place.'
if [[ "$resume" == resume ]] && find "$run_dir/attempts" -mindepth 2 -maxdepth 2 -name SUCCESS -print -quit | read -r; then
  fail 'The run already has a completed attempt.'
fi

work="$LOCAL_SCRATCH/supermimic"
mkdir -p -- "$work"/{archives,outputs,run,env,code,motions,tmp,cache}
export SM_RUN_ID=$6 SM_RUN_DIR=$run_dir SM_RESUME=$resume
export SM_OUTPUT_DIR="$work/outputs" SM_WORK_DIR="$work/run"
export TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache"
export HF_HOME="$work/cache/hf" HF_HUB_OFFLINE=1
export TORCH_HOME="$work/cache/torch" TORCH_EXTENSIONS_DIR="$work/cache/torch-extensions"
export TRITON_CACHE_DIR="$work/cache/triton" TORCHINDUCTOR_CACHE_DIR="$work/cache/torchinductor"
export CUDA_CACHE_PATH="$work/cache/cuda" MPLCONFIGDIR="$work/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$work/cache/pycache" PYTHON_EGG_CACHE="$work/cache/python-eggs"
export UV_CACHE_DIR="$work/cache/uv" PIP_CACHE_DIR="$work/cache/pip"
export WANDB_DIR="$work/outputs" WANDB_CACHE_DIR="$work/cache/wandb"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8
cd -- "$work"
ulimit -c 0

step_file="$work/training.step-id"
child_pid=
stop_code=0
launching=0
reason=setup_error

wait_for_exit() {
  local deadline=$((SECONDS + $1))
  while kill -0 "$child_pid" 2>/dev/null; do
    (( SECONDS < deadline )) || return 1
    sleep 1
  done
  wait "$child_pid" 2>/dev/null || true
}

stop_training() {
  [[ -n "$child_pid" ]] || return 0
  if ! kill -0 "$child_pid" 2>/dev/null; then
    wait "$child_pid" 2>/dev/null || true
    return 0
  fi
  local step_id='' deadline=$((SECONDS + 10))
  while [[ ! -s "$step_file" ]] && kill -0 "$child_pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 1
  done
  if [[ -s "$step_file" ]]; then read -r step_id < "$step_file"; fi
  if [[ "$step_id" =~ ^[0-9]+$ ]]; then
    timeout --kill-after=2s 10s /usr/bin/scancel --signal=TERM "$SLURM_JOB_ID.$step_id" || true
  else
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
  wait_for_exit 600 && return 0
  if [[ "$step_id" =~ ^[0-9]+$ ]]; then
    timeout --kill-after=2s 10s /usr/bin/scancel --signal=KILL "$SLURM_JOB_ID.$step_id" || true
    wait_for_exit 15 && return 0
  fi
  return 1
}

on_signal() {
  stop_code=$1
  reason=$2
  if (( launching == 0 )); then exit "$stop_code"; fi
}
trap 'on_signal 124 walltime_warning' USR1
trap 'on_signal 143 terminated' TERM
trap 'on_signal 130 interrupted' INT

finish() {
  local rc=$? last_iteration='' latest=''
  trap - EXIT
  trap '' USR1 TERM INT
  set +e
  (( stop_code == 0 )) || rc=$stop_code
  if ! stop_training; then
    printf 'ERROR: Training step still live; refusing stage-out of mutable logs.\n' >&2
    exit 1
  fi
  if (( rc == 0 )); then
    if timeout --kill-after=5s 180s /usr/bin/python3 -E "$work/code/training/air_launch.py" \
      verify-final "$run_dir" "$SM_OUTPUT_DIR" "$SLURM_JOB_ID"; then
      reason=completed
    else
      reason=verification_failed
      rc=1
    fi
  fi
  if [[ -L "$run_dir/checkpoints/latest" ]]; then
    latest=$(readlink -- "$run_dir/checkpoints/latest")
    if [[ "$latest" =~ ^step-([0-9]{9})$ ]]; then last_iteration=$((10#${BASH_REMATCH[1]})); fi
  fi
  if ! printf 'job_id=%s\nexit_code=%s\nreason=%s\nrun_id=%s\nmode=%s\nlast_published_iteration=%s\n' \
    "$SLURM_JOB_ID" "$rc" "$reason" "$SM_RUN_ID" "$mode" "$last_iteration" > "$SM_OUTPUT_DIR/job-status.txt"
  then
    printf 'ERROR: Cannot write job status on scratch.\n' >&2
    exit 1
  fi
  if ! timeout --kill-after=5s 200s /bin/bash --noprofile --norc -c '
    set -euo pipefail
    scratch_archive=$1; output=$2; attempt=$3; code=$4
    partial=$(mktemp -d "$(dirname -- "$attempt")/.$(basename -- "$attempt").XXXXXXXX")
    trap '\''rm -rf -- "$partial"'\'' EXIT
    tar -C "$output" -czf "$scratch_archive" .
    cp -- "$scratch_archive" "$partial/outputs.tar.gz"
    cp -- "$output/job-status.txt" "$partial/job-status.txt"
    [[ $(sha256sum -- "$scratch_archive" | cut -d " " -f 1) == $(sha256sum -- "$partial/outputs.tar.gz" | cut -d " " -f 1) ]]
    (cd -- "$partial"; sha256sum outputs.tar.gz job-status.txt > SHA256SUMS)
    if [[ $code == 0 ]]; then
      printf "Verified target checkpoint and job outputs.\n" > "$partial/SUCCESS"
    fi
    sync -d "$partial/"*
    sync -- "$partial"
    mv -nT -- "$partial" "$attempt"
    [[ ! -e "$partial" ]]
    sync -- "$(dirname -- "$attempt")"
  ' _ "$work/outputs.tar.gz" "$SM_OUTPUT_DIR" "$attempt_dir" "$rc"; then
    printf 'ERROR: Attempt stage-out failed; examine scheduler logs and the last durable checkpoint.\n' >&2
    exit 1
  fi
  printf 'Published attempt: %s (exit_code=%s)\n' "$attempt_dir" "$rc"
  exit "$rc"
}
trap finish EXIT
printf 'Run directory: %s\nJob ID: %s\nRecipe: %s (%s)\n' "$run_dir" "$SLURM_JOB_ID" "$mode" "$resume"

package_dir=$(realpath -e -- "$1")
motion_archive=$(realpath -e -- "$2")
[[ -d "$package_dir" && -f "$motion_archive" ]] || fail 'Missing staged package or motion archive.'
for member in environment.tar.gz code.tar.gz SHA256SUMS; do
  cp -- "$package_dir/$member" "$work/archives/$member"
done
(
  cd -- "$work/archives"
  sha256sum environment.tar.gz code.tar.gz | LC_ALL=C sort > actual
  LC_ALL=C sort SHA256SUMS > expected
  cmp -s actual expected || fail 'Environment or code package checksum mismatch.'
)
SM_CODE_SHA256=$(sha256sum -- "$work/archives/code.tar.gz" | cut -d ' ' -f 1)
SM_ENV_SHA256=$(sha256sum -- "$work/archives/environment.tar.gz" | cut -d ' ' -f 1)
export SM_CODE_SHA256 SM_ENV_SHA256
tar --no-same-owner --no-same-permissions -xzf "$work/archives/environment.tar.gz" -C "$work/env"
tar --no-same-owner --no-same-permissions -xzf "$work/archives/code.tar.gz" -C "$work/code"
rm -- "$work/archives/environment.tar.gz" "$work/archives/code.tar.gz"

cp -- "$motion_archive" "$work/archives/motions.tar.gz"
printf '%s  %s\n' "$3" "$work/archives/motions.tar.gz" | sha256sum -c --status ||
  fail 'Packed motion archive checksum mismatch on scratch.'
tar --no-same-owner --no-same-permissions -xzf "$work/archives/motions.tar.gz" -C "$work/motions"
rm -- "$work/archives/motions.tar.gz"

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
git -C "$work/code/supermimic" apply --check "$work/code/training/air_runner.patch"
git -C "$work/code/supermimic" apply "$work/code/training/air_runner.patch"
install -m 600 -- "$work/code/training/air_checkpoint.py" \
  "$work/code/supermimic/src/supermimic/runners/air_checkpoint.py"

cd -- "$SM_WORK_DIR"
reason=training_error
launching=1
/usr/bin/srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 /bin/bash --noprofile --norc -c '
  set -euo pipefail
  printf "%s\n" "$SLURM_STEP_ID" > "$1"
  shift
  exec "$@"
' _ "$step_file" "$work/env/bin/python" -u "$work/code/training/air_launch.py" \
  train "$mode" "$work/motions" "$work/code" "$4" "$3" > "$SM_OUTPUT_DIR/train.log" 2>&1 &
child_pid=$!
launching=0
(( stop_code == 0 )) || exit "$stop_code"
if wait "$child_pid"; then rc=0; else rc=$?; fi
child_pid=
exit "$rc"
