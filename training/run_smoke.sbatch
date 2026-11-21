#!/usr/bin/env bash
#SBATCH --job-name=supermimic-smoke
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --tmp=80G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@120

set -euo pipefail
work_root=${1:?WORK_ROOT}
bones_archive=${2:?BONES_G1_ARCHIVE}
[[ -d "$work_root/supermimic" && -f "$bones_archive" ]] || { echo "Missing work root or BONES archive" >&2; exit 2; }
[[ "$(git -C "$work_root/supermimic" rev-parse HEAD)" == 5fb92297ab4fe3d86629d9f0c635334791bb0d5f ]] || { echo "Unexpected SuperMimic commit" >&2; exit 2; }

scratch_root=${LOCAL_SCRATCH:?Slurm did not provide LOCAL_SCRATCH}
mkdir -p "$scratch_root/input" "$scratch_root/run"
cp --reflink=auto "$bones_archive" "$scratch_root/input/g1.tar.gz"
sha256sum "$bones_archive" > "$scratch_root/input/g1.tar.gz.sha256"
tar -xzf "$scratch_root/input/g1.tar.gz" -C "$scratch_root/input"
motion_dir="$scratch_root/input/g1"
[[ -d "$motion_dir" ]] || { echo "BONES archive did not contain g1/" >&2; exit 3; }

trap 'status=$?; if [[ -d "$scratch_root/run" ]]; then tar -czf "$work_root/smoke-${SLURM_JOB_ID:-local}.tar.gz" -C "$scratch_root/run" . || true; fi; exit "$status"' USR1 TERM EXIT
pushd "$scratch_root/run" >/dev/null
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}" uv --directory "$work_root/supermimic" run \
  "$work_root/supermimic/src/supermimic/scripts/train.py" SuperMimic-GenCoord-G1 \
  --motion-file "$motion_dir/csv/**/*.csv" \
  --num-envs 256 \
  --runner.max_iterations 200 \
  --runner.save_interval 50 \
  --runner.run_name "air-smoke-${SLURM_JOB_ID:-local}"
popd >/dev/null
