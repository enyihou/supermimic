#!/bin/bash
#SBATCH --job-name=supermimic-smoke
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --tmp=40G
#SBATCH --time=00:55:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@900

# Submit on gpu01 with lab-submit. The shared driver is the same reviewed
# run_full.sbatch submitted for the benchmark and official run.
set -euo pipefail
[[ $# == 7 ]] || {
  echo "Usage: run_smoke.sbatch SHARED_RUN_FULL_SCRIPT PACKAGE_DIR PACKED_MOTIONS_TGZ MOTION_SHA256 BONES_SHA256 PRIVATE_RUNS_ROOT RUN_ID" >&2
  exit 2
}
driver=$1
shift
[[ "$driver" == /* && -f "$driver" && -O "$driver" && ! -L "$driver" ]] ||
  { echo 'Expected owned, absolute, job-visible run_full.sbatch path' >&2; exit 2; }
exec /bin/bash "$driver" "$@" smoke fresh
