#!/usr/bin/env bash
#SBATCH --job-name=supermimic-full
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --tmp=160G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300

set -euo pipefail
work_root=${1:?WORK_ROOT}
bones_archive=${2:?BONES_G1_ARCHIVE}
archive_root=${3:?PRIVATE_ARCHIVE_ROOT}
[[ -d "$work_root/supermimic" && -f "$bones_archive" ]] || { echo "Missing work root or BONES archive" >&2; exit 2; }
[[ -d "$archive_root" && -w "$archive_root" ]] || { echo "Archive root is not a writable approved directory" >&2; exit 2; }
[[ "$(git -C "$work_root/supermimic" rev-parse HEAD)" == 5fb92297ab4fe3d86629d9f0c635334791bb0d5f ]] || { echo "Unexpected SuperMimic commit" >&2; exit 2; }

scratch_root=${LOCAL_SCRATCH:?Slurm did not provide LOCAL_SCRATCH}
run_id="${SLURM_JOB_ID:-local}-$(date -u +%Y%m%dT%H%M%SZ)"
run_root="$scratch_root/supermimic/$run_id"
backup_root="$archive_root/$run_id"
mkdir -p "$run_root/input" "$run_root/logs" "$backup_root"
cp --reflink=auto "$bones_archive" "$run_root/input/g1.tar.gz"
sha256sum "$bones_archive" | tee "$run_root/input/g1.tar.gz.sha256" "$backup_root/g1.tar.gz.sha256" >/dev/null
tar -xzf "$run_root/input/g1.tar.gz" -C "$run_root/input"
motion_dir="$run_root/input/g1"
[[ -d "$motion_dir" ]] || { echo "BONES archive did not contain g1/" >&2; exit 3; }

backup() {
  mkdir -p "$backup_root/checkpoints"
  find "$run_root/logs" -type f \( -name 'world_model.pt' -o -name 'actor.pt' -o -name 'critic.pt' -o -name '*.json' \) -print0 2>/dev/null \
    | tar --null -czf "$backup_root/checkpoints/latest.tgz" --files-from=- 2>/dev/null || true
  cp -f "$run_root/input/g1.tar.gz.sha256" "$backup_root/" 2>/dev/null || true
}
backup_loop() { while sleep 1800; do backup; done; }
backup_loop & backup_pid=$!
trap 'kill "$backup_pid" 2>/dev/null || true; backup; exit 143' USR1 TERM
trap 'kill "$backup_pid" 2>/dev/null || true; backup' EXIT

pushd "$run_root" >/dev/null
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}" uv --directory "$work_root/supermimic" run \
  "$work_root/supermimic/src/supermimic/scripts/train.py" SuperMimic-GenCoord-G1 \
  --motion-file "$motion_dir/csv/**/*.csv" \
  --num-envs 4096 \
  --runner.max_iterations 320000 \
  --runner.save_interval 1000 \
  --runner.run_name "air-full-${run_id}"
popd >/dev/null
backup
