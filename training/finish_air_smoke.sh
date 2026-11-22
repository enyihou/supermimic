#!/bin/bash
# Run detached on gpu01. Restart with the same arguments after confirming the worker stopped.
set -euo pipefail
umask 077
[[ $# -ge 4 && $# -le 6 ]] || {
  echo 'Usage: finish_air_smoke.sh SHARED_PROJECT RUN_ID JOB_ID PRIVATE_STATE [smoke|benchmark|full [BENCHMARK_AUDIT]]' >&2; exit 2;
}
shared=$1 run_id=$2 job=$3 state=$4
phase=${5:-smoke}
benchmark_args=()
case "$phase" in
  smoke) envs=256; steps=200; copy_steps=(--step 200) ;;
  benchmark) envs=4096; steps=1200; copy_steps=(--step 600 --step 1200) ;;
  full)
    [[ $# == 6 && "$6" == /* && -f "$6" && -O "$6" && ! -L "$6" ]] || {
      echo 'Full phase requires an existing benchmark audit from an absolute path.' >&2; exit 2;
    }
    envs=4096; steps=320000; copy_steps=(); benchmark_args=(--benchmark-audit "$6") ;;
  *) echo 'Expected smoke, benchmark, or full phase' >&2; exit 2 ;;
esac
[[ "$phase" == full || $# -le 5 ]]
code=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ "$shared" == /* && "$state" == /* && "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ && "$job" =~ ^[1-9][0-9]*$ ]]
[[ -d "$state" && -O "$state" && ! -L "$state" ]]
mode=$(stat -c '%a' -- "$state"); (( (8#$mode & 0777) == 0700 ))
[[ $(<"$state/job-id") == "$job" ]]
exec 9>"$state/finish.lock"
flock -n 9

record="$state/$job.slurm.json"
if [[ ! -f "$record" ]]; then
  python3 -B -u "$code/watch_slurm.py" "$job" "$record" --interval 10
fi

copy_log() (
  local stream=$1 src dest tmp
  src="$shared/logs/$job.$stream"
  dest="$state/$job.$stream"
  [[ -f "$src" && -O "$src" && ! -L "$src" ]]
  if [[ ! -e "$dest" && ! -L "$dest" ]]; then
    tmp=$(mktemp "$state/.$job.$stream.XXXXXXXX")
    trap 'rm -f -- "$tmp"' EXIT
    cp -- "$src" "$tmp"
    cmp -- "$src" "$tmp"
    sync -d -- "$tmp"
    mv -nT -- "$tmp" "$dest"
    sync -- "$state"
  fi
  [[ -f "$dest" && -O "$dest" && ! -L "$dest" && $(stat -c '%a' -- "$dest") == 600 ]]
  cmp -- "$src" "$dest"
)
for stream in out err; do
  if [[ -f "$shared/logs/$job.$stream" ]]; then copy_log "$stream"; fi
done

outcome=$(python3 -B -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
if record["job_id"] != sys.argv[2]:
    raise ValueError("terminal record job ID mismatch")
print(record["state"], record["exit_code"])
' "$record" "$job")
if [[ "$outcome" != 'COMPLETED 0:0' ]]; then
  if [[ -d "$shared/runs/$run_id/attempts/$job" ]]; then
    attempt="$shared/runs/$run_id/attempts/$job"
    if [[ "$phase" == full && "$outcome" == 'FAILED 124:0' ]]; then
      last=$(awk -F= '$1 == "last_published_iteration" {print $2}' "$attempt/job-status.txt")
      if [[ "$last" =~ ^[1-9][0-9]{0,5}$ ]] && (( last < steps )); then
        for (( step=32000; step<last; step+=32000 )); do copy_steps+=(--step "$step"); done
        python3 -B "$code/backup_air_attempt.py" "$shared/runs" "$run_id" "$job" "${copy_steps[@]}" --step "$last"
        python3 -B "$code/verify_air_resume.py" "$shared/runs" "$run_id" "$job" \
          --slurm-record "$record" --benchmark "$6"
        printf 'Full attempt %s: planned stop verified and backed up; no job submitted.\n' "$job"
        exit 0
      fi
    fi
    python3 -B "$code/backup_air_attempt.py" "$shared/runs" "$run_id" "$job"
  fi
  printf '%s %s: %s\n' "$phase" "$job" "$outcome" >&2
  exit 1
fi

read_input() {
  awk -F= -v key="$1" '$1 == key {print $2}' "$state/inputs.txt"
}
if [[ "$phase" == full ]]; then
  for (( step=32000; step<=steps; step+=32000 )); do copy_steps+=(--step "$step"); done
fi
python3 -B "$code/backup_air_attempt.py" "$shared/runs" "$run_id" "$job" "${copy_steps[@]}"
python3 -B "$code/audit_official.py" verify --run "$shared/runs/$run_id" --phase "$phase" \
  --job-id "$job" --slurm-record "$record" --dataset-sha256 "$(read_input dataset_sha256)" \
  --motion-sha256 "$(read_input motion_sha256)" --code-sha256 "$(read_input code_sha256)" \
  --environment-sha256 "$(read_input environment_sha256)" \
  --reproduction-commit "$(read_input reproduction_commit)" --num-envs "$envs" --iterations "$steps" \
  --output "$state/$phase-audit.json" "${benchmark_args[@]}"
printf '%s %s audited and backed up; inspect the public-safe audit before publishing.\n' "$phase" "$job"
