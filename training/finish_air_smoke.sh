#!/bin/bash
# Run detached on gpu01. Restart with the same arguments after confirming the worker stopped.
set -euo pipefail
umask 077
[[ $# == 4 ]] || { echo 'Usage: finish_air_smoke.sh SHARED_PROJECT RUN_ID JOB_ID PRIVATE_STATE' >&2; exit 2; }
shared=$1 run_id=$2 job=$3 state=$4
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
    python3 -B "$code/backup_air_attempt.py" "$shared/runs" "$run_id" "$job"
  fi
  printf 'Smoke %s: %s\n' "$job" "$outcome" >&2
  exit 1
fi

read_input() {
  awk -F= -v key="$1" '$1 == key {print $2}' "$state/inputs.txt"
}
python3 -B "$code/backup_air_attempt.py" "$shared/runs" "$run_id" "$job" --step 200
python3 -B "$code/audit_official.py" verify --run "$shared/runs/$run_id" --phase smoke \
  --job-id "$job" --slurm-record "$record" --dataset-sha256 "$(read_input dataset_sha256)" \
  --motion-sha256 "$(read_input motion_sha256)" --code-sha256 "$(read_input code_sha256)" \
  --environment-sha256 "$(read_input environment_sha256)" \
  --reproduction-commit "$(read_input reproduction_commit)" --num-envs 256 --iterations 200 \
  --output "$state/smoke-audit.json"
printf 'Smoke %s audited and backed up; inspect the public-safe audit before publishing.\n' "$job"
