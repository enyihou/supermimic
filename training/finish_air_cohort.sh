#!/bin/bash
# Detached gpu01 worker: audit the approximate cohort, then submit its benchmark.
set -euo pipefail
umask 077
[[ $# == 4 ]] || { echo 'Usage: finish_air_cohort.sh SHARED_PROJECT COHORT_ID JOB_ID BENCHMARK_RUN_ID' >&2; exit 2; }
shared=$1 cohort=$2 job=$3 benchmark=$4
account=$(id -un)
canonical="/public/$account/supermimic-archive"
state="$canonical/manifests/$cohort"
bench_state="$canonical/manifests/$benchmark"
ops=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
private() {
  local mode
  [[ -d "$1" && -O "$1" && ! -L "$1" ]] || return 1
  mode=$(stat -c '%a' -- "$1")
  (( (8#$mode & 0777) == 0700 ))
}
[[ "$shared" == "/data/$account/"* && $(realpath -e -- "$shared") == "$shared" &&
   "$cohort" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ && "$benchmark" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ &&
   "$benchmark" != "$cohort" && "$job" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid project or phase identifiers.'
for dir in "$shared" "$shared"/{logs,motions,runs,packages/f17e907,control/headless-v1} \
  "$canonical" "$canonical/manifests" "$state"; do private "$dir" || fail "Private directory expected: $dir"; done
[[ $(<"$state/job-id") == "$job" ]] || fail 'Cohort ID does not match the saved job.'
exec 9>"$state/finish.lock"
flock -n 9
record="$state/$job.slurm.json"

# Reuse the original detailed watcher if it is still live. If it died and the
# controller can still provide the exact job record, do not resubmit the job.
while [[ ! -s "$record" ]]; do
  pid=$(<"$state/watcher.pid")
  args=''
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then args=$(ps -p "$pid" -o args= || true); fi
  if [[ "$args" != *"/watch_slurm.py $job $record "* ]]; then
    if [[ ! -s "$record" ]]; then
      python3 -B -u "$ops/watch_slurm.py" "$job" "$record" --interval 10
    fi
    break
  fi
  sleep 12
done

for stream in out err; do
  src="$shared/logs/$job.$stream"
  dst="$state/$job.$stream"
  if [[ -f "$src" && -O "$src" && ! -L "$src" ]]; then
    if [[ ! -e "$dst" && ! -L "$dst" ]]; then
      temporary=$(mktemp "$state/.$job.$stream.XXXXXXXX")
      install -m 600 -- "$src" "$temporary"
      cmp -- "$src" "$temporary"
      sync -d -- "$temporary"
      mv -nT -- "$temporary" "$dst"
      [[ ! -e "$temporary" ]]
      sync -- "$state"
    fi
    [[ -f "$dst" && -O "$dst" && ! -L "$dst" && $(stat -c '%a' -- "$dst") == 600 ]]
    cmp -- "$src" "$dst"
  fi
done

audit="$state/cohort-audit.json"
if [[ ! -s "$audit" ]]; then
  python3 -B "$ops/audit_air_cohort.py" "$shared/motions/$cohort" "$job" --slurm-record "$record" --output "$audit"
fi
[[ -f "$audit" && -O "$audit" && ! -L "$audit" && $(stat -c '%a' -- "$audit") == 600 ]]
readarray -t hashes < <(python3 -B -c '
import json, re, sys
from pathlib import Path
row = json.loads(Path(sys.argv[1]).read_bytes())
if row["phase"] != "cohort" or row["allocation"]["job_id"] != sys.argv[2] or row["motion_manifest"]["approximate"] is not True:
    raise ValueError("The saved audit is not for this approximate cohort")
for name, size in (("dataset_sha256", 64), ("motion_archive_sha256", 64), ("code_sha256", 64),
                   ("environment_sha256", 64), ("reproduction_commit", 40)):
    value = row[name]
    if not re.fullmatch(r"[0-9a-f]{%d}" % size, value):
        raise ValueError("Malformed benchmark input fingerprint")
    print(value)
' "$audit" "$job")
[[ ${#hashes[@]} == 5 ]]
printf 'Cohort %s passed the audit and independent backup.\n' "$job"

if [[ ! -e "$bench_state" && ! -L "$bench_state" ]]; then mkdir -m 700 -- "$bench_state"; fi
private "$bench_state" || fail 'Benchmark state must be owner-only.'
exec 8>"$bench_state/submit.lock"
flock -n 8
if [[ -s "$bench_state/job-id" ]]; then
  printf 'Benchmark already submitted as %s; check its saved watcher.\n' "$(<"$bench_state/job-id")"
  exit
fi
[[ ! -e "$bench_state/submit-intent" && ! -e "$shared/runs/$benchmark" ]] ||
  fail 'Existing benchmark attempt; reconcile it before another submission.'
matches=$(squeue -h -u "$account" -o '%i|%j|%T' | awk -F '|' '$2 == "supermimic-benchmark" || $2 == "supermimic-full"')
[[ -z "$matches" ]] || fail "An existing training allocation must be inspected first: $matches"
if compgen -e | grep -E '^SBATCH_' >/dev/null; then fail 'Clear inherited SBATCH overrides before submitting.'; fi
batch="$shared/control/headless-v1/run_full.sbatch"
[[ -f "$batch" && -O "$batch" && ! -L "$batch" ]]
cmp -- "$ops/run_full.sbatch" "$batch"
motion="$shared/motions/$cohort/motions.tar.gz"
audit_sha=$(sha256sum -- "$audit" | cut -d ' ' -f1)
printf 'run_id=%s\ncohort_id=%s\ncohort_job_id=%s\ncohort_audit_sha256=%s\napproximate=true\ndataset_sha256=%s\nmotion_sha256=%s\ncode_sha256=%s\nenvironment_sha256=%s\nreproduction_commit=%s\n' \
  "$benchmark" "$cohort" "$job" "$audit_sha" "${hashes[@]}" > "$bench_state/inputs.txt"
printf '%s\n' "$benchmark" > "$bench_state/submit-intent"
sync -d -- "$bench_state/inputs.txt" "$bench_state/submit-intent"
sync -- "$bench_state"
raw_job=$(env -u BASH_ENV -u ENV lab-submit --kind gpu -- --parsable --chdir="$shared" \
  --job-name=supermimic-benchmark --time=02:00:00 --comment="$benchmark" \
  --output="$shared/logs/%j.out" --error="$shared/logs/%j.err" \
  "$batch" "$shared/packages/f17e907" "$motion" "${hashes[1]}" "${hashes[0]}" "$shared/runs" "$benchmark" benchmark fresh)
[[ "$raw_job" =~ ^([1-9][0-9]*)(\;[^[:space:]]+)?$ ]] || fail "Unexpected Slurm reply; reconcile before retry: $raw_job"
bench_job=${BASH_REMATCH[1]}
printf '%s\n' "$bench_job" > "$bench_state/job-id"
nohup /bin/bash "$ops/finish_air_smoke.sh" "$shared" "$benchmark" "$bench_job" "$bench_state" benchmark \
  8>&- 9>&- </dev/null >>"$bench_state/watcher.log" 2>&1 &
printf '%s\n' "$!" > "$bench_state/watcher.pid"
sync -d -- "$bench_state/job-id" "$bench_state/watcher.pid"
sync -- "$bench_state"
printf 'One-GPU benchmark job %s, run %s.\n' "$bench_job" "$benchmark"
