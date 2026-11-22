#!/bin/bash
# Run detached on gpu01, only after explicit approval of the documented cohort.
set -euo pipefail
umask 077
[[ $# == 4 && "$4" == --accept-approximate ]] || {
  echo 'Usage: submit_air_approx_cohort.sh SHARED_PROJECT COHORT_ID PRIVATE_STATE --accept-approximate' >&2
  exit 2
}
shared=$1 cohort=$2 state=$3
account=$(id -un)
canonical="/public/$account/supermimic-archive"
ops=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ceph="/data/$account"
package="$shared/packages/f17e907"
control="$shared/control/cohort-tools-v1"
source="$canonical/inputs/g1.tar.gz"
published="$shared/packages/bones-g1-2f59b20"
bones_sha=52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a
bones_bytes=23499973647
sonic_sha=b6b50d441a4876a382eb3ff33ba940c487d1647230b5ecacaa37cfcf521258da
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
private_directory() {
  local mode
  [[ -d "$1" && -O "$1" && ! -L "$1" ]] || return 1
  mode=$(stat -c '%a' -- "$1")
  (( (8#$mode & 0777) == 0700 ))
}
[[ "$cohort" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ && "$shared" == "$ceph/"* &&
   "$state" == "$canonical/manifests/$cohort" && $(realpath -e -- "$shared") == "$shared" ]] || fail 'Invalid project or state path.'
for dir in "$ceph" "$shared" "$shared"/{packages,logs,motions,runs} "$package" "$control" "$state" "$canonical"/inputs "$ops"; do
  private_directory "$dir" || fail "Expected owned private directory: $dir"
done
[[ -z "${SLURM_JOB_ID:-}" && $(findmnt -n -o FSTYPE -T "$source") == xfs && $(findmnt -n -o FSTYPE -T "$ceph") == ceph ]] ||
  fail 'Run from approved gpu01 login storage and Ceph, outside an allocation.'
exec 9>"$state/submit.lock"
flock -n 9
[[ ! -e "$state/submit-intent" && ! -e "$state/job-id" && ! -e "$shared/motions/$cohort" ]] ||
  fail 'Existing submission or publication; reconcile its exact job before any retry.'
check_queue() {
  local active
  active=$(squeue -h -u "$account" -o '%i|%j|%T' | awk -F '|' '$2 == "supermimic-prepare-motions"')
  [[ -z "$active" ]] || fail "Existing cohort job: $active"
}
check_queue
[[ -f "$source" && -O "$source" && ! -L "$source" && $(stat -c '%a %s' -- "$source") == "600 $bones_bytes" ]] ||
  fail 'Canonical BONES source does not match the verified private archive.'
printf '%s  %s\n' "$sonic_sha" "$control/sonic-filter.py" | sha256sum -c --strict
cmp -- "$ops/run_prepare_motions.sbatch" "$control/run_prepare_motions.sbatch"
printf '%s\n' '9e8e488d41a6aaed3af99c70372f42164e65505ea2cebd8d01e5090e81cf2172  environment.tar.gz' \
  '5698e6d903d27b3cd895c081cfe8a0a3069bd01d5a95c1e7c7e60c0118627f19  code.tar.gz' | cmp - "$package/SHA256SUMS"
quota=$(getfattr --only-values -n ceph.quota.max_bytes "$ceph" 2>/dev/null)
usage=$(getfattr --only-values -n ceph.dir.rbytes "$ceph" 2>/dev/null)
available=$(df -B1 --output=avail -- "$shared" | awk 'NR==2 {print $1}')
[[ "$quota" == 500000000000 && "$usage" =~ ^[0-9]+$ && "$available" =~ ^[0-9]+$ ]] || fail 'Could not establish Ceph headroom.'
(( quota - usage > bones_bytes + 60000000000 && available > bones_bytes + 60000000000 )) ||
  fail 'Insufficient Ceph space for the input and bounded packed output.'

if [[ ! -e "$published" && ! -L "$published" ]]; then
  partial=$(mktemp -d -- "$shared/packages/.bones-g1-2f59b20.XXXXXXXX")
  cleanup_stage() {
    if [[ -d "$partial" ]]; then
      for entry in g1.tar.gz SHA256SUMS; do
        if [[ -e "$partial/$entry" ]]; then unlink -- "$partial/$entry"; fi
      done
      rmdir -- "$partial"
    fi
  }
  trap cleanup_stage EXIT
  printf 'Staging the previously verified compressed BONES archive.\n'
  cp -- "$source" "$partial/g1.tar.gz"
  printf '%s  g1.tar.gz\n' "$bones_sha" > "$partial/SHA256SUMS"
  [[ $(stat -c '%s' -- "$partial/g1.tar.gz") == "$bones_bytes" ]]
  (cd -- "$partial"; sha256sum --check --strict SHA256SUMS)
  sync -d -- "$partial/g1.tar.gz" "$partial/SHA256SUMS"
  sync -- "$partial"
  mv -nT -- "$partial" "$published"
  [[ ! -e "$partial" ]] || fail 'Another process published the same archive.'
  trap - EXIT
  sync -- "$shared/packages"
else
  private_directory "$published" || fail 'Existing package is not private.'
  printf '%s  %s\n' "$bones_sha" "$published/g1.tar.gz" | sha256sum --check --strict
fi
check_queue
if compgen -e | grep -E '^SBATCH_' >/dev/null; then fail 'Clear inherited SBATCH overrides before submitting.'; fi
printf 'cohort_build_id=%s\nselection_id=supermimic-v3-appendix-e-approx-v1\napproximate=true\nbones_sha256=%s\nsonic_sha256=%s\npackage=%s\noutput=%s\n' \
  "$cohort" "$bones_sha" "$sonic_sha" "$package" "$shared/motions/$cohort" > "$state/inputs.txt"
printf '%s\n' "$cohort" > "$state/submit-intent"
sync -d -- "$state/inputs.txt" "$state/submit-intent"
sync -- "$state"
raw_job=$(env -u BASH_ENV -u ENV lab-submit --kind cpu -- --parsable --chdir="$shared" \
  --comment="$cohort" --output="$shared/logs/%j.out" --error="$shared/logs/%j.err" \
  "$control/run_prepare_motions.sbatch" "$package" "$published/g1.tar.gz" "$bones_sha" \
  "$shared/motions" "$cohort" --accept-approximate "$control/sonic-filter.py" "$sonic_sha")
[[ "$raw_job" =~ ^([1-9][0-9]*)(\;[^[:space:]]+)?$ ]] || fail "Unexpected Slurm reply; reconcile before retry: $raw_job"
job=${BASH_REMATCH[1]}
printf '%s\n' "$job" > "$state/job-id"
nohup /usr/bin/python3 -B -u "$ops/watch_slurm.py" "$job" "$state/$job.slurm.json" --interval 10 \
  9>&- </dev/null >>"$state/watcher.log" 2>&1 &
printf '%s\n' "$!" > "$state/watcher.pid"
sync -d -- "$state/job-id" "$state/watcher.pid"
sync -- "$state"
printf 'CPU cohort preparation job %s, run %s.\n' "$job" "$cohort"
