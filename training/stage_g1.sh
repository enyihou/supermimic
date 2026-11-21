#!/usr/bin/env bash
# Run on the local SSH server, after reading the current AIR server guide.
set -euo pipefail
umask 077
[[ $# == 3 ]] || { echo "Usage: bash $0 LOCAL_G1_TAR_GZ EXPECTED_SHA256 EXPECTED_BYTES" >&2; exit 2; }
archive=$1
sha=$2
bytes=$3
[[ "${archive##*/}" == g1.tar.gz && -f "$archive" && ! -L "$archive" ]]
[[ "$sha" =~ ^[a-f0-9]{64}$ && "$bytes" =~ ^[1-9][0-9]{0,11}$ ]]
[[ "$(stat -c %s -- "$archive")" == "$bytes" ]] || { echo "Local size mismatch" >&2; exit 3; }
exec 9>"$archive.air-transfer.lock"
flock -n 9
printf '%s  %s\n' "$sha" "$archive" | sha256sum --check --status || {
  echo "Local SHA-256 mismatch against pinned source" >&2
  exit 3
}
printf 'Verified local SHA-256: %s\n' "$sha"

# SSH authentication remains independent of Hugging Face. Never pass HF
# authentication or signed download URLs to the cluster, including via SendEnv.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH
ssh_options=(-o BatchMode=yes -o ConnectTimeout=20)
inputs=/public/houey/supermimic-archive/inputs
# shellcheck disable=SC2029 # Local args are a fixed path and validated hex/digits.
preflight=$(ssh "${ssh_options[@]}" air "bash -s -- '$inputs' '$sha' '$bytes'" <<'REMOTE'
set -euo pipefail
inputs=$1 sha=$2 bytes=$3
[[ -d "$inputs" && ! -L "$inputs" && -O "$inputs" && -w "$inputs" ]]
mode=$(stat -c %a "$inputs")
(( (8#$mode & 077) == 0 ))
cd "$inputs"
[[ ! -L g1.tar.gz && ! -L .g1.tar.gz.partial ]]
if [[ -e g1.tar.gz ]]; then
  [[ -f g1.tar.gz && -O g1.tar.gz && "$(stat -c %s g1.tar.gz)" == "$bytes" ]]
  printf '%s  g1.tar.gz\n' "$sha" | sha256sum --check --status
  echo already-verified
  exit
fi
existing=0
if [[ -e .g1.tar.gz.partial ]]; then
  [[ -f .g1.tar.gz.partial && -O .g1.tar.gz.partial ]]
  existing=$(stat -c '%b' .g1.tar.gz.partial)
fi
used_kib=$(xfs_quota -c 'quota -ub' /public | awk '
  $1 ~ /^\/dev\// && $2 ~ /^[0-9]+$/ { used = $2 }
  $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { used = $1 }
  END { print used }')
free=$(df -B1 --output=avail "$inputs" | awk 'NR==2 {print $1}')
[[ "$used_kib" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]]
remaining=$((bytes - existing * 512))
if (( remaining < 0 )); then remaining=0; fi
(( remaining + 1073741824 <= free &&
   used_kib * 1024 + remaining + 1073741824 <= 200000000000 )) || {
  echo 'Insufficient /public quota/disk headroom for transfer' >&2
  exit 4
}
echo transfer
REMOTE
)

finalize_transfer() {
  # Keep the here-document inside a function so the case dispatch cannot
  # accidentally consume a malformed terminator as branch syntax.
  ssh "${ssh_options[@]}" air "bash -s -- '$inputs' '$sha' '$bytes'" <<'REMOTE'
set -euo pipefail
umask 077
cd "$1"
sha=$2 bytes=$3
[[ -f .g1.tar.gz.partial && ! -L .g1.tar.gz.partial && -O .g1.tar.gz.partial ]]
[[ "$(stat -c %s .g1.tar.gz.partial)" == "$bytes" ]]
printf '%s  .g1.tar.gz.partial\n' "$sha" | sha256sum --check --status
chmod 600 .g1.tar.gz.partial
sync -d .g1.tar.gz.partial
mv -nT -- .g1.tar.gz.partial g1.tar.gz
[[ ! -e .g1.tar.gz.partial ]]
REMOTE
}

case "$preflight" in
  already-verified) ;;
  transfer)
    rsync --partial --inplace --chmod=F600 --times --info=progress2 \
      -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' -- \
      "$archive" "air:$inputs/.g1.tar.gz.partial"
    finalize_transfer
    ;;
  *) printf 'Unexpected AIR transfer preflight response\n' >&2; exit 4 ;;
esac
printf 'Verified AIR SHA-256: %s (%s bytes)\n' "$sha" "$bytes"
