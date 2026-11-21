#!/bin/bash
# On gpu01, assemble an immutable local input version for later approved stage-in.
set -euo pipefail
umask 077
[[ $# == 5 ]] || {
  echo "Usage: bash $0 ENVIRONMENT_TAR ENV_SHA CODE_TAR CODE_SHA NEW_PACKAGE_DIRECTORY" >&2
  exit 2
}
environment=$(realpath -e -- "$1")
code=$(realpath -e -- "$3")
dest=$(realpath -m -- "$5")
[[ "$2" =~ ^[a-f0-9]{64}$ && "$4" =~ ^[a-f0-9]{64}$ ]]
[[ -z "${SLURM_JOB_ID:-}" && -f "$environment" && -f "$code" ]]
[[ ! -e "$dest" && ! -L "$5" && ( "$dest" == "/home/$USER/"* || "$dest" == "/public/$USER/"* ) ]]
parent=$(dirname -- "$dest")
[[ -d "$parent" && -O "$parent" && "$(findmnt -n -o FSTYPE -T "$parent")" == xfs ]]
mode=$(stat -c %a -- "$parent")
(( (8#$mode & 0777) == 0700 ))
mount_root=/home
[[ "$dest" != /public/* ]] || mount_root=/public
used_kib=$(xfs_quota -c 'quota -ub' "$mount_root" | awk '
  $1 ~ /^\/dev\// && $2 ~ /^[0-9]+$/ { used = $2 }
  $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { used = $1 }
  END { print used }')
available=$(df -B1 --output=avail "$parent" | awk 'NR==2 {print $1}')
[[ "$used_kib" =~ ^[0-9]+$ && "$available" =~ ^[0-9]+$ ]]
reserve=$(( $(stat -c %s -- "$environment") + $(stat -c %s -- "$code") + 1073741824 ))
(( reserve <= available && used_kib * 1024 + reserve <= 200000000000 ))

stage=$(mktemp -d -- "$parent/.input-package.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
cp --reflink=auto -- "$environment" "$stage/environment.tar.gz"
cp --reflink=auto -- "$code" "$stage/code.tar.gz"
printf '%s  environment.tar.gz\n%s  code.tar.gz\n' "$2" "$4" > "$stage/SHA256SUMS"
(cd -- "$stage" && sha256sum --check --strict SHA256SUMS)
sync -d -- "$stage/"*
sync -- "$stage"
mv -nT -- "$stage" "$dest"
[[ ! -e "$stage" ]]
sync -- "$parent"
printf 'Local AIR package published: %s\n' "$dest"
