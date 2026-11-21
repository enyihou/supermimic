#!/usr/bin/env bash
# Export exact Git trees and only the reviewed launcher/helpers. No .git, env or data.
set -euo pipefail
umask 077
[[ $# == 2 ]] || { echo "Usage: bash $0 PINNED_CHECKOUT_ROOT CODE_TAR_GZ" >&2; exit 2; }
clones=$(realpath -e -- "$1")
out=$(realpath -m -- "$2")
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
revision=$(git -C "$here" rev-parse --verify HEAD)
[[ ! -e "$out" && ! -L "$out" ]]
build=$(mktemp -d -- "$(dirname -- "$out")/.code-build.XXXXXXXX")
trap 'rm -rf -- "$build"' EXIT
mkdir "$build/package" "$build/package/training"

supermimic=5fb92297ab4fe3d86629d9f0c635334791bb0d5f
mjlab=379c55e88b1dec73f93943e524a879b2ebbf45c5
export_pin() {
  local repo=$1 pin=$2
  [[ $(git -C "$clones/$repo" rev-parse HEAD) == "$pin" ]]
  git -C "$clones/$repo" archive --format=tar --prefix="$repo/" "$pin" |
    tar -xf - -C "$build/package" --no-same-owner --no-same-permissions
}
export_pin supermimic "$supermimic"
export_pin mjlab_research "$mjlab"
printf 'supermimic=%s\nmjlab_research=%s\n' "$supermimic" "$mjlab" > "$build/package/PINS.txt"
printf '%s\n' "$revision" > "$build/package/REPRODUCTION_COMMIT.txt"

for item in air_checkpoint.py air_launch.py air_runner.patch audit_official.py evaluate_heldout.py \
  lafan1_g1_eval_manifest.tsv inspect_bones.py pack_bones.py; do
  git -C "$here" show "$revision:training/$item" > "$build/package/training/$item"
done
git -C "$build/package/supermimic" apply --check "$build/package/training/air_runner.patch"
tar -C "$build/package" --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -czf "$build/code.tar.gz" \
  PINS.txt REPRODUCTION_COMMIT.txt supermimic mjlab_research training
sync -d "$build/code.tar.gz"
mv -nT -- "$build/code.tar.gz" "$out"
[[ ! -e "$build/code.tar.gz" ]]
sha256sum -- "$out"
