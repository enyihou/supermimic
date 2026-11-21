#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 WORK_ROOT" >&2
  exit 2
fi

work_root=$1
supermimic_commit=5fb92297ab4fe3d86629d9f0c635334791bb0d5f
mjlab_commit=379c55e88b1dec73f93943e524a879b2ebbf45c5
mkdir -p "$work_root"

clone_at() {
  local url=$1 destination=$2 commit=$3
  if [[ -d "$destination/.git" ]]; then
    git -C "$destination" fetch --depth 1 origin "$commit"
  else
    git clone --no-checkout "https://github.com/$url" "$destination"
  fi
  git -C "$destination" checkout --detach "$commit"
}

clone_at HybridRobotics/supermimic.git "$work_root/supermimic" "$supermimic_commit"
clone_at HybridRobotics/mjlab_research.git "$work_root/mjlab_research" "$mjlab_commit"

if [[ ! -f "$work_root/supermimic/pyproject.toml" || ! -f "$work_root/mjlab_research/pyproject.toml" ]]; then
  echo "Pinned repositories are incomplete" >&2
  exit 1
fi

printf 'supermimic=%s\nmjlab_research=%s\n' "$supermimic_commit" "$mjlab_commit" > "$work_root/PINS.txt"
printf 'Prepared pinned source checkouts in %s (no runtime installed).\n' "$work_root"
