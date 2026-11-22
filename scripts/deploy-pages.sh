#!/usr/bin/env bash
# Deploy a reviewed, committed public build of this repo. Never blend worktrees.
set -euo pipefail
[[ $# == 1 && "$1" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Usage: bash scripts/deploy-pages.sh FULL_SOURCE_COMMIT" >&2
  exit 2
}
source_commit=$1
root=$(git rev-parse --show-toplevel)
cd -- "$root"
[[ $(git branch --show-current) == source && "$(git rev-parse HEAD)" == "$source_commit" ]]
[[ -z "$(git status --porcelain --untracked-files=all -- public src scripts training index.html package.json package-lock.json tsconfig.json vite.config.ts)" ]] ||
  { echo "Public build input differs from the committed source." >&2; exit 3; }

git fetch --no-tags origin source main
[[ $(git rev-parse origin/source) == "$source_commit" ]] ||
  { echo "Push this exact source commit before deploying Pages." >&2; exit 3; }
pages_head=$(git rev-parse origin/main)
umask 077
temp=$(mktemp -d "${TMPDIR:-/tmp}/supermimic-deploy.XXXXXXXX")
pages="$temp/pages"
build="$temp/build"
cleanup() {
  if [[ -e "$pages/.git" ]]; then
    if ! git worktree remove --force "$pages"; then
      printf 'Could not remove temporary deploy worktree at %s\n' "$pages" >&2
      return 1
    fi
  fi
  rm -rf -- "$temp"
}
trap cleanup EXIT
GITHUB_PAGES=1 npm run build -- --outDir "$build"

[[ -s "$build/index.html" && -f "$build/reproduction/preflight.json" &&
   -f "$build/reproduction/status.json" && -f "$build/models/releases.json" ]]
node scripts/verify-release.mjs "$build"
bash scripts/verify-phase-status.sh "$build"
if [[ -n "$(find "$build" -type l -print -quit)" ]]; then
  echo 'Public builds must not contain symbolic links.' >&2; exit 4
fi
shopt -s dotglob globstar nullglob
asset_pattern='^(index\.html|assets/[A-Za-z0-9_.-]+\.(wasm|js|css)|licenses/[A-Za-z0-9_.-]+\.(md|txt)|reproduction/(preflight|status|smoke-audit|cohort-audit|benchmark-audit|full-audit)\.json|models/releases\.json|models/[a-z0-9-]+/(manifest\.json|actor\.onnx|scene\.mjbpack|ATTRIBUTION\.md|mjlab-LICENSE\.txt|Unitree-LICENSE\.txt))$'
for item in "$build"/**/*; do
  [[ -f "$item" ]] || continue
  path=${item#"$build/"}
  [[ "$path" =~ $asset_pattern ]] ||
    { printf 'Unexpected generated asset: %s\n' "$path" >&2; exit 4; }
  [[ "$(stat -c %s -- "$item")" -lt 104857600 ]] ||
    { printf 'Public asset exceeds GitHub file limit: %s\n' "$path" >&2; exit 4; }
done

git worktree add -q --detach "$pages" "$pages_head"
rsync -a --delete --exclude='/.git' --exclude='/.nojekyll' "$build/" "$pages/"
[[ -f "$pages/.nojekyll" ]]
git -C "$pages" add -A
git -C "$pages" diff --cached --check -- . ':!assets'
if git -C "$pages" diff --cached --quiet; then
  printf 'Pages already matches this build (%s).\n' "$pages_head"
  exit
fi
git -C "$pages" commit -q -m "Publish SuperMimic reproduction (${source_commit:0:12})"
git -C "$pages" push origin HEAD:main
git -C "$pages" rev-parse HEAD
