#!/usr/bin/env bash
# Verify the exact progress report served by a source or built public tree.
set -euo pipefail
[[ $# -le 1 ]] || { echo 'Usage: bash scripts/verify-phase-status.sh [PUBLIC_ROOT]' >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
public=$(realpath -e -- "${1:-$root/public}")
report="$public/reproduction/status.json"
[[ -f "$report" && ! -L "$report" && "$(stat -c %s -- "$report")" -le 262144 ]] || {
  echo 'Missing or oversized public progress report.' >&2
  exit 2
}
report_date=$(node -e '
  const date = require(process.argv[1]).updatedAt;
  if (typeof date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(date)) throw Error("Invalid progress date");
  process.stdout.write(date);
' "$report")
summary_args=(--preflight "$public/reproduction/preflight.json" --date "$report_date" --registry "$public/models/releases.json")
for phase in smoke benchmark full; do
  audit="$public/reproduction/$phase-audit.json"
  if [[ -e "$audit" || -L "$audit" ]]; then
    summary_args+=("--$phase" "$audit")
  fi
done
umask 077
temp=$(mktemp -d "${TMPDIR:-/tmp}/supermimic-status.XXXXXXXX")
trap 'rm -rf -- "$temp"' EXIT
python3 -B "$root/training/publish_phase_summary.py" "${summary_args[@]}" --output "$temp/status.json"
if ! cmp -s -- "$report" "$temp/status.json"; then
  echo 'Public status.json does not match the verified phase evidence. Regenerate it with training/publish_phase_summary.py.' >&2
  exit 1
fi
