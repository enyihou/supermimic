#!/bin/bash
set -euo pipefail
: "${DATA_DIR:?Use the AIR job template}"
: "${OUTPUT_DIR:?Use the AIR job template}"
: "${LOCAL_SCRATCH:?Use the allocated Slurm scratch}"
CODE_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$CODE_DIR/reproduction"
export WARP_CACHE_PATH="$LOCAL_SCRATCH/cache/warp"
mkdir -p "$WARP_CACHE_PATH"
cp -- "$DATA_DIR/inputs.json" "$OUTPUT_DIR/inputs.json"
python -u "$CODE_DIR/reproduction/scripts/data/convert_lafan_to_npz.py" \
    --input-dir "$DATA_DIR/csv" --output-dir "$DATA_DIR/motions" \
    > "$OUTPUT_DIR/conversion.log" 2>&1
exec python -u "$CODE_DIR/experiment.py" --manifest "$DATA_DIR/motions/manifest.json" \
    --teacher "$DATA_DIR/teacher.pt" --output "$OUTPUT_DIR" > "$OUTPUT_DIR/pilot.log" 2>&1
