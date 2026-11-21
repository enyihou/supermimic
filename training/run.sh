#!/bin/bash
# Run only from the verified code bundle on allocated node-local scratch.
set -euo pipefail
code=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python -u "$code/training/air_launch.py" train "$1" "$2" "$code" "$3" "$4"
