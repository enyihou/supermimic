#!/usr/bin/env bash
# Run on gpu01 local XFS after reading the current AIR guide and runbook.
# Do not run in a Slurm allocation or send Hugging Face authentication here.
set -euo pipefail
umask 077
[[ $# == 2 ]] || { echo "Usage: bash $0 NEW_BUILD_DIRECTORY RUNTIME_HASH_LOCK" >&2; exit 2; }
[[ $(uname -m) == x86_64 && -z "${SLURM_JOB_ID:-}" ]]
build=$(realpath -m -- "$1")
lock=$(realpath -e -- "$2")
[[ "$build" == "/home/$USER/"* || "$build" == "/public/$USER/"* ]]
mount_root=/home
[[ "$build" != /public/* ]] || mount_root=/public
parent=$(dirname -- "$build")
[[ $(findmnt -n -o FSTYPE -T "$parent") == xfs ]] ||
  { echo 'Build parent is not local XFS' >&2; exit 3; }
used_kib=$(xfs_quota -c 'quota -ub' "$mount_root" | awk '
  $1 ~ /^\/dev\// && $2 ~ /^[0-9]+$/ { used = $2 }
  $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { used = $1 }
  END { print used }')
free=$(df -B1 --output=avail "$parent" | awk 'NR==2 {print $1}')
[[ "$used_kib" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] ||
  { echo 'Could not establish AIR quota and disk headroom' >&2; exit 3; }
# Python, downloaded wheels, unpacked packages, and compressed environment coexist.
reserve=30000000000
(( reserve <= free && used_kib * 1024 + reserve <= 200000000000 )) ||
  { echo "Insufficient AIR local build headroom" >&2; exit 3; }
mkdir -m 700 -- "$build"

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH PYTHONPATH PYTHONHOME
unset CONDA_ENVS_PATH VIRTUAL_ENV
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="$build/conda-pkgs" UV_CACHE_DIR="$build/uv-cache"
export PIP_CACHE_DIR="$build/pip-cache"
conda=/opt/miniforge3/bin/conda
base_python=/opt/miniforge3/envs/py311/bin/python
tools="$build/pack-tools"
runtime="$build/runtime"
"$base_python" -m venv "$tools"
"$tools/bin/python" -m pip install --disable-pip-version-check --index-url https://pypi.org/simple \
  "uv==0.11.28" "conda-pack==0.9.2"
"$conda" create --yes --prefix "$runtime" --override-channels -c conda-forge 'python=3.13.14'

cp -- "$lock" "$build/requirements-air-cu126.lock"
"$tools/bin/uv" pip install --python "$runtime/bin/python" --only-binary :all: \
  --require-hashes -r "$build/requirements-air-cu126.lock"
"$tools/bin/uv" pip check --python "$runtime/bin/python"
"$tools/bin/uv" pip freeze --python "$runtime/bin/python" > "$build/pip-freeze.txt"
"$conda" list --prefix "$runtime" --explicit > "$build/conda-explicit.txt"
"$runtime/bin/python" -c 'import mujoco, numpy, torch, warp
print("Python CUDA runtime", torch.version.cuda)
print("MuJoCo", mujoco.__version__, "NumPy", numpy.__version__, "Warp", warp.__version__)
assert torch.version.cuda == "12.6"'
"$tools/bin/conda-pack" --prefix "$runtime" --output "$build/environment.partial.tar.gz"
sync -d "$build/environment.partial.tar.gz"
mv -- "$build/environment.partial.tar.gz" "$build/environment.tar.gz"
(cd "$build" && sha256sum environment.tar.gz requirements-air-cu126.lock > ENVIRONMENT_SHA256SUMS)
