#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -m pip install -r script/requirements.txt
if ! python -c 'import pytorch3d' >/dev/null 2>&1; then
  python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8'
fi
if ! python -c 'import curobo' >/dev/null 2>&1; then
  python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/curobo.git@v0.7.8'
fi
python script/patch_dependencies.py
python -m pip install --no-deps --no-build-isolation -e ./agent -e ./active_spatial_benchmark_xyz
echo 'Dependencies installed. Link assets using script/configure_assets.py, then run script/check_setup.py.'
