#!/usr/bin/env bash
# Create the fresh Linux/CUDA SfM environment entirely below DATA_ROOT.

set -euo pipefail
umask 027

readonly CODE_ROOT=/home/anas.khan/cv802_project/project1/ass1/sfm
readonly METHOD_ROOT=/l/users/anas.khan/cv_802_ass1/sfm
readonly ENV_PREFIX="$METHOD_ROOT/envs/headless-cuda"

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == *login* ]]; then
  echo "ERROR: setup_linux_cuda.sh requires the active Slurm GPU job" >&2
  exit 2
fi
if [[ ! -d "$METHOD_ROOT" || ! -w "$METHOD_ROOT" ]]; then
  echo "ERROR: SfM data root is missing or not writable: $METHOD_ROOT" >&2
  exit 2
fi
nvidia-smi -L

mkdir -p \
  "$METHOD_ROOT/envs" "$METHOD_ROOT/cache/pip" \
  "$METHOD_ROOT/cache/python" "$METHOD_ROOT/tmp" "$METHOD_ROOT/logs" \
  "$METHOD_ROOT/cache/xdg" "$METHOD_ROOT/cache/conda-pkgs" \
  "$METHOD_ROOT/cache/huggingface/hub" "$METHOD_ROOT/cache/torch" \
  "$METHOD_ROOT/cache/cuda" "$METHOD_ROOT/cache/matplotlib"
export PIP_CACHE_DIR="$METHOD_ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$METHOD_ROOT/cache/python"
export XDG_CACHE_HOME="$METHOD_ROOT/cache/xdg"
export CONDA_PKGS_DIRS="$METHOD_ROOT/cache/conda-pkgs"
export HF_HOME="$METHOD_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$METHOD_ROOT/cache/torch"
export CUDA_CACHE_PATH="$METHOD_ROOT/cache/cuda"
export MPLCONFIGDIR="$METHOD_ROOT/cache/matplotlib"
export TMPDIR="$METHOD_ROOT/tmp"
export PYTHONNOUSERSITE=1

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  python3.12 -m venv "$ENV_PREFIX"
fi
"$ENV_PREFIX/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_PREFIX/bin/python" -m pip install \
  --only-binary=:all: \
  --requirement "$CODE_ROOT/requirements-headless-linux-cuda.txt"

"$ENV_PREFIX/bin/python" - <<'PY'
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import pycolmap
import numpy
import PIL

has_cuda = getattr(pycolmap, "has_cuda", False)
has_cuda = bool(has_cuda() if callable(has_cuda) else has_cuda)
if not has_cuda:
    raise SystemExit("installed PyCOLMAP wheel does not expose CUDA")
receipt = {
    "schema": "cv802-sfm-environment/v2",
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "python": os.sys.version,
    "pycolmap": pycolmap.__version__,
    "pycolmap_has_cuda": has_cuda,
    "environment": os.sys.prefix,
    "numpy": numpy.__version__,
    "pillow": PIL.__version__,
    "requirements_sha256": hashlib.sha256(Path(
        "/home/anas.khan/cv802_project/project1/ass1/sfm/requirements-headless-linux-cuda.txt"
    ).read_bytes()).hexdigest(),
    "storage_environment": {name: os.environ[name] for name in (
        "PIP_CACHE_DIR", "PYTHONPYCACHEPREFIX", "TMPDIR", "XDG_CACHE_HOME",
        "CONDA_PKGS_DIRS", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME",
        "CUDA_CACHE_PATH", "MPLCONFIGDIR",
    )},
}
path = Path("/l/users/anas.khan/cv_802_ass1/sfm/environment-receipt.json")
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(receipt, sort_keys=True))
PY

"$ENV_PREFIX/bin/python" -m pip freeze \
  > "$METHOD_ROOT/environment-freeze.txt"
