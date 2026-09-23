#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DATA_ROOT="${CV802_DATA_ROOT:-/l/users/anas.khan/cv_802_ass1}"
DATA_ROOT="${PROJECT_DATA_ROOT}/mvs"
ENV_PREFIX=${DATA_ROOT}/envs/mvs-engine

if [[ "${PROJECT_DATA_ROOT}" != /* ]]; then
  echo "ERROR: CV802_DATA_ROOT must be an absolute path: ${PROJECT_DATA_ROOT}" >&2
  exit 2
fi
case "${CODE_ROOT}/" in
  "${PROJECT_DATA_ROOT}/"*)
    echo "ERROR: keep data/environments outside the source-code checkout." >&2
    exit 2
    ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" && "${CV802_ALLOW_NON_SLURM:-0}" != 1 ]]; then
  echo "ERROR: run inside the active GPU allocation; SLURM_JOB_ID is unset" >&2
  echo "For a personal machine only, set CV802_ALLOW_NON_SLURM=1 explicitly." >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" || ! -w "${DATA_ROOT}" ]]; then
  echo "ERROR: MVS data root is missing or not writable: ${DATA_ROOT}" >&2
  exit 2
fi
findmnt --target "${DATA_ROOT}" >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p \
  "${DATA_ROOT}/envs" \
  "${DATA_ROOT}/cache/conda-pkgs" \
  "${DATA_ROOT}/cache/pip" \
  "${DATA_ROOT}/runtime/tmp"

export CONDARC=/dev/null
export CONDA_ENVS_PATH=${DATA_ROOT}/envs
export CONDA_PKGS_DIRS=${DATA_ROOT}/cache/conda-pkgs
export PIP_CACHE_DIR=${DATA_ROOT}/cache/pip
export TMPDIR=${DATA_ROOT}/runtime/tmp

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  if ! command -v python3.12 >/dev/null 2>&1; then
    echo "ERROR: python3.12 is unavailable on this compute node" >&2
    exit 2
  fi
  # CIAI's legacy base Conda initializer can exit silently under strict shell
  # mode. A standard venv is sufficient because all MVS dependencies are
  # distributed as pinned binary wheels, including pycolmap-cuda12.
  python3.12 -m venv "${ENV_PREFIX}"
fi

"${ENV_PREFIX}/bin/python" -m pip install --upgrade "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
"${ENV_PREFIX}/bin/python" -m pip install \
  --only-binary=:all: \
  --requirement "${CODE_ROOT}/requirements.lock"

PYTHONPATH="${CODE_ROOT}/src" "${ENV_PREFIX}/bin/python" - <<'PY'
import pycolmap
from PIL import __version__ as pillow_version

assert pycolmap.__version__ == "4.2.0", pycolmap.__version__
print({
    "pycolmap_version": pycolmap.__version__,
    "pycolmap_has_cuda": getattr(pycolmap, "has_cuda", None),
    "pillow_version": pillow_version,
})
PY
