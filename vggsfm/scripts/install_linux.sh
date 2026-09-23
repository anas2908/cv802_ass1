#!/usr/bin/env bash
# Install the pinned official VGGSfM stack. Run this only inside the allocated GPU job.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DATA_ROOT="${CV802_DATA_ROOT:-/l/users/anas.khan/cv_802_ass1}"
METHOD_ROOT="${PROJECT_DATA_ROOT}/vggsfm"
OFFICIAL_ROOT="${METHOD_ROOT}/downloads/source/vggsfm"
LIGHTGLUE_ROOT="${METHOD_ROOT}/downloads/source/lightglue"
ENV_PREFIX="${METHOD_ROOT}/envs/vggsfm"
OFFICIAL_COMMIT="e1d9d2eb2b3575525792206fb94b2c749c58dc50"
LIGHTGLUE_COMMIT="2f23ca2ea9638cecad7f7220795210fc6b8353c3"
INSTALL_RUN_ID="${SLURM_JOB_ID:-local-$(date +%s)}"

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
  echo "ERROR: install_linux.sh must run inside the allocated Slurm GPU job." >&2
  echo "For a personal machine only, set CV802_ALLOW_NON_SLURM=1 explicitly." >&2
  exit 2
fi
if [[ ! -d "${PROJECT_DATA_ROOT}" || ! -w "${PROJECT_DATA_ROOT}" ]]; then
  echo "ERROR: data root is absent or not writable: ${PROJECT_DATA_ROOT}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: no allocated NVIDIA GPU is visible." >&2
  exit 2
fi

mkdir -p \
  "${METHOD_ROOT}/downloads/source" \
  "${METHOD_ROOT}/envs" \
  "${METHOD_ROOT}/cache/conda" \
  "${METHOD_ROOT}/cache/pip" \
  "${METHOD_ROOT}/cache/torch" \
  "${METHOD_ROOT}/cache/huggingface" \
  "${METHOD_ROOT}/cache/xdg" \
  "${METHOD_ROOT}/cache/cuda" \
  "${METHOD_ROOT}/cache/matplotlib" \
  "${METHOD_ROOT}/cache/pycache" \
  "${METHOD_ROOT}/cache/torch_extensions" \
  "${METHOD_ROOT}/cache/triton" \
  "${METHOD_ROOT}/cache/numba" \
  "${METHOD_ROOT}/cache/gradio" \
  "${METHOD_ROOT}/config/xdg" \
  "${METHOD_ROOT}/config/xdg-data" \
  "${METHOD_ROOT}/tmp" \
  "${METHOD_ROOT}/logs"

export CONDA_PKGS_DIRS="${METHOD_ROOT}/cache/conda"
export CONDA_ENVS_PATH="${METHOD_ROOT}/envs"
export CONDARC=/dev/null
export PIP_CACHE_DIR="${METHOD_ROOT}/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export TORCH_HOME="${METHOD_ROOT}/cache/torch"
export HF_HOME="${METHOD_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${METHOD_ROOT}/cache/huggingface/hub"
export HF_DATASETS_CACHE="${METHOD_ROOT}/cache/huggingface/datasets"
export XDG_CACHE_HOME="${METHOD_ROOT}/cache/xdg"
export XDG_CONFIG_HOME="${METHOD_ROOT}/config/xdg"
export XDG_DATA_HOME="${METHOD_ROOT}/config/xdg-data"
export CUDA_CACHE_PATH="${METHOD_ROOT}/cache/cuda"
export MPLCONFIGDIR="${METHOD_ROOT}/cache/matplotlib"
export PYTHONPYCACHEPREFIX="${METHOD_ROOT}/cache/pycache"
export TORCH_EXTENSIONS_DIR="${METHOD_ROOT}/cache/torch_extensions"
export TRITON_CACHE_DIR="${METHOD_ROOT}/cache/triton"
export NUMBA_CACHE_DIR="${METHOD_ROOT}/cache/numba"
export GRADIO_TEMP_DIR="${METHOD_ROOT}/cache/gradio"
export TMPDIR="${METHOD_ROOT}/tmp"
export TEMP="${METHOD_ROOT}/tmp"
export TMP="${METHOD_ROOT}/tmp"
export PYTHONNOUSERSITE=1
export PYTHONUSERBASE="${METHOD_ROOT}/envs/python-user-base"
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

INSTALL_LOG="${METHOD_ROOT}/logs/install-${INSTALL_RUN_ID}.log"
exec > >(tee -a "${INSTALL_LOG}") 2>&1

echo "Starting pinned VGGSfM install at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Run: ${INSTALL_RUN_ID}; Slurm job: ${SLURM_JOB_ID:-none}; node: $(hostname)"
nvidia-smi -L

if [[ -f /apps/local/conda_init.sh ]]; then
  # shellcheck source=/dev/null
  source /apps/local/conda_init.sh
elif ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is unavailable; install Miniconda/Anaconda or use the cluster initializer." >&2
  exit 2
fi

if [[ -e "${OFFICIAL_ROOT}" ]]; then
  if [[ ! -d "${OFFICIAL_ROOT}/.git" ]]; then
    echo "ERROR: official destination exists but is not a Git checkout: ${OFFICIAL_ROOT}" >&2
    exit 2
  fi
  FOUND_COMMIT="$(git -C "${OFFICIAL_ROOT}" rev-parse HEAD)"
  if [[ "${FOUND_COMMIT}" != "${OFFICIAL_COMMIT}" ]]; then
    echo "ERROR: existing official checkout is ${FOUND_COMMIT}, expected ${OFFICIAL_COMMIT}." >&2
    echo "Refusing to reset or overwrite it; resolve provenance explicitly." >&2
    exit 2
  fi
else
  OFFICIAL_STAGE="${METHOD_ROOT}/downloads/source/.vggsfm-${INSTALL_RUN_ID}.staging"
  if [[ -e "${OFFICIAL_STAGE}" ]]; then
    echo "ERROR: clone staging path already exists: ${OFFICIAL_STAGE}" >&2
    exit 2
  fi
  git clone --filter=blob:none --no-checkout \
    https://github.com/facebookresearch/vggsfm.git "${OFFICIAL_STAGE}"
  git -C "${OFFICIAL_STAGE}" checkout --detach "${OFFICIAL_COMMIT}"
  FOUND_COMMIT="$(git -C "${OFFICIAL_STAGE}" rev-parse HEAD)"
  [[ "${FOUND_COMMIT}" == "${OFFICIAL_COMMIT}" ]]
  mv "${OFFICIAL_STAGE}" "${OFFICIAL_ROOT}"
fi

if [[ -e "${LIGHTGLUE_ROOT}" ]]; then
  if [[ ! -d "${LIGHTGLUE_ROOT}/.git" ]]; then
    echo "ERROR: LightGlue destination is not a Git checkout: ${LIGHTGLUE_ROOT}" >&2
    exit 2
  fi
  FOUND_LIGHTGLUE="$(git -C "${LIGHTGLUE_ROOT}" rev-parse HEAD)"
  if [[ "${FOUND_LIGHTGLUE}" != "${LIGHTGLUE_COMMIT}" ]]; then
    echo "ERROR: existing LightGlue is ${FOUND_LIGHTGLUE}, expected ${LIGHTGLUE_COMMIT}." >&2
    exit 2
  fi
else
  LIGHTGLUE_STAGE="${METHOD_ROOT}/downloads/source/.lightglue-${INSTALL_RUN_ID}.staging"
  if [[ -e "${LIGHTGLUE_STAGE}" ]]; then
    echo "ERROR: clone staging path already exists: ${LIGHTGLUE_STAGE}" >&2
    exit 2
  fi
  git clone --filter=blob:none --no-checkout \
    https://github.com/jytime/LightGlue.git "${LIGHTGLUE_STAGE}"
  git -C "${LIGHTGLUE_STAGE}" checkout --detach "${LIGHTGLUE_COMMIT}"
  FOUND_LIGHTGLUE="$(git -C "${LIGHTGLUE_STAGE}" rev-parse HEAD)"
  [[ "${FOUND_LIGHTGLUE}" == "${LIGHTGLUE_COMMIT}" ]]
  mv "${LIGHTGLUE_STAGE}" "${LIGHTGLUE_ROOT}"
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create --yes --prefix "${ENV_PREFIX}" python=3.10
fi

ENV_PYTHON="${ENV_PREFIX}/bin/python"
if ! "${ENV_PYTHON}" -c \
  "import sys; assert sys.version_info[:2] == (3, 10), sys.version"; then
  echo "ERROR: existing environment is not Python 3.10: ${ENV_PREFIX}" >&2
  echo "Refusing to delete or replace it automatically." >&2
  exit 2
fi
if ! "${ENV_PYTHON}" -c \
  "import torch; assert torch.__version__.split('+')[0].startswith('2.1.'), torch.__version__" \
  >/dev/null 2>&1; then
  # This cluster ships Conda 4.11's classic solver, which can spend many
  # minutes exploring the PyTorch/CUDA solve without producing a plan.  Use
  # PyTorch's official, exactly pinned CUDA 12.1 wheels inside the same Conda
  # Python environment.  The wheel/cache still lives entirely in METHOD_ROOT.
  "${ENV_PYTHON}" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.1.0+cu121" "torchvision==0.16.0+cu121"
fi
"${ENV_PYTHON}" -m pip install \
  "fvcore==0.1.5.post20221221" "iopath==0.1.10"

"${ENV_PYTHON}" -m pip install \
  "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
# Visdom imports pkg_resources while building.  Build it against the pinned
# environment tooling instead of pip's isolated newest-setuptools sandbox,
# where pkg_resources has been removed.  The module is imported by the
# official runner even though visualization remains disabled headlessly.
"${ENV_PYTHON}" -m pip install --no-build-isolation "visdom==0.2.4"
"${ENV_PYTHON}" -m pip install \
  "numpy==1.26.3" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "opencv-python-headless==4.10.0.84" einops tqdm scipy plotly scikit-learn \
  'imageio[ffmpeg]' "gradio==4.44.1" trimesh huggingface_hub
"${ENV_PYTHON}" -m pip install \
  "pycolmap==3.10.0" "pyceres==2.3" "poselib==2.0.2"
"${ENV_PYTHON}" -m pip install --no-deps --editable "${LIGHTGLUE_ROOT}"
"${ENV_PYTHON}" -m pip install --editable "${OFFICIAL_ROOT}"

# LightGlue declares unbounded NumPy/OpenCV dependencies.  Keep the official
# VGGSfM NumPy 1.26 requirement and the headless OpenCV build authoritative;
# PyTorch 2.1 was compiled against the NumPy 1.x ABI.
"${ENV_PYTHON}" -m pip uninstall --yes opencv-python >/dev/null 2>&1 || true
"${ENV_PYTHON}" -m pip install --force-reinstall --no-deps \
  "numpy==1.26.3" "opencv-python-headless==4.10.0.84"

"${ENV_PYTHON}" -c \
  "import cv2, json, numpy, torch; assert numpy.__version__ == '1.26.3'; assert torch.cuda.is_available(); print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'numpy': numpy.__version__, 'opencv': cv2.__version__}))"
"${ENV_PYTHON}" "${CODE_ROOT}/scripts/write_install_receipt.py" \
  --output "${METHOD_ROOT}/install_receipt.json" \
  --official-root "${OFFICIAL_ROOT}" \
  --lightglue-root "${LIGHTGLUE_ROOT}" \
  --environment-python "${ENV_PYTHON}"

echo "VGGSfM install complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Receipt: ${METHOD_ROOT}/install_receipt.json"
