#!/usr/bin/env bash
# Idempotent install + official headless light-shirt inference for a Slurm GPU job.
set -Eeuo pipefail

CODE_ROOT="/home/anas.khan/cv802_project/project1/ass1/vggsfm"
METHOD_ROOT="/l/users/anas.khan/cv_802_ass1/vggsfm"
RUN_ID="${1:-light-shirt-vggsfm-v1}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: run_light_shirt.sh must execute inside the verified Slurm GPU job." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: no allocated NVIDIA GPU is visible." >&2
  exit 2
fi
if [[ ! -d "${METHOD_ROOT}/inputs/light_shirt/images" ]]; then
  echo "ERROR: prepare the independent input first: ${METHOD_ROOT}/inputs/light_shirt/images" >&2
  exit 2
fi

mkdir -p "${METHOD_ROOT}/logs" "${METHOD_ROOT}/cache/pycache" "${METHOD_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${METHOD_ROOT}/cache/pycache"
export TMPDIR="${METHOD_ROOT}/tmp"
export TEMP="${METHOD_ROOT}/tmp"
export TMP="${METHOD_ROOT}/tmp"

RUN_LOG="${METHOD_ROOT}/logs/${RUN_ID}-slurm-${SLURM_JOB_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

"${CODE_ROOT}/scripts/install_linux.sh"
PYTHON="${METHOD_ROOT}/envs/vggsfm/bin/python"
"${PYTHON}" -B "${CODE_ROOT}/run_vggsfm.py" doctor --require-ready
"${PYTHON}" -B "${CODE_ROOT}/run_vggsfm.py" plan \
  --dataset light_shirt \
  --run-id "${RUN_ID}" \
  --profile "${CODE_ROOT}/configs/light_shirt.json"
"${PYTHON}" -B "${CODE_ROOT}/run_vggsfm.py" run \
  --dataset light_shirt \
  --run-id "${RUN_ID}" \
  --profile "${CODE_ROOT}/configs/light_shirt.json" \
  --resume

echo "VGGSfM result: ${METHOD_ROOT}/outputs/${RUN_ID}/point_cloud.ply"
