#!/usr/bin/env bash
# Guarded all-290-view dark-shirt VGGSfM run inside an existing Slurm GPU job.
set -Eeuo pipefail

CODE_ROOT="/home/anas.khan/cv802_project/project1/ass1/vggsfm"
METHOD_ROOT="/l/users/anas.khan/cv_802_ass1/vggsfm"
DATASET="dark_shirt"
RUN_ID="dark-shirt-vggsfm-all290-a10040-fit-v2"
PROFILE="${CODE_ROOT}/configs/dark_shirt_all290_a10040_fit_v2.json"
PYTHON="${METHOD_ROOT}/envs/vggsfm/bin/python"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: run_dark_shirt.sh must execute inside the verified Slurm GPU job." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: no allocated NVIDIA GPU is visible." >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: fixed VGGSfM environment is missing: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${METHOD_ROOT}/outputs/light-shirt-vggsfm-v1/manifest.json" ]]; then
  echo "ERROR: the measured light reference must finish before the dark run." >&2
  exit 2
fi
if [[ ! -f "${METHOD_ROOT}/inputs/${DATASET}/input_receipt.json" ]]; then
  echo "ERROR: prepare and verify the method-local all-290 input first." >&2
  exit 2
fi

mkdir -p "${METHOD_ROOT}/logs" "${METHOD_ROOT}/cache/pycache" "${METHOD_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${METHOD_ROOT}/cache/pycache"
export TMPDIR="${METHOD_ROOT}/tmp"
export TEMP="${METHOD_ROOT}/tmp"
export TMP="${METHOD_ROOT}/tmp"

"${PYTHON}" -B - "${METHOD_ROOT}/inputs/${DATASET}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
receipt = json.loads((root / "input_receipt.json").read_text(encoding="utf-8"))
images = [path for path in (root / "images").rglob("*") if path.is_file()]
if receipt.get("dataset") != "dark_shirt":
    raise SystemExit("ERROR: input receipt dataset is not dark_shirt")
if receipt.get("selection") != "all_images":
    raise SystemExit("ERROR: dark run refuses a subset selection")
if receipt.get("image_count") != 290 or receipt.get("original_image_count") != 290:
    raise SystemExit("ERROR: dark input receipt must prove all 290 original views")
if receipt.get("mask_count") != 0:
    raise SystemExit("ERROR: reviewed dark VGGSfM profile is raw/unmasked to retain crutches")
if len(images) != 290 or any(path.is_symlink() for path in images):
    raise SystemExit("ERROR: expected exactly 290 method-local regular image files")
print("Verified independent dark input: 290/290 regular files, raw/unmasked")
PY

RUN_LOG="${METHOD_ROOT}/logs/${RUN_ID}-slurm-${SLURM_JOB_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

"${PYTHON}" -B "${CODE_ROOT}/run_vggsfm.py" doctor --require-ready
"${PYTHON}" -B "${CODE_ROOT}/run_vggsfm.py" plan \
  --dataset "${DATASET}" --run-id "${RUN_ID}" --profile "${PROFILE}"
"${PYTHON}" -B "${CODE_ROOT}/run_vggsfm.py" run \
  --dataset "${DATASET}" --run-id "${RUN_ID}" --profile "${PROFILE}"

echo "VGGSfM dark result: ${METHOD_ROOT}/outputs/${RUN_ID}/point_cloud.ply"
