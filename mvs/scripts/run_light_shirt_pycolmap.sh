#!/usr/bin/env bash
# Explicit no-SIF fallback using the pinned official PyCOLMAP CUDA wheel.

set -euo pipefail
umask 027

readonly CODE_ROOT=/home/anas.khan/cv802_project/project1/ass1
readonly DATA_ROOT=/l/users/anas.khan/cv_802_ass1
readonly MVS_CODE=$CODE_ROOT/mvs
readonly MVS_DATA=$DATA_ROOT/mvs
readonly PYTHON=$MVS_DATA/envs/mvs-engine/bin/python
readonly CONFIG=$MVS_CODE/configs/light_e10_1600_pycolmap.json
readonly HISTORICAL=$DATA_ROOT/sfm/historical/transferred/sfm
readonly E10=$HISTORICAL/reconstructions/experiments/E10_quality_exhaustive_guided_consensus90/light_shirt
readonly MASKS=$HISTORICAL/reconstructions/experiments/light_person_mask_cleanup

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == *login* ]]; then
  echo "ERROR: run this fallback inside the active Slurm GPU allocation" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: pinned MVS environment is absent: $PYTHON" >&2
  exit 2
fi

readonly RUNTIME=$MVS_DATA/runtime/pycolmap-$SLURM_JOB_ID
mkdir -p \
  "$RUNTIME/tmp" "$RUNTIME/xdg-cache" "$RUNTIME/xdg-runtime" \
  "$RUNTIME/pip-cache" "$RUNTIME/conda-pkgs" "$RUNTIME/torch-cache" \
  "$RUNTIME/huggingface-cache" "$MVS_DATA/logs"
chmod 700 "$RUNTIME/xdg-runtime"
export TMPDIR=$RUNTIME/tmp
export XDG_CACHE_HOME=$RUNTIME/xdg-cache
export XDG_RUNTIME_DIR=$RUNTIME/xdg-runtime
export PIP_CACHE_DIR=$RUNTIME/pip-cache
export CONDA_PKGS_DIRS=$RUNTIME/conda-pkgs
export TORCH_HOME=$RUNTIME/torch-cache
export HF_HOME=$RUNTIME/huggingface-cache
export PYTHONPYCACHEPREFIX=$RUNTIME/pycache
export OMP_NUM_THREADS=16

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -B "$MVS_CODE/run_mvs.py" stage-inputs \
  --experiment light_e10_colmap_mvs_1600 \
  --images-source "$E10/images" \
  --model-source "$E10/colmap/sparse/0" \
  --masks-source "$MASKS/masks" \
  --mask-manifest-source "$MASKS/mask_manifest.json" \
  --resume
"$PYTHON" -B "$MVS_CODE/run_mvs.py" plan --config "$CONFIG"
"$PYTHON" -B "$MVS_CODE/run_mvs.py" validate --config "$CONFIG" --runtime
"$PYTHON" -B "$MVS_CODE/run_mvs.py" run --config "$CONFIG" --resume
