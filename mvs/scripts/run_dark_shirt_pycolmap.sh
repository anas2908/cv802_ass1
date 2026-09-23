#!/usr/bin/env bash
# Start only after the coordinator confirms GPU availability in the existing job.
set -euo pipefail
umask 027
readonly PROJECT_CODE=/home/anas.khan/cv802_project/project1/ass1
readonly MVS_CODE=$PROJECT_CODE/mvs
readonly MVS_DATA=/l/users/anas.khan/cv_802_ass1/mvs
readonly MVS_PYTHON=$MVS_DATA/envs/mvs-engine/bin/python
readonly EXPERIMENT=dark_e3_colmap_mvs_1024_raw_v1
readonly CONFIG=$MVS_CODE/configs/dark_e3_1024_raw_pycolmap.json
readonly RUNTIME=$MVS_DATA/runtime/dark-$EXPERIMENT
readonly TELEMETRY_ID=${1:-command-v1}
if [[ ! "$TELEMETRY_ID" =~ ^command-v[1-9][0-9]*$ ]]; then
  echo "Telemetry argument must be command-v1, command-v2, etc." >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == *login* ]]; then
  echo "Run inside the existing authorized Slurm GPU step." >&2
  exit 2
fi
if [[ ! -f "$MVS_DATA/experiments/$EXPERIMENT/manifests/dark_preparation.json" ]]; then
  echo "Verified dark input preparation is required first." >&2
  exit 2
fi
mkdir -p "$RUNTIME/tmp" "$RUNTIME/cache" "$RUNTIME/xdg-runtime"
chmod 700 "$RUNTIME/xdg-runtime"
export TMPDIR=$RUNTIME/tmp TMP=$RUNTIME/tmp TEMP=$RUNTIME/tmp
export XDG_CACHE_HOME=$RUNTIME/cache XDG_RUNTIME_DIR=$RUNTIME/xdg-runtime
export PIP_CACHE_DIR=$RUNTIME/cache/pip CONDA_PKGS_DIRS=$RUNTIME/cache/conda
export TORCH_HOME=$RUNTIME/cache/torch HF_HOME=$RUNTIME/cache/huggingface
export MPLCONFIGDIR=$RUNTIME/cache/matplotlib PYTHONPYCACHEPREFIX=$RUNTIME/cache/pycache
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec "$MVS_PYTHON" -B "$PROJECT_CODE/scripts/hpc/measure_process.py" \
  --output "$MVS_DATA/telemetry/$EXPERIMENT-$TELEMETRY_ID" -- \
  "$MVS_PYTHON" -B "$MVS_CODE/run_mvs.py" run --config "$CONFIG" --resume
