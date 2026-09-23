#!/usr/bin/env bash
# Run from a CIAI login node. The foreground srun owns one A100 for at most 3h.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_SCRIPT="${SCRIPT_DIR}/ciai_gpu_ui_compute.sh"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec bash "${COMPUTE_SCRIPT}"
fi
if [[ "$(hostname -s)" != *login* ]]; then
  echo "ERROR: run this launcher on a CIAI login node or inside an existing Slurm job." >&2
  exit 2
fi

echo "Requesting one A100 40 GB GPU for up to 3 hours..."
echo "The terminal will remain attached to the job and will show progress and GPU use."
exec srun \
  --job-name=cv802-mvs-vgg-ui \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=96G \
  --gres=gpu:a100-sxm4-40gb:1 \
  --partition=cscc-gpu-p \
  --qos=gpu-debug-qos \
  --time=03:00:00 \
  --signal=B:USR1@300 \
  --exclude=gpu-05 \
  bash "${COMPUTE_SCRIPT}"
