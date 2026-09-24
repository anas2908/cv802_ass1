#!/usr/bin/env bash
# Internal half of run_ciai_gpu_ui.sh. It must execute on the allocated node.
set -Eeuo pipefail
umask 027

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CLUSTER_USER="${USER:?USER is not set}"
export CV802_DATA_ROOT="${CV802_DATA_ROOT:-/l/users/${CLUSTER_USER}/cv802_ass1-runs}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: no active Slurm allocation. Start scripts/run_ciai_gpu_ui.sh instead." >&2
  exit 2
fi
case "${CV802_DATA_ROOT}/" in
  "/l/users/${CLUSTER_USER}/"*) ;;
  *)
    echo "ERROR: CV802_DATA_ROOT must be under /l/users/${CLUSTER_USER}/" >&2
    exit 2
    ;;
esac

mkdir -p "${CV802_DATA_ROOT}/runtime/logs" "${CV802_DATA_ROOT}/runtime/tmp"
test -w "${CV802_DATA_ROOT}"
findmnt --target "${CV802_DATA_ROOT}"
lfs quota -h -u "${CLUSTER_USER}" /l || echo "WARNING: Lustre quota query was unavailable."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

PORT="${CV802_UI_PORT:-8780}"
if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "ERROR: CV802_UI_PORT must be an integer from 1024 to 65535." >&2
  exit 2
fi
TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
NODE="$(hostname -s)"
LOG="${CV802_DATA_ROOT}/runtime/logs/ciai-ui-${SLURM_JOB_ID}.log"
trap 'echo "WARNING: five minutes remain in this allocation. Rerun the launcher afterward to resume unfinished work."' USR1

echo
echo "============================================================"
echo "CV802 GPU UI is allocated on ${NODE}; job ${SLURM_JOB_ID}."
echo "Data root: ${CV802_DATA_ROOT}"
echo
echo "On your LAPTOP, open a second terminal and run:"
echo "ssh -N -L ${PORT}:${NODE}:${PORT} ${CLUSTER_USER}@10.127.79.236"
echo
echo "Then open this exact private URL in your laptop browser:"
echo "http://127.0.0.1:${PORT}/?token=${TOKEN}"
echo
echo "Keep both terminals open. Press Control-C here to release the GPU."
echo "============================================================"
echo

export PYTHONUNBUFFERED=1
export TMPDIR="${CV802_DATA_ROOT}/runtime/tmp"
python3 -B "${SCRIPT_DIR}/ciai_gpu_ui.py" \
  --data-root "${CV802_DATA_ROOT}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --token "${TOKEN}" 2>&1 | tee -a "${LOG}"
exit "${PIPESTATUS[0]}"
