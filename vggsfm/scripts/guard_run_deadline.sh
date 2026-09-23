#!/usr/bin/env bash
# Stop one official VGGSfM child before its existing Slurm allocation expires.
# The engine remains alive long enough to convert the child's non-zero exit
# into an immutable failed receipt. This script never requests an allocation.
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: guard_run_deadline.sh RUN_ID" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: deadline guard must run inside the existing Slurm job." >&2
  exit 2
fi

RUN_ID="$1"
case "${RUN_ID}" in
  *[!A-Za-z0-9._-]*|'') echo "ERROR: unsafe run id" >&2; exit 2 ;;
esac

METHOD_ROOT="/l/users/anas.khan/cv_802_ass1/vggsfm"
RECEIPT="${METHOD_ROOT}/experiments/${RUN_ID}/attempts/0001/receipt.json"
MANIFEST="${METHOD_ROOT}/outputs/${RUN_ID}/manifest.json"
LOG="${METHOD_ROOT}/logs/${RUN_ID}-deadline-guard.log"
MARGIN_SECONDS="${VGG_DEADLINE_MARGIN_SECONDS:-180}"
if [[ ! "${MARGIN_SECONDS}" =~ ^[0-9]+$ ]] || (( MARGIN_SECONDS < 60 )); then
  echo "ERROR: deadline margin must be an integer of at least 60 seconds" >&2
  exit 2
fi

mkdir -p "${METHOD_ROOT}/logs"
exec >>"${LOG}" 2>&1
echo "guard_start=$(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID} run=${RUN_ID} margin=${MARGIN_SECONDS}"

while true; do
  if [[ -f "${MANIFEST}" ]]; then
    echo "guard_exit=published at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
  fi
  if [[ -f "${RECEIPT}" ]] && grep -Eq '"status": "(complete|failed)"' "${RECEIPT}"; then
    echo "guard_exit=terminal_receipt at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
  fi

  END_TIME="$({ scontrol show job "${SLURM_JOB_ID}" -o || true; } | sed -n 's/.* EndTime=\([^ ]*\).*/\1/p')"
  if [[ -z "${END_TIME}" || "${END_TIME}" == "Unknown" ]]; then
    echo "ERROR: cannot read Slurm EndTime for job ${SLURM_JOB_ID}" >&2
    exit 3
  fi
  END_EPOCH="$(date -d "${END_TIME}" +%s)"
  NOW_EPOCH="$(date +%s)"
  REMAINING=$(( END_EPOCH - NOW_EPOCH ))
  if (( REMAINING <= MARGIN_SECONDS )); then
    mapfile -t PIDS < <(
      pgrep -u "$(id -u)" -f "/official_demo_compat.py .*SCENE_DIR=.*${RUN_ID}/" || true
    )
    if (( ${#PIDS[@]} != 1 )); then
      echo "ERROR: expected one official adapter child at deadline, found ${#PIDS[@]}: ${PIDS[*]-}" >&2
      exit 4
    fi
    echo "guard_action=SIGTERM pid=${PIDS[0]} remaining=${REMAINING} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    kill -TERM "${PIDS[0]}"
    # The parent engine normally writes its terminal failed receipt within a
    # few seconds. The allocation itself remains the final hard boundary.
    for _ in $(seq 1 30); do
      if [[ -f "${RECEIPT}" ]] && grep -Eq '"status": "(complete|failed)"' "${RECEIPT}"; then
        echo "guard_exit=terminal_after_signal at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        exit 0
      fi
      sleep 2
    done
    echo "ERROR: engine did not publish a terminal receipt after SIGTERM" >&2
    exit 5
  fi
  sleep 15
done
