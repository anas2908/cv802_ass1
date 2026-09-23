#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/home/anas.khan/cv802_project/project1/ass1/mvs
DATA_ROOT=/l/users/anas.khan/cv_802_ass1/mvs
DEPENDENCY_ROOT=${DATA_ROOT}/dependencies
CACHE_ROOT=${DATA_ROOT}/cache/singularity
TMP_ROOT=${DATA_ROOT}/runtime/singularity-tmp

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: fetch and verify the CUDA container inside the active GPU allocation" >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" || ! -w "${DATA_ROOT}" ]]; then
  echo "ERROR: MVS data root is missing or not writable: ${DATA_ROOT}" >&2
  exit 2
fi
findmnt --target "${DATA_ROOT}" >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# shellcheck disable=SC1091
source "${CODE_ROOT}/dependencies/COLMAP_CONTAINER.lock"
mkdir -p "${DEPENDENCY_ROOT}" "${CACHE_ROOT}" "${TMP_ROOT}"
export SINGULARITY_CACHEDIR=${CACHE_ROOT}
export SINGULARITY_TMPDIR=${TMP_ROOT}

SIF=${DEPENDENCY_ROOT}/${COLMAP_SIF_NAME}
SIF_SHA=${SIF}.sha256
OCI_DIGEST=${SIF}.oci-digest

CURRENT_OCI_DIGEST=$(skopeo inspect --format '{{.Digest}}' "${COLMAP_OCI_SOURCE}")
if [[ -f "${OCI_DIGEST}" ]]; then
  RECORDED_OCI_DIGEST=$(<"${OCI_DIGEST}")
  if [[ "${CURRENT_OCI_DIGEST}" != "${RECORDED_OCI_DIGEST}" ]]; then
    echo "ERROR: pinned registry tag digest changed; refusing replacement" >&2
    exit 2
  fi
else
  printf '%s\n' "${CURRENT_OCI_DIGEST}" >"${OCI_DIGEST}"
fi

if [[ ! -f "${SIF}" ]]; then
  TEMP_SIF=$(mktemp --tmpdir="${TMP_ROOT}" colmap-pull-XXXXXXXX.sif)
  trap 'rm -f "${TEMP_SIF}"' EXIT
  singularity pull --force "${TEMP_SIF}" "${COLMAP_OCI_SOURCE}"
  mv "${TEMP_SIF}" "${SIF}"
  trap - EXIT
fi

if [[ -f "${SIF_SHA}" ]]; then
  (cd "${DEPENDENCY_ROOT}" && sha256sum --check "$(basename "${SIF_SHA}")")
else
  (cd "${DEPENDENCY_ROOT}" && sha256sum "${COLMAP_SIF_NAME}" >"${COLMAP_SIF_NAME}.sha256")
fi

HELP_OUTPUT=$(singularity exec --nv --bind "${DATA_ROOT}:${DATA_ROOT}" "${SIF}" colmap -h)
printf '%s\n' "${HELP_OUTPUT}" | head -n 8
if [[ "${HELP_OUTPUT}" != *"COLMAP ${COLMAP_RELEASE}"* ]]; then
  echo "ERROR: container is not the locked COLMAP ${COLMAP_RELEASE} release" >&2
  exit 2
fi
singularity inspect --json "${SIF}" >"${SIF}.inspect.json"
