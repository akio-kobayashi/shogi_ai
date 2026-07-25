#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-uv}"
BACKEND="${1:-cpu}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
if [[ $# -gt 0 ]]; then shift; fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "this experiment environment targets Linux/WSL2" >&2
  exit 2
fi

case "${BACKEND}" in
  cpu)
    MIN_FREE_GB="${MIN_FREE_GB:-3}"
    TORCH_BACKEND="cpu"
    ;;
  cuda)
    MIN_FREE_GB="${MIN_FREE_GB:-15}"
    TORCH_BACKEND="${CUDA_BACKEND:-cu130}"
    ;;
  rocm)
    MIN_FREE_GB="${MIN_FREE_GB:-15}"
    TORCH_BACKEND="${ROCM_BACKEND:-rocm7.1}"
    ;;
  *)
    echo "unknown backend: ${BACKEND}; use cpu, cuda, or rocm" >&2
    exit 2
    ;;
esac

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi

cache_dir="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/uv}"
mkdir -p "${cache_dir}"
cache_free_kb="$(df -Pk "${cache_dir}" | awk 'NR == 2 {print $4}')"
project_free_kb="$(df -Pk "${SCRIPT_DIR}" | awk 'NR == 2 {print $4}')"
required_kb="$((MIN_FREE_GB * 1024 * 1024))"
if [[ "${SKIP_DISK_CHECK:-0}" != "1" ]] && {
  [[ "${cache_free_kb}" -lt "${required_kb}" ]] ||
  [[ "${project_free_kb}" -lt "${required_kb}" ]];
}; then
  echo "${BACKEND} setup requires at least ${MIN_FREE_GB} GiB free" >&2
  echo "cache: ${cache_dir} ($((${cache_free_kb} / 1024 / 1024)) GiB free)" >&2
  echo "project: ${SCRIPT_DIR} ($((${project_free_kb} / 1024 / 1024)) GiB free)" >&2
  echo "set UV_CACHE_DIR to a larger filesystem, or SKIP_DISK_CHECK=1 to override" >&2
  exit 2
fi

backend_marker="${SCRIPT_DIR}/.venv/.torch-backend"
environment_id="${BACKEND}:${TORCH_BACKEND}:${TORCH_VERSION}"
if [[ -f "${backend_marker}" ]]; then
  installed_environment="$(<"${backend_marker}")"
  if [[ "${installed_environment}" != "${environment_id}" ]]; then
    echo "existing .venv uses ${installed_environment}; remove .venv before switching to ${environment_id}" >&2
    exit 2
  fi
fi

"${UV_BIN}" sync \
  --project "${SCRIPT_DIR}" \
  --frozen \
  --inexact \
  "$@"

"${UV_BIN}" pip install \
  --python "${SCRIPT_DIR}/.venv/bin/python" \
  "torch==${TORCH_VERSION}" \
  --torch-backend "${TORCH_BACKEND}"

printf "%s\n" "${environment_id}" > "${backend_marker}"

"${SCRIPT_DIR}/.venv/bin/python" -c \
  "import cshogi, importlib.metadata as m, torch; print('cshogi:', m.version('cshogi')); print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('hip:', torch.version.hip); print('accelerator available:', torch.cuda.is_available())"

echo "environment ready: ${SCRIPT_DIR}/.venv (${environment_id})"
