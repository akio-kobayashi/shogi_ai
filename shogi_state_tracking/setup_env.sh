#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-uv}"
BACKEND="${1:-cpu}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
ROCM_TRITON_VERSION="${ROCM_TRITON_VERSION:-3.7.1}"
WSL_ROCM_VERSION="7.2"
WSL_TORCH_VERSION="2.9.1"
WSL_TRITON_VERSION="3.5.1"
if [[ $# -gt 0 ]]; then shift; fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "this experiment environment targets Linux/WSL2" >&2
  exit 2
fi

is_wsl=0
if [[ -r /proc/sys/kernel/osrelease ]] &&
  grep -qi microsoft /proc/sys/kernel/osrelease; then
  is_wsl=1
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
    if [[ "${is_wsl}" == "1" ]]; then
      # WSL does not expose native Linux KFD sysfs. Use AMD's validated WSL
      # wheels instead of uv's PyTorch.org ROCm backend.
      TORCH_BACKEND="rocm${WSL_ROCM_VERSION}-wsl"
    else
      TORCH_BACKEND="${ROCM_BACKEND:-rocm7.1}"
    fi
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
if [[ "${BACKEND}" == "rocm" && "${is_wsl}" == "1" ]]; then
  environment_id="${BACKEND}:${TORCH_BACKEND}:torch-${WSL_TORCH_VERSION}:triton-${WSL_TRITON_VERSION}"
elif [[ "${BACKEND}" == "rocm" ]]; then
  environment_id="${environment_id}:triton-${ROCM_TRITON_VERSION}"
fi
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

if [[ "${BACKEND}" == "rocm" && "${is_wsl}" == "1" ]]; then
  python_bin="${SCRIPT_DIR}/.venv/bin/python"
  python_tag="$("${python_bin}" -c \
    "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")"
  case "${python_tag}" in
    cp310|cp312)
      ;;
    *)
      echo "AMD ROCm ${WSL_ROCM_VERSION} WSL wheels require Python 3.10 or 3.12; found ${python_tag}" >&2
      exit 2
      ;;
  esac
  if [[ ! -e /dev/dxg ]]; then
    echo "WSL ROCm requires /dev/dxg; update/install the AMD Windows WSL driver" >&2
    exit 2
  fi
  if [[ ! -r /opt/rocm/lib/libhsa-runtime64.so.1 ]]; then
    echo "WSL ROCm runtime is missing: /opt/rocm/lib/libhsa-runtime64.so.1" >&2
    exit 2
  fi

  amd_wheel_base="https://repo.radeon.com/rocm/manylinux/rocm-rel-${WSL_ROCM_VERSION}"
  torch_wheel="${amd_wheel_base}/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-${python_tag}-${python_tag}-linux_x86_64.whl"
  triton_wheel="${amd_wheel_base}/triton-3.5.1%2Brocm7.2.0.gita272dfa8-${python_tag}-${python_tag}-linux_x86_64.whl"

  # PyTorch.org's native-Linux wheel expects /sys/class/kfd and aborts on WSL.
  # AMD validates these repo.radeon.com wheels together with NumPy 1.26.4.
  "${UV_BIN}" pip uninstall \
    --python "${python_bin}" \
    torch triton triton-rocm || true
  "${UV_BIN}" pip install \
    --python "${python_bin}" \
    "numpy==1.26.4" \
    "${torch_wheel}" \
    "${triton_wheel}"

  # On WSL, use the host ROCm runtime rather than the copy bundled in torch.
  find "${SCRIPT_DIR}/.venv/lib" \
    -path "*/site-packages/torch/lib/libhsa-runtime64.so*" \
    \( -type f -o -type l \) -delete
elif [[ "${BACKEND}" == "rocm" ]]; then
  # torch 2.13.0+rocm7.1 requires triton-rocm 3.7.1, but that wheel is
  # currently discoverable from PyTorch's aggregate index rather than
  # through uv's rocm7.1 backend index.
  "${UV_BIN}" pip install \
    --python "${SCRIPT_DIR}/.venv/bin/python" \
    "triton-rocm==${ROCM_TRITON_VERSION}" \
    --index "https://download.pytorch.org/whl"
  "${UV_BIN}" pip install \
    --python "${SCRIPT_DIR}/.venv/bin/python" \
    "torch==${TORCH_VERSION}" \
    --torch-backend "${TORCH_BACKEND}"
else
  "${UV_BIN}" pip install \
    --python "${SCRIPT_DIR}/.venv/bin/python" \
    "torch==${TORCH_VERSION}" \
    --torch-backend "${TORCH_BACKEND}"
fi

python_bin="${SCRIPT_DIR}/.venv/bin/python"

echo "[verify 1/5] cshogi import"
"${python_bin}" -u -c \
  "import cshogi, importlib.metadata as m; print('cshogi:', m.version('cshogi'), flush=True)"

echo "[verify 2/5] tensorboard import"
"${python_bin}" -u -c \
  "import tensorboard; print('tensorboard:', tensorboard.__version__, flush=True)"

echo "[verify 3/5] torch import"
"${python_bin}" -u -c \
  "import torch; print('torch:', torch.__version__, flush=True); print('cuda:', torch.version.cuda, flush=True); print('hip:', torch.version.hip, flush=True)"

echo "[verify 4/5] joint import (application order: torch, then cshogi)"
"${python_bin}" -u -c \
  "import torch, cshogi; print('joint import: ok', flush=True)"

echo "[verify 5/5] accelerator runtime"
if [[ "${BACKEND}" == "cpu" ]]; then
  "${python_bin}" -u -c \
    "import torch; print('accelerator available:', torch.cuda.is_available(), flush=True)"
else
  "${python_bin}" -u -c \
    "import torch; available = torch.cuda.is_available(); print('accelerator available:', available, flush=True); raise SystemExit(0 if available else 1)"
fi

# Record the environment only after both extension modules and the accelerator
# runtime have passed verification. A failed setup can then be rerun safely.
printf "%s\n" "${environment_id}" > "${backend_marker}"

echo "environment ready: ${SCRIPT_DIR}/.venv (${environment_id})"
