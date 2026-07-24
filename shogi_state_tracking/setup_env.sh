#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRIPT_DIR}/.uv-cache}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "this experiment environment targets Linux/WSL2" >&2
  exit 2
fi

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi

"${UV_BIN}" sync \
  --project "${SCRIPT_DIR}" \
  --frozen \
  "$@"

"${SCRIPT_DIR}/.venv/bin/python" -c \
  "import cshogi, importlib.metadata as m, torch; print('cshogi:', m.version('cshogi')); print('torch:', torch.__version__)"

echo "environment ready: ${SCRIPT_DIR}/.venv"
