#!/usr/bin/env bash
# 学習中または学習後のTensorBoardを，指定したresultsディレクトリから起動する。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TENSORBOARD_BIN="${TENSORBOARD_BIN:-${SCRIPT_DIR}/.venv/bin/tensorboard}"
[[ $# -ge 1 ]] || { echo "Usage: $0 RESULTS_DIR [TensorBoard options]" >&2; exit 2; }
RESULTS_DIR="$1"; shift
[[ -x "${TENSORBOARD_BIN}" ]] || { echo "tensorboard is unavailable: run ./setup_env.sh first" >&2; exit 1; }
exec "${TENSORBOARD_BIN}" --logdir "${RESULTS_DIR}" --host "${TENSORBOARD_HOST:-127.0.0.1}" --port "${TENSORBOARD_PORT:-6006}" "$@"
