#!/usr/bin/env bash
# 実験結果から，転送用の小さな分析packageを作る．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_FALLBACK:-python3}"
fi

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/package_analysis_results.py" "$@"
