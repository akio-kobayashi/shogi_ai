#!/usr/bin/env bash
# 複数のresults rootを検査してfactorized-v3分析archiveへまとめる．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_FALLBACK:-python3}"
fi

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/collect_factorized_analysis.py" "$@"
