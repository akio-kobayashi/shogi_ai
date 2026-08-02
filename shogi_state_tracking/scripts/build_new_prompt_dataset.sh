#!/usr/bin/env bash
# 新prompt datasetはCSAを置くデータセット作成機だけで構築する．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Pythonが見つかりません: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SOURCE_JSONL_DIR OUTPUT_DATASET_DIR [builder options]" >&2
  exit 2
fi

SOURCE_DIR="$1"
OUTPUT_DIR="$2"
shift 2

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_new_prompt_dataset.py" \
  --input-dir "${SOURCE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" \
  --dataset-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_new_prompt_oracle.py" \
  --dataset-dir "${OUTPUT_DIR}"
