#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/scoped_datasets}"

TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/datasets/train.jsonl}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${PROJECT_DIR}/annotate_position_scopes.py" \
  --train-jsonl "${TRAIN_JSONL}" \
  --input-dir "${DATA_DIR}/datasets" \
  --output-dir "${OUTPUT_DIR}"

echo "scoped datasets written to ${OUTPUT_DIR}"
