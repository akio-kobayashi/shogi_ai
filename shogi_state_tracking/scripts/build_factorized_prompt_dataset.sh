#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
INPUT_DIR="${1:?usage: $0 INPUT_NEW_PROMPT_DATASET OUTPUT_DATASET}"
OUTPUT_DIR="${2:?output dataset directory is required}"
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/build_factorized_prompt_dataset.py" \
  --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/validate_factorized_v3_dataset.py" \
  --dataset-dir "${OUTPUT_DIR}"
