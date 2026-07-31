#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
INPUT_JSONL="${INPUT_JSONL:-${PROJECT_DIR}/data/datasets/evaluation.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${PROJECT_DIR}/data/check_probe/evaluation.jsonl}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-10000}"
MAX_PREFIX_MOVES="${MAX_PREFIX_MOVES:-221}"
START_PLIES="${START_PLIES:-0,24,25,32,33}"
MIN_SUFFIX_MOVES="${MIN_SUFFIX_MOVES:-40}"
SEED="${SEED:-20260724}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "input JSONL is unavailable: ${INPUT_JSONL}" >&2
  exit 2
fi

exec "${PYTHON_BIN}" "${PROJECT_DIR}/create_check_probe_dataset.py" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --samples-per-class "${SAMPLES_PER_CLASS}" \
  --max-prefix-moves "${MAX_PREFIX_MOVES}" \
  --start-plies "${START_PLIES}" \
  --min-suffix-moves "${MIN_SUFFIX_MOVES}" \
  --seed "${SEED}" \
  "$@"
