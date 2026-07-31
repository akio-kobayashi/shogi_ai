#!/usr/bin/env bash
set -euo pipefail

# 王手probe用の学習・検証・評価集合を，元のsplitを跨がずに作る。
# 各split内で正例（王手）と負例（非王手）を同数にする。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data/check_probe}"
MAX_PREFIX_MOVES="${MAX_PREFIX_MOVES:-221}"
START_PLIES="${START_PLIES:-0,24,25,32,33}"
MIN_SUFFIX_MOVES="${MIN_SUFFIX_MOVES:-40}"
SEED="${SEED:-20260724}"
TRAIN_SAMPLES_PER_CLASS="${TRAIN_SAMPLES_PER_CLASS:-20000}"
VALIDATION_SAMPLES_PER_CLASS="${VALIDATION_SAMPLES_PER_CLASS:-5000}"
EVALUATION_SAMPLES_PER_CLASS="${EVALUATION_SAMPLES_PER_CLASS:-10000}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi

build_split() {
  local split="$1"
  local samples="$2"
  local input="${DATASET_DIR}/${split}.jsonl"
  local output="${OUTPUT_DIR}/${split}.jsonl"
  if [[ ! -f "${input}" ]]; then
    echo "input JSONL is unavailable: ${input}" >&2
    exit 2
  fi
  echo "[check-probe dataset] split=${split} samples_per_class=${samples}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/create_check_probe_dataset.py" \
    --input-jsonl "${input}" \
    --output-jsonl "${output}" \
    --samples-per-class "${samples}" \
    --max-prefix-moves "${MAX_PREFIX_MOVES}" \
    --start-plies "${START_PLIES}" \
    --min-suffix-moves "${MIN_SUFFIX_MOVES}" \
    --seed "${SEED}"
}

build_split train "${TRAIN_SAMPLES_PER_CLASS}"
build_split validation "${VALIDATION_SAMPLES_PER_CLASS}"
build_split evaluation "${EVALUATION_SAMPLES_PER_CLASS}"
