#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_DIR}/data/datasets/train.jsonl}"
VALIDATION_JSONL="${VALIDATION_JSONL:-${PROJECT_DIR}/data/datasets/validation.jsonl}"
EVALUATION_JSONL="${EVALUATION_JSONL:-${PROJECT_DIR}/data/datasets/evaluation.jsonl}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
AMP="${AMP:-auto}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
if [[ "${MAX_SEQ_LEN}" != "512" ]]; then
  echo "MAX_SEQ_LEN must be 512 for this script" >&2
  exit 2
fi

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/transformer_size_compare_512}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_PROBES="${RUN_PROBES:-1}"
PROBE_MODE="${PROBE_MODE:-standard}"
PROBE_OUTPUT_SUFFIX="${PROBE_OUTPUT_SUFFIX:-probes}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
LOG_FILE="${LOG_FILE:-${RESULTS_ROOT}/run_with_probes.log}"

if [[ "${LOGGING_INITIALIZED:-0}" -ne 1 ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  export LOGGING_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

mkdir -p "${RESULTS_ROOT}"

if [[ "${RUN_TRAIN}" -ne 0 ]]; then
  echo "start training stage"
  MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
  RESULTS_ROOT="${RESULTS_ROOT}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
  AMP="${AMP}" \
  TRAIN_JSONL="${TRAIN_JSONL}" \
  VALIDATION_JSONL="${VALIDATION_JSONL}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  EPOCHS="${EPOCHS}" \
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \
  EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA}" \
  PROGRESS_EVERY="${PROGRESS_EVERY}" \
    "${PROJECT_DIR}/scripts/compare_transformer_sizes_maxseq512.sh"
else
  echo "skip training stage (RUN_TRAIN=0)"
fi

if [[ "${RUN_PROBES}" -eq 0 ]]; then
  echo "run complete: training only"
  exit 0
fi

for size in small base large; do
  checkpoint="${RESULTS_ROOT}/seed_${SEED}/${size}/best.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "checkpoint not found: ${checkpoint}" >&2
    echo "set RUN_TRAIN=1 or provide pretrained checkpoints" >&2
    exit 2
  fi

  probe_output_dir="${RESULTS_ROOT}/seed_${SEED}/${size}/${PROBE_OUTPUT_SUFFIX}"
  echo "run probe for size=${size} checkpoint=${checkpoint}"

  PYTHON_BIN="${PYTHON_BIN}" \
  CHECKPOINT="${checkpoint}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_JSONL="${TRAIN_JSONL}" \
  VALIDATION_JSONL="${VALIDATION_JSONL}" \
  EVALUATION_JSONL="${EVALUATION_JSONL}" \
  OUTPUT_DIR="${probe_output_dir}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_probe_evaluation.sh" "${PROBE_MODE}"
done

echo "run_complete results_root=${RESULTS_ROOT}"
