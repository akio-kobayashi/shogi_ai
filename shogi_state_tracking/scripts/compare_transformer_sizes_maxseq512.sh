#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
if [[ "${MAX_SEQ_LEN}" != "512" ]]; then
  echo "MAX_SEQ_LEN_512 must be 512; forced for this script" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_DIR}/data/datasets/train.jsonl}"
VALIDATION_JSONL="${VALIDATION_JSONL:-${PROJECT_DIR}/data/datasets/validation.jsonl}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
AMP="${AMP:-auto}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/transformer_size_compare_512}"
LOG_FILE="${LOG_FILE:-${RESULTS_ROOT}/run.log}"

mkdir -p "${RESULTS_ROOT}"
if [[ "${LOGGING_INITIALIZED:-0}" -ne 1 ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  export LOGGING_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "run_start project_dir=${PROJECT_DIR} results_root=${RESULTS_ROOT} max_seq_len=${MAX_SEQ_LEN}"

for size in small base large; do
  case "${size}" in
    small)
      D_MODEL=128
      N_LAYERS=4
      N_HEADS=4
      D_FF=512
      BATCH_SIZE=8
      ;;
    base)
      D_MODEL=256
      N_LAYERS=8
      N_HEADS=8
      D_FF=1024
      BATCH_SIZE=4
      ;;
    large)
      D_MODEL=384
      N_LAYERS=12
      N_HEADS=12
      D_FF=1536
      BATCH_SIZE=2
      ;;
    *)
      echo "unknown size: ${size}" >&2
      exit 2
      ;;
  esac

  output_dir="${RESULTS_ROOT}/seed_${SEED}/${size}"
  mkdir -p "${output_dir}"

  echo "size=${size} d_model=${D_MODEL} n_layers=${N_LAYERS} n_heads=${N_HEADS} d_ff=${D_FF} batch=${BATCH_SIZE} output_dir=${output_dir}"

  PYTHON_BIN="${PYTHON_BIN}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_JSONL="${TRAIN_JSONL}" \
  VALIDATION_JSONL="${VALIDATION_JSONL}" \
  OUTPUT_DIR="${output_dir}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
  AMP="${AMP}" \
  MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  EPOCHS="${EPOCHS}" \
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \
  EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA}" \
  PROGRESS_EVERY="${PROGRESS_EVERY}" \
    "${PROJECT_DIR}/scripts/run_training.sh" \
      pretrain vanilla \
      --d-model "${D_MODEL}" \
      --n-layers "${N_LAYERS}" \
      --n-heads "${N_HEADS}" \
      --d-ff "${D_FF}" \
      "$@"

done

echo "run_complete results_root=${RESULTS_ROOT} log_file=${LOG_FILE}"
