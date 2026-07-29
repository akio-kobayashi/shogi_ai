#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_TYPE="${1:-vanilla}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT_DIR}/results/training/${MODEL_TYPE}/seed_20260724/best.pt}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
TRAIN_GAMES="${TRAIN_GAMES:-${PROJECT_DIR}/data/datasets/train.jsonl}"
VALIDATION_GAMES="${VALIDATION_GAMES:-${PROJECT_DIR}/data/datasets/validation.jsonl}"
EVALUATION_GAMES="${EVALUATION_GAMES:-${PROJECT_DIR}/data/datasets/evaluation.jsonl}"
TRACE_DIR="${TRACE_DIR:-${PROJECT_DIR}/data/traces/${MODEL_TYPE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/cot/${MODEL_TYPE}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/run.log}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
RUN_EVALUATION="${RUN_EVALUATION:-0}"
RUN_PROBES="${RUN_PROBES:-1}"
GENERATE_EVALUATION_TRACE="${GENERATE_EVALUATION_TRACE:-0}"
POSITIONS_PER_GAME="${POSITIONS_PER_GAME:-4}"
LINES="${LINES:-3}"
LINE_BATCH_SIZE="${LINE_BATCH_SIZE:-0}"
LINE_LENGTH="${LINE_LENGTH:-4}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_GAMES="${MAX_GAMES:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

mkdir -p "${TRACE_DIR}" "${OUTPUT_DIR}"
if [[ "${LOGGING_INITIALIZED:-0}" -ne 1 ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  export LOGGING_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi
echo "run_start model_type=${MODEL_TYPE} log_file=${LOG_FILE} log_level=${LOG_LEVEL}"

SPLITS=(train validation)
if [[ "${GENERATE_EVALUATION_TRACE}" -eq 1 ]]; then
  SPLITS+=(evaluation)
fi

for split in "${SPLITS[@]}"
do
  case "${split}" in
    train) INPUT="${TRAIN_GAMES}" ;;
    validation) INPUT="${VALIDATION_GAMES}" ;;
    evaluation) INPUT="${EVALUATION_GAMES}" ;;
  esac
  "${PYTHON_BIN}" "${PROJECT_DIR}/generate_reasoning_traces.py" \
    --checkpoint "${BASE_CHECKPOINT}" \
    --vocab "${VOCAB_PATH}" \
    --input-jsonl "${INPUT}" \
    --output-jsonl "${TRACE_DIR}/${split}.jsonl" \
    --positions-per-game "${POSITIONS_PER_GAME}" \
    --lines "${LINES}" \
    --line-batch-size "${LINE_BATCH_SIZE}" \
    --line-length "${LINE_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-games "${MAX_GAMES}" \
    --progress-every "${PROGRESS_EVERY}" \
    --log-file "${TRACE_DIR}/${split}.log" \
    --summary-json "${TRACE_DIR}/${split}.summary.json" \
    --log-level "${LOG_LEVEL}" \
    --seed "${SEED}" \
    --device "${DEVICE}"
done

INIT_CHECKPOINT="${BASE_CHECKPOINT}" \
VOCAB_PATH="${VOCAB_PATH}" \
TRAIN_JSONL="${TRACE_DIR}/train.jsonl" \
VALIDATION_JSONL="${TRACE_DIR}/validation.jsonl" \
OUTPUT_DIR="${OUTPUT_DIR}/training" \
SEED="${SEED}" \
DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_training.sh" cot "${MODEL_TYPE}" "$@"

if [[ "${RUN_EVALUATION}" -eq 1 ]]; then
  BASE_CHECKPOINT="${BASE_CHECKPOINT}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_GAMES="${TRAIN_GAMES}" \
  VALIDATION_GAMES="${VALIDATION_GAMES}" \
  EVALUATION_GAMES="${EVALUATION_GAMES}" \
  TRACE_DIR="${TRACE_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  RUN_PROBES="${RUN_PROBES}" \
  POSITIONS_PER_GAME="${POSITIONS_PER_GAME}" \
  LINES="${LINES}" \
  LINE_LENGTH="${LINE_LENGTH}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_P="${TOP_P}" \
  MAX_GAMES="${MAX_GAMES}" \
  PROGRESS_EVERY="${PROGRESS_EVERY}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_cot_evaluation.sh" "${MODEL_TYPE}"
else
  echo "trace generation and CoT training completed"
  echo "evaluation trace generation and evaluation are deferred; run scripts/run_cot_evaluation.sh ${MODEL_TYPE} when ready"
fi
echo "run_complete model_type=${MODEL_TYPE} log_file=${LOG_FILE}"
