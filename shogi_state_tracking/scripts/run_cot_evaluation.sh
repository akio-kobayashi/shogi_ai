#!/usr/bin/env bash
set -euo pipefail

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
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/evaluation.log}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
COT_CHECKPOINT="${COT_CHECKPOINT:-${OUTPUT_DIR}/training/best.pt}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
RUN_PROBES="${RUN_PROBES:-1}"
PROBE_MODE="${PROBE_MODE:-standard}"
GENERATE_EVALUATION_TRACE="${GENERATE_EVALUATION_TRACE:-1}"
POSITIONS_PER_GAME="${POSITIONS_PER_GAME:-4}"
LINES="${LINES:-3}"
LINE_BATCH_SIZE="${LINE_BATCH_SIZE:-0}"
LINE_LENGTH="${LINE_LENGTH:-4}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_GAMES="${MAX_GAMES:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi

for required_file in \
  "${COT_CHECKPOINT}" \
  "${VOCAB_PATH}"
do
  if [[ ! -f "${required_file}" ]]; then
    echo "required file is unavailable: ${required_file}" >&2
    exit 2
  fi
done

mkdir -p "${TRACE_DIR}" "${OUTPUT_DIR}/evaluation"
if [[ "${LOGGING_INITIALIZED:-0}" -ne 1 ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  export LOGGING_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi
echo "run_start model_type=${MODEL_TYPE} log_file=${LOG_FILE} log_level=${LOG_LEVEL}"

if [[ ! -f "${TRACE_DIR}/evaluation.jsonl" ]]; then
  if [[ "${GENERATE_EVALUATION_TRACE}" -ne 1 ]]; then
    echo "evaluation trace is unavailable: ${TRACE_DIR}/evaluation.jsonl" >&2
    echo "set GENERATE_EVALUATION_TRACE=1 to create it now" >&2
    exit 2
  fi
  for required_file in "${BASE_CHECKPOINT}" "${EVALUATION_GAMES}"
  do
    if [[ ! -f "${required_file}" ]]; then
      echo "required file is unavailable for evaluation trace generation: ${required_file}" >&2
      exit 2
    fi
  done
  GENERATE_COMMAND=(
    "${PYTHON_BIN}" "${PROJECT_DIR}/generate_reasoning_traces.py"
    --checkpoint "${BASE_CHECKPOINT}"
    --vocab "${VOCAB_PATH}"
    --input-jsonl "${EVALUATION_GAMES}"
    --output-jsonl "${TRACE_DIR}/evaluation.jsonl"
    --positions-per-game "${POSITIONS_PER_GAME}"
    --lines "${LINES}"
    --line-batch-size "${LINE_BATCH_SIZE}"
    --line-length "${LINE_LENGTH}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-games "${MAX_GAMES}"
    --progress-every "${PROGRESS_EVERY}"
    --log-file "${TRACE_DIR}/evaluation.log"
    --summary-json "${TRACE_DIR}/evaluation.summary.json"
    --log-level "${LOG_LEVEL}"
    --seed "${SEED}"
    --device "${DEVICE}"
  )
  echo "generating deferred evaluation trace"
  printf "command:"
  printf " %q" "${GENERATE_COMMAND[@]}"
  printf "\n"
  "${GENERATE_COMMAND[@]}"
fi

if [[ ! -f "${TRACE_DIR}/evaluation.jsonl" ]]; then
  echo "evaluation trace generation did not produce ${TRACE_DIR}/evaluation.jsonl" >&2
  exit 2
fi

COMMAND=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/evaluate_reasoning.py"
  --checkpoint "${COT_CHECKPOINT}"
  --vocab "${VOCAB_PATH}"
  --trace-jsonl "${TRACE_DIR}/evaluation.jsonl"
  --output-dir "${OUTPUT_DIR}/evaluation"
  --seed "${SEED}"
  --device "${DEVICE}"
  "$@"
)

echo "evaluating CoT reasoning"
printf "command:"
printf " %q" "${COMMAND[@]}"
printf "\n"
"${COMMAND[@]}"

if [[ "${RUN_PROBES}" -eq 1 ]]; then
  CHECKPOINT="${BASE_CHECKPOINT}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_JSONL="${TRAIN_GAMES}" \
  VALIDATION_JSONL="${VALIDATION_GAMES}" \
  EVALUATION_JSONL="${EVALUATION_GAMES}" \
  OUTPUT_DIR="${OUTPUT_DIR}/probes-answer-only" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_probe_evaluation.sh" "${PROBE_MODE}"

  CHECKPOINT="${COT_CHECKPOINT}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_JSONL="${TRAIN_GAMES}" \
  VALIDATION_JSONL="${VALIDATION_GAMES}" \
  EVALUATION_JSONL="${EVALUATION_GAMES}" \
  OUTPUT_DIR="${OUTPUT_DIR}/probes-cot" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_probe_evaluation.sh" "${PROBE_MODE}"
fi
echo "run_complete model_type=${MODEL_TYPE} log_file=${LOG_FILE}"
