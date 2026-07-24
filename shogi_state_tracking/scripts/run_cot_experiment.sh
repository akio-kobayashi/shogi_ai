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
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
RUN_PROBES="${RUN_PROBES:-1}"

mkdir -p "${TRACE_DIR}" "${OUTPUT_DIR}"

for split in train validation evaluation
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

"${PYTHON_BIN}" "${PROJECT_DIR}/evaluate_reasoning.py" \
  --checkpoint "${OUTPUT_DIR}/training/best.pt" \
  --vocab "${VOCAB_PATH}" \
  --trace-jsonl "${TRACE_DIR}/evaluation.jsonl" \
  --output-dir "${OUTPUT_DIR}/evaluation" \
  --seed "${SEED}" \
  --device "${DEVICE}"

if [[ "${RUN_PROBES}" -eq 1 ]]; then
  CHECKPOINT="${BASE_CHECKPOINT}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_JSONL="${TRAIN_GAMES}" \
  VALIDATION_JSONL="${VALIDATION_GAMES}" \
  EVALUATION_JSONL="${EVALUATION_GAMES}" \
  OUTPUT_DIR="${OUTPUT_DIR}/probes-answer-only" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_probe_evaluation.sh" standard

  CHECKPOINT="${OUTPUT_DIR}/training/best.pt" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_JSONL="${TRAIN_GAMES}" \
  VALIDATION_JSONL="${VALIDATION_GAMES}" \
  EVALUATION_JSONL="${EVALUATION_GAMES}" \
  OUTPUT_DIR="${OUTPUT_DIR}/probes-cot" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_probe_evaluation.sh" standard
fi
