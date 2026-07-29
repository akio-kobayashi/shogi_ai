#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
TRAIN_GAMES="${TRAIN_GAMES:-${PROJECT_DIR}/data/datasets/train.jsonl}"
VALIDATION_GAMES="${VALIDATION_GAMES:-${PROJECT_DIR}/data/datasets/validation.jsonl}"
EVALUATION_GAMES="${EVALUATION_GAMES:-${PROJECT_DIR}/data/datasets/evaluation.jsonl}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/results/2x2}"
LOG_FILE="${LOG_FILE:-${EXPERIMENT_DIR}/evaluation.log}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
RUN_PROBES="${RUN_PROBES:-1}"
PROBE_MODE="${PROBE_MODE:-standard}"

mkdir -p "${EXPERIMENT_DIR}"
if [[ "${LOGGING_INITIALIZED:-0}" -ne 1 ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  export LOGGING_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi
echo "run_start experiment_dir=${EXPERIMENT_DIR} log_file=${LOG_FILE}"

for model_type in vanilla t2mlr
do
  model_root="${EXPERIMENT_DIR}/seed_${SEED}/${model_type}"
  BASE_CHECKPOINT="${model_root}/answer-only/best.pt" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_GAMES="${TRAIN_GAMES}" \
  VALIDATION_GAMES="${VALIDATION_GAMES}" \
  EVALUATION_GAMES="${EVALUATION_GAMES}" \
  TRACE_DIR="${model_root}/traces" \
  OUTPUT_DIR="${model_root}/cot" \
  RUN_PROBES="${RUN_PROBES}" \
  PROBE_MODE="${PROBE_MODE}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_cot_evaluation.sh" "${model_type}" "$@"
done

"${PYTHON_BIN}" "${PROJECT_DIR}/summarize_2x2.py" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --seed "${SEED}"
echo "run_complete experiment_dir=${EXPERIMENT_DIR} log_file=${LOG_FILE}"
