#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
TRAIN_GAMES="${TRAIN_GAMES:-${PROJECT_DIR}/data/datasets/train.jsonl}"
VALIDATION_GAMES="${VALIDATION_GAMES:-${PROJECT_DIR}/data/datasets/validation.jsonl}"
EVALUATION_GAMES="${EVALUATION_GAMES:-${PROJECT_DIR}/data/datasets/evaluation.jsonl}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/results/2x2}"
LOG_FILE="${LOG_FILE:-${EXPERIMENT_DIR}/run.log}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
RUN_EVALUATION="${RUN_EVALUATION:-0}"
RUN_PROBES="${RUN_PROBES:-1}"

mkdir -p "${EXPERIMENT_DIR}"
if [[ "${LOGGING_INITIALIZED:-0}" -ne 1 ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  export LOGGING_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi
echo "run_start experiment_dir=${EXPERIMENT_DIR} log_file=${LOG_FILE}"

for model_type in vanilla t2mlr
do
  base_dir="${EXPERIMENT_DIR}/seed_${SEED}/${model_type}/answer-only"
  extra_arguments=()
  if [[ "${model_type}" == "vanilla" ]]; then
    extra_arguments+=(--match-t2mlr)
  fi

  PYTHON_BIN="${PYTHON_BIN}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_JSONL="${TRAIN_GAMES}" \
  VALIDATION_JSONL="${VALIDATION_GAMES}" \
  OUTPUT_DIR="${base_dir}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_training.sh" \
      pretrain "${model_type}" "${extra_arguments[@]}" "$@"

  PYTHON_BIN="${PYTHON_BIN}" \
  BASE_CHECKPOINT="${base_dir}/best.pt" \
  VOCAB_PATH="${VOCAB_PATH}" \
  TRAIN_GAMES="${TRAIN_GAMES}" \
  VALIDATION_GAMES="${VALIDATION_GAMES}" \
  EVALUATION_GAMES="${EVALUATION_GAMES}" \
  TRACE_DIR="${EXPERIMENT_DIR}/seed_${SEED}/${model_type}/traces" \
  OUTPUT_DIR="${EXPERIMENT_DIR}/seed_${SEED}/${model_type}/cot" \
  RUN_EVALUATION="${RUN_EVALUATION}" \
  RUN_PROBES="${RUN_PROBES}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_cot_experiment.sh" "${model_type}" "$@"
done

if [[ "${RUN_EVALUATION}" -eq 1 ]]; then
  "${PYTHON_BIN}" "${PROJECT_DIR}/summarize_2x2.py" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --seed "${SEED}"
else
  echo "training and trace generation completed"
  echo "evaluation is deferred; run scripts/run_2x2_evaluation.sh when ready"
fi
echo "run_complete experiment_dir=${EXPERIMENT_DIR} log_file=${LOG_FILE}"
