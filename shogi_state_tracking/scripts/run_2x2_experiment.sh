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
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"

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
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
    "${PROJECT_DIR}/scripts/run_cot_experiment.sh" "${model_type}" "$@"
done

"${PYTHON_BIN}" "${PROJECT_DIR}/summarize_2x2.py" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --seed "${SEED}"
