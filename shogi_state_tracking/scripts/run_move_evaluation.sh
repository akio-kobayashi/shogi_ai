#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/checkpoints/model.pt}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
EVALUATION_JSONL="${EVALUATION_JSONL:-${PROJECT_DIR}/data/datasets/evaluation.jsonl}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi

CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_NAME="${CHECKPOINT_NAME%.*}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/moves/${CHECKPOINT_NAME}/seed_${SEED}}"

for required_file in "${CHECKPOINT}" "${VOCAB_PATH}" "${EVALUATION_JSONL}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "required file is unavailable: ${required_file}" >&2
    exit 2
  fi
done

COMMAND=(
  "${PYTHON_BIN}"
  "${PROJECT_DIR}/evaluate_move_metrics.py"
  --checkpoint "${CHECKPOINT}"
  --vocab "${VOCAB_PATH}"
  --evaluation-jsonl "${EVALUATION_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --progress-every "${PROGRESS_EVERY}"
)
COMMAND+=("$@")

echo "evaluation_start_mode: fixed_start_ply_0"
echo "checkpoint: ${CHECKPOINT}"
echo "output: ${OUTPUT_DIR}"
printf "command:"
printf " %q" "${COMMAND[@]}"
printf "\n"

"${COMMAND[@]}"
