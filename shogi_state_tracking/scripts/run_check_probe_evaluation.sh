#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# 凍結済みdecoderの隠れ表現から，手番側の玉が王手されているかを線形復号する。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-standard}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/checkpoints/model.pt}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
CHECK_PROBE_DIR="${CHECK_PROBE_DIR:-${PROJECT_DIR}/data/check_probe}"
TRAIN_JSONL="${TRAIN_JSONL:-${CHECK_PROBE_DIR}/train.jsonl}"
VALIDATION_JSONL="${VALIDATION_JSONL:-${CHECK_PROBE_DIR}/validation.jsonl}"
EVALUATION_JSONL="${EVALUATION_JSONL:-${CHECK_PROBE_DIR}/evaluation.jsonl}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"

case "${MODE}" in
  standard) SOURCES="${SOURCES:-final,recurrent,token_embedding}" ;;
  all-layers) SOURCES="${SOURCES:-layers,recurrent,token_embedding}" ;;
  *)
    echo "unknown mode: ${MODE}; expected standard or all-layers" >&2
    exit 2
    ;;
esac

for required_file in "${PYTHON_BIN}" "${CHECKPOINT}" "${VOCAB_PATH}" \
  "${TRAIN_JSONL}" "${VALIDATION_JSONL}" "${EVALUATION_JSONL}"
do
  if [[ ! -e "${required_file}" ]]; then
    echo "required file is unavailable: ${required_file}" >&2
    exit 2
  fi
done

CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_NAME="${CHECKPOINT_NAME%.*}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/check-probes/${CHECKPOINT_NAME}/${MODE}/seed_${SEED}}"

COMMAND=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/evaluate_check_probe.py"
  --checkpoint "${CHECKPOINT}"
  --vocab "${VOCAB_PATH}"
  --train-jsonl "${TRAIN_JSONL}"
  --validation-jsonl "${VALIDATION_JSONL}"
  --evaluation-jsonl "${EVALUATION_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --sources "${SOURCES}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --seed "${SEED}"
  --device "${DEVICE}"
)
COMMAND+=("$@")

echo "mode: ${MODE}"
echo "checkpoint: ${CHECKPOINT}"
echo "sources: ${SOURCES}"
echo "output: ${OUTPUT_DIR}"
printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
