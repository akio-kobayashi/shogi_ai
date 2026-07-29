#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE="${1:-pretrain}"
MODEL_TYPE="${2:-vanilla}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# metadataのtrain p95（固定prefix 99 + 221手）に対応する既定値。
# GPUメモリに余裕がなければ256、coverageを優先する場合は352へ上書きする。
MAX_SEQ_LEN="${MAX_SEQ_LEN:-320}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

case "${STAGE}" in
  pretrain)
    TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_DIR}/data/datasets/train.jsonl}"
    VALIDATION_JSONL="${VALIDATION_JSONL:-${PROJECT_DIR}/data/datasets/validation.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/training/${MODEL_TYPE}/seed_${SEED}}"
    INIT_ARGUMENTS=()
    ;;
  cot)
    TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_DIR}/data/traces/train.jsonl}"
    VALIDATION_JSONL="${VALIDATION_JSONL:-${PROJECT_DIR}/data/traces/validation.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/cot/${MODEL_TYPE}/seed_${SEED}}"
    if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
      echo "cot stage requires INIT_CHECKPOINT" >&2
      exit 2
    fi
    INIT_ARGUMENTS=(--init-checkpoint "${INIT_CHECKPOINT}")
    ;;
  *)
    echo "unknown stage: ${STAGE}; use pretrain or cot" >&2
    exit 2
    ;;
esac

COMMAND=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/train_model.py"
  --stage "${STAGE}"
  --model-type "${MODEL_TYPE}"
  --vocab "${VOCAB_PATH}"
  --train-jsonl "${TRAIN_JSONL}"
  --validation-jsonl "${VALIDATION_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --progress-every "${PROGRESS_EVERY}"
  "${INIT_ARGUMENTS[@]}"
  "$@"
)

printf "command:"
printf " %q" "${COMMAND[@]}"
printf "\n"
"${COMMAND[@]}"
