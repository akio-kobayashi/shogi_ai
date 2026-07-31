#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-standard}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/checkpoints/model.pt}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_DIR}/data/datasets/train.jsonl}"
VALIDATION_JSONL="${VALIDATION_JSONL:-${PROJECT_DIR}/data/datasets/validation.jsonl}"
EVALUATION_JSONL="${EVALUATION_JSONL:-${PROJECT_DIR}/data/datasets/evaluation.jsonl}"

SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
POSITIONS_PER_GAME="${POSITIONS_PER_GAME:-16}"
SAMPLES_PER_GAME="${SAMPLES_PER_GAME:-1}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
INCLUDE_INITIAL_STATE="${INCLUDE_INITIAL_STATE:-0}"
EVALUATION_START_PLIES="${EVALUATION_START_PLIES:-0,24,25,32,33}"
EVALUATION_MIN_SUFFIX_MOVES="${EVALUATION_MIN_SUFFIX_MOVES:-40}"

UNTRAINED=0
case "${MODE}" in
  standard)
    SOURCES="${SOURCES:-final,recurrent,token_embedding}"
    ;;
  all-layers)
    SOURCES="${SOURCES:-layers,recurrent,token_embedding}"
    ;;
  untrained)
    SOURCES="${SOURCES:-final,recurrent,token_embedding}"
    UNTRAINED=1
    ;;
  all-layers-untrained)
    SOURCES="${SOURCES:-layers,recurrent,token_embedding}"
    UNTRAINED=1
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    echo "modes: standard, all-layers, untrained, all-layers-untrained" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi

for required_file in \
  "${CHECKPOINT}" \
  "${VOCAB_PATH}" \
  "${TRAIN_JSONL}" \
  "${VALIDATION_JSONL}" \
  "${EVALUATION_JSONL}"
do
  if [[ ! -f "${required_file}" ]]; then
    echo "required file is unavailable: ${required_file}" >&2
    exit 2
  fi
done

CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_NAME="${CHECKPOINT_NAME%.*}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/probes/${CHECKPOINT_NAME}/${MODE}/seed_${SEED}}"

COMMAND=(
  "${PYTHON_BIN}"
  "${PROJECT_DIR}/evaluate_probes.py"
  --checkpoint "${CHECKPOINT}"
  --vocab "${VOCAB_PATH}"
  --train-jsonl "${TRAIN_JSONL}"
  --validation-jsonl "${VALIDATION_JSONL}"
  --evaluation-jsonl "${EVALUATION_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --sources "${SOURCES}"
  --positions-per-game "${POSITIONS_PER_GAME}"
  --samples-per-game "${SAMPLES_PER_GAME}"
  --probe-epochs "${PROBE_EPOCHS}"
  --patience "${PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --evaluation-start-plies "${EVALUATION_START_PLIES}"
  --min-suffix-moves "${EVALUATION_MIN_SUFFIX_MOVES}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --skip-language-model
)

if [[ "${INCLUDE_INITIAL_STATE}" -eq 1 ]]; then
  COMMAND+=(--include-initial-state)
else
  COMMAND+=(--exclude-initial-state)
fi

if [[ "${UNTRAINED}" -eq 1 ]]; then
  COMMAND+=(--untrained)
fi

# 残りの引数はevaluate_probes.pyへ渡し、既定値を一時的に上書きできる。
COMMAND+=("$@")

echo "mode: ${MODE}"
echo "checkpoint: ${CHECKPOINT}"
echo "sources: ${SOURCES}"
echo "include_initial_state: ${INCLUDE_INITIAL_STATE}"
echo "evaluation_start_plies: ${EVALUATION_START_PLIES}"
echo "evaluation_min_suffix_moves: ${EVALUATION_MIN_SUFFIX_MOVES}"
echo "output: ${OUTPUT_DIR}"
printf "command:"
printf " %q" "${COMMAND[@]}"
printf "\n"

"${COMMAND[@]}"
