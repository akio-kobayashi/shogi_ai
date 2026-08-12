#!/usr/bin/env bash
# factorized_v3 baselineをLLaMA型またはVanilla decoderで学習する．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
DATASET_DIR="${1:-${SCRIPT_DIR}/new_data}"
RESULTS_DIR="${2:-${SCRIPT_DIR}/factorized_results}"
shift $(( $# >= 2 ? 2 : $# ))

MODEL_TYPE="${MODEL_TYPE:-llama}"
MODEL_SIZE="${MODEL_SIZE:-base}"
ANNOTATION_MODE="${ANNOTATION_MODE:-vanilla}"
ANNOTATION_PROBABILITY="${ANNOTATION_PROBABILITY:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2560}"
MAX_MOVES="${MAX_MOVES:-512}"
MAX_HINTS="${MAX_HINTS:-512}"
EPOCHS="${EPOCHS:-50}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-type) MODEL_TYPE="$2"; shift 2 ;;
    --model-type=*) MODEL_TYPE="${1#*=}"; shift ;;
    --model-size) MODEL_SIZE="$2"; shift 2 ;;
    --model-size=*) MODEL_SIZE="${1#*=}"; shift ;;
    --annotation-mode) ANNOTATION_MODE="$2"; shift 2 ;;
    --annotation-mode=*) ANNOTATION_MODE="${1#*=}"; shift ;;
    --annotation-probability) ANNOTATION_PROBABILITY="$2"; shift 2 ;;
    --annotation-probability=*) ANNOTATION_PROBABILITY="${1#*=}"; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
case "${MODEL_TYPE}" in llama|vanilla) ;; *) echo "--model-type must be llama or vanilla" >&2; exit 2 ;; esac
case "${MODEL_SIZE}" in small|base|large) ;; *) echo "--model-size must be small, base, or large" >&2; exit 2 ;; esac

OUTPUT_DIR="${RESULTS_DIR}/${MODEL_TYPE}-${MODEL_SIZE}/implicit-initial/${ANNOTATION_MODE}-p${ANNOTATION_PROBABILITY}/seed-${SEED:-20260802}"
VOCAB="${DATASET_DIR}/vocab.json"
mkdir -p "${OUTPUT_DIR}"
[[ -f "${VOCAB}" ]] || { echo "missing ${VOCAB}; run build_factorized_prompt_dataset.sh first" >&2; exit 2; }
MANIFEST_ARGS=()
if [[ -f "${DATASET_DIR}/dataset_manifest.json" ]] \
  && grep -q '"move_encoding"[[:space:]]*:[[:space:]]*"factorized_v3_no_eom"' "${DATASET_DIR}/dataset_manifest.json" \
  && grep -q '"stage_1_2_input_mode"[[:space:]]*:[[:space:]]*"implicit_standard_initial"' "${DATASET_DIR}/dataset_manifest.json" \
  && grep -q '"terminal_encoding"[[:space:]]*:[[:space:]]*"eos_on_complete_decisive_game_v1"' "${DATASET_DIR}/dataset_manifest.json"; then
  MANIFEST_ARGS=(--dataset-manifest "${DATASET_DIR}/dataset_manifest.json")
else
  echo "missing or obsolete factorized_v3 dataset_manifest.json; rebuild with scripts/setup_factorized_v3_data.sh" >&2
  exit 2
fi

set +e
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_new_prompt.py" \
  --train-jsonl "${DATASET_DIR}/train.jsonl" \
  --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
  --vocab "${VOCAB}" \
  "${MANIFEST_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-type "${MODEL_TYPE}" \
  --model-size "${MODEL_SIZE}" \
  --move-encoding factorized_v3_no_eom \
  --state-prompt-mode implicit_initial \
  --start-selection fixed_initial \
  --annotation-mode "${ANNOTATION_MODE}" \
  --annotation-probability "${ANNOTATION_PROBABILITY}" \
  --eos-loss-weight "${EOS_LOSS_WEIGHT:-1.0}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --max-moves "${MAX_MOVES}" \
  --max-hints "${MAX_HINTS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
