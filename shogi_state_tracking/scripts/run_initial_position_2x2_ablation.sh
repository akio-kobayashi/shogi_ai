#!/usr/bin/env bash
# A: explicit/p0, B: implicit/p0, C: explicit/RAP, D: implicit/RAPを学習し評価する。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || {
    echo "MEMORY_MAX is required on Linux; use e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
DATASET_DIR="${1:?usage: $0 ABLATION_DATASET RESULTS_DIR [--model-type llama|vanilla] [--model-size SIZE]}"
RESULTS_DIR="${2:?results directory is required}"
shift 2

MODEL_TYPE="${MODEL_TYPE:-llama}"
MODEL_SIZE="${MODEL_SIZE:-base}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2560}"
MAX_MOVES="${MAX_MOVES:-512}"
MAX_HINTS="${MAX_HINTS:-192}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
RAP_PROBABILITY="${RAP_PROBABILITY:-0.15}"
HINT_LOSS_WEIGHT="${HINT_LOSS_WEIGHT:-1.0}"
SEEDS="${SEEDS:-20260802}"
EVAL_STAGE="${EVAL_STAGE:-all}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
DEVICE="${DEVICE:-auto}"
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-type) MODEL_TYPE="$2"; shift 2 ;;
    --model-type=*) MODEL_TYPE="${1#*=}"; shift ;;
    --model-size) MODEL_SIZE="$2"; shift 2 ;;
    --model-size=*) MODEL_SIZE="${1#*=}"; shift ;;
    *) EXTRA_TRAIN_ARGS+=("$1"); shift ;;
  esac
done
case "${MODEL_TYPE}" in llama|vanilla) ;; *) echo "model type must be llama or vanilla" >&2; exit 2 ;; esac
case "${MODEL_SIZE}" in small|base|large) ;; *) echo "model size must be small, base, or large" >&2; exit 2 ;; esac
case "${EVAL_STAGE}" in moves|token|probes|all) ;; *) echo "EVAL_STAGE must be moves, token, probes, or all" >&2; exit 2 ;; esac

for required in train validation evaluation; do
  [[ -f "${DATASET_DIR}/${required}.jsonl" ]] || { echo "missing ${DATASET_DIR}/${required}.jsonl" >&2; exit 2; }
done
TRAIN_JSONL="${DATASET_DIR}/train.runtime.jsonl"
VALIDATION_JSONL="${DATASET_DIR}/validation.runtime.jsonl"
[[ -f "${TRAIN_JSONL}" ]] || TRAIN_JSONL="${DATASET_DIR}/train.jsonl"
[[ -f "${VALIDATION_JSONL}" ]] || VALIDATION_JSONL="${DATASET_DIR}/validation.jsonl"
VOCAB="${DATASET_DIR}/vocab.json"
[[ -f "${VOCAB}" ]] || { echo "missing ${VOCAB}" >&2; exit 2; }
MANIFEST_ARGS=()
[[ -f "${DATASET_DIR}/dataset_manifest.json" ]] && MANIFEST_ARGS=(--dataset-manifest "${DATASET_DIR}/dataset_manifest.json")

CONDITION_IDS=(A B C D)
STATE_MODES=(explicit implicit_initial explicit implicit_initial)
ANNOTATION_MODES=(vanilla vanilla rap rap)
ANNOTATION_PROBABILITIES=(0.0 0.0 "${RAP_PROBABILITY}" "${RAP_PROBABILITY}")

IFS=',' read -r -a SEED_VALUES <<< "${SEEDS}"
for seed in "${SEED_VALUES[@]}"; do
  for index in "${!CONDITION_IDS[@]}"; do
    condition="${CONDITION_IDS[$index]}"
    state_mode="${STATE_MODES[$index]}"
    annotation_mode="${ANNOTATION_MODES[$index]}"
    probability="${ANNOTATION_PROBABILITIES[$index]}"
    label="${condition}_${state_mode}_${annotation_mode}_p${probability}"
    output="${RESULTS_DIR}/${MODEL_TYPE}-${MODEL_SIZE}/${label}/seed-${seed}"
    mkdir -p "${output}"
    echo "[$(date '+%F %T')] condition=${condition} state=${state_mode} annotation=${annotation_mode} p=${probability} seed=${seed}" | tee -a "${output}/pipeline.log"

    if [[ "${FORCE_TRAIN}" == 1 || ! -f "${output}/best.pt" ]]; then
      set +e
      "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_new_prompt.py" \
        --train-jsonl "${TRAIN_JSONL}" \
        --validation-jsonl "${VALIDATION_JSONL}" \
        --vocab "${VOCAB}" \
        "${MANIFEST_ARGS[@]}" \
        --output-dir "${output}" \
        --model-type "${MODEL_TYPE}" \
        --model-size "${MODEL_SIZE}" \
        --move-encoding factorized_v2 \
        --state-prompt-mode "${state_mode}" \
        --start-selection fixed_initial \
        --annotation-mode "${annotation_mode}" \
        --annotation-probability "${probability}" \
        --hint-loss-weight "${HINT_LOSS_WEIGHT}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --max-moves "${MAX_MOVES}" \
        --max-hints "${MAX_HINTS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --epochs "${EPOCHS}" \
        --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
        --seed "${seed}" \
        --device "${DEVICE}" \
        "${EXTRA_TRAIN_ARGS[@]}" 2>&1 | tee -a "${output}/train.log"
      status=${PIPESTATUS[0]}
      set -e
      [[ ${status} -eq 0 ]] || exit "${status}"
    else
      echo "best.pt exists; training skipped" | tee -a "${output}/pipeline.log"
    fi

    DEVICE="${DEVICE}" "${SCRIPT_DIR}/scripts/run_factorized_evaluation.sh" \
      "${output}/best.pt" "${DATASET_DIR}" "${VOCAB}" "${output}/evaluation" "${EVAL_STAGE}"
  done
done
