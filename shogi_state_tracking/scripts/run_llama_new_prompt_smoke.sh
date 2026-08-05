#!/usr/bin/env bash
# LLaMA型decoderの疎通確認。既存run_new_prompt_smoke.sh（Vanilla）は変更しない。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ -z "${BATCH_SIZE:-}" && -n "${batch_size:-}" ]]; then
  BATCH_SIZE="${batch_size}"
fi
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1280}"
MAX_MOVES="${MAX_MOVES:-512}"
MAX_HINTS="${MAX_HINTS:-320}"
[[ $# -ge 2 ]] || { echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train options]" >&2; exit 2; }
DATASET_DIR="$1"; RESULTS_DIR="$2"; shift 2
SEED="${SEED:-20260802}"; EPOCHS="${SMOKE_EPOCHS:-1}"
if [[ -z "${MODEL_SIZE:-}" && -n "${SCALE_SIZES:-}" ]]; then
  MODEL_SIZE="${SCALE_SIZES}"
fi
MODEL_SIZE="${MODEL_SIZE:-small}"
case "${MODEL_SIZE}" in
  small|base|large) ;;
  *) echo "MODEL_SIZE must be small, base, or large" >&2; exit 2 ;;
esac
printf 'selected llama smoke model size: %s\n' "${MODEL_SIZE}" >&2
printf 'batch size: %s\n' "${BATCH_SIZE}" >&2
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"
for condition in vanilla partial_action random_control; do
  probability=0.0; [[ "${condition}" != "vanilla" ]] && probability="${SMOKE_ANNOTATION_RATE:-0.3}"
  output="${RESULTS_DIR}/llama-${MODEL_SIZE}/${condition}/p${probability}/seed-${SEED}"
  train_args=("$@")
  [[ -f "${output}/last.pt" ]] && train_args+=(--resume)
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
    --train-jsonl "${DATASET_DIR}/train.jsonl" --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
    --vocab "${DATASET_DIR}/vocab.json" --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
    --output-dir "${output}" --model-type llama --model-size "${MODEL_SIZE}" --annotation-mode "${condition}" \
    --annotation-probability "${probability}" --max-seq-len "${MAX_SEQ_LEN}" --max-moves "${MAX_MOVES}" --max-hints "${MAX_HINTS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --dropout 0.0 --epochs "${EPOCHS}" --seed "${SEED}" "${train_args[@]}"
done
