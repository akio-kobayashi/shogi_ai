#!/usr/bin/env bash
# 注釈率ablationで選んだp*をsmall/largeで比較し，容量依存性を調べる。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/scripts/lib_new_prompt_launcher.sh"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
# GPUメモリに合わせて呼出し側から上書きできる。1280 tokenの既定は安全側に1。
if [[ -z "${BATCH_SIZE:-}" && -n "${batch_size:-}" ]]; then
  BATCH_SIZE="${batch_size}"
fi
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DROPOUT="${DROPOUT:-0.0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1280}"
MAX_MOVES="${MAX_MOVES:-512}"
MAX_HINTS="${MAX_HINTS:-320}"
[[ $# -ge 2 ]] || { echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train options]" >&2; exit 2; }
DATASET_DIR="$1"; RESULTS_DIR="$2"; shift 2
new_prompt_extract_launcher_args "$@"
IFS=',' read -r -a SEEDS <<< "${SEEDS:-20260802}"
new_prompt_resolve_model_sizes "small,large"
SIZES=("${NEW_PROMPT_MODEL_SIZES[@]}")
SCALE_ANNOTATION_RATE="${SCALE_ANNOTATION_RATE:?run_new_prompt_rate_ablation.sh後に採用率を指定してください。例：SCALE_ANNOTATION_RATE=0.3}"
printf 'selected vanilla model sizes: %s\n' "${SIZES[*]}" >&2
printf 'batch size: %s\n' "${BATCH_SIZE}" >&2
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"
for size in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for condition in vanilla partial_action random_control; do
      probability="${SCALE_ANNOTATION_RATE}"
      if [[ "${condition}" == "vanilla" ]]; then
        probability=0.0
      fi
      output="${RESULTS_DIR}/vanilla-${size}/${condition}/p${probability}/seed-${seed}"
      train_args=()
      if ((${#NEW_PROMPT_EXTRA_ARGS[@]})); then train_args=("${NEW_PROMPT_EXTRA_ARGS[@]}"); fi
      if [[ -f "${output}/last.pt" ]]; then
        train_args+=(--resume)
      fi
      printf 'run model_type=vanilla model_size=%s condition=%s probability=%s seed=%s output_dir=%s\n' "${size}" "${condition}" "${probability}" "${seed}" "${output}"
      "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
        --train-jsonl "${DATASET_DIR}/train.jsonl" --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
        --vocab "${DATASET_DIR}/vocab.json" --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
        --output-dir "${output}" --model-type vanilla --model-size "${size}" --annotation-mode "${condition}" \
        --annotation-probability "${probability}" --max-seq-len "${MAX_SEQ_LEN}" --max-moves "${MAX_MOVES}" --max-hints "${MAX_HINTS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
        --dropout "${DROPOUT}" --seed "${seed}" "${train_args[@]+"${train_args[@]}"}"
    done
  done
done
