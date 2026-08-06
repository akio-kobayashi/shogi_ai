#!/usr/bin/env bash
# LLaMA型decoderの注釈率ablation。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/scripts/lib_new_prompt_launcher.sh"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
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
new_prompt_resolve_single_model_size base
MODEL_SIZE="${NEW_PROMPT_MODEL_SIZE}"
IFS=',' read -r -a SEEDS <<< "${SEEDS:-20260802}"
IFS=',' read -r -a RATES <<< "${ANNOTATION_RATES:-0.1,0.3,0.5}"
printf 'selected llama rate-ablation model size: %s\n' "${MODEL_SIZE}" >&2
printf 'batch size: %s\n' "${BATCH_SIZE}" >&2
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"
run_one () {
  local condition="$1" probability="$2" seed="$3"
  local output="${RESULTS_DIR}/llama-${MODEL_SIZE}/${condition}/p${probability}/seed-${seed}"
  local train_args=()
  if ((${#NEW_PROMPT_EXTRA_ARGS[@]})); then
    train_args=("${NEW_PROMPT_EXTRA_ARGS[@]}")
  fi
  if [[ -f "${output}/last.pt" ]]; then
    train_args+=(--resume)
  fi
  printf 'run model_type=llama model_size=%s condition=%s probability=%s seed=%s output_dir=%s\n' "${MODEL_SIZE}" "${condition}" "${probability}" "${seed}" "${output}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
    --train-jsonl "${DATASET_DIR}/train.jsonl" --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
    --vocab "${DATASET_DIR}/vocab.json" --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
    --output-dir "${output}" --model-type llama --model-size "${MODEL_SIZE}" --annotation-mode "${condition}" \
    --annotation-probability "${probability}" --max-seq-len "${MAX_SEQ_LEN}" --max-moves "${MAX_MOVES}" --max-hints "${MAX_HINTS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --dropout "${DROPOUT}" --seed "${seed}" "${train_args[@]+"${train_args[@]}"}"
}
for seed in "${SEEDS[@]}"; do
  run_one vanilla 0.0 "${seed}"
  for rate in "${RATES[@]}"; do
    run_one partial_action "${rate}" "${seed}"
    run_one random_control "${rate}" "${seed}"
  done
done
