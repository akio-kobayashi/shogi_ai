#!/usr/bin/env bash
# 旧一括launcher。主実験には使わず，smoke／rate-ablation／scale-comparisonを順に使う。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/scripts/lib_new_prompt_launcher.sh"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ -z "${BATCH_SIZE:-}" && -n "${batch_size:-}" ]]; then
  BATCH_SIZE="${batch_size}"
fi
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DROPOUT="${DROPOUT:-0.0}"
MAIN_MAX_SEQ_LEN="${MAIN_MAX_SEQ_LEN:-1280}"
MAIN_MAX_MOVES="${MAIN_MAX_MOVES:-512}"
MAIN_MAX_HINTS="${MAIN_MAX_HINTS:-320}"
EPOCHS="${EPOCHS:-50}"
MAX_STEPS="${MAX_STEPS:-0}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train_new_prompt.py options]" >&2
  exit 2
fi

DATASET_DIR="$1"
RESULTS_DIR="$2"
shift 2
new_prompt_extract_launcher_args "$@"
if [[ -n "${NEW_PROMPT_CLI_NUM_WORKERS}" ]]; then
  NUM_WORKERS="${NEW_PROMPT_CLI_NUM_WORKERS}"
fi
SEEDS_TEXT="${SEEDS:-20260802}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_TEXT}"
new_prompt_resolve_model_sizes "small,base,large"
SIZES=("${NEW_PROMPT_MODEL_SIZES[@]}")
# 1280 tokenで512指手を保つ互換設定。p=1は上限によって実効率が歪むため不可。
PARTIAL_ACTION_PROBABILITY="${PARTIAL_ACTION_PROBABILITY:-0.3}"

printf 'selected vanilla model sizes: %s\n' "${SIZES[*]}" >&2
printf 'batch size: %s\n' "${BATCH_SIZE}" >&2
printf 'epochs: %s, max_steps: %s, num_workers: %s\n' "${EPOCHS}" "${MAX_STEPS}" "${NUM_WORKERS}" >&2

mkdir -p "${RESULTS_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" \
  --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"

for size in "${SIZES[@]}"; do
  for condition in vanilla partial_action random_control; do
    for seed in "${SEEDS[@]}"; do
      probability=0.0
      if [[ "${condition}" != "vanilla" ]]; then
        probability="${PARTIAL_ACTION_PROBABILITY}"
      fi
      output="${RESULTS_DIR}/vanilla-${size}/${condition}/seed-${seed}"
      train_args=()
      log_path="${output}/train.log"
      mkdir -p "${output}"
      if ((${#NEW_PROMPT_EXTRA_ARGS[@]})); then train_args=("${NEW_PROMPT_EXTRA_ARGS[@]}"); fi
      if [[ -f "${output}/last.pt" ]]; then
        if [[ "${RESUME:-0}" != "1" ]]; then
          echo "existing checkpoint: ${output}/last.pt; use RESUME=1 to continue, or choose a new RESULTS_DIR" >&2
          exit 2
        fi
        train_args+=(--resume)
      fi
      printf 'run model_type=vanilla model_size=%s condition=%s seed=%s output_dir=%s\n' "${size}" "${condition}" "${seed}" "${output}"
      new_prompt_run_logged "${log_path}" "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
      --train-jsonl "${DATASET_DIR}/train.jsonl" \
      --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
      --vocab "${DATASET_DIR}/vocab.json" \
      --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
      --output-dir "${output}" \
      --model-type vanilla \
      --model-size "${size}" \
      --annotation-mode "${condition}" \
      --annotation-probability "${probability}" \
      --max-seq-len "${MAIN_MAX_SEQ_LEN}" --max-moves "${MAIN_MAX_MOVES}" --max-hints "${MAIN_MAX_HINTS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
      --dropout "${DROPOUT}" --epochs "${EPOCHS}" --max-steps "${MAX_STEPS}" --seed "${seed}" "${train_args[@]+"${train_args[@]}"}"
    done
  done
done
