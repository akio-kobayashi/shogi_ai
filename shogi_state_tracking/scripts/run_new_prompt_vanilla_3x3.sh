#!/usr/bin/env bash
# 旧一括launcher。主実験には使わず，smoke／rate-ablation／scale-comparisonを順に使う。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ -z "${BATCH_SIZE:-}" && -n "${batch_size:-}" ]]; then
  BATCH_SIZE="${batch_size}"
fi
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DROPOUT="${DROPOUT:-0.0}"
MAIN_MAX_SEQ_LEN="${MAIN_MAX_SEQ_LEN:-1280}"
MAIN_MAX_MOVES="${MAIN_MAX_MOVES:-512}"
MAIN_MAX_HINTS="${MAIN_MAX_HINTS:-320}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train_new_prompt.py options]" >&2
  exit 2
fi

DATASET_DIR="$1"
RESULTS_DIR="$2"
shift 2
SEEDS_TEXT="${SEEDS:-20260802}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_TEXT}"
IFS=',' read -r -a SIZES <<< "${SCALE_SIZES:-small,base,large}"
# 1280 tokenで512指手を保つ互換設定。p=1は上限によって実効率が歪むため不可。
PARTIAL_ACTION_PROBABILITY="${PARTIAL_ACTION_PROBABILITY:-0.3}"

for size in "${SIZES[@]}"; do
  case "${size}" in
    small|base|large) ;;
    *) echo "SCALE_SIZES must contain only small, base, or large: ${size}" >&2; exit 2 ;;
  esac
done
printf 'selected vanilla model sizes: %s\n' "${SIZES[*]}" >&2
printf 'batch size: %s\n' "${BATCH_SIZE}" >&2

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
      train_args=("$@")
      if [[ -f "${output}/last.pt" ]]; then train_args+=(--resume); fi
      printf 'run model_type=vanilla model_size=%s condition=%s seed=%s output_dir=%s\n' "${size}" "${condition}" "${seed}" "${output}"
      "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
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
      --dropout "${DROPOUT}" --seed "${seed}" "${train_args[@]+"${train_args[@]}"}"
    done
  done
done
