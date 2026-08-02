#!/usr/bin/env bash
# 注釈なしと全注釈p=1の差をsmall/largeで再現し，容量依存性を調べる。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
[[ $# -ge 2 ]] || { echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train options]" >&2; exit 2; }
DATASET_DIR="$1"; RESULTS_DIR="$2"; shift 2
IFS=',' read -r -a SEEDS <<< "${SEEDS:-20260802}"
IFS=',' read -r -a SIZES <<< "${SCALE_SIZES:-small,large}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"
for size in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for condition in vanilla partial_action random_control; do
      probability=1.0; [[ "${condition}" == "vanilla" ]] && probability=0.0
      output="${RESULTS_DIR}/vanilla-${size}/${condition}/p${probability}/seed-${seed}"
      resume=(); [[ -f "${output}/last.pt" ]] && resume=(--resume)
      "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
        --train-jsonl "${DATASET_DIR}/train.jsonl" --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
        --vocab "${DATASET_DIR}/vocab.json" --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
        --output-dir "${output}" --model-size "${size}" --annotation-mode "${condition}" \
        --annotation-probability "${probability}" --max-moves 128 --max-hints 128 \
        --seed "${seed}" "${resume[@]}" "$@"
    done
  done
done
