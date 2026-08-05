#!/usr/bin/env bash
# 注釈率ablationで選んだp*をsmall/largeで比較し，容量依存性を調べる。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
# GPUメモリに合わせて呼出し側から上書きできる。1280 tokenの既定は安全側に1。
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1280}"
MAX_MOVES="${MAX_MOVES:-512}"
MAX_HINTS="${MAX_HINTS:-320}"
[[ $# -ge 2 ]] || { echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train options]" >&2; exit 2; }
DATASET_DIR="$1"; RESULTS_DIR="$2"; shift 2
IFS=',' read -r -a SEEDS <<< "${SEEDS:-20260802}"
IFS=',' read -r -a SIZES <<< "${SCALE_SIZES:-small,large}"
SCALE_ANNOTATION_RATE="${SCALE_ANNOTATION_RATE:?run_new_prompt_rate_ablation.sh後に採用率を指定してください。例：SCALE_ANNOTATION_RATE=0.3}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"
for size in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for condition in vanilla partial_action random_control; do
      probability="${SCALE_ANNOTATION_RATE}"; [[ "${condition}" == "vanilla" ]] && probability=0.0
      output="${RESULTS_DIR}/vanilla-${size}/${condition}/p${probability}/seed-${seed}"
      resume=(); [[ -f "${output}/last.pt" ]] && resume=(--resume)
      "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
        --train-jsonl "${DATASET_DIR}/train.jsonl" --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
        --vocab "${DATASET_DIR}/vocab.json" --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
        --output-dir "${output}" --model-size "${size}" --annotation-mode "${condition}" \
        --annotation-probability "${probability}" --max-seq-len "${MAX_SEQ_LEN}" --max-moves "${MAX_MOVES}" --max-hints "${MAX_HINTS}" --batch-size "${BATCH_SIZE}" \
        --seed "${seed}" "${resume[@]}" "$@"
    done
  done
done
