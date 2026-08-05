#!/usr/bin/env bash
# LLaMA型decoderについて，rate ablationで選んだp*の規模依存性を測る。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
[[ $# -ge 2 ]] || { echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train options]" >&2; exit 2; }
DATASET_DIR="$1"; RESULTS_DIR="$2"; shift 2
IFS=',' read -r -a SEEDS <<< "${SEEDS:-20260802}"
IFS=',' read -r -a SIZES <<< "${SCALE_SIZES:-small,large}"
SCALE_ANNOTATION_RATE="${SCALE_ANNOTATION_RATE:?run_llama_new_prompt_rate_ablation.sh後に採用率を指定してください。例：SCALE_ANNOTATION_RATE=0.3}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"
for size in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for condition in vanilla partial_action random_control; do
      probability="${SCALE_ANNOTATION_RATE}"; [[ "${condition}" == "vanilla" ]] && probability=0.0
      output="${RESULTS_DIR}/llama-${size}/${condition}/p${probability}/seed-${seed}"
      resume=(); [[ -f "${output}/last.pt" ]] && resume=(--resume)
      "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
        --train-jsonl "${DATASET_DIR}/train.jsonl" --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
        --vocab "${DATASET_DIR}/vocab.json" --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
        --output-dir "${output}" --model-type llama --model-size "${size}" --annotation-mode "${condition}" \
        --annotation-probability "${probability}" --max-seq-len 512 --max-moves 192 --max-hints 110 \
        --dropout 0.0 --seed "${seed}" "${resume[@]}" "$@"
    done
  done
done
