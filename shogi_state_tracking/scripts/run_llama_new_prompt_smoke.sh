#!/usr/bin/env bash
# LLaMA型decoderの疎通確認。既存run_new_prompt_smoke.sh（Vanilla）は変更しない。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
[[ $# -ge 2 ]] || { echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train options]" >&2; exit 2; }
DATASET_DIR="$1"; RESULTS_DIR="$2"; shift 2
SEED="${SEED:-20260802}"; EPOCHS="${SMOKE_EPOCHS:-1}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"
for condition in vanilla partial_action random_control; do
  probability=0.0; [[ "${condition}" != "vanilla" ]] && probability="${SMOKE_ANNOTATION_RATE:-0.3}"
  output="${RESULTS_DIR}/llama-small/${condition}/p${probability}/seed-${SEED}"
  resume=(); [[ -f "${output}/last.pt" ]] && resume=(--resume)
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
    --train-jsonl "${DATASET_DIR}/train.jsonl" --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
    --vocab "${DATASET_DIR}/vocab.json" --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
    --output-dir "${output}" --model-type llama --model-size small --annotation-mode "${condition}" \
    --annotation-probability "${probability}" --max-seq-len 512 --max-moves 192 --max-hints 110 \
    --dropout 0.0 --epochs "${EPOCHS}" --seed "${SEED}" "${resume[@]}" "$@"
done
