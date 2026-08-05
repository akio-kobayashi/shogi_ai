#!/usr/bin/env bash
# baseモデルで注釈率pを変えるablation。Vanilla p=0は各seedにつき一度だけ学習する。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
# GPUメモリに合わせて呼出し側から上書きできる。1280 tokenの既定は安全側に1。
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1280}"
MAX_MOVES="${MAX_MOVES:-512}"
MAX_HINTS="${MAX_HINTS:-320}"
[[ $# -ge 2 ]] || { echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train options]" >&2; exit 2; }
DATASET_DIR="$1"; RESULTS_DIR="$2"; shift 2
EXTRA_ARGS=("$@")
IFS=',' read -r -a SEEDS <<< "${SEEDS:-20260802}"
# 1280 token文脈で実指手数を512に固定する。p=0.5でも期待注釈数は約256で，
# 上限320にほぼ達しない。p=1は注釈上限によって実効率が歪むので含めない。
IFS=',' read -r -a RATES <<< "${ANNOTATION_RATES:-0.1,0.3,0.5}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"
run_one () {
  local condition="$1" probability="$2" seed="$3"
  local output="${RESULTS_DIR}/vanilla-base/${condition}/p${probability}/seed-${seed}"
  local resume=(); [[ -f "${output}/last.pt" ]] && resume=(--resume)
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
    --train-jsonl "${DATASET_DIR}/train.jsonl" --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
    --vocab "${DATASET_DIR}/vocab.json" --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
    --output-dir "${output}" --model-size base --annotation-mode "${condition}" \
    --annotation-probability "${probability}" --max-seq-len "${MAX_SEQ_LEN}" --max-moves "${MAX_MOVES}" --max-hints "${MAX_HINTS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --seed "${seed}" "${resume[@]}" "${EXTRA_ARGS[@]}"
}
for seed in "${SEEDS[@]}"; do
  run_one vanilla 0.0 "${seed}"
  for rate in "${RATES[@]}"; do
    run_one partial_action "${rate}" "${seed}"
    run_one random_control "${rate}" "${seed}"
  done
done
