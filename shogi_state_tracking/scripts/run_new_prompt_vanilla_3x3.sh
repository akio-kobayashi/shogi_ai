#!/usr/bin/env bash
# 3規模 × 3条件のVanilla主実験．dataset作成機ではなく計算機で実行する．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train_new_prompt.py options]" >&2
  exit 2
fi

DATASET_DIR="$1"
RESULTS_DIR="$2"
shift 2
SEEDS_TEXT="${SEEDS:-20260802}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_TEXT}"
# Partial-actionとRandom controlは，注釈位置数以外の条件を揃えるため同じ確率を使う。
# 例：PARTIAL_ACTION_PROBABILITY=0.1 scripts/run_new_prompt_vanilla_3x3.sh ...
PARTIAL_ACTION_PROBABILITY="${PARTIAL_ACTION_PROBABILITY:-0.30}"

mkdir -p "${RESULTS_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" \
  --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"

for size in small base large; do
  for condition in vanilla partial_action random_control; do
    for seed in "${SEEDS[@]}"; do
      probability=0.0
      if [[ "${condition}" != "vanilla" ]]; then
        probability="${PARTIAL_ACTION_PROBABILITY}"
      fi
      output="${RESULTS_DIR}/vanilla-${size}/${condition}/seed-${seed}"
      mkdir -p "${output}"
      resume_args=()
      if [[ -f "${output}/last.pt" ]]; then resume_args=(--resume); fi
      echo "run model=${size} condition=${condition} seed=${seed} output=${output}"
      "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
      --train-jsonl "${DATASET_DIR}/train.jsonl" \
      --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
      --vocab "${DATASET_DIR}/vocab.json" \
      --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
      --output-dir "${output}" \
      --model-size "${size}" \
      --annotation-mode "${condition}" \
      --annotation-probability "${probability}" \
      --seed "${seed}" "${resume_args[@]}" \
      "$@"
    done
  done
done
