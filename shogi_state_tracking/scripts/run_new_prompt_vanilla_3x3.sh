#!/usr/bin/env bash
# 一括実行用の互換launcher。主実験はsmoke／rate-ablation／scale-comparisonへ分ける。
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
# 主比較は，注釈なしVanillaと，全非駒打ち指手へ注釈を挿入するPartial-actionである。
# Random controlは同じ全挿入量で現在局面との対応だけを壊す補助対照とする。
# p<1は主比較後のablation専用。例：PARTIAL_ACTION_PROBABILITY=0.1 ...
PARTIAL_ACTION_PROBABILITY="${PARTIAL_ACTION_PROBABILITY:-1.0}"
# p=1の主比較では，選んだ履歴内の全非駒打ち指手を注釈できるようH=K=128を使う。
# 512 token文脈では，prompt + H + 2K + boundary が収まる保守的な上限である。
MAIN_MAX_MOVES="${MAIN_MAX_MOVES:-128}"
MAIN_MAX_HINTS="${MAIN_MAX_HINTS:-128}"

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
      --max-moves "${MAIN_MAX_MOVES}" --max-hints "${MAIN_MAX_HINTS}" \
      --seed "${seed}" "${resume_args[@]}" \
      "$@"
    done
  done
done
