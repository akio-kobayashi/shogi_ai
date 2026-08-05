#!/usr/bin/env bash
# 旧一括launcher。主実験には使わず，smoke／rate-ablation／scale-comparisonを順に使う。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
# 旧launcherも長文脈に合わせる。呼出し側でBATCH_SIZE=2のように上書きできる。
BATCH_SIZE="${BATCH_SIZE:-1}"
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
# 1280 tokenで512指手を保つ互換設定。p=1は上限によって実効率が歪むため不可。
PARTIAL_ACTION_PROBABILITY="${PARTIAL_ACTION_PROBABILITY:-0.3}"

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
      --max-seq-len "${MAIN_MAX_SEQ_LEN}" --max-moves "${MAIN_MAX_MOVES}" --max-hints "${MAIN_MAX_HINTS}" --batch-size "${BATCH_SIZE}" \
      --seed "${seed}" "${resume_args[@]}" \
      "$@"
    done
  done
done
