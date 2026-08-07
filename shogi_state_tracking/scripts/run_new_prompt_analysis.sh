#!/usr/bin/env bash
# 学習済み9 checkpointに対して，学習を再開せず指手評価と層別線形プローブを行う．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATASET_DIR RESULTS_DIR" >&2
  exit 2
fi
DATASET_DIR="$1"; RESULTS_DIR="$2"; shift 2
EVALUATION_MAX_GAMES="${EVALUATION_MAX_GAMES:-5000}"
EVALUATION_CANDIDATES_PER_GAME="${EVALUATION_CANDIDATES_PER_GAME:-3}"
PROBE_MAX_TRAIN_SAMPLES="${PROBE_MAX_TRAIN_SAMPLES:-12000}"
PROBE_MAX_VALIDATION_SAMPLES="${PROBE_MAX_VALIDATION_SAMPLES:-3000}"
PROBE_MAX_EVALUATION_SAMPLES="${PROBE_MAX_EVALUATION_SAMPLES:-5000}"
EVALUATION_SEED="${EVALUATION_SEED:-20260802}"
mkdir -p "${RESULTS_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" \
  --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"

shopt -s globstar nullglob
checkpoints=("${RESULTS_DIR}"/**/best.pt)
[[ ${#checkpoints[@]} -gt 0 ]] || { echo "no best.pt below: ${RESULTS_DIR}" >&2; exit 1; }
for checkpoint in "${checkpoints[@]}"; do
  run_dir="$(dirname "${checkpoint}")"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_new_prompt_moves.py" \
    --checkpoint "${checkpoint}" --vocab "${DATASET_DIR}/vocab.json" \
    --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
    --max-games "${EVALUATION_MAX_GAMES}" \
    --candidates-per-game "${EVALUATION_CANDIDATES_PER_GAME}" \
    --output "${run_dir}/move_metrics.json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_new_prompt_probes.py" \
    --checkpoint "${checkpoint}" --vocab "${DATASET_DIR}/vocab.json" \
    --train-jsonl "${DATASET_DIR}/train.jsonl" \
    --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
    --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
    --max-train-samples "${PROBE_MAX_TRAIN_SAMPLES}" \
    --max-validation-samples "${PROBE_MAX_VALIDATION_SAMPLES}" \
    --max-evaluation-samples "${PROBE_MAX_EVALUATION_SAMPLES}" \
    --seed "${EVALUATION_SEED}" \
    --output-dir "${run_dir}/probes"
  if [[ "${run_dir}" == *"/partial_action/"* ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_new_prompt_token_probe.py" \
      --checkpoint "${checkpoint}" --vocab "${DATASET_DIR}/vocab.json" \
      --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
      --output "${run_dir}/token_probe_metrics.json"
  fi
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_new_prompt_results.py" \
  --results-dir "${RESULTS_DIR}" --output-dir "${RESULTS_DIR}/summary"
"${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_new_prompt_results.py" \
  --results-dir "${RESULTS_DIR}" \
  --summary-json "${RESULTS_DIR}/summary/summary.json" \
  --output-dir "${RESULTS_DIR}/figures"
