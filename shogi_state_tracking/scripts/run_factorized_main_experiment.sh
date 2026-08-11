#!/usr/bin/env bash
# 主実験：同一dataset・語彙・モデル設定でRAPなし／ありだけを比較する．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${1:?usage: $0 DATASET_DIR RESULTS_DIR}"
RESULTS_DIR="${2:?results directory is required}"
shift 2

MODEL_TYPE="${MODEL_TYPE:-llama}"
MODEL_SIZE="${MODEL_SIZE:-base}"
RAP_PROBABILITY="${RAP_PROBABILITY:-0.15}"
SEEDS="${SEEDS:-20260802}"
EVAL_STAGE="${EVAL_STAGE:-main}"

case "${RAP_PROBABILITY}" in
  0|0.0|0.00) echo "RAP_PROBABILITY must be greater than zero" >&2; exit 2 ;;
esac

RAP_RATES="0.0,${RAP_PROBABILITY}" \
MODEL_TYPE="${MODEL_TYPE}" MODEL_SIZE="${MODEL_SIZE}" SEEDS="${SEEDS}" \
EVAL_STAGE="${EVAL_STAGE}" \
  "${SCRIPT_DIR}/scripts/run_factorized_rap_ablation.sh" \
  "${DATASET_DIR}" "${RESULTS_DIR}" "$@"
