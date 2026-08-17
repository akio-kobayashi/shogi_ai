#!/usr/bin/env bash
# 既存last.ptの固定epoch継続と，標準・Lishogi評価を連続実行する．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${1:?usage: $0 DATASET_DIR SOURCE_RESULTS_DIR FIXED_RESULTS_DIR LISHOGI_DATASET_DIR}"
SOURCE_RESULTS_DIR="${2:?source results directory is required}"
FIXED_RESULTS_DIR="${3:?fixed-epoch results directory is required}"
LISHOGI_DATASET_DIR="${4:?Lishogi evaluation dataset directory is required}"

"${SCRIPT_DIR}/scripts/run_factorized_fixed_epoch_training.sh" \
  "${DATASET_DIR}" "${SOURCE_RESULTS_DIR}" "${FIXED_RESULTS_DIR}"

"${SCRIPT_DIR}/scripts/run_factorized_fixed_epoch_evaluation.sh" \
  "${DATASET_DIR}" "${FIXED_RESULTS_DIR}" "${LISHOGI_DATASET_DIR}"
