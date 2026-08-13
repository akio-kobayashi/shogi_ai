#!/usr/bin/env bash
# AP：駒打ちを除く全通常移動へ移動前駒種を注釈して学習・評価する．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${1:?usage: $0 DATASET_DIR RESULTS_DIR}"
RESULTS_DIR="${2:?results directory is required}"
shift 2

RAP_RATES=1.0 \
  "${SCRIPT_DIR}/scripts/run_factorized_rap_ablation.sh" \
  "${DATASET_DIR}" "${RESULTS_DIR}" "$@"
