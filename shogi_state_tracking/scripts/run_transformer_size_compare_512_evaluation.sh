#!/usr/bin/env bash
set -euo pipefail

# 既存の small/base/large の best.pt を使い，学習を行わずに
# 指手・状態・王手 probe の数値評価と集計だけを実行する。
#
# 例:
#   scripts/run_transformer_size_compare_512_evaluation.sh
#   RESULTS_ROOT=results/model-comparison \
#     scripts/run_transformer_size_compare_512_evaluation.sh
#   PREPARE_CHECK_PROBE_DATA=0 \
#     scripts/run_transformer_size_compare_512_evaluation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TRAIN=0 \
RUN_MOVE_EVALUATION="${RUN_MOVE_EVALUATION:-1}" \
RUN_PROBES="${RUN_PROBES:-1}" \
RUN_CHECK_PROBES="${RUN_CHECK_PROBES:-1}" \
PREPARE_CHECK_PROBE_DATA="${PREPARE_CHECK_PROBE_DATA:-1}" \
RUN_VISUALIZATIONS="${RUN_VISUALIZATIONS:-0}" \
RUN_SUMMARY="${RUN_SUMMARY:-1}" \
"${PROJECT_DIR}/scripts/run_transformer_size_compare_512.sh"
