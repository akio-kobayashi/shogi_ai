#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/transformer_size_compare_512}"
LOG_FILE="${LOG_FILE:-${RESULTS_ROOT}/run_compare_512.log}"
SEED="${SEED:-20260724}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MOVE_EVALUATION="${RUN_MOVE_EVALUATION:-1}"
RUN_PROBES="${RUN_PROBES:-1}"
RUN_CHECK_PROBES="${RUN_CHECK_PROBES:-1}"
PREPARE_CHECK_PROBE_DATA="${PREPARE_CHECK_PROBE_DATA:-1}"
# 可視化にはgame_id・ply・駒種の選定が必要なため，既定では実行しない。
RUN_VISUALIZATIONS="${RUN_VISUALIZATIONS:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"

if [[ "${MAX_SEQ_LEN}" != "512" ]]; then
  echo "MAX_SEQ_LEN must be 512 for this workflow"
  exit 2
fi

mkdir -p "${RESULTS_ROOT}"
if [[ "${LOGGING_INITIALIZED:-0}" -ne 1 ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  export LOGGING_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "run_start project_dir=${PROJECT_DIR} results_root=${RESULTS_ROOT} max_seq_len=${MAX_SEQ_LEN} seed=${SEED}"

MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SEED="${SEED}" \
RUN_TRAIN="${RUN_TRAIN}" \
RUN_MOVE_EVALUATION="${RUN_MOVE_EVALUATION}" \
RUN_PROBES="${RUN_PROBES}" \
RUN_CHECK_PROBES="${RUN_CHECK_PROBES}" \
PREPARE_CHECK_PROBE_DATA="${PREPARE_CHECK_PROBE_DATA}" \
RUN_VISUALIZATIONS="${RUN_VISUALIZATIONS}" \
  "${PROJECT_DIR}/scripts/compare_transformer_sizes_maxseq512_with_probes.sh"

if [[ "${RUN_SUMMARY}" -eq 1 ]]; then
  SUMMARY_COMMAND=(
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/summarize_transformer_sizes_maxseq512.py"
    --experiment-dir "${RESULTS_ROOT}"
    --seed "${SEED}"
  )
  if [[ "${RUN_MOVE_EVALUATION}" -eq 1 ]]; then
    SUMMARY_COMMAND+=(--include-moves)
  fi
  if [[ "${RUN_CHECK_PROBES}" -eq 1 ]]; then
    SUMMARY_COMMAND+=(--include-check-probes)
  fi
  "${SUMMARY_COMMAND[@]}"
else
  echo "summary skipped (RUN_SUMMARY=0)"
fi

echo "run_complete results_root=${RESULTS_ROOT}"
