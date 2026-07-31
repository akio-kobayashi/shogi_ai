#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_DIR}/data/vocab.json}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_DIR}/data/datasets/train.jsonl}"
VALIDATION_JSONL="${VALIDATION_JSONL:-${PROJECT_DIR}/data/datasets/validation.jsonl}"
EVALUATION_JSONL="${EVALUATION_JSONL:-${PROJECT_DIR}/data/datasets/evaluation.jsonl}"
SEED="${SEED:-20260724}"
DEVICE="${DEVICE:-auto}"
AMP="${AMP:-auto}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
if [[ "${MAX_SEQ_LEN}" != "512" ]]; then
  echo "MAX_SEQ_LEN must be 512 for this script" >&2
  exit 2
fi

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/transformer_size_compare_512}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MOVE_EVALUATION="${RUN_MOVE_EVALUATION:-1}"
RUN_PROBES="${RUN_PROBES:-1}"
RUN_CHECK_PROBES="${RUN_CHECK_PROBES:-1}"
PREPARE_CHECK_PROBE_DATA="${PREPARE_CHECK_PROBE_DATA:-1}"
RUN_VISUALIZATIONS="${RUN_VISUALIZATIONS:-0}"
PROBE_MODE="${PROBE_MODE:-all-layers}"
CHECK_PROBE_MODE="${CHECK_PROBE_MODE:-all-layers}"
PROBE_OUTPUT_SUFFIX="${PROBE_OUTPUT_SUFFIX:-probes}"
MOVE_OUTPUT_SUFFIX="${MOVE_OUTPUT_SUFFIX:-moves}"
CHECK_PROBE_OUTPUT_SUFFIX="${CHECK_PROBE_OUTPUT_SUFFIX:-check_probes}"
CHECK_PROBE_DIR="${CHECK_PROBE_DIR:-${PROJECT_DIR}/data/check_probe}"
CHECK_MAX_PREFIX_MOVES="${CHECK_MAX_PREFIX_MOVES:-$((MAX_SEQ_LEN - 99))}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
LOG_FILE="${LOG_FILE:-${RESULTS_ROOT}/run_with_probes.log}"

if [[ "${LOGGING_INITIALIZED:-0}" -ne 1 ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  export LOGGING_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

mkdir -p "${RESULTS_ROOT}"

if [[ "${RUN_TRAIN}" -ne 0 ]]; then
  echo "start training stage"
  MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
  RESULTS_ROOT="${RESULTS_ROOT}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
  AMP="${AMP}" \
  TRAIN_JSONL="${TRAIN_JSONL}" \
  VALIDATION_JSONL="${VALIDATION_JSONL}" \
  VOCAB_PATH="${VOCAB_PATH}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  EPOCHS="${EPOCHS}" \
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \
  EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA}" \
  PROGRESS_EVERY="${PROGRESS_EVERY}" \
    "${PROJECT_DIR}/scripts/compare_transformer_sizes_maxseq512.sh"
else
  echo "skip training stage (RUN_TRAIN=0)"
fi

if [[ "${RUN_CHECK_PROBES}" -eq 1 && "${PREPARE_CHECK_PROBE_DATA}" -eq 1 ]]; then
  echo "prepare balanced check-probe datasets max_prefix_moves=${CHECK_MAX_PREFIX_MOVES}"
  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_DIR="${PROJECT_DIR}/data/datasets" \
  OUTPUT_DIR="${CHECK_PROBE_DIR}" \
  MAX_PREFIX_MOVES="${CHECK_MAX_PREFIX_MOVES}" \
    "${PROJECT_DIR}/scripts/create_check_probe_datasets.sh"
elif [[ "${RUN_CHECK_PROBES}" -eq 1 ]]; then
  echo "use existing check-probe datasets: ${CHECK_PROBE_DIR}"
fi

for size in small base large; do
  checkpoint="${RESULTS_ROOT}/seed_${SEED}/${size}/best.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "checkpoint not found: ${checkpoint}" >&2
    echo "set RUN_TRAIN=1 or provide pretrained checkpoints" >&2
    exit 2
  fi

  size_dir="${RESULTS_ROOT}/seed_${SEED}/${size}"

  if [[ "${RUN_MOVE_EVALUATION}" -eq 1 ]]; then
    move_output_dir="${size_dir}/${MOVE_OUTPUT_SUFFIX}"
    echo "run move evaluation for size=${size} checkpoint=${checkpoint}"
    PYTHON_BIN="${PYTHON_BIN}" \
    CHECKPOINT="${checkpoint}" \
    VOCAB_PATH="${VOCAB_PATH}" \
    EVALUATION_JSONL="${EVALUATION_JSONL}" \
    OUTPUT_DIR="${move_output_dir}" \
    DEVICE="${DEVICE}" \
      "${PROJECT_DIR}/scripts/run_move_evaluation.sh"
  else
    echo "skip move evaluation for size=${size} (RUN_MOVE_EVALUATION=0)"
  fi

  if [[ "${RUN_PROBES}" -eq 1 ]]; then
    probe_output_dir="${size_dir}/${PROBE_OUTPUT_SUFFIX}"
    echo "run state probe for size=${size} checkpoint=${checkpoint} mode=${PROBE_MODE}"
    PYTHON_BIN="${PYTHON_BIN}" \
    CHECKPOINT="${checkpoint}" \
    VOCAB_PATH="${VOCAB_PATH}" \
    TRAIN_JSONL="${TRAIN_JSONL}" \
    VALIDATION_JSONL="${VALIDATION_JSONL}" \
    EVALUATION_JSONL="${EVALUATION_JSONL}" \
    OUTPUT_DIR="${probe_output_dir}" \
    DEVICE="${DEVICE}" \
      "${PROJECT_DIR}/scripts/run_probe_evaluation.sh" "${PROBE_MODE}"
  else
    echo "skip state probe for size=${size} (RUN_PROBES=0)"
  fi

  if [[ "${RUN_CHECK_PROBES}" -eq 1 ]]; then
    check_output_dir="${size_dir}/${CHECK_PROBE_OUTPUT_SUFFIX}"
    echo "run check probe for size=${size} checkpoint=${checkpoint} mode=${CHECK_PROBE_MODE}"
    PYTHON_BIN="${PYTHON_BIN}" \
    CHECKPOINT="${checkpoint}" \
    VOCAB_PATH="${VOCAB_PATH}" \
    CHECK_PROBE_DIR="${CHECK_PROBE_DIR}" \
    OUTPUT_DIR="${check_output_dir}" \
    DEVICE="${DEVICE}" \
      "${PROJECT_DIR}/scripts/run_check_probe_evaluation.sh" "${CHECK_PROBE_MODE}"
  else
    echo "skip check probe for size=${size} (RUN_CHECK_PROBES=0)"
  fi
done

if [[ "${RUN_VISUALIZATIONS}" -eq 1 ]]; then
  cat >&2 <<'EOF'
RUN_VISUALIZATIONS=1 was requested, but no visualizations were run.
Major-piece and king plots require a deliberately selected game_id, ply, piece,
and trained linear_probes.pt. Run visualize_major_piece_probe.py after inspecting
the numerical metrics; do not select examples automatically.
EOF
else
  echo "visualizations deferred: select representative game_id/ply/piece after numerical evaluation"
fi

echo "run_complete results_root=${RESULTS_ROOT}"
