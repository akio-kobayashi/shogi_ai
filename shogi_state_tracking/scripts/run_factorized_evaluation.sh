#!/usr/bin/env bash
# cheapな指手評価，token probe，学習を伴う線形プローブを分離して実行する。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || {
    echo "MEMORY_MAX is required on Linux; use e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi
CHECKPOINT="${1:?usage: $0 CHECKPOINT DATASET_DIR VOCAB OUTPUT_DIR [moves|probes|all]}"
DATASET_DIR="${2:?dataset directory is required}"
VOCAB="${3:?factorized vocabulary is required}"
OUTPUT_DIR="${4:?output directory is required}"
STAGE="${5:-all}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
mkdir -p "${OUTPUT_DIR}"

if [[ "${STAGE}" == moves || "${STAGE}" == all ]]; then
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_moves.py" \
    --checkpoint "${CHECKPOINT}" \
    --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
    --vocab "${VOCAB}" \
    --output "${OUTPUT_DIR}/move_metrics.json" \
    --batch-size "${EVAL_BATCH_SIZE:-64}" \
    --max-queries "${MAX_EVAL_QUERIES:-30000}" \
    --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/move_evaluation.log"
fi

if [[ "${STAGE}" == token || "${STAGE}" == all ]]; then
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_token_probe.py" \
    --checkpoint "${CHECKPOINT}" \
    --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
    --vocab "${VOCAB}" \
    --output "${OUTPUT_DIR}/token_probe_metrics.json" \
    --batch-size "${TOKEN_PROBE_BATCH_SIZE:-128}" \
    --max-queries "${MAX_TOKEN_PROBE_QUERIES:-30000}" \
    --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/token_probe.log"
fi

if [[ "${STAGE}" == probes || "${STAGE}" == all ]]; then
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_new_prompt_probes.py" \
    --checkpoint "${CHECKPOINT}" \
    --vocab "${VOCAB}" \
    --train-jsonl "${DATASET_DIR}/train.jsonl" \
    --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
    --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
    --output-dir "${OUTPUT_DIR}/probes" \
    --batch-size "${PROBE_BATCH_SIZE:-128}" \
    --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/probe_evaluation.log"
fi

case "${STAGE}" in moves|token|probes|all) ;; *) echo "stage must be moves, token, probes, or all" >&2; exit 2 ;; esac
