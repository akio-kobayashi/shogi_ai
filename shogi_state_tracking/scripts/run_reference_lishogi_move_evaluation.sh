#!/usr/bin/env bash
# reference checkpointを非BOT Lishogi評価局面で評価する。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || {
    echo "MEMORY_MAX is required on Linux; e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi

CHECKPOINT="${1:?usage: $0 CHECKPOINT LISHOGI_EVAL_DATASET [OUTPUT_DIR]}"
EVAL_DATASET="${2:?evaluation dataset directory is required}"
RUN_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
OUTPUT_DIR="${3:-${RUN_DIR}/evaluation/lishogi-non-bot/moves}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
VOCAB="${VOCAB:-${EVAL_DATASET}/vocab.json}"

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint does not exist: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${EVAL_DATASET}/evaluation.jsonl" ]] || { echo "evaluation.jsonl does not exist: ${EVAL_DATASET}" >&2; exit 2; }
[[ -f "${VOCAB}" ]] || { echo "vocab does not exist: ${VOCAB}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_moves.py" \
  --checkpoint "${CHECKPOINT}" \
  --evaluation-jsonl "${EVAL_DATASET}/evaluation.jsonl" \
  --vocab "${VOCAB}" \
  --output "${OUTPUT_DIR}/move_metrics.json" \
  --history-distances "${EVAL_HISTORY_DISTANCES:-0,8,32}" \
  --primary-history-distances "${PRIMARY_HISTORY_DISTANCES:-8,32}" \
  --max-games "${MAX_EVAL_GAMES:-500}" \
  --max-queries "${MAX_EVAL_QUERIES:-30000}" \
  --batch-size "${EVAL_BATCH_SIZE:-64}" \
  --length-bucket-pool-batches "${EVAL_LENGTH_BUCKET_POOL_BATCHES:-16}" \
  --beam-micro-batch-size "${BEAM_MICRO_BATCH_SIZE:-8}" \
  --amp "${EVAL_AMP:-auto}" \
  --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/move_evaluation.log"

echo "Lishogi move evaluation complete: ${OUTPUT_DIR}/move_metrics.json"
