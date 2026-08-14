#!/usr/bin/env bash
# 学習済みfactorized_v3モデルへ，チェス先行研究対応のStart/End評価だけを実行する．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || {
    echo "MEMORY_MAX is required on Linux; use e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi

CHECKPOINT="${1:?usage: $0 CHECKPOINT DATASET_DIR VOCAB [OUTPUT_DIR]}"
DATASET_DIR="${2:?dataset directory is required}"
VOCAB="${3:?factorized vocabulary is required}"
DEFAULT_OUTPUT="$(dirname "${CHECKPOINT}")/evaluation/chess-protocol"
OUTPUT_DIR="${4:-${DEFAULT_OUTPUT}}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint does not exist: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATASET_DIR}/evaluation.jsonl" ]] || { echo "evaluation data does not exist: ${DATASET_DIR}/evaluation.jsonl" >&2; exit 2; }
[[ -f "${VOCAB}" ]] || { echo "vocabulary does not exist: ${VOCAB}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_chess_protocol.py" \
  --checkpoint "${CHECKPOINT}" \
  --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
  --vocab "${VOCAB}" \
  --output "${OUTPUT_DIR}/chess_protocol_metrics.json" \
  --min-prefix-plies "${CHESS_PROTOCOL_MIN_PREFIX_PLIES:-51}" \
  --max-prefix-plies "${CHESS_PROTOCOL_MAX_PREFIX_PLIES:-100}" \
  --max-instances "${CHESS_PROTOCOL_MAX_INSTANCES:-1000}" \
  --max-games "${CHESS_PROTOCOL_MAX_GAMES:-0}" \
  --batch-size "${CHESS_PROTOCOL_BATCH_SIZE:-128}" \
  --length-bucket-pool-batches "${EVAL_LENGTH_BUCKET_POOL_BATCHES:-16}" \
  --seed "${SEED:-20260802}" \
  --amp "${EVAL_AMP:-auto}" \
  --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/chess_protocol_evaluation.log"

