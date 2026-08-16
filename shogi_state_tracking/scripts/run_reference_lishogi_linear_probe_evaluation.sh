#!/usr/bin/env bash
# reference checkpointの線形状態probeを機械棋譜で学習し，非BOT Lishogi棋譜で評価する。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || {
    echo "MEMORY_MAX is required on Linux; e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi

CHECKPOINT="${1:?usage: $0 CHECKPOINT TRAIN_DATASET LISHOGI_EVAL_DATASET [OUTPUT_DIR]}"
TRAIN_DATASET="${2:?training dataset directory is required}"
EVAL_DATASET="${3:?Lishogi evaluation dataset directory is required}"
RUN_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
OUTPUT_DIR="${4:-${RUN_DIR}/evaluation/lishogi-non-bot/linear-probes}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
VOCAB="${VOCAB:-${TRAIN_DATASET}/vocab.json}"

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint does not exist: ${CHECKPOINT}" >&2; exit 2; }
for split in train validation; do
  [[ -f "${TRAIN_DATASET}/${split}.jsonl" ]] || {
    echo "${split}.jsonl does not exist: ${TRAIN_DATASET}" >&2
    exit 2
  }
done
[[ -f "${EVAL_DATASET}/evaluation.jsonl" ]] || { echo "evaluation.jsonl does not exist: ${EVAL_DATASET}" >&2; exit 2; }
[[ -f "${VOCAB}" ]] || { echo "vocab does not exist: ${VOCAB}" >&2; exit 2; }
if [[ -f "${EVAL_DATASET}/vocab.json" ]] && ! cmp -s "${VOCAB}" "${EVAL_DATASET}/vocab.json"; then
  echo "training and Lishogi evaluation vocabularies differ" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_new_prompt_probes.py" \
  --checkpoint "${CHECKPOINT}" \
  --vocab "${VOCAB}" \
  --train-jsonl "${TRAIN_DATASET}/train.jsonl" \
  --validation-jsonl "${TRAIN_DATASET}/validation.jsonl" \
  --evaluation-jsonl "${EVAL_DATASET}/evaluation.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --sources "${PROBE_SOURCES:-layers}" \
  --history-distances "${PROBE_HISTORY_DISTANCES:-8,32}" \
  --max-train-samples "${PROBE_MAX_TRAIN_SAMPLES:-12000}" \
  --max-validation-samples "${PROBE_MAX_VALIDATION_SAMPLES:-3000}" \
  --max-evaluation-samples "${PROBE_MAX_EVALUATION_SAMPLES:-5000}" \
  --batch-size "${PROBE_BATCH_SIZE:-128}" \
  --length-bucket-pool-batches "${PROBE_LENGTH_BUCKET_POOL_BATCHES:-16}" \
  --probe-epochs "${PROBE_EPOCHS:-30}" \
  --patience "${PROBE_PATIENCE:-5}" \
  --alignment-check-samples "${STATE_PROBE_ALIGNMENT_CHECK_SAMPLES:-8}" \
  --amp "${EVAL_AMP:-auto}" \
  --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/probe_evaluation.log"

echo "Lishogi linear-probe evaluation complete: ${OUTPUT_DIR}/probe_metrics.json"
