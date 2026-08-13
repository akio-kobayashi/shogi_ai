#!/usr/bin/env bash
# 学習済みfactorized_v3モデルに対し，線形probe学習から持ち駒遷移・駒打ち評価まで独立実行する。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || {
    echo "MEMORY_MAX is required on Linux; use e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi

CHECKPOINT="${1:?usage: $0 CHECKPOINT DATASET_DIR VOCAB OUTPUT_DIR}"
DATASET_DIR="${2:?dataset directory is required}"
VOCAB="${3:?factorized vocabulary is required}"
OUTPUT_DIR="${4:?output directory is required}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
PROBE_DIR="${OUTPUT_DIR}/linear-probes"

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint does not exist: ${CHECKPOINT}" >&2; exit 2; }
for split in train validation evaluation; do
  [[ -f "${DATASET_DIR}/${split}.jsonl" ]] || { echo "dataset split does not exist: ${DATASET_DIR}/${split}.jsonl" >&2; exit 2; }
done
[[ -f "${VOCAB}" ]] || { echo "vocabulary does not exist: ${VOCAB}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"

if [[ "${REUSE_LINEAR_PROBES:-0}" != 1 || ! -f "${PROBE_DIR}/linear_probes.pt" ]]; then
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_new_prompt_probes.py" \
    --checkpoint "${CHECKPOINT}" \
    --vocab "${VOCAB}" \
    --train-jsonl "${DATASET_DIR}/train.jsonl" \
    --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
    --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
    --output-dir "${PROBE_DIR}" \
    --sources "${HAND_PROBE_SOURCES:-layers}" \
    --history-distances "${HAND_PROBE_HISTORY_DISTANCES:-8,32}" \
    --max-train-samples "${HAND_PROBE_MAX_TRAIN_SAMPLES:-12000}" \
    --max-validation-samples "${HAND_PROBE_MAX_VALIDATION_SAMPLES:-3000}" \
    --max-evaluation-samples "${HAND_PROBE_MAX_EVALUATION_SAMPLES:-5000}" \
    --batch-size "${HAND_PROBE_BATCH_SIZE:-128}" \
    --length-bucket-pool-batches "${HAND_PROBE_LENGTH_BUCKET_POOL_BATCHES:-16}" \
    --alignment-check-samples "${HAND_PROBE_ALIGNMENT_CHECK_SAMPLES:-8}" \
    --amp "${EVAL_AMP:-auto}" \
    --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/linear_probe_training.log"
fi

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_hand_dynamics.py" \
  --checkpoint "${CHECKPOINT}" \
  --linear-probes "${PROBE_DIR}/linear_probes.pt" \
  --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
  --vocab "${VOCAB}" \
  --output "${OUTPUT_DIR}/hand_dynamics_metrics.json" \
  --sources "${HAND_DYNAMICS_SOURCES:-available}" \
  --max-events "${MAX_HAND_EVENTS:-10000}" \
  --max-drop-queries "${MAX_DROP_QUERIES:-5000}" \
  --max-games "${MAX_HAND_EVAL_GAMES:-5000}" \
  --batch-size "${HAND_EVAL_BATCH_SIZE:-64}" \
  --beam-micro-batch-size "${BEAM_MICRO_BATCH_SIZE:-8}" \
  --amp "${EVAL_AMP:-auto}" \
  --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/hand_dynamics_evaluation.log"

echo "hand evaluation complete: ${OUTPUT_DIR}/hand_dynamics_metrics.json"
