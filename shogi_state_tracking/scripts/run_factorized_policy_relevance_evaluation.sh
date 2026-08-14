#!/usr/bin/env bash
# 次手関連マスの復号精度と、プローブ方向への局所的介入効果を評価する。
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
RUN_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
OUTPUT_DIR="${4:-${RUN_DIR}/evaluation/policy-relevance}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
PROBE_DIR="${STATE_PROBE_DIR:-${RUN_DIR}/evaluation/probes}"
PROBE_ARTIFACT="${PROBE_DIR}/linear_probes.pt"

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint does not exist: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATASET_DIR}/evaluation.jsonl" ]] || { echo "evaluation split does not exist: ${DATASET_DIR}/evaluation.jsonl" >&2; exit 2; }
[[ -f "${VOCAB}" ]] || { echo "vocabulary does not exist: ${VOCAB}" >&2; exit 2; }
[[ -f "${PROBE_ARTIFACT}" ]] || {
  echo "linear probe artifact does not exist: ${PROBE_ARTIFACT}" >&2
  echo "run run_factorized_evaluation.sh ... probes first, or set STATE_PROBE_DIR" >&2
  exit 2
}

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_policy_relevance.py" \
  --checkpoint "${CHECKPOINT}" \
  --linear-probes "${PROBE_ARTIFACT}" \
  --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
  --vocab "${VOCAB}" \
  --output "${OUTPUT_DIR}/policy_relevance_metrics.json" \
  --sources "${POLICY_RELEVANCE_SOURCES:-available}" \
  --history-distances "${POLICY_RELEVANCE_HISTORY_DISTANCES:-8,32}" \
  --max-examples "${MAX_POLICY_RELEVANCE_EXAMPLES:-5000}" \
  --batch-size "${POLICY_RELEVANCE_BATCH_SIZE:-64}" \
  --steering-sources "${POLICY_STEERING_SOURCES:-middle,late,penultimate}" \
  --steering-strengths "${POLICY_STEERING_STRENGTHS:-0.5,1.0,2.0}" \
  --max-steering-examples "${MAX_POLICY_STEERING_EXAMPLES:-1000}" \
  --seed "${EVALUATION_SEED:-20260802}" \
  --amp "${EVAL_AMP:-auto}" \
  --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/policy_relevance_evaluation.log"

echo "policy-relevance evaluation complete: ${OUTPUT_DIR}/policy_relevance_metrics.json"
