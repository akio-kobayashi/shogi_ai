#!/usr/bin/env bash
# 凍結したreference LLaMAで駒打ち時の持ち駒信頼度・attention・遮断を連続評価する。
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
OUTPUT_DIR="${4:-${RUN_DIR}/evaluation/drop-relevance}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ -n "${STATE_PROBE_DIR:-}" ]]; then
  PROBE_DIR="${STATE_PROBE_DIR}"
elif [[ -f "${RUN_DIR}/evaluation/probes/linear_probes.pt" ]]; then
  PROBE_DIR="${RUN_DIR}/evaluation/probes"
else
  PROBE_DIR="${RUN_DIR}/evaluation/hand-evaluation/linear-probes"
fi
PROBE_ARTIFACT="${PROBE_DIR}/linear_probes.pt"

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint does not exist: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATASET_DIR}/evaluation.jsonl" ]] || { echo "evaluation split does not exist" >&2; exit 2; }
[[ -f "${DATASET_DIR}/validation.jsonl" ]] || { echo "validation split does not exist" >&2; exit 2; }
[[ -f "${VOCAB}" ]] || { echo "vocabulary does not exist: ${VOCAB}" >&2; exit 2; }
[[ -f "${PROBE_ARTIFACT}" ]] || {
  echo "linear probe artifact does not exist: ${PROBE_ARTIFACT}" >&2
  echo "run the ordinary factorized probe evaluation first, or set STATE_PROBE_DIR" >&2
  exit 2
}

mkdir -p "${OUTPUT_DIR}/figures" "${OUTPUT_DIR}/logs"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_drop_relevance.py" \
  --checkpoint "${CHECKPOINT}" \
  --linear-probes "${PROBE_ARTIFACT}" \
  --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
  --calibration-jsonl "${DATASET_DIR}/validation.jsonl" \
  --vocab "${VOCAB}" \
  --output "${OUTPUT_DIR}/confidence_trajectory.json" \
  --sources "${DROP_RELEVANCE_SOURCES:-available}" \
  --window "${DROP_RELEVANCE_WINDOW:-16}" \
  --max-drops "${MAX_DROP_RELEVANCE_EXAMPLES:-5000}" \
  --max-calibration-examples "${MAX_DROP_CALIBRATION_EXAMPLES:-5000}" \
  --batch-size "${DROP_RELEVANCE_BATCH_SIZE:-64}" \
  --seed "${EVALUATION_SEED:-20260802}" \
  --amp "${EVAL_AMP:-auto}" \
  --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/logs/confidence.log"

if [[ "${SKIP_DROP_ATTENTION:-0}" != 1 ]]; then
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_drop_attention.py" \
    --checkpoint "${CHECKPOINT}" \
    --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
    --vocab "${VOCAB}" \
    --output "${OUTPUT_DIR}/attention_metrics.json" \
    --max-pairs "${MAX_DROP_ATTENTION_PAIRS:-1000}" \
    --max-ablation-pairs "${MAX_DROP_ABLATION_PAIRS:-250}" \
    --ablation-layers "${DROP_ABLATION_LAYERS:-middle,late,all}" \
    --seed "${EVALUATION_SEED:-20260802}" \
    --amp "${ATTENTION_AMP:-off}" \
    --device "${DEVICE:-auto}" 2>&1 | tee "${OUTPUT_DIR}/logs/attention.log"
fi

visualize_args=(
  --trajectory "${OUTPUT_DIR}/confidence_trajectory.json"
  --output-dir "${OUTPUT_DIR}/figures"
)
[[ ! -f "${OUTPUT_DIR}/attention_metrics.json" ]] || visualize_args+=(--attention "${OUTPUT_DIR}/attention_metrics.json")
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/visualize_factorized_drop_relevance.py" "${visualize_args[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/visualization.log"

echo "reference drop-relevance experiment complete: ${OUTPUT_DIR}"
