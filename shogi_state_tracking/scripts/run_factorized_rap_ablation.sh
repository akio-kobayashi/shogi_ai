#!/usr/bin/env bash
# 追加ablation：factorized_v3のRAP挿入率を比較する．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || { echo "MEMORY_MAX is required on Linux" >&2; exit 2; }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi
DATASET_DIR="${1:?usage: $0 DATASET_DIR RESULTS_DIR}"
RESULTS_DIR="${2:?results directory is required}"
shift 2
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODEL_TYPE="${MODEL_TYPE:-llama}"
MODEL_SIZE="${MODEL_SIZE:-base}"
RAP_RATES="${RAP_RATES:-0.0,0.05,0.15,0.30,1.0}"
SEEDS="${SEEDS:-20260802}"
EVAL_STAGE="${EVAL_STAGE:-moves}"
VOCAB="${DATASET_DIR}/vocab.json"
MANIFEST="${DATASET_DIR}/dataset_manifest.json"
for path in "${DATASET_DIR}/train.jsonl" "${DATASET_DIR}/validation.jsonl" "${DATASET_DIR}/evaluation.jsonl" "${VOCAB}" "${MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
grep -q '"move_encoding"[[:space:]]*:[[:space:]]*"factorized_v3_no_eom"' "${MANIFEST}" || {
  echo "obsolete dataset: rebuild factorized_v3 first" >&2; exit 2;
}

IFS=',' read -r -a rates <<< "${RAP_RATES}"
IFS=',' read -r -a seeds <<< "${SEEDS}"
for seed in "${seeds[@]}"; do
  for rate in "${rates[@]}"; do
    mode=rap
    [[ "${rate}" == 0 || "${rate}" == 0.0 || "${rate}" == 0.00 ]] && mode=vanilla
    output="${RESULTS_DIR}/${MODEL_TYPE}-${MODEL_SIZE}/${mode}-p${rate}/seed-${seed}"
    if [[ "${FORCE_TRAIN:-0}" == 1 || ! -f "${output}/best.pt" ]]; then
      SEED="${seed}" MODEL_TYPE="${MODEL_TYPE}" MODEL_SIZE="${MODEL_SIZE}" ANNOTATION_MODE="${mode}" \
        ANNOTATION_PROBABILITY="${rate}" BATCH_SIZE="${BATCH_SIZE:-8}" NUM_WORKERS="${NUM_WORKERS:-0}" \
        MAX_SEQ_LEN="${MAX_SEQ_LEN:-2560}" MAX_MOVES="${MAX_MOVES:-512}" MAX_HINTS="${MAX_HINTS:-512}" \
        EPOCHS="${EPOCHS:-50}" "${SCRIPT_DIR}/scripts/run_factorized_baseline.sh" \
        "${DATASET_DIR}" "${RESULTS_DIR}" --seed "${seed}" "$@"
    fi
    DEVICE="${DEVICE:-auto}" "${SCRIPT_DIR}/scripts/run_factorized_evaluation.sh" \
      "${output}/best.pt" "${DATASET_DIR}" "${VOCAB}" "${output}/evaluation" "${EVAL_STAGE}"
  done
done
