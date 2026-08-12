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
EVAL_STAGE="${EVAL_STAGE:-main}"
VOCAB="${DATASET_DIR}/vocab.json"
MANIFEST="${DATASET_DIR}/dataset_manifest.json"

# 出力パスを組み立てる前にモデル指定を解釈する．従来はこれらをそのまま
# 学習シェルへ渡していたため，学習はsmall，評価パスはbaseという不整合が
# 起こり得た．その他の学習引数だけをEXTRA_ARGSとして転送する．
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-type) MODEL_TYPE="$2"; shift 2 ;;
    --model-type=*) MODEL_TYPE="${1#*=}"; shift ;;
    --model-size) MODEL_SIZE="$2"; shift 2 ;;
    --model-size=*) MODEL_SIZE="${1#*=}"; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
case "${MODEL_TYPE}" in llama|vanilla) ;; *) echo "--model-type must be llama or vanilla" >&2; exit 2 ;; esac
case "${MODEL_SIZE}" in small|base|large) ;; *) echo "--model-size must be small, base, or large" >&2; exit 2 ;; esac
echo "factorized experiment configuration: model_type=${MODEL_TYPE} model_size=${MODEL_SIZE} rap_rates=${RAP_RATES} seeds=${SEEDS}" >&2

for path in "${DATASET_DIR}/train.jsonl" "${DATASET_DIR}/validation.jsonl" "${DATASET_DIR}/evaluation.jsonl" "${VOCAB}" "${MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
grep -q '"move_encoding"[[:space:]]*:[[:space:]]*"factorized_v3_no_eom"' "${MANIFEST}" || {
  echo "obsolete dataset: rebuild factorized_v3 first" >&2; exit 2;
}
grep -q '"stage_1_2_input_mode"[[:space:]]*:[[:space:]]*"implicit_standard_initial"' "${MANIFEST}" || {
  echo "obsolete dataset: rebuild factorized_v3 with implicit standard-initial manifest" >&2; exit 2;
}
grep -q '"terminal_encoding"[[:space:]]*:[[:space:]]*"eos_on_complete_decisive_game_v1"' "${MANIFEST}" || {
  echo "obsolete dataset: rebuild factorized_v3 with complete-game EOS supervision" >&2; exit 2;
}

IFS=',' read -r -a rates <<< "${RAP_RATES}"
IFS=',' read -r -a seeds <<< "${SEEDS}"
for seed in "${seeds[@]}"; do
  for rate in "${rates[@]}"; do
    mode=rap
    [[ "${rate}" == 0 || "${rate}" == 0.0 || "${rate}" == 0.00 ]] && mode=vanilla
    output="${RESULTS_DIR}/${MODEL_TYPE}-${MODEL_SIZE}/implicit-initial/${mode}-p${rate}/seed-${seed}"
    if [[ "${FORCE_TRAIN:-0}" == 1 || ! -f "${output}/best.pt" ]]; then
      SEED="${seed}" MODEL_TYPE="${MODEL_TYPE}" MODEL_SIZE="${MODEL_SIZE}" ANNOTATION_MODE="${mode}" \
        ANNOTATION_PROBABILITY="${rate}" BATCH_SIZE="${BATCH_SIZE:-8}" NUM_WORKERS="${NUM_WORKERS:-0}" \
        MAX_SEQ_LEN="${MAX_SEQ_LEN:-2560}" MAX_MOVES="${MAX_MOVES:-512}" MAX_HINTS="${MAX_HINTS:-512}" \
        EPOCHS="${EPOCHS:-50}" "${SCRIPT_DIR}/scripts/run_factorized_baseline.sh" \
        "${DATASET_DIR}" "${RESULTS_DIR}" --seed "${seed}" "${EXTRA_ARGS[@]}"
    fi
    [[ -f "${output}/best.pt" ]] || {
      echo "training did not produce ${output}/best.pt; evaluation is aborted. Use a fresh results directory and rerun training." >&2
      exit 1
    }
    DEVICE="${DEVICE:-auto}" "${SCRIPT_DIR}/scripts/run_factorized_evaluation.sh" \
      "${output}/best.pt" "${DATASET_DIR}" "${VOCAB}" "${output}/evaluation" "${EVAL_STAGE}"
  done
done
