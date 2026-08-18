#!/usr/bin/env bash
# 既存last.ptから全条件を同じepochまで継続する．元resultsは変更しない．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || {
    echo "MEMORY_MAX is required on Linux; use e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi

DATASET_DIR="${1:?usage: $0 DATASET_DIR SOURCE_RESULTS_DIR FIXED_RESULTS_DIR}"
SOURCE_RESULTS_DIR="${2:?source results directory is required}"
FIXED_RESULTS_DIR="${3:?fixed-epoch results directory is required}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODEL_TYPE="${MODEL_TYPE:-llama}"
MODEL_SIZE="${MODEL_SIZE:-reference}"
# q=1.0は当初仕様どおりAP（学習・評価とも常時駒種注釈）として扱う。
# fixed50と収集器の既定条件を一致させ，APだけ欠けるartifactを作らない。
RAP_RATES="${RAP_RATES:-0.0,0.15,0.25,1.0}"
SEEDS="${SEEDS:-20260802}"
TARGET_EPOCHS="${TARGET_EPOCHS:-50}"

[[ "${SOURCE_RESULTS_DIR}" != "${FIXED_RESULTS_DIR}" ]] || {
  echo "SOURCE_RESULTS_DIR and FIXED_RESULTS_DIR must differ; the original results are never modified" >&2
  exit 2
}
for path in "${DATASET_DIR}/train.jsonl" "${DATASET_DIR}/validation.jsonl" \
  "${DATASET_DIR}/evaluation.jsonl" "${DATASET_DIR}/vocab.json" "${DATASET_DIR}/dataset_manifest.json"; do
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
case "${MODEL_TYPE}" in llama|vanilla) ;; *) echo "MODEL_TYPE must be llama or vanilla" >&2; exit 2 ;; esac
case "${MODEL_SIZE}" in small|base|large|reference) ;; *) echo "MODEL_SIZE must be small, base, large, or reference" >&2; exit 2 ;; esac
[[ "${TARGET_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || { echo "TARGET_EPOCHS must be a positive integer" >&2; exit 2; }

checkpoint_epoch() {
  "${PYTHON_BIN}" -c \
    'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("epoch", -1)))' "$1"
}

IFS=',' read -r -a rates <<< "${RAP_RATES}"
IFS=',' read -r -a seeds <<< "${SEEDS}"
for seed in "${seeds[@]}"; do
  for rate in "${rates[@]}"; do
    mode=rap
    [[ "${rate}" == 0 || "${rate}" == 0.0 || "${rate}" == 0.00 ]] && mode=vanilla
    [[ "${rate}" == 1 || "${rate}" == 1.0 || "${rate}" == 1.00 ]] && mode=ap
    run_variant=""
    [[ "${mode}" == rap ]] && run_variant="proportional-rap-v1"
    [[ "${mode}" == ap ]] && run_variant="proportional-annotation-v1"
    condition="${mode}-p${rate}"
    [[ -n "${run_variant}" ]] && condition="${condition}-${run_variant}"

    relative_run="${MODEL_TYPE}-${MODEL_SIZE}/implicit-initial/${condition}/seed-${seed}"
    source_run="${SOURCE_RESULTS_DIR}/${relative_run}"
    fixed_run="${FIXED_RESULTS_DIR}/${relative_run}"
    source_last="${source_run}/last.pt"
    source_best="${source_run}/best.pt"
    fixed_last="${fixed_run}/last.pt"
    [[ -f "${source_last}" || -f "${fixed_last}" ]] || {
      echo "missing source checkpoint: ${source_last}" >&2
      exit 2
    }

    # 初回だけ元runを固定epoch用rootへ複製する．以後は固定epoch側から再開する．
    # 元のbest.pt，last.pt，履歴，評価結果には一切書き込まない．
    if [[ ! -f "${fixed_last}" ]]; then
      [[ -f "${source_best}" ]] || { echo "missing source checkpoint: ${source_best}" >&2; exit 2; }
      mkdir -p "${fixed_run}"
      cp -p "${source_last}" "${fixed_last}"
      for name in best.pt training_history.json run_manifest.json; do
        [[ ! -f "${source_run}/${name}" ]] || cp -p "${source_run}/${name}" "${fixed_run}/${name}"
      done
      echo "initialized fixed-epoch run from ${source_run}"
    fi
    # 中断した初期化を再実行した場合も，train_new_prompt.pyが要求するbest.ptを補う．
    if [[ ! -f "${fixed_run}/best.pt" ]]; then
      [[ -f "${source_best}" ]] || { echo "missing source checkpoint: ${source_best}" >&2; exit 2; }
      cp -p "${source_best}" "${fixed_run}/best.pt"
    fi

    current_epoch="$(checkpoint_epoch "${fixed_last}")"
    if (( current_epoch > TARGET_EPOCHS )); then
      echo "${fixed_last} is already at epoch ${current_epoch}, beyond target ${TARGET_EPOCHS}" >&2
      exit 2
    fi
    if (( current_epoch == TARGET_EPOCHS )); then
      echo "fixed-epoch training already complete: ${fixed_last} (epoch ${current_epoch})"
      continue
    fi

    echo "continuing ${condition}, seed=${seed}: epoch ${current_epoch} -> ${TARGET_EPOCHS}"
    SEED="${seed}" MODEL_TYPE="${MODEL_TYPE}" MODEL_SIZE="${MODEL_SIZE}" \
      ANNOTATION_MODE="${mode}" RUN_VARIANT="${run_variant}" ANNOTATION_PROBABILITY="${rate}" \
      BATCH_SIZE="${BATCH_SIZE:-8}" NUM_WORKERS="${NUM_WORKERS:-0}" \
      MAX_SEQ_LEN="${MAX_SEQ_LEN:-2560}" MAX_MOVES="${MAX_MOVES:-512}" MAX_HINTS="${MAX_HINTS:-512}" \
      EPOCHS="${TARGET_EPOCHS}" "${SCRIPT_DIR}/scripts/run_factorized_baseline.sh" \
      "${DATASET_DIR}" "${FIXED_RESULTS_DIR}" --seed "${seed}" --resume --early-stopping-patience 0

    completed_epoch="$(checkpoint_epoch "${fixed_last}")"
    [[ "${completed_epoch}" == "${TARGET_EPOCHS}" ]] || {
      echo "training ended at epoch ${completed_epoch}; expected ${TARGET_EPOCHS}: ${fixed_last}" >&2
      exit 1
    }
  done
done

echo "fixed-epoch training complete: ${FIXED_RESULTS_DIR}"
