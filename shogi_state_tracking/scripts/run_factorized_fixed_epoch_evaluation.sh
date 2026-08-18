#!/usr/bin/env bash
# 固定epochのlast.ptを標準データとLishogi非BOTデータで同一条件評価する．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || {
    echo "MEMORY_MAX is required on Linux; use e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi

DATASET_DIR="${1:?usage: $0 DATASET_DIR FIXED_RESULTS_DIR LISHOGI_DATASET_DIR}"
FIXED_RESULTS_DIR="${2:?fixed-epoch results directory is required}"
LISHOGI_DATASET_DIR="${3:?Lishogi evaluation dataset directory is required}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODEL_TYPE="${MODEL_TYPE:-llama}"
MODEL_SIZE="${MODEL_SIZE:-reference}"
# q=1.0は当初仕様どおりAPとして評価する。fixed50学習・収集器と同じ4条件を既定にする。
RAP_RATES="${RAP_RATES:-0.0,0.15,0.25,1.0}"
SEEDS="${SEEDS:-20260802}"
TARGET_EPOCHS="${TARGET_EPOCHS:-50}"
STANDARD_EVAL_STAGE="${STANDARD_EVAL_STAGE:-main}"
VOCAB="${DATASET_DIR}/vocab.json"

for path in "${DATASET_DIR}/train.jsonl" "${DATASET_DIR}/validation.jsonl" \
  "${DATASET_DIR}/evaluation.jsonl" "${VOCAB}" "${DATASET_DIR}/dataset_manifest.json" \
  "${LISHOGI_DATASET_DIR}/evaluation.jsonl"; do
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
if [[ -f "${LISHOGI_DATASET_DIR}/vocab.json" ]] && ! cmp -s "${VOCAB}" "${LISHOGI_DATASET_DIR}/vocab.json"; then
  echo "standard and Lishogi vocabularies differ" >&2
  exit 2
fi

checkpoint_epoch() {
  "${PYTHON_BIN}" -c \
    'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("epoch", -1)))' "$1"
}

IFS=',' read -r -a rates <<< "${RAP_RATES}"
IFS=',' read -r -a seeds <<< "${SEEDS}"

resolve_condition() {
  local rate="$1"
  local mode=rap
  [[ "${rate}" == 0 || "${rate}" == 0.0 || "${rate}" == 0.00 ]] && mode=vanilla
  [[ "${rate}" == 1 || "${rate}" == 1.0 || "${rate}" == 1.00 ]] && mode=ap
  local run_variant=""
  [[ "${mode}" == rap ]] && run_variant="proportional-rap-v1"
  [[ "${mode}" == ap ]] && run_variant="proportional-annotation-v1"
  RESOLVED_CONDITION="${mode}-p${rate}"
  [[ -n "${run_variant}" ]] && RESOLVED_CONDITION="${RESOLVED_CONDITION}-${run_variant}"
}

# 逐次評価の途中で後続条件の欠落が判明すると，先頭条件だけを含む部分artifactが
# 残る。全条件のfixed50 checkpointを先に検査し，1件でも欠ければ評価を開始しない。
preflight_failed=0
for seed in "${seeds[@]}"; do
  for rate in "${rates[@]}"; do
    resolve_condition "${rate}"
    run_dir="${FIXED_RESULTS_DIR}/${MODEL_TYPE}-${MODEL_SIZE}/implicit-initial/${RESOLVED_CONDITION}/seed-${seed}"
    checkpoint="${run_dir}/last.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "missing fixed-epoch checkpoint: ${checkpoint}" >&2
      preflight_failed=1
      continue
    fi
    actual_epoch="$(checkpoint_epoch "${checkpoint}")"
    if [[ "${actual_epoch}" != "${TARGET_EPOCHS}" ]]; then
      echo "refusing unequal-budget evaluation: ${checkpoint} is epoch ${actual_epoch}, expected ${TARGET_EPOCHS}" >&2
      preflight_failed=1
    fi
  done
done
[[ "${preflight_failed}" == 0 ]] || {
  echo "fixed-epoch evaluation preflight failed; no condition was evaluated" >&2
  exit 2
}

for seed in "${seeds[@]}"; do
  for rate in "${rates[@]}"; do
    resolve_condition "${rate}"
    condition="${RESOLVED_CONDITION}"

    run_dir="${FIXED_RESULTS_DIR}/${MODEL_TYPE}-${MODEL_SIZE}/implicit-initial/${condition}/seed-${seed}"
    checkpoint="${run_dir}/last.pt"
    actual_epoch="$(checkpoint_epoch "${checkpoint}")"

    echo "[standard] ${condition}, seed=${seed}, checkpoint=last.pt, epoch=${actual_epoch}"
    "${SCRIPT_DIR}/scripts/run_factorized_evaluation.sh" \
      "${checkpoint}" "${DATASET_DIR}" "${VOCAB}" "${run_dir}/evaluation" "${STANDARD_EVAL_STAGE}"

    echo "[Lishogi moves] ${condition}, seed=${seed}"
    VOCAB="${VOCAB}" "${SCRIPT_DIR}/scripts/run_reference_lishogi_move_evaluation.sh" \
      "${checkpoint}" "${LISHOGI_DATASET_DIR}" "${run_dir}/evaluation/lishogi-non-bot/moves"

    echo "[Lishogi linear probes] ${condition}, seed=${seed}"
    VOCAB="${VOCAB}" "${SCRIPT_DIR}/scripts/run_reference_lishogi_linear_probe_evaluation.sh" \
      "${checkpoint}" "${DATASET_DIR}" "${LISHOGI_DATASET_DIR}" \
      "${run_dir}/evaluation/lishogi-non-bot/linear-probes"
  done
done

echo "fixed-epoch standard and Lishogi evaluation complete: ${FIXED_RESULTS_DIR}"
