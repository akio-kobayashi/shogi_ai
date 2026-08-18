#!/usr/bin/env bash
# reference LLaMAの同一prefix行動条件実験を3主条件＋AP別枠で実行する。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Linux && "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" != 1 && "${ALLOW_UNBOUNDED_MEMORY:-0}" != 1 ]]; then
  [[ -n "${MEMORY_MAX:-}" ]] || {
    echo "MEMORY_MAX is required on Linux; use e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
  exec "${SCRIPT_DIR}/scripts/run_memory_bounded.sh" "$0" "$@"
fi

DATASET_DIR="${1:?usage: $0 DATASET_DIR RESULTS_DIR}"
RESULTS_DIR="${2:?reference results directory is required}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
VOCAB="${VOCAB:-${DATASET_DIR}/vocab.json}"
SEEDS="${SEEDS:-20260802}"
CHECKPOINT_NAME="${ACTION_CONDITION_CHECKPOINT:-last.pt}"
TARGET_EPOCHS="${TARGET_EPOCHS:-50}"

IFS=',' read -r -a seed_values <<< "${SEEDS}"
if [[ "${REQUIRE_MULTIPLE_MODEL_SEEDS:-0}" == 1 && "${#seed_values[@]}" -lt 3 ]]; then
  echo "final inference requires at least three independently trained model seeds" >&2
  exit 2
fi

for path in "${DATASET_DIR}/evaluation.jsonl" "${DATASET_DIR}/validation.jsonl" "${VOCAB}"; do
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done

checkpoint_epoch() {
  "${PYTHON_BIN}" -c \
    'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("epoch", -1)))' "$1"
}

probe_artifact() {
  local run_dir="$1"
  if [[ -f "${run_dir}/evaluation/probes/linear_probes.pt" ]]; then
    printf '%s\n' "${run_dir}/evaluation/probes/linear_probes.pt"
  elif [[ -f "${run_dir}/evaluation/hand-evaluation/linear-probes/linear_probes.pt" ]]; then
    printf '%s\n' "${run_dir}/evaluation/hand-evaluation/linear-probes/linear_probes.pt"
  else
    return 1
  fi
}

run_condition() {
  local condition="$1"
  local protocol="$2"
  local category="$3"
  local seed="$4"
  local run_dir="${RESULTS_DIR}/llama-reference/implicit-initial/${condition}/seed-${seed}"
  local checkpoint="${run_dir}/${CHECKPOINT_NAME}"
  local probes
  local evaluation_matching_seed=$((10#${seed} + 2))
  local output_dir="${run_dir}/evaluation/action-condition/${category}"
  local actual_epoch
  [[ -f "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
  actual_epoch="$(checkpoint_epoch "${checkpoint}")"
  [[ "${actual_epoch}" == "${TARGET_EPOCHS}" ]] || {
    echo "refusing unequal-budget action evaluation: ${checkpoint} is epoch ${actual_epoch}, expected ${TARGET_EPOCHS}" >&2
    exit 2
  }
  if ! probes="$(probe_artifact "${run_dir}")"; then
    echo "linear probe artifact is missing; fitting standard state probes for ${condition}, seed=${seed}" >&2
    "${SCRIPT_DIR}/scripts/run_factorized_evaluation.sh" \
      "${checkpoint}" "${DATASET_DIR}" "${VOCAB}" "${run_dir}/evaluation" probes
    probes="$(probe_artifact "${run_dir}")" || {
      echo "state-probe evaluation did not produce a linear probe artifact for ${run_dir}" >&2
      exit 2
    }
  fi
  mkdir -p "${output_dir}/figures" "${output_dir}/logs"
  echo "action-condition: condition=${condition} protocol=${protocol} seed=${seed} checkpoint=${CHECKPOINT_NAME} epoch=${actual_epoch}" >&2
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_action_condition.py" \
    --checkpoint "${checkpoint}" \
    --linear-probes "${probes}" \
    --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
    --vocab "${VOCAB}" \
    --output "${output_dir}/action_condition_metrics.json" \
    --evaluation-input-mode "${protocol}" \
    --sources "${ACTION_CONDITION_SOURCES:-available}" \
    --max-pairs "${MAX_ACTION_CONDITION_PAIRS:-5000}" \
    --max-calibration-examples "${MAX_ACTION_CALIBRATION_EXAMPLES:-5000}" \
    --batch-size "${ACTION_CONDITION_BATCH_SIZE:-64}" \
    --seed "${seed}" \
    --amp "${EVAL_AMP:-auto}" \
    --device "${DEVICE:-auto}" 2>&1 | tee "${output_dir}/logs/evaluation.log"

  # 主解析．評価対局をゲーム単位で再分割し，pre／DROP／通常手の各位置を
  # 均等に学習したprobeと位置間交差評価で，既存h_pre probeの分布ずれを監査する。
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_action_condition_robustness.py" \
    --checkpoint "${checkpoint}" \
    --linear-probes "${probes}" \
    --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
    --vocab "${VOCAB}" \
    --output "${output_dir}/action_condition_robustness.json" \
    --probe-output "${output_dir}/branch_hand_probes.pt" \
    --evaluation-input-mode "${protocol}" \
    --sources "${ACTION_ROBUSTNESS_SOURCES:-middle,late,final}" \
    --max-probe-pairs "${MAX_ACTION_PROBE_PAIRS:-1500}" \
    --max-calibration-pairs "${MAX_ACTION_ROBUSTNESS_CALIBRATION_PAIRS:-500}" \
    --max-evaluation-pairs "${MAX_ACTION_ROBUSTNESS_EVALUATION_PAIRS:-2000}" \
    --normal-branches "${ACTION_NORMAL_BRANCHES:-3}" \
    --probe-epochs "${ACTION_PROBE_EPOCHS:-20}" \
    --probe-patience "${ACTION_PROBE_PATIENCE:-3}" \
    --batch-size "${ACTION_CONDITION_BATCH_SIZE:-64}" \
    --seed "${seed}" \
    --amp "${EVAL_AMP:-auto}" \
    --device "${DEVICE:-auto}" 2>&1 | tee "${output_dir}/logs/robustness.log"

  # Attention観測と直接接続遮断はoracle APには適用せず，主3条件だけで行う。
  # SKIP_ACTION_CAUSAL_AUDIT=1で高コスト部分だけ後回しにできる。
  if [[ "${category}" == primary && "${SKIP_ACTION_CAUSAL_AUDIT:-0}" != 1 ]]; then
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_drop_attention.py" \
      --checkpoint "${checkpoint}" \
      --evaluation-jsonl "${DATASET_DIR}/evaluation.jsonl" \
      --vocab "${VOCAB}" \
      --output "${output_dir}/action_condition_attention_ablation.json" \
      --max-pairs "${MAX_ACTION_ATTENTION_PAIRS:-1000}" \
      --max-ablation-pairs "${MAX_ACTION_ABLATION_PAIRS:-250}" \
      --ablation-layers "${ACTION_ABLATION_LAYERS:-middle,late,all}" \
      --game-partition evaluation \
      --partition-seed "${seed}" \
      --seed "${evaluation_matching_seed}" \
      --amp "${ATTENTION_AMP:-off}" \
      --device "${DEVICE:-auto}" 2>&1 | tee "${output_dir}/logs/attention_ablation.log"
  fi

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/visualize_factorized_action_condition.py" \
    --metrics "${output_dir}/action_condition_metrics.json" \
    --robustness "${output_dir}/action_condition_robustness.json" \
    --output-dir "${output_dir}/figures" 2>&1 | tee "${output_dir}/logs/visualization.log"
}

for seed in "${seed_values[@]}"; do
  # 同じ注釈なし評価protocolで比較する主条件。
  run_condition "vanilla-p0.0" "no-annotation" "primary" "${seed}"
  run_condition "rap-p0.15-proportional-rap-v1" "no-annotation" "primary" "${seed}"
  run_condition "rap-p0.25-proportional-rap-v1" "no-annotation" "primary" "${seed}"

  # APはoracle native入力と，注釈除去による分布外感度分析を混ぜずに保存する。
  run_condition "ap-p1.0-proportional-annotation-v1" "native" "oracle-native" "${seed}"
  run_condition "ap-p1.0-proportional-annotation-v1" "no-annotation" "sensitivity-no-annotation" "${seed}"
done

SUMMARY_DIR="${RESULTS_DIR}/action-condition-reference-summary"
mkdir -p "${SUMMARY_DIR}"
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/collect_factorized_action_condition_matrix.py" \
  --results-dir "${RESULTS_DIR}" \
  --seeds "${SEEDS}" \
  --output "${SUMMARY_DIR}/action_condition_matrix.json"

echo "reference action-condition matrix complete: ${RESULTS_DIR}" >&2
