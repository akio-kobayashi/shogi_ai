#!/usr/bin/env bash
# 未実行4評価の成果物を1条件×1シードで小さく作る試走。
#
# chess-protocol／hand-dynamics／policy-relevance／drop-relevance は
# まだ一度も生成されておらず，JSONの中身を確認できていないため
# summarize_factorized_study.py に抽出器を書けていない。
# 本スクリプトで実物を作り，その構造を見てから抽出器を追加する。
#
# 本番の規模では走らせない。標本数を絞ってあるので数値は暫定値であり，
# 論文へは使用しない。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDITION="${CONDITION:-vanilla-p0.0}"
SEED="${SEED:-20260802}"
DATA_DIR="${DATA_DIR:-factorized_v3_eos_data}"
RESULTS_DIR="${RESULTS_DIR:-factorized_v3_eos_results_reference_fixed50}"
VOCAB="${VOCAB:-${DATA_DIR}/vocab.json}"
RUN_DIR="${RESULTS_DIR}/llama-reference/implicit-initial/${CONDITION}/seed-${SEED}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/last.pt}"

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint does not exist: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_DIR}/evaluation.jsonl" ]] || { echo "missing ${DATA_DIR}/evaluation.jsonl" >&2; exit 2; }

echo "condition:  ${CONDITION}/seed-${SEED}" >&2
echo "checkpoint: ${CHECKPOINT}" >&2
echo "output:     ${RUN_DIR}/evaluation" >&2

# policy-relevance と drop-relevance は probes が作る linear_probes.pt を使う。
# 無ければ probes 段階を先に流す。
if [[ ! -f "${RUN_DIR}/evaluation/probes/linear_probes.pt" ]]; then
  echo "linear_probes.pt is missing; running the probes stage first" >&2
  ONLY_STAGES=probes "${SCRIPT_DIR}/scripts/run_factorized_full_evaluation.sh" \
    "${CHECKPOINT}" "${DATA_DIR}" "${VOCAB}"
fi

# 標本数を絞る。構造を見るのが目的であり，数値の精度は求めない。
export CHESS_PROTOCOL_MAX_INSTANCES="${CHESS_PROTOCOL_MAX_INSTANCES:-50}"
export MAX_HAND_EVAL_GAMES="${MAX_HAND_EVAL_GAMES:-50}"
export MAX_HAND_EVENTS="${MAX_HAND_EVENTS:-200}"
export MAX_DROP_QUERIES="${MAX_DROP_QUERIES:-200}"
export MAX_POLICY_RELEVANCE_EXAMPLES="${MAX_POLICY_RELEVANCE_EXAMPLES:-50}"
export MAX_POLICY_STEERING_EXAMPLES="${MAX_POLICY_STEERING_EXAMPLES:-20}"
export MAX_DROP_RELEVANCE_EXAMPLES="${MAX_DROP_RELEVANCE_EXAMPLES:-50}"
export MAX_DROP_CALIBRATION_EXAMPLES="${MAX_DROP_CALIBRATION_EXAMPLES:-100}"
export MAX_DROP_ATTENTION_PAIRS="${MAX_DROP_ATTENTION_PAIRS:-20}"
export MAX_DROP_ABLATION_PAIRS="${MAX_DROP_ABLATION_PAIRS:-5}"

ONLY_STAGES="${ONLY_STAGES:-chess,hand-dynamics,policy-relevance,drop-relevance}" \
FORCE_EVAL="${FORCE_EVAL:-1}" \
  "${SCRIPT_DIR}/scripts/run_factorized_full_evaluation.sh" \
  "${CHECKPOINT}" "${DATA_DIR}" "${VOCAB}" || true

echo >&2
echo "########## 生成された成果物 ##########" >&2
for relative in \
  "chess-protocol/chess_protocol_metrics.json" \
  "hand-evaluation/hand_dynamics_metrics.json" \
  "policy-relevance/policy_relevance_metrics.json" \
  "drop-relevance/confidence_trajectory.json" \
  "drop-relevance/attention_metrics.json"
do
  path="${RUN_DIR}/evaluation/${relative}"
  if [[ -f "${path}" ]]; then
    printf 'OK      %s\n' "${relative}" >&2
  else
    printf 'MISSING %s\n' "${relative}" >&2
  fi
done

echo >&2
echo "次の手順：上記のJSONの構造を確認し，summarize_factorized_study.py へ抽出器を追加する" >&2
echo "  ls ${RUN_DIR}/evaluation/{chess-protocol,hand-evaluation,policy-relevance,drop-relevance}" >&2
