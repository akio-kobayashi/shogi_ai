#!/usr/bin/env bash
# 学習済みの12 runすべてについて，評価集合の全指手を教師強制で採点する。
#
#   ./scripts/run_full_history_move_evaluation.sh
#   ./scripts/run_full_history_move_evaluation.sh --results-dir DIR --data-dir DIR
#
# 実行環境での動作確認は --smoke を使う。1条件1シードだけを20対局で回し，
# 採点位置の自己検査と，既存の抽出評価との突き合わせを行う。
#
#   ./scripts/run_full_history_move_evaluation.sh --smoke
#
# 既存のeval段階とは独立に，checkpointだけを使って後から実行できる。成果物は
# 各runの evaluation/full_history_move_metrics.json である。
#
# 既存の抽出評価（8手・32手の2点）を置き換えるものではない。生成が必要な指標は
# 含まれないため，両方を並べて読む。metricsのcross_check_ply_8 / _32が
# 抽出評価と同じ対象なので，値の一致を確認できる。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="${PYTHON_FALLBACK:-python3}"

DATA_DIR="${DATA_DIR:-factorized_v3_eos_data}"
RESULTS_DIR="${RESULTS_DIR:-factorized_v3_eos_results_reference_fixed50}"
CONDITIONS="${CONDITIONS:-vanilla-p0.0,rap-p0.15-proportional-rap-v1,rap-p0.25-proportional-rap-v1,ap-p1.0-proportional-annotation-v1}"
SEEDS="${SEEDS:-20260802,20260803,20260804}"
MAX_GAMES="${MAX_GAMES:-0}"
GAMES_PER_BATCH="${GAMES_PER_BATCH:-8}"
FORCE="${FORCE:-0}"
SELF_CHECK_GAMES="${SELF_CHECK_GAMES:-2}"
SELF_CHECK_TOLERANCE="${SELF_CHECK_TOLERANCE:-1e-3}"
# 自己検査は既定でfp32。採点位置の検証を数値精度から切り離すため。
SELF_CHECK_AMP="${SELF_CHECK_AMP:-off}"
# ファイルの有無だけで判定すると，評価器を変えた後も古い成果物が残る。
# 評価器の版番号（EVALUATOR_VERSION）より古い成果物だけを作り直す。
stale() {
  local artifact="$1" code=0
  [[ -f "${artifact}" ]] || return 0
  "${PYTHON_BIN}" "${SCRIPT_DIR}/artifact_versions.py" check "${artifact}" 2>/dev/null || code=$?
  [[ "${code}" -eq 1 || "${code}" -eq 3 ]]
}
COMPARE="${COMPARE:-1}"
SMOKE=0
OUTPUT_NAME="${OUTPUT_NAME:-full_history_move_metrics.json}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir) RESULTS_DIR="${2:?}"; shift 2 ;;
    --data-dir)    DATA_DIR="${2:?}"; shift 2 ;;
    --conditions)  CONDITIONS="${2:?}"; shift 2 ;;
    --seeds)       SEEDS="${2:?}"; shift 2 ;;
    --max-games)   MAX_GAMES="${2:?}"; shift 2 ;;
    --force)       FORCE=1; shift ;;
    --smoke)       SMOKE=1; shift ;;
    --no-compare)  COMPARE=0; shift ;;
    -h|--help)     sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "${SMOKE}" == 1 ]]; then
  # 動作確認用。既存成果物を壊さないよう出力名を分け，毎回作り直す。
  CONDITIONS="${SMOKE_CONDITION:-vanilla-p0.0}"
  SEEDS="${SMOKE_SEED:-20260802}"
  MAX_GAMES="${SMOKE_GAMES:-20}"
  SELF_CHECK_GAMES="${SMOKE_SELF_CHECK_GAMES:-2}"
  OUTPUT_NAME="full_history_move_metrics.smoke.json"
  FORCE=1
  echo "smoke: ${CONDITIONS} / seed ${SEEDS} / ${MAX_GAMES} games -> ${OUTPUT_NAME}" >&2
fi

VOCAB="${VOCAB:-${DATA_DIR}/vocab.json}"
EVALUATION_JSONL="${EVALUATION_JSONL:-${DATA_DIR}/evaluation.jsonl}"
[[ -f "${VOCAB}" ]] || { echo "vocab not found: ${VOCAB}" >&2; exit 2; }
[[ -f "${EVALUATION_JSONL}" ]] || { echo "evaluation jsonl not found: ${EVALUATION_JSONL}" >&2; exit 2; }

IFS=',' read -r -a condition_values <<< "${CONDITIONS}"
IFS=',' read -r -a seed_values <<< "${SEEDS}"

declare -a FAILED=() MISSING=() DONE=()
for condition in "${condition_values[@]}"; do
  for seed in "${seed_values[@]}"; do
    run_dir="${RESULTS_DIR}/llama-reference/implicit-initial/${condition}/seed-${seed}"
    checkpoint="${run_dir}/last.pt"
    output="${run_dir}/evaluation/${OUTPUT_NAME}"
    sampled="${run_dir}/evaluation/move_metrics.json"
    compare_args=()
    if [[ "${COMPARE}" == 1 && -f "${sampled}" ]]; then
      compare_args=(--compare-with "${sampled}")
    fi
    if [[ ! -f "${checkpoint}" ]]; then
      MISSING+=("${condition}/seed-${seed}")
      echo "skip ${condition}/seed-${seed}: ${checkpoint} is missing" >&2
      continue
    fi
    if [[ -f "${output}" && "${FORCE}" != 1 ]]; then
      if stale "${output}"; then
        echo "stale ${condition}/seed-${seed}: produced before the evaluator changed; rerunning" >&2
      else
        DONE+=("${condition}/seed-${seed}")
        echo "cached ${condition}/seed-${seed}: ${output}" >&2
        continue
      fi
    fi
    echo >&2
    echo "---------- ${condition}/seed-${seed} ----------" >&2
    mkdir -p "${run_dir}/evaluation"
    if "${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate_factorized_full_history_moves.py" \
      --checkpoint "${checkpoint}" \
      --evaluation-jsonl "${EVALUATION_JSONL}" \
      --vocab "${VOCAB}" \
      --output "${output}" \
      --max-games "${MAX_GAMES}" \
      --games-per-batch "${GAMES_PER_BATCH}" \
      --self-check-games "${SELF_CHECK_GAMES}" \
      --self-check-tolerance "${SELF_CHECK_TOLERANCE}" \
      --self-check-amp "${SELF_CHECK_AMP}" \
      "${compare_args[@]}" \
      --device "${DEVICE:-auto}" \
      --amp "${EVAL_AMP:-auto}" 2>&1 | tee "${run_dir}/evaluation/${OUTPUT_NAME%.json}.log"; then
      DONE+=("${condition}/seed-${seed}")
    else
      FAILED+=("${condition}/seed-${seed}")
    fi
  done
done

echo >&2
echo "completed: ${#DONE[@]}  missing checkpoint: ${#MISSING[@]}  failed: ${#FAILED[@]}" >&2
[[ "${#MISSING[@]}" -eq 0 ]] || echo "missing: ${MISSING[*]}" >&2
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "failed: ${FAILED[*]}" >&2
  exit 1
fi
