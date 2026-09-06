#!/usr/bin/env bash
# factorized-v3の再収集を段階ごとに駆動する最上位スクリプト。
#
# 学習に約8日，評価に数日かかるため段階を分ける。各段階は単独で再実行でき，
# 完了済みの作業を繰り返さない。同じコマンドを何度打っても同じ状態へ収束する。
#
#   ./scripts/run_full_study.sh --stage data
#   ./scripts/run_full_study.sh --stage train
#   ./scripts/run_full_study.sh --stage eval
#   ./scripts/run_full_study.sh --stage collect
#
# MEMORY_MAXは各段階が呼ぶ下位スクリプトが要求する。ここでは検査だけ行う。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="${PYTHON_FALLBACK:-python3}"

STAGES=""
# 既定は現行の実ディレクトリである。新しいrootを切る場合だけ明示的に上書きする。
DATA_DIR="${DATA_DIR:-factorized_v3_eos_data}"
RESULTS_DIR="${RESULTS_DIR:-factorized_v3_eos_results_reference_fixed50}"
SOURCE_RESULTS_DIR="${SOURCE_RESULTS_DIR:-factorized_v3_eos_results_reference}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
CONDITIONS="${CONDITIONS:-vanilla-p0.0,rap-p0.15-proportional-rap-v1,rap-p0.25-proportional-rap-v1,ap-p1.0-proportional-annotation-v1}"
SEEDS="${SEEDS:-20260802,20260803,20260804}"
RAP_RATES="${RAP_RATES:-0.0,0.15,0.25,1.0}"
MODEL_TYPE="${MODEL_TYPE:-llama}"
MODEL_SIZE="${MODEL_SIZE:-reference}"
TARGET_EPOCHS="${TARGET_EPOCHS:-50}"
DATA_INPUT="${DATA_INPUT:-}"

usage() {
  cat >&2 <<'EOF'
usage: run_full_study.sh --stage {data|train|eval|collect|summarize|report}[,...]
                         [--dataset-dir DIR] [--results-dir DIR] [--source-results-dir DIR]
                         [--output-root DIR] [--conditions LIST] [--seeds LIST]
                         [--data-input METADATA_CSV] [--dry-run]

既定のディレクトリ（現行の実構成）
  --dataset-dir         factorized_v3_eos_data
  --results-dir         factorized_v3_eos_results_reference_fixed50   固定50エポック
  --source-results-dir  factorized_v3_eos_results_reference           early stopping
  --output-root         --results-dir と同じ（summary，paper，監査reportの出力先）

stages
  data       datasetのmanifestとvocab hashを凍結する．--data-inputを与えた場合だけ再構築する
  train      early stopping学習ののち固定エポックへ揃える．主3条件は全seed，APは先頭seedのみ
  eval       全checkpointへcheckpoint単位の11評価を適用し，study単位のaction-conditionを1回実行する
  collect    分析archiveへまとめる
  summarize  条件×シードで集約する
  report     論文の表と図を生成する
EOF
  exit 2
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)       STAGES="${2:?--stage requires a value}"; shift 2 ;;
    --dataset-dir) DATA_DIR="${2:?--dataset-dir requires a value}"; shift 2 ;;
    --results-dir) RESULTS_DIR="${2:?--results-dir requires a value}"; shift 2 ;;
    --source-results-dir) SOURCE_RESULTS_DIR="${2:?--source-results-dir requires a value}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?--output-root requires a value}"; shift 2 ;;
    --conditions)  CONDITIONS="${2:?--conditions requires a value}"; shift 2 ;;
    --seeds)       SEEDS="${2:?--seeds requires a value}"; shift 2 ;;
    --data-input)  DATA_INPUT="${2:?--data-input requires a value}"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[[ -n "${STAGES}" ]] || usage

OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_DIR}}"
STUDY_ROOT="${RESULTS_DIR}"
VOCAB="${VOCAB:-${DATA_DIR}/vocab.json}"
IFS=',' read -r -a seed_values <<< "${SEEDS}"
IFS=',' read -r -a condition_values <<< "${CONDITIONS}"

for stage in ${STAGES//,/ }; do
  case "${stage}" in
    data|train|eval|collect|summarize|report) ;;
    *) echo "unknown stage: ${stage}" >&2; usage ;;
  esac
done

announce() { echo; echo "########## ${1} ##########" >&2; }
run() {
  echo "+ $*" >&2
  [[ "${DRY_RUN}" == 1 ]] && return 0
  "$@"
}
require_memory_max() {
  [[ "$(uname -s)" != Linux || -n "${MEMORY_MAX:-}" || "${ALLOW_UNBOUNDED_MEMORY:-0}" == 1 ]] || {
    echo "MEMORY_MAX is required on Linux; use e.g. MEMORY_MAX=100G MEMORY_HIGH=90G $0 ..." >&2
    exit 2
  }
}

stage_data() {
  announce "data"
  if [[ -n "${DATA_INPUT}" ]]; then
    if [[ -f "${DATA_DIR}/dataset_manifest.json" && "${FORCE_DATA:-0}" != 1 ]]; then
      # 再構築するとmanifest hashが変わり，既存checkpointが比較不能になる。
      echo "refusing to rebuild an existing dataset: ${DATA_DIR}" >&2
      echo "existing checkpoints become incomparable once the manifest hash changes;" >&2
      echo "set FORCE_DATA=1 only if you intend to retrain every condition" >&2
      exit 2
    fi
    run "${SCRIPT_DIR}/scripts/setup_factorized_v3_data.sh" "${DATA_INPUT}" "${DATA_DIR}"
  else
    [[ -f "${DATA_DIR}/dataset_manifest.json" ]] || {
      echo "dataset is missing: ${DATA_DIR}; pass --data-input to build it" >&2
      exit 2
    }
    echo "using the existing dataset: ${DATA_DIR}" >&2
  fi
  # 凍結の記録．学習開始後にdatasetが変わっていないことを後から照合できる。
  run "${PYTHON_BIN}" -u "${SCRIPT_DIR}/freeze_dataset.py" "${DATA_DIR}"
}

stage_train() {
  announce "train"
  require_memory_max
  [[ -f "${DATA_DIR}/dataset_manifest.json" ]] || {
    echo "dataset is missing: ${DATA_DIR}; run --stage data first" >&2
    exit 2
  }
  # 1段目：early stoppingつきに学習する。best.ptがある条件は再学習しない。
  run env RAP_RATES="${RAP_RATES}" SEEDS="${SEEDS}" \
    MODEL_TYPE="${MODEL_TYPE}" MODEL_SIZE="${MODEL_SIZE}" \
    "${SCRIPT_DIR}/scripts/run_factorized_rap_ablation.sh" \
    "${DATA_DIR}" "${SOURCE_RESULTS_DIR}"
  # 2段目：不足エポックだけ継続し，条件間で学習量を揃える。
  run env RAP_RATES="${RAP_RATES}" SEEDS="${SEEDS}" \
    MODEL_TYPE="${MODEL_TYPE}" MODEL_SIZE="${MODEL_SIZE}" TARGET_EPOCHS="${TARGET_EPOCHS}" \
    "${SCRIPT_DIR}/scripts/run_factorized_fixed_epoch_training.sh" \
    "${DATA_DIR}" "${SOURCE_RESULTS_DIR}" "${RESULTS_DIR}"
}

stage_eval() {
  announce "eval"
  require_memory_max
  local failed=() missing=() condition seed run_dir checkpoint
  for condition in "${condition_values[@]}"; do
    while IFS= read -r seed; do
      run_dir="${RESULTS_DIR}/llama-reference/implicit-initial/${condition}/seed-${seed}"
      checkpoint="${run_dir}/last.pt"
      if [[ ! -f "${checkpoint}" ]]; then
        missing+=("${condition}/seed-${seed}")
        echo "skip ${condition}/seed-${seed}: ${checkpoint} is missing" >&2
        continue
      fi
      echo >&2
      echo "---------- ${condition}/seed-${seed} ----------" >&2
      run "${SCRIPT_DIR}/scripts/run_factorized_full_evaluation.sh" \
        "${checkpoint}" "${DATA_DIR}" "${VOCAB}" || failed+=("${condition}/seed-${seed}")
    done < <(printf '%s\n' "${seed_values[@]}")
  done

  # action-conditionは4条件を横断するstudy単位の評価なので，全checkpointの後に1回だけ実行する。
  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "skipping study-level action-condition: ${#missing[@]} checkpoint(s) are missing" >&2
  else
    announce "eval: action-condition (study-level)"
    run env SEEDS="${SEEDS}" TARGET_EPOCHS="${TARGET_EPOCHS}" \
      "${SCRIPT_DIR}/scripts/run_reference_action_condition_experiment.sh" \
      "${DATA_DIR}" "${RESULTS_DIR}" || failed+=("action-condition")
  fi

  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "missing checkpoints:" >&2; printf '  - %s\n' "${missing[@]}" >&2
  fi
  if [[ "${#failed[@]}" -gt 0 ]]; then
    echo "failed evaluations:" >&2; printf '  - %s\n' "${failed[@]}" >&2
    return 1
  fi
  [[ "${#missing[@]}" -eq 0 ]] || return 1
  return 0
}

stage_collect() {
  announce "collect"
  local archive="${OUTPUT_ROOT}/analysis_bundle.tar.gz"
  # archiveは再生成できる派生物なので既定で上書きする。段階を冪等に保つため。
  local force=()
  [[ "${COLLECT_FORCE:-1}" == 1 ]] && force=(--force)
  run "${PYTHON_BIN}" -u "${SCRIPT_DIR}/collect_factorized_analysis.py" \
    "${archive}" "${RESULTS_DIR}" \
    --dataset-dir "${DATA_DIR}" \
    --include-probe-artifacts \
    "${force[@]}" \
    ${COLLECT_EXTRA_ARGS:-}
  echo "archive: ${archive}" >&2
}


stage_summarize() {
  announce "summarize"
  local script="${SCRIPT_DIR}/summarize_factorized_study.py"
  [[ -f "${script}" ]] || {
    echo "not implemented yet: ${script}" >&2
    echo "see STUDY_PIPELINE_DESIGN.md section 5.4" >&2
    exit 3
  }
  # 集約はresults rootを直接読む。archiveは配布用であり，展開は不要である。
  run "${PYTHON_BIN}" -u "${script}" \
    --bundle "${SUMMARIZE_INPUT:-${RESULTS_DIR}}" \
    --conditions "${CONDITIONS}" \
    --seeds "${SEEDS}" \
    --output "${OUTPUT_ROOT}/summary"
}

stage_report() {
  announce "report"
  local script="${SCRIPT_DIR}/render_paper_tables.py"
  [[ -f "${script}" ]] || {
    echo "not implemented yet: ${script}" >&2
    echo "see STUDY_PIPELINE_DESIGN.md section 5.5" >&2
    exit 3
  }
  run "${PYTHON_BIN}" -u "${script}" \
    --summary "${OUTPUT_ROOT}/summary/study_summary.json" \
    --output "${OUTPUT_ROOT}/paper"
}

mkdir -p "${OUTPUT_ROOT}"
echo "dataset:    ${DATA_DIR}" >&2
echo "results:    ${RESULTS_DIR}" >&2
echo "source:     ${SOURCE_RESULTS_DIR}" >&2
echo "output:     ${OUTPUT_ROOT}" >&2
echo "conditions: ${CONDITIONS}" >&2
echo "seeds:      ${SEEDS}" >&2
echo "stages:     ${STAGES}" >&2
[[ "${DRY_RUN}" == 1 ]] && echo "dry run: commands are printed but not executed" >&2

declare -a FAILED_STAGES=()
# 段階の失敗をここで捕まえる。&&で繋ぐとset -eの対象から外れ，
# 失敗したまま最後まで進んでcompleteと表示されてしまう。
attempt() {
  local name="$1"; shift
  if "$@"; then
    return 0
  fi
  FAILED_STAGES+=("${name}")
  echo "stage failed: ${name}" >&2
  return 1
}

for stage in ${STAGES//,/ }; do
  case "${stage}" in
    data)      attempt data stage_data ;;
    train)     attempt train stage_train ;;
    eval)      attempt eval stage_eval ;;
    collect)   attempt collect stage_collect ;;
    summarize) attempt summarize stage_summarize ;;
    report)    attempt report stage_report ;;
  esac
done

echo >&2
if [[ "${#FAILED_STAGES[@]}" -gt 0 ]]; then
  echo "run_full_study FAILED: ${FAILED_STAGES[*]}" >&2
  exit 1
fi
echo "run_full_study complete: ${STAGES}" >&2
