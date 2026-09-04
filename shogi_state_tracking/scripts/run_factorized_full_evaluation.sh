#!/usr/bin/env bash
# 1つのcheckpointに対する評価を依存順に実行し，stage_log.jsonへ結果を記録する。
# action-conditionは4条件を横断するstudy単位の評価なので本スクリプトには含めない。
# run_reference_action_condition_experiment.shを全checkpointの評価後に1回実行する。
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
OUTPUT_DIR="${4:-${RUN_DIR}/evaluation}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
LISHOGI_EVAL_DATASET="${LISHOGI_EVAL_DATASET:-data/lishogi-non-bot-factorized-eval}"
PROBE_TRAIN_DATASET="${PROBE_TRAIN_DATASET:-${DATASET_DIR}}"
STAGE_LOG="${OUTPUT_DIR}/stage_log.json"

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint does not exist: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATASET_DIR}/evaluation.jsonl" ]] || { echo "missing ${DATASET_DIR}/evaluation.jsonl" >&2; exit 2; }
[[ -f "${VOCAB}" ]] || { echo "vocabulary does not exist: ${VOCAB}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"

# 条件はcheckpointのパスから決める。APはoracle条件なのでdrop-relevanceを実行しない。
CONDITION="$(basename "$(dirname "${RUN_DIR}")")"
IS_ORACLE=0
[[ "${CONDITION}" == ap-* ]] && IS_ORACLE=1

# 段階名 -> 完了を判定する成果物（依存順）。
STAGE_NAMES=(
  moves token probes terminal chess
  lishogi-moves lishogi-probes
  hand-dynamics policy-relevance drop-relevance
)
stage_artifact() {
  case "$1" in
    moves)            printf '%s\n' "move_metrics.json" "distribution_baselines.json" ;;
    token)            printf '%s\n' "token_probe_metrics.json" ;;
    probes)           printf '%s\n' "probes/probe_metrics.json" "probes/linear_probes.pt" ;;
    terminal)         printf '%s\n' "terminal-probe/action_probe_metrics.json" ;;
    chess)            printf '%s\n' "chess-protocol/chess_protocol_metrics.json" ;;
    lishogi-moves)    printf '%s\n' "lishogi-non-bot/moves/move_metrics.json" ;;
    lishogi-probes)   printf '%s\n' "lishogi-non-bot/linear-probes/probe_metrics.json" ;;
    hand-dynamics)    printf '%s\n' "hand-evaluation/hand_dynamics_metrics.json" ;;
    policy-relevance) printf '%s\n' "policy-relevance/policy_relevance_metrics.json" ;;
    drop-relevance)   printf '%s\n' "drop-relevance/confidence_trajectory.json" "drop-relevance/attention_metrics.json" ;;
  esac
}

selected() {
  local name="$1"
  if [[ -n "${ONLY_STAGES:-}" ]]; then
    [[ ",${ONLY_STAGES}," == *",${name},"* ]] || return 1
  fi
  [[ ",${SKIP_STAGES:-}," != *",${name},"* ]] || return 1
  return 0
}

complete() {
  local name="$1" relative
  while IFS= read -r relative; do
    [[ -e "${OUTPUT_DIR}/${relative}" ]] || return 1
  done < <(stage_artifact "${name}")
  return 0
}

declare -a LOG_ENTRIES=()
declare -a FAILED=()
PROBES_OK=1

record() {
  LOG_ENTRIES+=("$(printf '{"stage":"%s","status":"%s","exit_code":%s,"started":"%s","finished":"%s","seconds":%s}' \
    "$1" "$2" "$3" "$4" "$5" "$6")")
}

run_stage() {
  local name="$1"; shift
  if ! selected "${name}"; then
    record "${name}" skipped 0 "" "" 0
    echo "skip ${name} (not selected)" >&2
    return 0
  fi
  if [[ "${FORCE_EVAL:-0}" != 1 ]] && complete "${name}"; then
    record "${name}" cached 0 "" "" 0
    echo "skip ${name} (artifacts present; set FORCE_EVAL=1 to rerun)" >&2
    return 0
  fi
  # probesが失敗したら，linear_probes.ptを必要とする段階は実行しない。
  case "${name}" in
    hand-dynamics|policy-relevance|drop-relevance)
      if [[ "${PROBES_OK}" != 1 ]]; then
        record "${name}" blocked 0 "" "" 0
        FAILED+=("${name} (blocked: probes stage did not produce linear_probes.pt)")
        echo "skip ${name} (blocked by failed probes stage)" >&2
        return 0
      fi ;;
  esac

  local started finished elapsed status code
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local begin=${SECONDS}
  echo "=== ${name} ===" >&2
  code=0
  "$@" || code=$?
  elapsed=$((SECONDS - begin))
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "${code}" -eq 0 ]] && complete "${name}"; then
    status=ok
  else
    status=failed
    [[ "${code}" -eq 0 ]] && code=1
    FAILED+=("${name} (exit ${code})")
    [[ "${name}" == probes ]] && PROBES_OK=0
  fi
  record "${name}" "${status}" "${code}" "${started}" "${finished}" "${elapsed}"
  echo "=== ${name}: ${status} in ${elapsed}s ===" >&2
}

evaluation_stage() {
  ALLOW_UNBOUNDED_MEMORY=1 "${SCRIPT_DIR}/scripts/run_factorized_evaluation.sh" \
    "${CHECKPOINT}" "${DATASET_DIR}" "${VOCAB}" "${OUTPUT_DIR}" "$1"
}

run_stage moves    evaluation_stage moves
run_stage token    evaluation_stage token
run_stage probes   evaluation_stage probes
run_stage terminal evaluation_stage terminal
run_stage chess    evaluation_stage chess

run_stage lishogi-moves \
  env ALLOW_UNBOUNDED_MEMORY=1 "${SCRIPT_DIR}/scripts/run_reference_lishogi_move_evaluation.sh" \
  "${CHECKPOINT}" "${LISHOGI_EVAL_DATASET}" "${OUTPUT_DIR}/lishogi-non-bot/moves"

run_stage lishogi-probes \
  env ALLOW_UNBOUNDED_MEMORY=1 "${SCRIPT_DIR}/scripts/run_reference_lishogi_linear_probe_evaluation.sh" \
  "${CHECKPOINT}" "${PROBE_TRAIN_DATASET}" "${LISHOGI_EVAL_DATASET}" \
  "${OUTPUT_DIR}/lishogi-non-bot/linear-probes"

run_stage hand-dynamics \
  env ALLOW_UNBOUNDED_MEMORY=1 REUSE_LINEAR_PROBES="${REUSE_LINEAR_PROBES:-0}" \
  "${SCRIPT_DIR}/scripts/run_factorized_hand_evaluation.sh" \
  "${CHECKPOINT}" "${DATASET_DIR}" "${VOCAB}" "${OUTPUT_DIR}/hand-evaluation"

run_stage policy-relevance \
  env ALLOW_UNBOUNDED_MEMORY=1 STATE_PROBE_DIR="${OUTPUT_DIR}/probes" \
  "${SCRIPT_DIR}/scripts/run_factorized_policy_relevance_evaluation.sh" \
  "${CHECKPOINT}" "${DATASET_DIR}" "${VOCAB}" "${OUTPUT_DIR}/policy-relevance"

if [[ "${IS_ORACLE}" == 1 ]]; then
  record drop-relevance excluded 0 "" "" 0
  echo "skip drop-relevance (oracle AP condition carries piece annotations in history)" >&2
else
  run_stage drop-relevance \
    env ALLOW_UNBOUNDED_MEMORY=1 STATE_PROBE_DIR="${OUTPUT_DIR}/probes" \
    "${SCRIPT_DIR}/scripts/run_reference_drop_relevance_experiment.sh" \
    "${CHECKPOINT}" "${DATASET_DIR}" "${VOCAB}" "${OUTPUT_DIR}/drop-relevance"
fi

{
  printf '{\n  "format_version": 1,\n'
  printf '  "checkpoint": "%s",\n' "${CHECKPOINT}"
  printf '  "condition": "%s",\n' "${CONDITION}"
  printf '  "dataset_dir": "%s",\n' "${DATASET_DIR}"
  printf '  "git_commit": "%s",\n' "$(git -C "${SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf '  "completed": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "failures": %d,\n' "${#FAILED[@]}"
  printf '  "stages": [\n'
  for index in "${!LOG_ENTRIES[@]}"; do
    printf '    %s' "${LOG_ENTRIES[index]}"
    [[ "${index}" -lt $((${#LOG_ENTRIES[@]} - 1)) ]] && printf ','
    printf '\n'
  done
  printf '  ]\n}\n'
} > "${STAGE_LOG}"

echo "stage log: ${STAGE_LOG}" >&2
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "full evaluation finished with ${#FAILED[@]} failed stage(s):" >&2
  printf '  - %s\n' "${FAILED[@]}" >&2
  exit 1
fi
echo "full evaluation complete: ${OUTPUT_DIR}" >&2
