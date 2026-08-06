#!/usr/bin/env bash
# Vanilla decoderの疎通確認。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/scripts/lib_new_prompt_launcher.sh"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ -z "${BATCH_SIZE:-}" && -n "${batch_size:-}" ]]; then
  BATCH_SIZE="${batch_size}"
fi
BATCH_SIZE="${BATCH_SIZE:-1}"
# 新prompt datasetは全recordをメモリ上へmaterializeするため，smokeではworker複製を避ける。
NUM_WORKERS="${NUM_WORKERS:-0}"
DROPOUT="${DROPOUT:-0.0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1280}"
MAX_MOVES="${MAX_MOVES:-512}"
MAX_HINTS="${MAX_HINTS:-320}"

smoke_failure_report() {
  local status="$1" output_dir="$2" log_path="$3"
  printf 'smoke training process failed: exit_status=%s output_dir=%s\n' "${status}" "${output_dir}" >&2
  for memory_file in \
    /sys/fs/cgroup/memory.events \
    /sys/fs/cgroup/memory.current \
    /sys/fs/cgroup/memory.max \
    /sys/fs/cgroup/memory/memory.oom_control \
    /sys/fs/cgroup/memory/memory.limit_in_bytes \
    /sys/fs/cgroup/memory/memory.usage_in_bytes \
    /sys/fs/cgroup/memory/memory.failcnt; do
    if [[ -r "${memory_file}" ]]; then
      printf 'cgroup %s:\n' "${memory_file}" >&2
      cat "${memory_file}" >&2 || true
    fi
  done
  if command -v free >/dev/null 2>&1; then
    free -h >&2 || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader >&2 || printf 'nvidia-smi failed\n' >&2
  else
    printf 'nvidia-smi is unavailable\n' >&2
  fi
  if [[ -f "${log_path}" ]]; then
    printf 'last 40 lines of %s:\n' "${log_path}" >&2
    tail -n 40 "${log_path}" >&2 || true
  fi
}
[[ $# -ge 2 ]] || { echo "Usage: $0 DATASET_DIR RESULTS_DIR [extra train options]" >&2; exit 2; }
DATASET_DIR="$1"; RESULTS_DIR="$2"; shift 2
new_prompt_extract_launcher_args "$@"
SEED="${SEED:-20260802}"; EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-32}"
new_prompt_resolve_single_model_size small
MODEL_SIZE="${NEW_PROMPT_MODEL_SIZE}"
printf 'selected vanilla smoke model size: %s\n' "${MODEL_SIZE}" >&2
printf 'batch size: %s\n' "${BATCH_SIZE}" >&2
printf 'smoke max_steps: %s, validation: full split, num_workers: %s\n' "${SMOKE_MAX_STEPS}" "${NUM_WORKERS}" >&2
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_new_prompt_dataset.py" --dataset-dir "${DATASET_DIR}" --output "${RESULTS_DIR}/artifact_verification.json"
for condition in vanilla partial_action random_control; do
  probability=0.0; [[ "${condition}" != "vanilla" ]] && probability="${SMOKE_ANNOTATION_RATE:-0.3}"
  output="${RESULTS_DIR}/vanilla-${MODEL_SIZE}/${condition}/p${probability}/seed-${SEED}"
  mkdir -p "${output}"
  log_path="${output}/train.log"
  train_args=()
  if ((${#NEW_PROMPT_EXTRA_ARGS[@]})); then train_args=("${NEW_PROMPT_EXTRA_ARGS[@]}"); fi
  [[ -f "${output}/last.pt" ]] && train_args+=(--resume)
  printf 'run model_type=vanilla model_size=%s condition=%s seed=%s output_dir=%s\n' "${MODEL_SIZE}" "${condition}" "${SEED}" "${output}"
  set +e
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_new_prompt.py" \
    --train-jsonl "${DATASET_DIR}/train.jsonl" --validation-jsonl "${DATASET_DIR}/validation.jsonl" \
    --vocab "${DATASET_DIR}/vocab.json" --dataset-manifest "${DATASET_DIR}/dataset_manifest.json" \
    --output-dir "${output}" --model-type vanilla --model-size "${MODEL_SIZE}" --annotation-mode "${condition}" \
    --annotation-probability "${probability}" --max-seq-len "${MAX_SEQ_LEN}" --max-moves "${MAX_MOVES}" --max-hints "${MAX_HINTS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --dropout "${DROPOUT}" --epochs "${EPOCHS}" --max-steps "${SMOKE_MAX_STEPS}" --seed "${SEED}" "${train_args[@]+"${train_args[@]}"}" 2>&1 | tee -a "${log_path}"
  status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    smoke_failure_report "${status}" "${output}" "${log_path}"
    exit "${status}"
  fi
done
