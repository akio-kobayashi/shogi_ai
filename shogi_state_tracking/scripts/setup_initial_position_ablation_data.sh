#!/usr/bin/env bash
# metadata.csvからCSAを変換し，A–D共通の平手初期局面datasetまで構築する。
# 既存のnew-prompt JSONLディレクトリを第1引数へ渡す旧形式も維持する。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
INPUT="${1:?usage: $0 METADATA_CSV_OR_NEW_PROMPT_DATASET OUTPUT_ABLATION_DATASET [create_dataset build options]}"
OUTPUT_DIR="${2:?output ablation dataset is required}"
shift 2

if [[ -f "${INPUT}" && "${INPUT}" == *.csv ]]; then
  BUILD_ROOT="${DATA_BUILD_ROOT:-${OUTPUT_DIR}.build}"
  SOURCE_DIR="${BUILD_ROOT}/source-jsonl"
  NEW_PROMPT_DIR="${BUILD_ROOT}/new-prompt"
  mkdir -p "${BUILD_ROOT}" "${OUTPUT_DIR}"

  echo "[1/3] metadata.csv + CSA -> source JSONL" >&2
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/create_dataset.py" build \
    --metadata-csv "${INPUT}" \
    --output-dir "${SOURCE_DIR}" \
    "$@"

  echo "[2/3] source JSONL -> new-prompt dataset" >&2
  "${SCRIPT_DIR}/scripts/build_new_prompt_dataset.sh" \
    "${SOURCE_DIR}" "${NEW_PROMPT_DIR}"

  INPUT_DIR="${NEW_PROMPT_DIR}"
else
  if [[ $# -gt 0 ]]; then
    echo "additional create_dataset options are accepted only when the first argument is metadata.csv" >&2
    exit 2
  fi
  INPUT_DIR="${INPUT}"
  mkdir -p "${OUTPUT_DIR}"
fi

echo "[3/3] new-prompt dataset -> initial-position A-D dataset" >&2
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/build_initial_position_ablation_dataset.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}.setup.log"
