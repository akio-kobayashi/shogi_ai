#!/usr/bin/env bash
# metadata.csvからfactorized_v3 datasetを構築する．実データのある計算機で実行する．
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
INPUT="${1:?usage: $0 METADATA_CSV_OR_NEW_PROMPT_DATASET OUTPUT_DATASET [create_dataset build options]}"
OUTPUT_DIR="${2:?output dataset is required}"
shift 2

if [[ -f "${INPUT}" && "${INPUT}" == *.csv ]]; then
  BUILD_ROOT="${DATA_BUILD_ROOT:-${OUTPUT_DIR}.build}"
  SOURCE_DIR="${BUILD_ROOT}/source-jsonl"
  NEW_PROMPT_DIR="${BUILD_ROOT}/new-prompt"
  mkdir -p "${BUILD_ROOT}" "${OUTPUT_DIR}"
  # create_dataset.py buildはOUTPUT_DIR/datasets/{train,validation,evaluation}.jsonlへ出力する．
  SOURCE_JSONL_DIR="${SOURCE_DIR}/datasets"
  if [[ "${REUSE_SOURCE_DATASET:-0}" == 1 ]] \
    && [[ -f "${SOURCE_JSONL_DIR}/train.jsonl" ]] \
    && [[ -f "${SOURCE_JSONL_DIR}/validation.jsonl" ]] \
    && [[ -f "${SOURCE_JSONL_DIR}/evaluation.jsonl" ]]; then
    echo "reusing source JSONL: ${SOURCE_JSONL_DIR}" >&2
  else
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/create_dataset.py" build --metadata-csv "${INPUT}" --output-dir "${SOURCE_DIR}" "$@"
  fi
  for split in train validation evaluation; do
    [[ -f "${SOURCE_JSONL_DIR}/${split}.jsonl" ]] || {
      echo "create_dataset output is missing: ${SOURCE_JSONL_DIR}/${split}.jsonl" >&2
      exit 2
    }
  done
  "${SCRIPT_DIR}/scripts/build_new_prompt_dataset.sh" "${SOURCE_JSONL_DIR}" "${NEW_PROMPT_DIR}"
  INPUT_DIR="${NEW_PROMPT_DIR}"
else
  [[ $# -eq 0 ]] || { echo "extra build options require metadata.csv" >&2; exit 2; }
  INPUT_DIR="${INPUT}"
fi

mkdir -p "${OUTPUT_DIR}"
"${SCRIPT_DIR}/scripts/build_factorized_prompt_dataset.sh" "${INPUT_DIR}" "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}.setup.log"
