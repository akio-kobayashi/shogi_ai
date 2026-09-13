#!/usr/bin/env bash
# 実験結果のtar.gzから，学生が分析に使うCSVを作る。
#
#   ./scripts/prepare_student_data.sh results.tar.gz student_data
#
# 出力（student_data/）
#   runs.csv     1行が1モデル（条件×シード）
#   summary.csv  1行が1条件。3シードの平均と標準偏差
#   columns.csv  列の意味
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="${PYTHON_FALLBACK:-python3}"

ARCHIVE="${1:?usage: $0 RESULTS.tar.gz [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-student_data}"
WORK_DIR="${WORK_DIR:-${OUTPUT_DIR}/.extracted}"

[[ -f "${ARCHIVE}" ]] || { echo "archive does not exist: ${ARCHIVE}" >&2; exit 2; }

echo "展開中: ${ARCHIVE}" >&2
mkdir -p "${WORK_DIR}"
tar xzf "${ARCHIVE}" -C "${WORK_DIR}"

# tar.gzの中身はanalysis_bundle/以下か，results rootそのもののことがある。
BUNDLE="$(find "${WORK_DIR}" -maxdepth 2 -type d -name analysis_bundle | head -1)"
[[ -n "${BUNDLE}" ]] || BUNDLE="$(find "${WORK_DIR}" -maxdepth 2 -type d -name 'llama-reference' -exec dirname {} \; | head -1)"
[[ -n "${BUNDLE}" ]] || { echo "results root not found in ${WORK_DIR}" >&2; exit 2; }
echo "results root: ${BUNDLE}" >&2

echo "集約中" >&2
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/summarize_factorized_study.py" \
  --bundle "${BUNDLE}" --output "${OUTPUT_DIR}/summary"

echo "学生向けCSVを作成中" >&2
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/make_student_csv.py" \
  --summary "${OUTPUT_DIR}/summary/study_summary.json" --output "${OUTPUT_DIR}"

echo >&2
echo "完了: ${OUTPUT_DIR}" >&2
ls -1 "${OUTPUT_DIR}"/*.csv >&2
echo >&2
echo "詳細な数値が必要な場合は ${OUTPUT_DIR}/summary/by_run.csv（全指標）を見てください" >&2
