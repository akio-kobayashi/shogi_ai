#!/usr/bin/env bash
set -euo pipefail

# Package portable experiment inputs.  Do not copy .venv or .uv-cache: those
# contain accelerator- and host-specific binaries and are recreated by setup_env.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUNDLE_ROOT="shogi_state_tracking"

usage() {
  cat <<'EOF'
Usage: scripts/package_experiment.sh [options] OUTPUT.tar.gz

Create a portable bundle containing tracked source files and generated data.
The destination machine must recreate .venv with setup_env.sh.

Options:
  --no-data             omit the data/ directory
  --include-results     include results/ (otherwise omitted)
  --include-checkpoints include checkpoints/ (otherwise omitted)
  --force               replace an existing archive
  -h, --help            show this help

Examples:
  scripts/package_experiment.sh /mnt/transfer/shogi-study.tar.gz
  scripts/package_experiment.sh --include-checkpoints /mnt/transfer/run.tar.gz
EOF
}

include_data=1
include_results=0
include_checkpoints=0
force=0
output=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-data) include_data=0; shift ;;
    --include-results) include_results=1; shift ;;
    --include-checkpoints) include_checkpoints=1; shift ;;
    --force) force=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "${output}" ]]; then
        echo "only one output archive may be specified" >&2
        exit 2
      fi
      output="$1"
      shift
      ;;
  esac
done

if [[ -z "${output}" ]]; then
  usage >&2
  exit 2
fi

if [[ "${output}" != /* ]]; then
  output="$(pwd)/${output}"
fi
output_dir="$(dirname "${output}")"
mkdir -p "${output_dir}"
if [[ -e "${output}" && "${force}" != "1" ]]; then
  echo "archive already exists: ${output} (use --force to replace it)" >&2
  exit 2
fi

if ! command -v git >/dev/null 2>&1 || ! git -C "${PROJECT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "package creation requires a Git checkout: ${PROJECT_DIR}" >&2
  exit 2
fi
if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required" >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python is required to write the checksum manifest: ${PYTHON_BIN}" >&2
  exit 2
fi

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/shogi-state-package.XXXXXX")"
trap 'rm -rf "${tmp_root}"' EXIT
stage="${tmp_root}/${BUNDLE_ROOT}"
mkdir -p "${stage}"

copy_relative() {
  local relative="$1"
  local source="${PROJECT_DIR}/${relative}"
  local destination="${stage}/${relative}"
  if [[ ! -e "${source}" && ! -L "${source}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${destination}")"
  cp -a "${source}" "${destination}"
}

# Git-tracked files are the source of truth for code and documentation.  This
# avoids accidentally packaging local virtual environments or private files.
while IFS= read -r -d '' relative; do
  copy_relative "${relative}"
done < <(git -C "${PROJECT_DIR}" ls-files -z)

# The generated dataset is intentionally not expected to be tracked by Git.
if [[ "${include_data}" == "1" ]]; then
  if [[ -d "${PROJECT_DIR}/data" ]]; then
    cp -a "${PROJECT_DIR}/data" "${stage}/data"
    data_mode="included"
  else
    data_mode="requested-but-missing"
    echo "warning: data/ does not exist; packaging source only" >&2
  fi
else
  data_mode="omitted"
fi

artifacts="none"
if [[ "${include_results}" == "1" && -d "${PROJECT_DIR}/results" ]]; then
  cp -a "${PROJECT_DIR}/results" "${stage}/results"
  artifacts="results"
fi
if [[ "${include_checkpoints}" == "1" && -d "${PROJECT_DIR}/checkpoints" ]]; then
  cp -a "${PROJECT_DIR}/checkpoints" "${stage}/checkpoints"
  if [[ "${artifacts}" == "none" ]]; then artifacts="checkpoints"; else artifacts="${artifacts},checkpoints"; fi
fi

# Ensure the migration utilities are present even before the next Git commit.
copy_relative "scripts/migration_manifest.py"
copy_relative "scripts/package_experiment.sh"
copy_relative "scripts/restore_experiment.sh"
copy_relative "evaluate_move_metrics.py"
copy_relative "scripts/run_move_evaluation.sh"
copy_relative "visualize_major_piece_probe.py"

source_commit="$(git -C "${PROJECT_DIR}" rev-parse --short=12 HEAD 2>/dev/null || printf '%s' unknown)"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/migration_manifest.py" write \
  --root "${stage}" \
  --output "${stage}/MIGRATION_MANIFEST.json" \
  --source-commit "${source_commit}" \
  --data-mode "${data_mode}" \
  --artifacts "${artifacts}"

tar -czf "${output}" -C "${tmp_root}" "${BUNDLE_ROOT}"

echo "created: ${output}"
echo "source commit: ${source_commit}"
echo "data: ${data_mode}; artifacts: ${artifacts}"
echo "warning: generated data may contain player names and source paths; do not share the archive publicly without review" >&2
