#!/usr/bin/env bash
set -euo pipefail

# Restore a package made by package_experiment.sh without overwriting an
# existing non-empty directory.  Verification happens before the final move.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUNDLE_ROOT="shogi_state_tracking"

usage() {
  cat <<'EOF'
Usage: scripts/restore_experiment.sh ARCHIVE.tar.gz DESTINATION

Extract and checksum-verify a migration bundle. DESTINATION must not be a
non-empty directory. The archive contains no .venv; run setup_env.sh after
restoration and select cpu/cuda/rocm for the destination host.

Options:
  --skip-verify        skip checksum verification (not recommended)
  -h, --help           show this help
EOF
}

skip_verify=0
archive=""
destination=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-verify) skip_verify=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -z "${archive}" ]]; then archive="$1"
      elif [[ -z "${destination}" ]]; then destination="$1"
      else echo "too many positional arguments" >&2; exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "${archive}" || -z "${destination}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "${archive}" ]]; then
  echo "archive does not exist: ${archive}" >&2
  exit 2
fi
if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required" >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python is required for checksum verification: ${PYTHON_BIN}" >&2
  exit 2
fi

# Reject absolute paths and parent traversal before extraction.  The archive
# is normally produced locally, but this check makes the restore operation
# safe when the file came from another host.
while IFS= read -r entry; do
  case "${entry}" in
    /*|../*|*/../*|..|*/..)
      echo "unsafe archive entry: ${entry}" >&2
      exit 2
      ;;
  esac
done < <(tar -tzf "${archive}")

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/shogi-state-restore.XXXXXX")"
trap 'rm -rf "${tmp_root}"' EXIT
tar -xzf "${archive}" -C "${tmp_root}"
restored="${tmp_root}/${BUNDLE_ROOT}"
if [[ ! -f "${restored}/MIGRATION_MANIFEST.json" ]]; then
  echo "invalid bundle: ${BUNDLE_ROOT}/MIGRATION_MANIFEST.json is missing" >&2
  exit 2
fi

if [[ "${skip_verify}" != "1" ]]; then
  "${PYTHON_BIN}" "${restored}/scripts/migration_manifest.py" verify \
    --root "${restored}" \
    --manifest "${restored}/MIGRATION_MANIFEST.json"
fi

if [[ -e "${destination}" ]]; then
  if [[ ! -d "${destination}" || -n "$(find "${destination}" -mindepth 1 -print -quit)" ]]; then
    echo "destination exists and is not empty: ${destination}" >&2
    echo "choose another destination; restore never overwrites existing files" >&2
    exit 2
  fi
  rmdir "${destination}"
fi
mkdir -p "$(dirname "${destination}")"
mv "${restored}" "${destination}"

echo "restored: ${destination}"
echo "next step (Linux/WSL2):"
echo "  cd ${destination}"
echo "  ./setup_env.sh cpu    # or cuda/rocm for the destination host"
echo "then verify the dataset and run the experiment scripts."
