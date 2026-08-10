#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trace="${1:?memory trace path is required}"
shift
[[ $# -gt 0 ]] || { echo "guarded command is required" >&2; exit 2; }

"$@" &
command_pid=$!
python3 "${SCRIPT_DIR}/monitor_process_memory.py" \
  --pid "${command_pid}" --output "${trace}" \
  --interval "${MEMORY_LOG_INTERVAL:-5}" &
monitor_pid=$!

set +e
wait "${command_pid}"
status=$?
set -e
wait "${monitor_pid}" 2>/dev/null || true
exit "${status}"
