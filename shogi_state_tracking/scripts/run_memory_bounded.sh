#!/usr/bin/env bash
# Linux cgroup v2で実験全体を隔離し，OS全体のswap枯渇を防ぐ。
set -euo pipefail

[[ $# -gt 0 ]] || {
  echo "usage: MEMORY_MAX=100G [MEMORY_HIGH=90G] [MEMORY_SWAP_MAX=2G] $0 COMMAND [ARGS...]" >&2
  exit 2
}
[[ -n "${MEMORY_MAX:-}" ]] || {
  echo "MEMORY_MAX is required; refusing an unbounded experiment" >&2
  exit 2
}

if [[ "${SHOGI_MEMORY_GUARD_ACTIVE:-0}" == 1 ]]; then
  exec "$@"
fi
[[ "$(uname -s)" == Linux ]] || {
  echo "memory guard requires Linux cgroup v2" >&2
  exit 2
}
[[ -f /sys/fs/cgroup/cgroup.controllers ]] || {
  echo "cgroup v2 is unavailable; refusing an unbounded experiment" >&2
  exit 2
}
command -v systemd-run >/dev/null 2>&1 || {
  echo "systemd-run is unavailable; refusing an unbounded experiment" >&2
  exit 2
}

unit="shogi-state-tracking-$USER-$$"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trace="${MEMORY_TRACE:-memory-${unit}.jsonl}"
properties=(--property="MemoryMax=${MEMORY_MAX}" --property="OOMPolicy=stop")
[[ -z "${MEMORY_HIGH:-}" ]] || properties+=(--property="MemoryHigh=${MEMORY_HIGH}")
properties+=(--property="MemorySwapMax=${MEMORY_SWAP_MAX:-0}")

echo "memory guard: unit=${unit} high=${MEMORY_HIGH:-unset} max=${MEMORY_MAX} swap_max=${MEMORY_SWAP_MAX:-0}" >&2
set +e
systemd-run --user --scope --unit="${unit}" \
  "${properties[@]}" \
  --setenv=SHOGI_MEMORY_GUARD_ACTIVE=1 \
  "${script_dir}/memory_guard_inner.sh" "${trace}" "$@"
status=$?
set -e

report="${MEMORY_REPORT:-memory-${unit}.txt}"
if command -v systemctl >/dev/null 2>&1; then
  systemctl --user show "${unit}.scope" \
    --property=Result \
    --property=MemoryCurrent \
    --property=MemoryPeak \
    --property=MemorySwapCurrent \
    --property=MemorySwapPeak \
    --property=OOMPolicy >"${report}" 2>&1 || true
  echo "memory guard report: ${report}" >&2
fi
echo "memory time series: ${trace}" >&2
exit "${status}"
