#!/usr/bin/env bash
set -euo pipefail
echo "The initial-position 2x2 experiment was retired. Running the factorized_v3 RAP-rate experiment." >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_factorized_rap_ablation.sh" "$@"
