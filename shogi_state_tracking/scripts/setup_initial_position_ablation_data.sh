#!/usr/bin/env bash
set -euo pipefail
echo "The initial-position ablation dataset was retired. Building factorized_v3 instead." >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_factorized_v3_data.sh" "$@"
