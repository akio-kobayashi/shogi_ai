#!/usr/bin/env python3
"""廃止済みinitial-position ablationへの誤実行を防ぐ互換entry point．"""

raise SystemExit(
    "initial-position ablation was retired; use build_factorized_prompt_dataset.py "
    "or scripts/setup_factorized_v3_data.sh"
)
