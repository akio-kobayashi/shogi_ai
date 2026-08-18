#!/usr/bin/env python3
"""reference行動条件実験の3主条件とAP別枠を1つのJSONへ整理する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY = (
    ("vanilla", "vanilla-p0.0", "primary"),
    ("rap_0.15", "rap-p0.15-proportional-rap-v1", "primary"),
    ("rap_0.25", "rap-p0.25-proportional-rap-v1", "primary"),
)
ORACLE = (
    ("ap_native", "ap-p1.0-proportional-annotation-v1", "oracle-native"),
    ("ap_no_annotation", "ap-p1.0-proportional-annotation-v1", "sensitivity-no-annotation"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="行動条件実験matrixの収集")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seeds", default="20260802")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def compact(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = {}
    for source, values in payload.get("metrics", {}).items():
        metrics[source] = values.get("within_prefix_contrasts", {})
    robustness_path = path.with_name("action_condition_robustness.json")
    attention_path = path.with_name("action_condition_attention_ablation.json")
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    robust_metrics = {}
    for source, values in robustness.get("metrics", {}).items():
        robust_metrics[source] = {
            "pooled_probe_within_prefix": values.get("pooled_probe_within_prefix"),
            "cross_position_generalization": values.get("cross_position_generalization"),
            "behavior_after_drop": values.get("behavior_after_drop"),
        }
    attention_payload = (
        json.loads(attention_path.read_text(encoding="utf-8"))
        if attention_path.is_file() else None
    )
    return {
        "source": str(path),
        "checkpoint": payload.get("checkpoint"),
        "protocol": payload.get("protocol"),
        "matching": payload.get("matching"),
        "matching_balance": payload.get("matching_balance"),
        "branch_summary": payload.get("branch_summary"),
        "within_prefix_contrasts": metrics,
        "robustness_source": str(robustness_path),
        "split_audit": robustness.get("split_audit"),
        "causal_prefix_full_audit": robustness.get("causal_prefix_full_audit"),
        "robustness": robust_metrics,
        "attention_ablation_source": str(attention_path) if attention_path.is_file() else None,
        "attention_ablation": None if attention_payload is None else {
            "no_mask_forward_max_absolute_logit_error": attention_payload.get("no_mask_forward_max_absolute_logit_error"),
            "matching": attention_payload.get("matching"),
            "attention": attention_payload.get("attention"),
            "ablation": attention_payload.get("ablation"),
        },
    }


def main():
    args = parse_args()
    root = Path(args.results_dir)
    seeds = [value.strip() for value in args.seeds.split(",") if value.strip()]
    result = {
        "format_version": 1,
        "design": {
            "primary": [name for name, _, _ in PRIMARY],
            "oracle_separate": [name for name, _, _ in ORACLE],
            "pool_ap_with_primary": False,
        },
        "primary": {},
        "oracle": {},
    }
    missing = []
    for section, conditions in (("primary", PRIMARY), ("oracle", ORACLE)):
        for name, condition, category in conditions:
            result[section][name] = {}
            for seed in seeds:
                path = (
                    root / "llama-reference" / "implicit-initial" / condition / "seed-{}".format(seed)
                    / "evaluation" / "action-condition" / category / "action_condition_metrics.json"
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue
                robustness_path = path.with_name("action_condition_robustness.json")
                if not robustness_path.is_file():
                    missing.append(str(robustness_path))
                    continue
                attention_path = path.with_name("action_condition_attention_ablation.json")
                if section == "primary" and not attention_path.is_file():
                    missing.append(str(attention_path))
                    continue
                result[section][name][seed] = compact(path)
    if missing:
        raise FileNotFoundError("action-condition matrix is incomplete:\n" + "\n".join(missing))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "action_condition_matrix_complete", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
