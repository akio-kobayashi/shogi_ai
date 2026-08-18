#!/usr/bin/env python3
"""同一prefix行動条件評価を依存なしSVGで可視化する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualize_factorized_drop_relevance import layer_number, write_lines


def parse_args():
    parser = argparse.ArgumentParser(description="行動条件付き状態表現のSVG可視化")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def contrast_points(metrics, sources, group, field):
    points = []
    for source in sources:
        value = metrics[source]["within_prefix_contrasts"].get(group, {}).get(field, {}).get("mean")
        if value is not None:
            points.append((layer_number(source), float(value)))
    return points


def branch_points(metrics, sources, group, branch, relevance):
    points = []
    for source in sources:
        value = (
            metrics[source].get("branch_metrics", {}).get(group, {})
            .get(branch, {}).get(relevance, {}).get("mean_true_count_probability")
        )
        if value is not None:
            points.append((layer_number(source), float(value)))
    return points


def main():
    args = parse_args()
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    sources = sorted(metrics, key=layer_number)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    series = []
    for relevance, field in (
        ("relevant", "relevant_count_drop_minus_normal"),
        ("irrelevant", "irrelevant_count_drop_minus_normal"),
    ):
        points = contrast_points(metrics, sources, "all", field)
        if points:
            series.append((relevance, points))
    if series:
        write_lines(
            output / "within_prefix_action_contrast.svg", series,
            "Same-prefix action contrast", "decoder layer",
            "true-count probability: <DROP> minus normal prefix",
        )

    series = []
    field = "selective_count_difference_in_differences"
    for group in ("all", "actual_drop", "actual_normal"):
        points = contrast_points(metrics, sources, group, field)
        if points:
            series.append((group, points))
    if series:
        write_lines(
            output / "selective_action_condition.svg", series,
            "Selective action-conditioned hand representation", "decoder layer",
            "difference-in-differences of true-count probability",
        )

    series = []
    for group in ("actual_drop", "actual_normal"):
        for branch in ("pre", "drop", "normal"):
            points = branch_points(metrics, sources, group, branch, "relevant")
            if points:
                series.append(("{} {}".format(group, branch), points))
    if series:
        write_lines(
            output / "relevant_hand_by_branch.svg", series,
            "Relevant hand information by branch", "decoder layer",
            "probability assigned to the true hand count",
        )

    print(json.dumps({
        "event": "action_condition_visualization_complete", "output_dir": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
