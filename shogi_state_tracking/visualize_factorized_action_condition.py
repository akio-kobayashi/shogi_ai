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
    parser.add_argument("--robustness")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_heatmap(path, rows, columns, values, title):
    cell = 76; left = 145; top = 70
    width = left + cell * len(columns) + 20; height = top + cell * len(rows) + 55
    valid = [value for row in values for value in row if value is not None]
    lower = min(valid) if valid else 0.0; upper = max(valid) if valid else 1.0
    span = max(upper - lower, 1e-12)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(width, height),
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="{}" y="25" text-anchor="middle" font-family="sans-serif" font-size="15">{}</text>'.format(width / 2, title)]
    for column, label in enumerate(columns):
        parts.append('<text x="{}" y="55" text-anchor="middle" font-family="sans-serif" font-size="12">{}</text>'.format(left + (column + .5) * cell, label))
    for row, label in enumerate(rows):
        parts.append('<text x="{}" y="{}" text-anchor="end" dominant-baseline="middle" font-family="sans-serif" font-size="12">{}</text>'.format(left - 8, top + (row + .5) * cell, label))
        for column, value in enumerate(values[row]):
            normalized = 0.0 if value is None else (value - lower) / span
            red = int(245 - 150 * normalized); green = int(248 - 70 * normalized); blue = int(255 - 10 * normalized)
            x = left + column * cell; y = top + row * cell
            parts.append('<rect x="{}" y="{}" width="{}" height="{}" fill="rgb({},{},{})" stroke="#777"/>'.format(x, y, cell, cell, red, green, blue))
            label_value = "NA" if value is None else "{:.3f}".format(value)
            parts.append('<text x="{}" y="{}" text-anchor="middle" dominant-baseline="middle" font-family="monospace" font-size="12">{}</text>'.format(x + cell / 2, y + cell / 2, label_value))
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


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

    if args.robustness:
        robust = json.loads(Path(args.robustness).read_text(encoding="utf-8"))["metrics"]
        robust_sources = sorted(robust, key=layer_number)
        contrast = []
        behavior = []
        for source in robust_sources:
            difference = robust[source].get("pooled_probe_within_prefix", {}).get("difference", {}).get("mean")
            if difference is not None:
                contrast.append((layer_number(source), float(difference)))
            group = robust[source].get("behavior_after_drop", {}).get("actual_drop", {})
            probability = group.get("mean_correct_piece_probability_after_drop")
            if probability is not None:
                behavior.append((layer_number(source), float(probability)))
            cross = robust[source].get("cross_position_generalization", {})
            families = [name for name in ("pre", "drop", "normal", "pooled") if name in cross]
            values = [[
                cross[family].get("tested_at", {}).get(branch, {}).get("relevant_count_accuracy")
                for branch in ("pre", "drop", "normal")
            ] for family in families]
            if values:
                write_heatmap(output / "cross_position_{}.svg".format(source), families,
                              ("test pre", "test drop", "test normal"), values,
                              "Cross-position probe accuracy: {}".format(source))
        if contrast:
            write_lines(output / "robust_pooled_action_contrast.svg", [("pooled probe", contrast)],
                        "Branch-balanced same-prefix contrast", "decoder layer",
                        "true-count probability: DROP minus mean normal")
        if behavior:
            write_lines(output / "drop_piece_behavior.svg", [("model output", behavior)],
                        "Correct piece probability after DROP", "decoder layer",
                        "restricted probability of recorded drop piece")

    print(json.dumps({
        "event": "action_condition_visualization_complete", "output_dir": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
