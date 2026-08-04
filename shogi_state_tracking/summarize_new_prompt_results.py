#!/usr/bin/env python3
"""3規模×条件（将来は複数seed）の主要metricをCSVとMarkdownへ集約する。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="新prompt実験の結果を集約する")
    parser.add_argument("--results-dir", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); rows = []
    for path in sorted(Path(args.results_dir).rglob("move_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8")); metrics = payload["metrics"]
        model_root = next((parent for parent in path.parents if parent.name.startswith("vanilla-")), None)
        if model_root is None: continue
        model_size = model_root.name.removeprefix("vanilla-")
        relative = path.relative_to(model_root)
        condition = relative.parts[0]
        row = {"model_size": model_size, "condition": condition, "run_dir": str(path.parent), **metrics, **metrics["legality"]}
        manifest_path = path.parent / "run_manifest.json"
        if manifest_path.exists():
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_args = run_manifest.get("args", {})
            row["seed"] = run_args.get("seed")
            # pはrate ablationの独立変数であり，conditionだけで集約してはならない。
            row["annotation_probability"] = run_args.get("annotation_probability")
        probe_path = path.parent / "probes" / "probe_metrics.json"
        if probe_path.exists():
            probe = json.loads(probe_path.read_text(encoding="utf-8")); final = probe["probe_results"].get("layer_12", {})
            for name, value in final.get("evaluation", {}).items():
                if isinstance(value, (int, float)): row["probe_final_" + name] = value
        token_path = path.parent / "token_probe_metrics.json"
        if token_path.exists(): row.update({"token_" + key: value for key, value in json.loads(token_path.read_text(encoding="utf-8"))["metrics"].items()})
        rows.append(row)
    if not rows: raise ValueError("no move_metrics.json below results-dir")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# 新prompt実験サマリー", "", "| model | condition | p | top-1 | top-5 | legal top-1 | legal mass | probe full state |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {model_size} | {condition} | {annotation_probability} | {top1_accuracy:.4f} | {top5_accuracy:.4f} | {top1_legal_rate:.4f} | {mean_legal_probability_mass:.4f} | {probe_final_full_state_exact_match:.4f} |".format(**{**row, "annotation_probability": row.get("annotation_probability", "?"), "probe_final_full_state_exact_match": row.get("probe_final_full_state_exact_match", float("nan"))}))
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    grouped = {}
    numeric = ("top1_accuracy", "top5_accuracy", "top1_legal_rate", "mean_legal_probability_mass", "probe_final_full_state_exact_match")
    for row in rows:
        grouped.setdefault((row["model_size"], row["condition"], row.get("annotation_probability")), []).append(row)
    aggregate = []
    for (model_size, condition, probability), values in sorted(grouped.items()):
        row = {"model_size": model_size, "condition": condition, "annotation_probability": probability, "runs": len(values)}
        for name in numeric:
            observations = [float(value[name]) for value in values if isinstance(value.get(name), (int, float))]
            if observations:
                row[name + "_mean"] = statistics.fmean(observations)
                row[name + "_std"] = statistics.stdev(observations) if len(observations) > 1 else 0.0
        aggregate.append(row)
    aggregate_fields = sorted({key for row in aggregate for key in row})
    with (output / "summary_by_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields); writer.writeheader(); writer.writerows(aggregate)
    (output / "summary_by_condition.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output / "summary.md")


if __name__ == "__main__": main()
