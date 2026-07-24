#!/usr/bin/env python3
"""2モデル×CoT有無の出力をlong形式の比較表へまとめる。"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="run_2x2_experiment.shの結果をJSON/CSVへ集約する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="一部の条件が未完了でも、存在する結果だけを集約する",
    )
    return parser.parse_args()


def flatten_numbers(value, prefix: str = "") -> Iterable[tuple[str, float]]:
    """JSON中の数値leafをdot区切りの名前へ変換する。"""
    if isinstance(value, bool):
        yield prefix, float(value)
    elif isinstance(value, (int, float)):
        yield prefix, float(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
            yield from flatten_numbers(child, child_prefix)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def condition_files(root: Path, model_type: str) -> Dict[str, tuple[str, Path]]:
    model_root = root / model_type
    return {
        "answer_only_training": (
            "answer-only",
            model_root / "answer-only" / "training_history.json",
        ),
        "answer_only_probes": (
            "answer-only",
            model_root / "cot" / "probes-answer-only" / "probe_metrics.json",
        ),
        "cot_training": (
            "cot",
            model_root / "cot" / "training" / "training_history.json",
        ),
        "cot_probes": (
            "cot",
            model_root / "cot" / "probes-cot" / "probe_metrics.json",
        ),
        "cot_reasoning": (
            "cot",
            model_root / "cot" / "evaluation" / "reasoning_metrics.json",
        ),
    }


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    root = experiment_dir / "seed_{}".format(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    source_files: Dict[str, str] = {}
    for model_type in ("vanilla", "t2mlr"):
        for source, (training_condition, path) in condition_files(
            root, model_type
        ).items():
            source_key = "{}.{}".format(model_type, source)
            source_files[source_key] = str(path)
            if not path.exists():
                missing.append(str(path))
                continue
            payload = load_json(path)
            for metric, value in flatten_numbers(payload):
                rows.append(
                    {
                        "seed": args.seed,
                        "model_type": model_type,
                        "training_condition": training_condition,
                        "source": source,
                        "metric": metric,
                        "value": value,
                    }
                )

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "missing experiment outputs:\n{}".format("\n".join(missing))
        )
    rows.sort(
        key=lambda row: (
            str(row["model_type"]),
            str(row["training_condition"]),
            str(row["source"]),
            str(row["metric"]),
        )
    )

    csv_path = output_dir / "comparison_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "model_type",
                "training_condition",
                "source",
                "metric",
                "value",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": 1,
        "experiment_dir": str(experiment_dir),
        "seed": args.seed,
        "rows": rows,
        "source_files": source_files,
        "missing_files": missing,
    }
    json_path = output_dir / "comparison_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "json": str(json_path),
                "metrics": len(rows),
                "missing_files": len(missing),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
