#!/usr/bin/env python3
"""small/base/largeサイズ比較実験結果を長形式CSV/JSONへ集約する。"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="small/base/largeサイズ比較実験の集約を作成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="比較実験ルート（例: results/transformer_size_compare_512）",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=("small", "base", "large"),
        help="集約対象のサイズ名",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="出力先。未指定なら実験ディレクトリ/seed_x/summary を使用",
    )
    parser.add_argument(
        "--probe-suffix",
        default="probes",
        help="比較時に使うprobe出力ディレクトリ名",
    )
    parser.add_argument("--move-suffix", default="moves")
    parser.add_argument("--check-probe-suffix", default="check_probes")
    parser.add_argument(
        "--include-moves",
        action="store_true",
        help="指手・合法手評価を集約対象へ加え，欠損時はエラーにする",
    )
    parser.add_argument(
        "--include-check-probes",
        action="store_true",
        help="王手probeを集約対象へ加え，欠損時はエラーにする",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="一部サイズ・ログ不足でも集約を継続する",
    )
    parser.add_argument(
        "--include-non-number",
        action="store_true",
        help="数値以外の要素も0/1フラグに変換して収集する",
    )
    return parser.parse_args()


def flatten_numbers(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, bool):
        yield prefix, 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        yield prefix, float(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_numbers(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from flatten_numbers(child, child_prefix)


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_size_rows(
    experiment_root: Path,
    seed: int,
    size: str,
    probe_suffix: str,
    move_suffix: str,
    check_probe_suffix: str,
    include_moves: bool,
    include_check_probes: bool,
    include_non_number: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    size_dir = experiment_root / f"seed_{seed}" / size

    training_path = size_dir / "training_history.json"
    probe_path = size_dir / probe_suffix / "probe_metrics.json"
    move_path = size_dir / move_suffix / "move_metrics.json"
    check_probe_path = size_dir / check_probe_suffix / "check_probe_metrics.json"

    if training_path.exists():
        training_payload = load_json(training_path)
        for metric, value in flatten_numbers(training_payload):
            rows.append(
                {
                    "seed": seed,
                    "size": size,
                    "artifact": str(training_path),
                    "artifact_type": "training",
                    "metric": metric,
                    "value": value,
                }
            )

    if probe_path.exists():
        probe_payload = load_json(probe_path)
        for metric, value in flatten_numbers(probe_payload):
            rows.append(
                {
                    "seed": seed,
                    "size": size,
                    "artifact": str(probe_path),
                    "artifact_type": "probes",
                    "metric": metric,
                    "value": value,
                }
            )

    for artifact_type, path, enabled in (
        ("moves", move_path, include_moves),
        ("check_probes", check_probe_path, include_check_probes),
    ):
        if enabled and path.exists():
            payload = load_json(path)
            for metric, value in flatten_numbers(payload):
                rows.append(
                    {
                        "seed": seed,
                        "size": size,
                        "artifact": str(path),
                        "artifact_type": artifact_type,
                        "metric": metric,
                        "value": value,
                    }
                )

    if include_non_number:
        for artifact_type, path in (
            ("training", training_path),
            ("probes", probe_path),
            ("moves", move_path),
            ("check_probes", check_probe_path),
        ):
            if not path.exists():
                continue
            payload = load_json(path)
            rows.append(
                {
                    "seed": seed,
                    "size": size,
                    "artifact": str(path),
                    "artifact_type": artifact_type,
                    "metric": "exists",
                    "value": 1.0,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else experiment_dir / f"seed_{args.seed}" / "summary"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    missing: list[str] = []

    for size in args.sizes:
        size_dir = experiment_dir / f"seed_{args.seed}" / size
        training_path = size_dir / "training_history.json"
        probe_path = size_dir / args.probe_suffix / "probe_metrics.json"
        move_path = size_dir / args.move_suffix / "move_metrics.json"
        check_probe_path = size_dir / args.check_probe_suffix / "check_probe_metrics.json"

        if not training_path.exists():
            missing.append(str(training_path))
        if not probe_path.exists():
            missing.append(str(probe_path))
        if args.include_moves and not move_path.exists():
            missing.append(str(move_path))
        if args.include_check_probes and not check_probe_path.exists():
            missing.append(str(check_probe_path))

        rows.extend(
            collect_size_rows(
                experiment_dir,
                args.seed,
                size,
                args.probe_suffix,
                args.move_suffix,
                args.check_probe_suffix,
                args.include_moves,
                args.include_check_probes,
                args.include_non_number,
            )
        )

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "missing required files:\n" + "\n".join(sorted(set(missing)))
        )

    rows.sort(key=lambda item: (item["size"], item["artifact_type"], item["metric"]))

    csv_path = output_root / "size_compare_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "size",
                "artifact_type",
                "artifact",
                "metric",
                "value",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_root / "size_compare_summary.json"
    payload = {
        "experiment_dir": str(experiment_dir),
        "seed": args.seed,
        "sizes": list(args.sizes),
        "probe_suffix": args.probe_suffix,
        "move_suffix": args.move_suffix,
        "check_probe_suffix": args.check_probe_suffix,
        "include_moves": args.include_moves,
        "include_check_probes": args.include_check_probes,
        "rows": rows,
        "missing_files": sorted(set(missing)),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        json.dumps(
            {
                "output_csv": str(csv_path),
                "output_json": str(summary_path),
                "rows": len(rows),
                "missing": len(missing),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
