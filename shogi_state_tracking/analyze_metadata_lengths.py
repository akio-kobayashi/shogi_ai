#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""metadata.csvの対局長分布からTransformer系列長を決める。

``create_dataset.py split`` と同じ既定の抽出条件を使い、train／validation／
evaluation eligibleごとの ``total_moves`` を調べる。CSA本体は読まないため、
CSAがない計算機でも実行できる。

この実験の入力は固定局面96トークンに制御トークン3個を加えた系列である。
したがって、推奨する ``max_seq_len`` は次式で計算する。

    fixed_overhead + 対局長の指定分位点

指定分位点を超える対局は、学習時のrandom-start windowingで切り出される。
最大対局長に合わせると、少数の極端に長い対局がattentionのメモリを支配するため、
このスクリプトでは分位点を使う。
"""

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


DATE_IN_PATH = re.compile(r"/(20[0-9]{2})/([0-9]{2})/([0-9]{2})/")
DATE_IN_FILENAME = re.compile(r"(20[0-9]{2})([0-9]{2})([0-9]{2})[0-9]{6}")


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def extract_game_date(row: Mapping[str, str]) -> dt.date:
    """create_dataset.pyと同じ優先順位で対局日を得る。"""
    for column in ("game_date", "date"):
        value = row.get(column, "").strip()
        if value:
            return parse_date(value[:10])

    path = row["file_path"].replace("\\", "/")
    match = DATE_IN_PATH.search(path) or DATE_IN_FILENAME.search(path)
    if not match:
        raise ValueError("対局日をfile_pathから抽出できません: {}".format(row["file_path"]))
    return dt.date(*(int(part) for part in match.groups()))


def iter_lengths(
    metadata_csv: Path,
    min_date: dt.date,
    max_date: Optional[dt.date],
    validation_from: dt.date,
    evaluation_from: dt.date,
    min_rating: float,
    min_moves: int,
    include_draws: bool,
) -> tuple[Dict[str, List[int]], Counter]:
    """条件に合う対局の手数をsplit別にストリームで集計する。

    metadata全体を辞書のリストへ展開しない。76 MB程度のmetadataでも、解析時の
    一時的なメモリ使用量を手数配列と重複キーだけに抑える。
    """
    lengths: Dict[str, List[int]] = {
        "all": [],
        "train": [],
        "validation": [],
        "evaluation_eligible": [],
    }
    rejected: Counter = Counter()
    seen_keys: Set[tuple[str, int]] = set()

    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "file_path",
            "kif_index",
            "rating_b",
            "rating_w",
            "game_result",
            "total_moves",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "metadata CSVに必要な列がありません: {}".format(
                    ", ".join(sorted(missing))
                )
            )

        for source in reader:
            try:
                row = dict(source)
                game_date = extract_game_date(row)
                kif_index = int(row["kif_index"])
                rating_b = float(row["rating_b"])
                rating_w = float(row["rating_w"])
                total_moves = int(row["total_moves"])
                game_result = int(row["game_result"])
            except (KeyError, TypeError, ValueError):
                rejected["invalid_metadata"] += 1
                continue

            key = (row["file_path"], kif_index)
            if key in seen_keys:
                rejected["duplicate"] += 1
                continue
            seen_keys.add(key)

            if game_date < min_date or (
                max_date is not None and game_date > max_date
            ):
                rejected["date"] += 1
                continue
            if rating_b < min_rating or rating_w < min_rating:
                rejected["rating"] += 1
                continue
            if total_moves < min_moves:
                rejected["moves"] += 1
                continue
            if not include_draws and game_result == 0:
                rejected["draw"] += 1
                continue

            lengths["all"].append(total_moves)
            if game_date < validation_from:
                split = "train"
            elif game_date < evaluation_from:
                split = "validation"
            else:
                split = "evaluation_eligible"
            lengths[split].append(total_moves)

    return lengths, rejected


def nearest_rank(values: Sequence[int], quantile: float) -> int:
    """線形補間を避けた再現性のあるnearest-rank分位点。"""
    if not values:
        raise ValueError("分位点を空の配列から計算できません")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    position = max(1, math.ceil(quantile * len(values)))
    return values[min(len(values), position) - 1]


def ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("round_to must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def summarize_lengths(
    values: Iterable[int],
    fixed_overhead: int,
    recommendation_quantile: float,
    round_to: int,
    candidate_lengths: Sequence[int],
) -> Dict[str, object]:
    values = sorted(values)
    if not values:
        return {"games": 0, "total_moves": 0, "quantiles": {}}

    selected_moves = nearest_rank(values, recommendation_quantile)
    raw_seq_len = fixed_overhead + selected_moves
    recommended_seq_len = ceil_to_multiple(raw_seq_len, round_to)
    quantiles = {
        "p50": nearest_rank(values, 0.50),
        "p75": nearest_rank(values, 0.75),
        "p90": nearest_rank(values, 0.90),
        "p95": nearest_rank(values, 0.95),
        "p97": nearest_rank(values, 0.97),
        "p99": nearest_rank(values, 0.99),
        "p100": values[-1],
    }
    candidates: Dict[str, Dict[str, object]] = {}
    for seq_len in candidate_lengths:
        move_limit = seq_len - fixed_overhead
        covered = sum(move <= move_limit for move in values)
        candidates[str(seq_len)] = {
            "move_window": move_limit,
            "covered_games": covered,
            "coverage": covered / len(values),
            "tail_games": len(values) - covered,
        }

    return {
        "games": len(values),
        "total_moves": sum(values),
        "mean_moves": sum(values) / len(values),
        "min_moves": values[0],
        "max_moves": values[-1],
        "quantiles": quantiles,
        "recommendation": {
            "quantile": recommendation_quantile,
            "quantile_moves": selected_moves,
            "fixed_overhead": fixed_overhead,
            "raw_seq_len": raw_seq_len,
            "max_seq_len": recommended_seq_len,
            "move_window": recommended_seq_len - fixed_overhead,
            "coverage": sum(
                move <= recommended_seq_len - fixed_overhead for move in values
            )
            / len(values),
        },
        "candidate_lengths": candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="metadata.csvの対局長からmax_seq_lenを推定する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--min-date", default="2022-01-01")
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--min-rating", type=float, default=3000.0)
    parser.add_argument("--min-moves", type=int, default=80)
    parser.add_argument("--include-draws", action="store_true")
    parser.add_argument("--validation-from", default="2024-10-01")
    parser.add_argument("--evaluation-from", default="2025-01-01")
    parser.add_argument("--fixed-overhead", type=int, default=99)
    parser.add_argument(
        "--recommendation-quantile",
        type=float,
        default=0.95,
        help="指定分位点以上をwindowing対象にする",
    )
    parser.add_argument(
        "--round-to",
        type=int,
        default=32,
        help="GPUメモリと比較しやすい系列長の丸め幅",
    )
    parser.add_argument(
        "--candidate-lengths",
        default="256,288,320,352,384",
        help="coverageを併記するmax_seq_lenのカンマ区切りリスト",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        min_date = parse_date(args.min_date)
        max_date = parse_date(args.max_date) if args.max_date else None
        validation_from = parse_date(args.validation_from)
        evaluation_from = parse_date(args.evaluation_from)
        if validation_from >= evaluation_from:
            raise ValueError("validation-fromはevaluation-fromより前でなければなりません")
        candidate_lengths = [
            int(value.strip())
            for value in args.candidate_lengths.split(",")
            if value.strip()
        ]
        lengths, rejected = iter_lengths(
            Path(args.metadata_csv),
            min_date=min_date,
            max_date=max_date,
            validation_from=validation_from,
            evaluation_from=evaluation_from,
            min_rating=args.min_rating,
            min_moves=args.min_moves,
            include_draws=args.include_draws,
        )
        split_summaries = {
            name: summarize_lengths(
                values,
                fixed_overhead=args.fixed_overhead,
                recommendation_quantile=args.recommendation_quantile,
                round_to=args.round_to,
                candidate_lengths=candidate_lengths,
            )
            for name, values in lengths.items()
        }
        if not split_summaries["train"].get("games"):
            raise ValueError("条件に合うtrain対局がありません")
        result = {
            "metadata_csv": str(Path(args.metadata_csv)),
            "filters": {
                "min_date": args.min_date,
                "max_date": args.max_date,
                "min_rating": args.min_rating,
                "min_moves": args.min_moves,
                "include_draws": args.include_draws,
                "validation_from": args.validation_from,
                "evaluation_from": args.evaluation_from,
            },
            "rejected": dict(sorted(rejected.items())),
            "fixed_overhead": args.fixed_overhead,
            "round_to": args.round_to,
            "primary_recommendation": {
                "basis": "train",
                **split_summaries["train"]["recommendation"],
            },
            "splits": split_summaries,
            "note": (
                "total_movesの分布だけでは千日手の有無は判定できない。"
                "また推奨max_seq_lenは固定局面prefixを含む。"
            ),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, csv.Error) as exc:
        print("エラー: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
