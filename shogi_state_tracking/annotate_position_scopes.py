#!/usr/bin/env python3
"""既存JSONLへ局面未見性のスコープを付与する。

学習済みモデルや既存の語彙を変更せず、initial_sfenとmove_tokensをcshogiで再生して
train JSONLの局面集合と照合する。position_scopeの追加後も、モデル入力は従来どおり
開始局面96トークンと指し手列だけである。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from create_dataset import annotate_position_scopes, import_cshogi, make_position_hash


def read_records(path: Path) -> Iterable[Tuple[int, Dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("{}:{} はJSON objectではありません".format(path, line_number))
            yield line_number, record


def replay_position_hashes(record: Mapping[str, object], cshogi_module) -> List[str]:
    """開始SFENと指し手列から、初期局面を含むハッシュ列を作る。"""
    if "initial_sfen" not in record:
        raise ValueError("game {} にinitial_sfenがありません".format(record.get("game_id", "?")))
    moves = list(record.get("move_tokens", []))
    board = cshogi_module.Board(str(record["initial_sfen"]))
    hashes = [make_position_hash(board.sfen())]
    for ply, move_token in enumerate(moves, 1):
        try:
            move = board.move_from_usi(str(move_token))
            if not board.is_legal(move):
                raise ValueError("illegal move")
            board.push(move)
        except Exception as exc:
            raise ValueError(
                "game {} の第{}手を再生できません: {}".format(
                    record.get("game_id", "?"), ply, move_token
                )
            ) from exc
        hashes.append(make_position_hash(board.sfen()))
    return hashes


def collect_train_position_hashes(path: Path, cshogi_module) -> Set[str]:
    hashes: Set[str] = set()
    records = 0
    for line_number, record in read_records(path):
        try:
            hashes.update(replay_position_hashes(record, cshogi_module))
        except Exception as exc:
            raise ValueError("{}:{}: {}".format(path, line_number, exc)) from exc
        records += 1
    if records == 0:
        raise ValueError("学習JSONLが空です: {}".format(path))
    return hashes


def annotate_jsonl(
    input_path: Path,
    output_path: Path,
    train_position_hashes: Set[str],
    split: str,
    strict: bool = False,
) -> Dict[str, object]:
    cshogi = import_cshogi()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    position_counts: Counter = Counter()
    trajectory_counts: Counter = Counter()
    written = 0
    errors: List[Dict[str, object]] = []

    with output_path.open("w", encoding="utf-8") as output_handle:
        for line_number, source in read_records(input_path):
            try:
                record = dict(source)
                hashes = replay_position_hashes(record, cshogi)
                record["position_hashes"] = hashes
                annotate_position_scopes(
                    record,
                    train_position_hashes=train_position_hashes,
                    is_training=split == "train",
                )
                record.pop("position_hashes", None)
                record["schema_version"] = max(2, int(record.get("schema_version", 1)))
                record["player_scope"] = str(
                    record.get("player_scope", record.get("engine_scope", ""))
                )
                record["engine_scope"] = str(
                    record.get("engine_scope", record["player_scope"])
                )
                position_counts.update(record["position_scope_by_ply"])
                trajectory_counts.update([record["trajectory_scope"]])
                json.dump(record, output_handle, ensure_ascii=False, separators=(",", ":"))
                output_handle.write("\n")
                written += 1
            except Exception as exc:
                if strict:
                    raise
                errors.append(
                    {
                        "line": line_number,
                        "game_id": source.get("game_id", ""),
                        "error": str(exc),
                    }
                )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "split": split,
        "requested_records": written + len(errors),
        "written_records": written,
        "errors": errors,
        "position_scope_counts": dict(sorted(position_counts.items())),
        "trajectory_scope_counts": dict(sorted(trajectory_counts.items())),
        "train_position_count": len(train_position_hashes),
    }
    return summary


def annotate_directory(
    train_jsonl: Path,
    input_dir: Path,
    output_dir: Path,
    strict: bool = False,
) -> Dict[str, object]:
    """trainの局面集合を一度だけ作り、3 splitをまとめて処理する。"""
    cshogi = import_cshogi()
    train_hashes = collect_train_position_hashes(train_jsonl, cshogi)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, object] = {}
    for split in ("train", "validation", "evaluation"):
        input_path = input_dir / "{}.jsonl".format(split)
        if not input_path.exists():
            raise OSError("input JSONL not found: {}".format(input_path))
        summaries[split] = annotate_jsonl(
            input_path,
            output_dir / "{}.jsonl".format(split),
            train_position_hashes=train_hashes,
            split=split,
            strict=strict,
        )
        summary_path = output_dir / "{}.scope_summary.json".format(split)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summaries[split], handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train_position_count": len(train_hashes),
        "splits": summaries,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="既存JSONLへseen/unseen position scopeを付与する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-jsonl", required=True, help="局面集合の基準にするtrain JSONL")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-jsonl", help="スコープを付与するJSONL")
    mode.add_argument("--input-dir", help="train/validation/evaluation JSONLを含むディレクトリ")
    parser.add_argument("--output-jsonl", help="付与後JSONL。入力を上書きしない")
    parser.add_argument("--output-dir", help="3 splitの付与後JSONLを置くディレクトリ")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "evaluation"),
        help="trainでは全局面をseen_positionとして扱う",
    )
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--strict", action="store_true", help="再生失敗時に停止する")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.input_dir:
            if args.output_dir is None or args.output_jsonl or args.split or args.summary_json:
                raise ValueError(
                    "--input-dirでは--output-dirだけを指定し、--output-jsonl/--split/--summary-jsonは指定しないでください"
                )
            summary = annotate_directory(
                Path(args.train_jsonl),
                Path(args.input_dir),
                Path(args.output_dir),
                strict=args.strict,
            )
            has_errors = any(
                bool(split_summary["errors"])
                for split_summary in summary["splits"].values()
            )
        else:
            if args.output_jsonl is None or args.split is None or args.output_dir:
                raise ValueError(
                    "単一JSONLモードでは--output-jsonlと--splitが必要です"
                )
            cshogi = import_cshogi()
            train_hashes = collect_train_position_hashes(Path(args.train_jsonl), cshogi)
            summary = annotate_jsonl(
                Path(args.input_jsonl),
                Path(args.output_jsonl),
                train_position_hashes=train_hashes,
                split=args.split,
                strict=args.strict,
            )
            has_errors = bool(summary["errors"])
            if args.summary_json:
                summary_path = Path(args.summary_json)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with summary_path.open("w", encoding="utf-8") as handle:
                    json.dump(summary, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if not has_errors else 1
    except (OSError, RuntimeError, ValueError) as exc:
        print("エラー: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
