#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将棋状態追跡実験用データセットをCSA棋譜から作成する。

モデルへ与える系列は、開始局面を表す固定長96トークン
（盤面81 + 持ち駒14 + 手番1）と、1手1トークンのUSI指し手列からなる。
途中局面は学習データへ出力しない。
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


REQUIRED_COLUMNS = {
    "file_path",
    "kif_index",
    "black_player",
    "white_player",
    "rating_b",
    "rating_w",
    "game_result",
    "total_moves",
}

MANIFEST_FIELDS = [
    "game_id",
    "game_date",
    "split",
    "engine_scope",
    "file_path",
    "kif_index",
    "black_player",
    "white_player",
    "rating_b",
    "rating_w",
    "game_result",
    "total_moves",
]

SPECIAL_TOKENS = [
    "<PAD>",
    "<BOS>",
    "<MOVES>",
    "<EOS>",
    "<THINK>",
    "<SEP>",
    "</THINK>",
    "<ANSWER>",
]
PIECE_NAMES = {
    1: "P",
    2: "L",
    3: "N",
    4: "S",
    5: "B",
    6: "R",
    7: "G",
    8: "K",
    9: "PP",
    10: "PL",
    11: "PN",
    12: "PS",
    13: "PB",
    14: "PR",
}
HAND_ORDER = ("P", "L", "N", "S", "G", "B", "R")
DATE_IN_PATH = re.compile(r"/(20[0-9]{2})/([0-9]{2})/([0-9]{2})/")
DATE_IN_FILENAME = re.compile(r"(20[0-9]{2})([0-9]{2})([0-9]{2})[0-9]{6}")


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def extract_game_date(row: Mapping[str, str]) -> dt.date:
    """メタデータ列またはCSAパスから対局日を得る。"""
    for column in ("game_date", "date"):
        value = row.get(column, "").strip()
        if value:
            return parse_date(value[:10])

    path = row["file_path"].replace("\\", "/")
    match = DATE_IN_PATH.search(path) or DATE_IN_FILENAME.search(path)
    if not match:
        raise ValueError("対局日をfile_pathから抽出できません: {}".format(row["file_path"]))
    return dt.date(*(int(part) for part in match.groups()))


def make_game_id(file_path: str, kif_index: int) -> str:
    source_key = "{}#{}".format(file_path, kif_index)
    return hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:20]


def read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError("metadata CSVに必要な列がありません: {}".format(", ".join(sorted(missing))))
        return list(reader)


def filter_metadata(
    rows: Iterable[Mapping[str, str]],
    min_date: dt.date,
    max_date: Optional[dt.date],
    min_rating: float,
    min_moves: int,
    include_draws: bool,
) -> Tuple[List[Dict[str, str]], Counter]:
    """現在の実験条件でmetadataを絞り込む。"""
    selected: List[Dict[str, str]] = []
    rejected = Counter()
    seen_keys: Set[Tuple[str, int]] = set()

    for source in rows:
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

        if game_date < min_date or (max_date is not None and game_date > max_date):
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

        row.update(
            {
                "game_id": make_game_id(row["file_path"], kif_index),
                "game_date": game_date.isoformat(),
                "kif_index": str(kif_index),
                "rating_b": str(rating_b),
                "rating_w": str(rating_w),
                "total_moves": str(total_moves),
                "game_result": str(game_result),
            }
        )
        selected.append(row)

    return selected, rejected


def assign_splits(
    rows: Sequence[Dict[str, str]],
    validation_from: dt.date,
    evaluation_from: dt.date,
) -> Dict[str, List[Dict[str, str]]]:
    if validation_from >= evaluation_from:
        raise ValueError("--validation-from は --evaluation-from より前でなければなりません")

    splits: Dict[str, List[Dict[str, str]]] = {
        "train": [],
        "validation": [],
        "evaluation": [],
    }
    for row in rows:
        game_date = parse_date(row["game_date"])
        if game_date < validation_from:
            split = "train"
        elif game_date < evaluation_from:
            split = "validation"
        else:
            split = "evaluation"
        row["split"] = split
        splits[split].append(row)

    train_engines = {
        engine
        for row in splits["train"]
        for engine in (row["black_player"], row["white_player"])
    }
    for split, split_rows in splits.items():
        for row in split_rows:
            if split == "train":
                scope = "train"
            else:
                seen_black = row["black_player"] in train_engines
                seen_white = row["white_player"] in train_engines
                if seen_black and seen_white:
                    scope = "open"
                elif seen_black or seen_white:
                    scope = "mixed"
                else:
                    scope = "closed"
            row["engine_scope"] = scope

    return splits


def deterministic_scope_sample(
    rows: Sequence[Dict[str, str]],
    total_games: int,
    seed: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, int]]]:
    """open/mixed/closedを均等化した再現可能な評価標本を作る。

    0以下を指定した場合は上限を設けない。乱数ライブラリの実装差を避けるため、
    seedとgame_idのSHA-1値で並べる。
    """
    scopes = ("open", "mixed", "closed")
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["engine_scope"]].append(row)

    if total_games > 0:
        base, remainder = divmod(total_games, len(scopes))
        quotas = {
            scope: base + (1 if index < remainder else 0)
            for index, scope in enumerate(scopes)
        }
    else:
        quotas = {scope: len(grouped.get(scope, [])) for scope in scopes}

    sampled: List[Dict[str, str]] = []
    summary: Dict[str, Dict[str, int]] = {}
    for scope in scopes:
        scope_rows = grouped.get(scope, [])
        quota = quotas[scope]
        if len(scope_rows) > quota:
            scope_rows = sorted(
                scope_rows,
                key=lambda row: hashlib.sha1(
                    "{}:{}".format(seed, row["game_id"]).encode("utf-8")
                ).hexdigest(),
            )[:quota]
        else:
            scope_rows = list(scope_rows)
        sampled.extend(scope_rows)
        summary[scope] = {
            "eligible": len(grouped.get(scope, [])),
            "selected": len(scope_rows),
        }

    sampled.sort(key=lambda row: (row["game_date"], row["game_id"]))
    return sampled, summary


def write_manifest(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_splits(
    source_count: int,
    eligible_count: int,
    splits: Mapping[str, Sequence[Mapping[str, str]]],
    rejected: Counter,
    evaluation_sampling: Mapping[str, Mapping[str, int]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    split_summary: Dict[str, object] = {}
    for split, rows in splits.items():
        scopes = Counter(row["engine_scope"] for row in rows)
        split_summary[split] = {
            "games": len(rows),
            "moves": sum(int(row["total_moves"]) for row in rows),
            "engine_scopes": dict(sorted(scopes.items())),
        }
    return {
        "source_games": source_count,
        "eligible_games_before_evaluation_sampling": eligible_count,
        "selected_games": sum(len(rows) for rows in splits.values()),
        "rejected": dict(sorted(rejected.items())),
        "filters": {
            "min_date": args.min_date,
            "max_date": args.max_date,
            "min_rating": args.min_rating,
            "min_moves": args.min_moves,
            "include_draws": args.include_draws,
            "validation_from": args.validation_from,
            "evaluation_from": args.evaluation_from,
            "evaluation_games": args.evaluation_games,
            "sampling_seed": args.sampling_seed,
        },
        "evaluation_sampling": evaluation_sampling,
        "splits": split_summary,
    }


def split_dataset(args: argparse.Namespace) -> Dict[str, object]:
    metadata_path = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    rows = read_metadata(metadata_path)
    selected, rejected = filter_metadata(
        rows,
        min_date=parse_date(args.min_date),
        max_date=parse_date(args.max_date) if args.max_date else None,
        min_rating=args.min_rating,
        min_moves=args.min_moves,
        include_draws=args.include_draws,
    )
    splits = assign_splits(
        selected,
        validation_from=parse_date(args.validation_from),
        evaluation_from=parse_date(args.evaluation_from),
    )
    eligible_count = sum(len(split_rows) for split_rows in splits.values())
    eligible_evaluation = list(splits["evaluation"])
    sampled_evaluation, evaluation_sampling = deterministic_scope_sample(
        eligible_evaluation,
        total_games=args.evaluation_games,
        seed=args.sampling_seed,
    )
    splits["evaluation"] = sampled_evaluation

    manifests_dir = output_dir / "manifests"
    write_manifest(manifests_dir / "evaluation_eligible.csv", eligible_evaluation)
    for split, split_rows in splits.items():
        write_manifest(manifests_dir / "{}.csv".format(split), split_rows)
        if split != "train":
            for scope in ("open", "mixed", "closed"):
                scope_rows = [row for row in split_rows if row["engine_scope"] == scope]
                write_manifest(
                    manifests_dir / "{}_{}.csv".format(split, scope),
                    scope_rows,
                )

    summary = summarize_splits(
        len(rows),
        eligible_count,
        splits,
        rejected,
        evaluation_sampling,
        args,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("manifestに必要な列がありません: {}".format(", ".join(sorted(missing))))
        return list(reader)


def import_cshogi():
    try:
        import cshogi  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "exportにはcshogiが必要です。wsl2/requirements.txt相当の環境で実行してください。"
        ) from exc
    return cshogi


def encode_initial_state(board, cshogi_module) -> List[str]:
    """cshogiの1a,1b,...,9i順で固定長96トークンを作る。"""
    tokens: List[str] = []
    for piece in board.pieces:
        if piece == 0:
            tokens.append("SQ_EMPTY")
            continue
        color = "W" if piece >= 16 else "B"
        piece_type = piece % 16
        tokens.append("SQ_{}_{}".format(color, PIECE_NAMES[piece_type]))

    for color_index in (cshogi_module.BLACK, cshogi_module.WHITE):
        for count in board.pieces_in_hand[color_index]:
            tokens.append("HAND_{}".format(int(count)))

    tokens.append(
        "TURN_BLACK" if board.turn == cshogi_module.BLACK else "TURN_WHITE"
    )
    if len(tokens) != 96:
        raise AssertionError("開始局面トークン数が96ではありません: {}".format(len(tokens)))
    return tokens


def remap_path(path: str, prefix_from: Optional[str], prefix_to: Optional[str]) -> Path:
    if bool(prefix_from) != bool(prefix_to):
        raise ValueError("--path-prefix-from と --path-prefix-to は同時に指定してください")
    if not prefix_from:
        return Path(path)

    normalized_path = path.replace("\\", "/")
    normalized_from = str(prefix_from).replace("\\", "/").rstrip("/")
    if not (
        normalized_path == normalized_from
        or normalized_path.startswith(normalized_from + "/")
    ):
        return Path(path)
    suffix = normalized_path[len(normalized_from) :].lstrip("/")
    return Path(str(prefix_to)) / suffix


def build_record(row: Mapping[str, str], game, cshogi_module) -> Dict[str, object]:
    board = cshogi_module.Board(game.sfen)
    initial_tokens = encode_initial_state(board, cshogi_module)
    moves: List[str] = []
    for ply, move in enumerate(game.moves, 1):
        if not board.is_legal(move):
            raise ValueError("第{}手が開始局面から合法に再生できません".format(ply))
        moves.append(cshogi_module.move_to_usi(move))
        board.push(move)

    expected_moves = int(row["total_moves"])
    if len(moves) != expected_moves:
        raise ValueError(
            "metadataの手数{}とCSAの手数{}が一致しません".format(expected_moves, len(moves))
        )

    return {
        "schema_version": 1,
        "game_id": row["game_id"],
        "split": row["split"],
        "engine_scope": row["engine_scope"],
        "game_date": row["game_date"],
        "initial_sfen": game.sfen,
        "initial_state_tokens": initial_tokens,
        "move_tokens": moves,
        "black_player": row["black_player"],
        "white_player": row["white_player"],
        "rating_b": float(row["rating_b"]),
        "rating_w": float(row["rating_w"]),
        "game_result": int(row["game_result"]),
    }


def base_vocabulary() -> List[str]:
    tokens = list(SPECIAL_TOKENS)
    tokens.append("SQ_EMPTY")
    for color in ("B", "W"):
        for piece_type in range(1, 15):
            tokens.append("SQ_{}_{}".format(color, PIECE_NAMES[piece_type]))
    tokens.extend("HAND_{}".format(count) for count in range(19))
    tokens.extend(("TURN_BLACK", "TURN_WHITE"))
    return tokens


def all_usi_move_tokens() -> List[str]:
    """任意局面の合法手を必ず表現できる固定USI指手語彙を返す。

    from/toだけでは駒種が分からないため、通常移動は盤面上の異なる2マスの全組合せを
    含める。`+`付きも同様に含め、駒打ちは7駒種と81マスの全組合せを含める。
    局面によって不可能なトークンも語彙には入るが、合法性は局面側で判定する。
    """
    squares = [
        "{}{}".format(file_index, rank)
        for file_index in range(1, 10)
        for rank in "abcdefghi"
    ]
    moves: List[str] = []
    for source in squares:
        for destination in squares:
            if source == destination:
                continue
            moves.append(source + destination)
            moves.append(source + destination + "+")
    for piece in HAND_ORDER:
        moves.extend("{}*{}".format(piece, square) for square in squares)
    return moves


def export_manifest(
    manifest_path: Path,
    output_path: Path,
    errors_path: Path,
    prefix_from: Optional[str],
    prefix_to: Optional[str],
    strict: bool,
    limit: Optional[int],
) -> Dict[str, object]:
    cshogi = import_cshogi()
    rows = load_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["file_path"]].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    move_vocabulary: Set[str] = set()
    written_games = 0
    written_moves = 0
    error_rows: List[Dict[str, str]] = []

    with output_path.open("w", encoding="utf-8") as output_handle:
        for source_path, file_rows in grouped.items():
            resolved_path = remap_path(source_path, prefix_from, prefix_to)
            try:
                games = cshogi.Parser.parse_file(str(resolved_path))
                if games is None:
                    raise ValueError("CSAパーサが棋譜を返しませんでした")
            except Exception as exc:
                if strict:
                    raise
                for row in file_rows:
                    error_rows.append(
                        {
                            "game_id": row["game_id"],
                            "file_path": source_path,
                            "kif_index": row["kif_index"],
                            "error": str(exc),
                        }
                    )
                continue

            for row in file_rows:
                try:
                    index = int(row["kif_index"])
                    if index < 0 or index >= len(games):
                        raise IndexError("kif_index {} が範囲外です".format(index))
                    record = build_record(row, games[index], cshogi)
                    json.dump(record, output_handle, ensure_ascii=False, separators=(",", ":"))
                    output_handle.write("\n")
                    move_vocabulary.update(record["move_tokens"])
                    written_games += 1
                    written_moves += len(record["move_tokens"])
                except Exception as exc:
                    if strict:
                        raise
                    error_rows.append(
                        {
                            "game_id": row["game_id"],
                            "file_path": source_path,
                            "kif_index": row["kif_index"],
                            "error": str(exc),
                        }
                    )

    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["game_id", "file_path", "kif_index", "error"]
        )
        writer.writeheader()
        writer.writerows(error_rows)

    return {
        "manifest": str(manifest_path),
        "output": str(output_path),
        "requested_games": len(rows),
        "written_games": written_games,
        "written_moves": written_moves,
        "errors": len(error_rows),
        "move_vocabulary": sorted(move_vocabulary),
    }


def write_vocabulary(path: Path, move_tokens: Iterable[str]) -> None:
    fixed_moves = all_usi_move_tokens()
    observed_moves = set(move_tokens)
    unsupported = observed_moves - set(fixed_moves)
    if unsupported:
        raise ValueError(
            "固定USI語彙で表現できない指手があります: {}".format(
                ", ".join(sorted(unsupported)[:10])
            )
        )
    tokens = base_vocabulary() + fixed_moves
    vocabulary = {
        "schema_version": 3,
        "token_to_id": {token: index for index, token in enumerate(tokens)},
        "special_tokens": SPECIAL_TOKENS,
        "state_token_count": 96,
        "board_order": "1a,1b,...,1i,2a,...,9i",
        "hand_order": [
            "{}_{}".format(color, piece)
            for color in ("BLACK", "WHITE")
            for piece in HAND_ORDER
        ],
        "move_encoding": "one USI move per token",
        "move_vocabulary": "fixed syntactic superset of all legal USI moves",
        "observed_move_count": len(observed_moves),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(vocabulary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def export_dataset(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    manifests_dir = output_dir / "manifests"
    datasets_dir = output_dir / "datasets"
    errors_dir = output_dir / "errors"
    all_move_tokens: Set[str] = set()
    summaries: Dict[str, object] = {}

    for split in ("train", "validation", "evaluation"):
        summary = export_manifest(
            manifest_path=manifests_dir / "{}.csv".format(split),
            output_path=datasets_dir / "{}.jsonl".format(split),
            errors_path=errors_dir / "{}.csv".format(split),
            prefix_from=args.path_prefix_from,
            prefix_to=args.path_prefix_to,
            strict=args.strict,
            limit=args.limit,
        )
        all_move_tokens.update(summary.pop("move_vocabulary"))
        summaries[split] = summary

    write_vocabulary(output_dir / "vocab.json", all_move_tokens)
    result = {
        "splits": summaries,
        "observed_move_vocabulary_size": len(all_move_tokens),
        "fixed_move_vocabulary_size": len(all_usi_move_tokens()),
    }
    with (output_dir / "export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def print_summary(summary: Mapping[str, object]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def add_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata-csv", required=True, help="CSA一覧を含むmetadata.csv")
    parser.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--min-date", default="2022-01-01")
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--min-rating", type=float, default=3000.0)
    parser.add_argument("--min-moves", type=int, default=80)
    parser.add_argument(
        "--include-draws",
        action="store_true",
        help="指定しない場合はgame_result=0を除外する",
    )
    parser.add_argument("--validation-from", default="2024-10-01")
    parser.add_argument("--evaluation-from", default="2025-01-01")
    parser.add_argument(
        "--evaluation-games",
        type=int,
        default=5000,
        help="open/mixed/closedを均等化した評価対局数の合計。0は上限なし",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=20260724,
        help="評価集合の決定論的抽出に用いるseed",
    )


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True, help="split済みmanifestを含む出力先")
    parser.add_argument("--path-prefix-from", default=None, help="metadata内CSAパスの置換元")
    parser.add_argument("--path-prefix-to", default=None, help="実環境でのCSAルート")
    parser.add_argument("--strict", action="store_true", help="1件でも変換失敗したら停止する")
    parser.add_argument("--limit", type=int, default=None, help="各splitの先頭N局だけ変換する")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="将棋状態追跡Transformer用データセットを作成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="metadataを抽出・時系列分割する")
    add_split_arguments(split_parser)

    export_parser = subparsers.add_parser("export", help="manifest中のCSAをJSONLへ変換する")
    add_export_arguments(export_parser)

    build_parser = subparsers.add_parser("build", help="splitとexportを連続実行する")
    add_split_arguments(build_parser)
    build_parser.add_argument("--path-prefix-from", default=None)
    build_parser.add_argument("--path-prefix-to", default=None)
    build_parser.add_argument("--strict", action="store_true")
    build_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "split":
            print_summary(split_dataset(args))
        elif args.command == "export":
            print_summary(export_dataset(args))
        else:
            split_summary = split_dataset(args)
            export_summary = export_dataset(args)
            print_summary({"split": split_summary, "export": export_summary})
    except (OSError, RuntimeError, ValueError) as exc:
        print("エラー: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
