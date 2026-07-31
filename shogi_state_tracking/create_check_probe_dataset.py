#!/usr/bin/env python3
"""JSONLの各splitから，王手状態を線形probeで測るための均衡サブデータセットを作る。

モデル学習には使わない。各行は「開始局面＋そこまでの教師指手列」と，その直後に
手番側の玉が王手されているかを表す``in_check``ラベルからなる。
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, MutableSequence, Tuple

from create_dataset import import_cshogi
from data import parse_start_plies
from preprocess import materialize_segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JSONLから均衡化した王手probe用状態集合を抽出する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10000,
        help="全開始条件を合わせた王手／非王手それぞれの最大状態数。0なら全件を保存する",
    )
    parser.add_argument(
        "--max-prefix-moves",
        type=int,
        default=221,
        help="開始局面から使う最大指手数。checkpointのmax_seq_lenに合わせる",
    )
    parser.add_argument(
        "--start-plies",
        default="0,24,25,32,33",
        help="状態promptとして使う開始plyをcomma区切りで指定する",
    )
    parser.add_argument(
        "--min-suffix-moves",
        type=int,
        default=40,
        help="各開始局面の後に必要な最小指手数",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def reservoir_add(
    values: MutableSequence[Mapping[str, object]],
    item: Mapping[str, object],
    seen: int,
    limit: int,
    rng: random.Random,
) -> int:
    """ラベルごとの一様reservoir sampling。limit=0は全件保存。"""
    seen += 1
    if limit == 0:
        values.append(item)
    elif len(values) < limit:
        values.append(item)
    else:
        replacement = rng.randrange(seen)
        if replacement < limit:
            values[replacement] = item
    return seen


def position_scope(record: Mapping[str, object], ply: int) -> str:
    scopes = record.get("position_scope_by_ply", [])
    if isinstance(scopes, list) and ply < len(scopes):
        return str(scopes[ply])
    return str(record.get("position_scope", "unknown_position_scope"))


def make_state_record(
    record: Mapping[str, object],
    start_ply: int,
    prefix_moves: List[str],
    absolute_ply: int,
    in_check: bool,
    current_sfen: str,
    start_sfen: str,
    initial_tokens: List[str],
) -> Dict[str, object]:
    source_game_id = str(record.get("game_id", "unknown_game"))
    return {
        "schema_version": 1,
        "game_id": "{}:check-start-{}-ply-{}".format(
            source_game_id, start_ply, absolute_ply
        ),
        "source_game_id": source_game_id,
        "split": "{}_check_probe".format(record.get("split", "unknown")),
        "start_ply": start_ply,
        "probe_ply": absolute_ply,
        "in_check": bool(in_check),
        "initial_sfen": start_sfen,
        "initial_state_tokens": initial_tokens,
        # 後段ではこのprefixの最後の指手位置の表現を使う。
        "move_tokens": prefix_moves,
        "target_sfen": current_sfen,
        "player_scope": str(record.get("player_scope", record.get("engine_scope", ""))),
        "position_scope": position_scope(record, absolute_ply),
        "trajectory_scope": str(record.get("trajectory_scope", "unknown_position_scope")),
    }


def build_dataset(args: argparse.Namespace) -> Dict[str, object]:
    if (
        args.samples_per_class < 0
        or args.max_prefix_moves <= 0
        or args.min_suffix_moves <= 0
    ):
        raise ValueError("sampling and suffix limits must be positive")
    cshogi = import_cshogi()
    start_plies = parse_start_plies(args.start_plies)
    rng = random.Random(args.seed)
    # 条件ごとの王手頻度差が結果に混ざらないよう，開始plyごとに均衡化する。
    selected = {
        start_ply: {True: [], False: []} for start_ply in start_plies
    }
    seen = {start_ply: {True: 0, False: 0} for start_ply in start_plies}
    limit_per_start = (
        0
        if args.samples_per_class == 0
        else (args.samples_per_class + len(start_plies) - 1) // len(start_plies)
    )
    games = 0

    with Path(args.input_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            initial_sfen = str(record.get("initial_sfen", record.get("start_sfen", "")))
            if not initial_sfen:
                raise ValueError("{}:{} has no initial_sfen".format(args.input_jsonl, line_number))
            moves = [str(move) for move in record.get("move_tokens", [])]
            games += 1
            for start_ply in start_plies:
                if len(moves) - start_ply < args.min_suffix_moves:
                    continue
                segment = materialize_segment(
                    record,
                    start_ply=start_ply,
                    max_suffix_moves=args.max_prefix_moves,
                )
                board = cshogi.Board(str(segment["start_sfen"]))
                initial_tokens = list(segment["initial_state_tokens"])
                segment_moves = [str(move) for move in segment["move_tokens"]]
                # state_0は明示promptそのものなので除外する。
                for relative_ply, move_usi in enumerate(segment_moves, 1):
                    move = board.move_from_usi(move_usi)
                    if not board.is_legal(move):
                        raise ValueError(
                            "{}:{} illegal move at ply {}: {}".format(
                                args.input_jsonl,
                                line_number,
                                start_ply + relative_ply,
                                move_usi,
                            )
                        )
                    board.push(move)
                    label = bool(board.is_check())
                    item = make_state_record(
                        record,
                        start_ply,
                        segment_moves[:relative_ply],
                        start_ply + relative_ply,
                        label,
                        board.sfen(),
                        str(segment["start_sfen"]),
                        initial_tokens,
                    )
                    seen[start_ply][label] = reservoir_add(
                        selected[start_ply][label],
                        item,
                        seen[start_ply][label],
                        limit_per_start,
                        rng,
                    )

    # 正例不足時にも，ある開始条件の非王手だけが過剰に残らないようにする。
    counts_by_start = {}
    output_records = []
    for start_ply in start_plies:
        count = min(
            len(selected[start_ply][True]), len(selected[start_ply][False])
        )
        for label in (True, False):
            rng.shuffle(selected[start_ply][label])
            del selected[start_ply][label][count:]
            output_records.extend(selected[start_ply][label])
        counts_by_start[str(start_ply)] = {
            "candidate_in_check": seen[start_ply][True],
            "candidate_not_in_check": seen[start_ply][False],
            "selected_per_class": count,
        }
    rng.shuffle(output_records)

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in output_records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {
        "format_version": 1,
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(output),
        "games": games,
        "candidate_states_by_start_ply": counts_by_start,
        "selected_states_per_class": sum(
            value["selected_per_class"] for value in counts_by_start.values()
        ),
        "selected_states": len(output_records),
        "max_prefix_moves": args.max_prefix_moves,
        "start_plies": start_plies,
        "min_suffix_moves": args.min_suffix_moves,
        "seed": args.seed,
    }


def main() -> int:
    args = parse_args()
    summary = build_dataset(args)
    summary_path = Path(args.output_jsonl).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
