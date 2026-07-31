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

from create_dataset import encode_initial_state, import_cshogi


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
        help="王手／非王手それぞれの最大状態数。0なら全件を保存する",
    )
    parser.add_argument(
        "--max-prefix-moves",
        type=int,
        default=221,
        help="開始局面から使う最大指手数。checkpointのmax_seq_lenに合わせる",
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
    moves: List[str],
    ply: int,
    in_check: bool,
    current_sfen: str,
    initial_tokens: List[str],
) -> Dict[str, object]:
    source_game_id = str(record.get("game_id", "unknown_game"))
    return {
        "schema_version": 1,
        "game_id": "{}:check-ply-{}".format(source_game_id, ply),
        "source_game_id": source_game_id,
        "split": "{}_check_probe".format(record.get("split", "unknown")),
        "probe_ply": ply,
        "in_check": bool(in_check),
        "initial_sfen": str(record.get("initial_sfen", record.get("start_sfen", ""))),
        "initial_state_tokens": initial_tokens,
        # 後段ではこのprefixの最後の指手位置の表現を使う。
        "move_tokens": moves[:ply],
        "target_sfen": current_sfen,
        "player_scope": str(record.get("player_scope", record.get("engine_scope", ""))),
        "position_scope": position_scope(record, ply),
        "trajectory_scope": str(record.get("trajectory_scope", "unknown_position_scope")),
    }


def build_dataset(args: argparse.Namespace) -> Dict[str, object]:
    if args.samples_per_class < 0 or args.max_prefix_moves <= 0:
        raise ValueError("samples-per-class must be nonnegative and max-prefix-moves positive")
    cshogi = import_cshogi()
    rng = random.Random(args.seed)
    selected: Dict[bool, List[Mapping[str, object]]] = {True: [], False: []}
    seen = {True: 0, False: 0}
    games = 0

    with Path(args.input_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            initial_sfen = str(record.get("initial_sfen", record.get("start_sfen", "")))
            if not initial_sfen:
                raise ValueError("{}:{} has no initial_sfen".format(args.input_jsonl, line_number))
            board = cshogi.Board(initial_sfen)
            initial_tokens = list(record.get("initial_state_tokens", []))
            if len(initial_tokens) != 96:
                initial_tokens = encode_initial_state(board, cshogi)
            moves = [str(move) for move in record.get("move_tokens", [])]
            games += 1
            # state_0は明示promptそのものなので除外する。
            for ply, move_usi in enumerate(moves[: args.max_prefix_moves], 1):
                move = board.move_from_usi(move_usi)
                if not board.is_legal(move):
                    raise ValueError(
                        "{}:{} illegal move at ply {}: {}".format(
                            args.input_jsonl, line_number, ply, move_usi
                        )
                    )
                board.push(move)
                label = bool(board.is_check())
                item = make_state_record(
                    record, moves, ply, label, board.sfen(), initial_tokens
                )
                seen[label] = reservoir_add(
                    selected[label], item, seen[label], args.samples_per_class, rng
                )

    # 両クラスを同数へ揃える。正例不足時にも，非王手だけが過剰に残らないようにする。
    count = min(len(selected[True]), len(selected[False]))
    for label in (True, False):
        rng.shuffle(selected[label])
        del selected[label][count:]
    output_records = selected[True] + selected[False]
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
        "candidate_states": {"in_check": seen[True], "not_in_check": seen[False]},
        "selected_states_per_class": count,
        "selected_states": len(output_records),
        "max_prefix_moves": args.max_prefix_moves,
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
