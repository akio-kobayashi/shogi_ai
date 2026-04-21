# -*- coding: utf-8 -*-
"""
build-h5 で生成した HDF5 データセットの構造と内容を検査するスクリプト。

主な確認内容:
- groups / dataset 構造
- positions dtype と candidates dtype
- actual_move の保存状況
- candidates の探索条件メタ
- 必要に応じた packed sfen からの局面復元と合法手確認
"""

import argparse
import sys
from pathlib import Path

import cshogi
import h5py
import numpy as np


EXPECTED_POSITION_FIELDS = {
    "ply",
    "psv",
    "actual_move",
    "is_check",
    "features",
    "candidates",
}

EXPECTED_CANDIDATE_FIELDS = {
    "search_depth",
    "search_nodes",
    "search_movetime",
    "multipv",
    "move",
    "score",
    "is_mate",
}


def _print(msg: str) -> None:
    print(msg)


def _fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def packed_sfen_field_view(record: np.void) -> np.ndarray:
    """構造化レコードの `psv` フィールドを PackedSfenValue の 1 要素配列に直す。"""
    return np.asarray(record["psv"], dtype=cshogi.PackedSfenValue).reshape(1)


def inspect_schema(h5_path: Path) -> bool:
    ok = True
    with h5py.File(h5_path, "r") as f:
        group_names = sorted(f.keys())
        _print(f"groups: {len(group_names)}")
        if not group_names:
            _fail("HDF5 に game group がありません。")
            return False

        first_group = f[group_names[0]]
        if "positions" not in first_group:
            _fail("最初の game group に positions dataset がありません。")
            return False

        positions = first_group["positions"]
        position_fields = set(positions.dtype.names or [])
        _print(f"position fields: {sorted(position_fields)}")

        missing_position = EXPECTED_POSITION_FIELDS - position_fields
        if missing_position:
            _fail(f"positions に必要なフィールドが足りません: {sorted(missing_position)}")
            ok = False

        candidate_dtype = positions.dtype["candidates"].metadata.get("vlen") if positions.dtype["candidates"].metadata else None
        if candidate_dtype is None or candidate_dtype.names is None:
            _fail("candidates が vlen structured dtype ではありません。")
            return False

        candidate_fields = set(candidate_dtype.names)
        _print(f"candidate fields: {sorted(candidate_fields)}")

        missing_candidate = EXPECTED_CANDIDATE_FIELDS - candidate_fields
        if missing_candidate:
            _fail(f"candidates に必要なフィールドが足りません: {sorted(missing_candidate)}")
            ok = False

    return ok


def inspect_samples(h5_path: Path, max_games: int, max_positions: int, check_moves: bool) -> bool:
    ok = True
    checked_positions = 0
    non_empty_candidate_positions = 0
    search_depth_values = set()

    with h5py.File(h5_path, "r") as f:
        board = cshogi.Board()
        for game_index, game_name in enumerate(sorted(f.keys())):
            if game_index >= max_games:
                break

            game_group = f[game_name]
            attrs = dict(game_group.attrs.items())
            _print(f"\n[{game_name}] attrs keys: {sorted(attrs.keys())}")

            if "positions" not in game_group:
                _fail(f"{game_name} に positions dataset がありません。")
                ok = False
                continue

            positions = game_group["positions"]
            limit = min(len(positions), max_positions)
            _print(f"{game_name}: positions={len(positions)}, checking={limit}")

            for pos_index in range(limit):
                pos = positions[pos_index]
                checked_positions += 1

                ply = int(pos["ply"])
                actual_move = int(pos["actual_move"])
                candidates = pos["candidates"]

                _print(
                    f"  pos[{pos_index}] ply={ply} actual_move={actual_move} "
                    f"candidates={len(candidates)} is_check={bool(pos['is_check'])}"
                )

                if actual_move == 0:
                    _fail(f"{game_name} pos[{pos_index}] の actual_move が 0 です。")
                    ok = False

                if len(candidates) > 0:
                    non_empty_candidate_positions += 1
                    for cand in candidates:
                        search_depth_values.add(int(cand["search_depth"]))

                    preview = candidates[: min(3, len(candidates))]
                    for cand in preview:
                        _print(
                            "    "
                            f"depth={int(cand['search_depth'])} "
                            f"nodes={int(cand['search_nodes'])} "
                            f"movetime={int(cand['search_movetime'])} "
                            f"multipv={int(cand['multipv'])} "
                            f"move={int(cand['move'])} "
                            f"score={int(cand['score'])} "
                            f"is_mate={bool(cand['is_mate'])}"
                        )
                else:
                    _fail(f"{game_name} pos[{pos_index}] の candidates が空です。")
                    ok = False

                if check_moves:
                    try:
                        board.set_psfen(packed_sfen_field_view(pos))
                    except Exception as exc:
                        _fail(f"{game_name} pos[{pos_index}] の psv 復元に失敗しました: {exc}")
                        ok = False
                        continue

                    actual_legal = board.is_legal(actual_move)
                    if not actual_legal:
                        _fail(f"{game_name} pos[{pos_index}] の actual_move が合法手ではありません。 move={actual_move}")
                        ok = False

                    if len(candidates) > 0:
                        best_move = int(candidates[0]["move"])
                        best_legal = board.is_legal(best_move)
                        if not best_legal:
                            _fail(f"{game_name} pos[{pos_index}] の先頭 candidate が合法手ではありません。 move={best_move}")
                            ok = False

    _print("\nsummary:")
    _print(f"  checked_positions={checked_positions}")
    _print(f"  non_empty_candidate_positions={non_empty_candidate_positions}")
    _print(f"  observed_search_depths={sorted(search_depth_values)}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="build-h5 で生成した HDF5 データセットを検査する。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_h5", help="検査対象の HDF5 ファイル")
    parser.add_argument("--max-games", type=int, default=2, help="検査する game group 数")
    parser.add_argument("--max-positions", type=int, default=5, help="各 game group で検査する局面数")
    parser.add_argument("--check-moves", action="store_true", help="psv から局面を復元し、actual_move と先頭 candidate の合法性を確認する")
    args = parser.parse_args()

    h5_path = Path(args.input_h5)
    if not h5_path.exists():
        sys.exit(f"エラー: 指定された HDF5 ファイルが見つかりません: {h5_path}")

    schema_ok = inspect_schema(h5_path)
    sample_ok = inspect_samples(h5_path, args.max_games, args.max_positions, args.check_moves)

    if not (schema_ok and sample_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
