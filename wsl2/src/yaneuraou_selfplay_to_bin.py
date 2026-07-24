#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YaneuraOu の自己対局 .sfen/jsonl から、create_dataset.py 互換の
PackedSfenValue .bin を生成する。

デフォルトでは YaneuraOu/selfplay/*.sfen を読み込み、静止局面だけを
shogi_ai/wsl2/output_data_selfplay/yaneuraou_sfen_static/{train,val}.bin
へ出力する。

使い方例:
  python yaneuraou_selfplay_to_bin.py

  python yaneuraou_selfplay_to_bin.py \
    --input-pattern /home/akio/GitHub/YaneuraOu/selfplay/*.sfen \
    --output-dir /home/akio/GitHub/shogi_ai/wsl2/output_data_selfplay/static \
    --quiet-level 3 \
    --min-ply 20

  python yaneuraou_selfplay_to_bin.py \
    --input-pattern /home/akio/GitHub/YaneuraOu/selfplay/20260523202833T16_n200000.jsonl \
    --input-format jsonl \
    --output /home/akio/GitHub/shogi_ai/wsl2/output_data_selfplay/20260523202833T16_n200000.bin
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[3]
LOCAL_CSHOGI_REPO = WORKSPACE_ROOT / "cshogi"
if LOCAL_CSHOGI_REPO.exists():
    sys.path.insert(0, str(LOCAL_CSHOGI_REPO))

import cshogi

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_INPUT_PATTERN = str(WORKSPACE_ROOT / "YaneuraOu" / "selfplay" / "*.sfen")
DEFAULT_OUTPUT_DIR = str(
    WORKSPACE_ROOT / "shogi_ai" / "wsl2" / "output_data_selfplay" / "yaneuraou_sfen_static"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YaneuraOu selfplay .sfen/jsonl を PackedSfenValue .bin へ変換する。"
    )
    parser.add_argument(
        "--input-pattern",
        default=DEFAULT_INPUT_PATTERN,
        help="入力 .sfen/jsonl を探す glob。省略時は YaneuraOu/selfplay/*.sfen。",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "sfen", "jsonl"],
        default="auto",
        help="入力形式。auto では拡張子で判定する。",
    )
    parser.add_argument(
        "--output",
        help="単一の .bin 出力先。--output-dir とは同時指定不可。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="train.bin / val.bin を出力するディレクトリ。--output とは同時指定不可。",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="--output-dir 指定時の検証データ比率。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="分割用乱数シード。",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=1,
        help="採用する最小手数。",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=999,
        help="採用する最大手数。",
    )
    parser.add_argument(
        "--quiet-level",
        choices=["none", "1", "2", "3"],
        default="3",
        help=(
            "静止局面フィルタ。none=無効, 1=終局/王手/反復除外, "
            "2=1手詰めも除外, 3=得な取り・王手候補・成り筋・玉周辺の危険も除外。"
        ),
    )
    parser.add_argument(
        "--score-clip",
        type=int,
        default=32000,
        help="評価値の絶対値クリップ。0 以下なら int16 範囲外を除外する。",
    )
    args = parser.parse_args()

    if args.output and args.output_dir != DEFAULT_OUTPUT_DIR:
        parser.error("--output と --output-dir は同時指定できません。")
    if args.output:
        args.output_dir = None
    if not (0.0 <= args.val_split <= 1.0):
        parser.error("--val-split は 0.0 以上 1.0 以下で指定してください。")
    if args.min_ply <= 0:
        parser.error("--min-ply は 1 以上で指定してください。")
    if args.min_ply > args.max_ply:
        parser.error("--min-ply は --max-ply 以下である必要があります。")
    return args


def resolve_input_paths(input_pattern: str, input_format: str) -> list[Path]:
    raw_matches = sorted(glob.glob(input_pattern))
    if not raw_matches:
        for suffix in (".sfen", ".jsonl"):
            if input_pattern.endswith(suffix):
                continue
            raw_matches = sorted(glob.glob(f"{input_pattern}{suffix}"))
            if raw_matches:
                break

    allowed_suffixes = {".sfen", ".jsonl"} if input_format == "auto" else {f".{input_format}"}
    paths = [Path(p) for p in raw_matches if Path(p).suffix in allowed_suffixes]
    if not paths:
        sys.exit(f"エラー: 入力ファイルが見つかりません: {input_pattern}")
    return paths


def set_initial_position(board: cshogi.Board, initial_position: str) -> None:
    initial_position = initial_position.strip()
    if initial_position == "startpos":
        board.reset()
        return
    if initial_position.startswith("sfen "):
        board.set_sfen(initial_position[5:])
        return
    board.set_sfen(initial_position)


def parse_game_result(result_label: str) -> int:
    if result_label == "P1_WIN":
        return 1
    if result_label == "P2_WIN":
        return 2
    return 0


QUIET_POSITION_PIECE_VALUES = {
    cshogi.PAWN: 100,
    cshogi.LANCE: 300,
    cshogi.KNIGHT: 320,
    cshogi.SILVER: 480,
    cshogi.GOLD: 520,
    cshogi.BISHOP: 850,
    cshogi.ROOK: 950,
    cshogi.KING: 15000,
    cshogi.PROM_PAWN: 420,
    cshogi.PROM_LANCE: 400,
    cshogi.PROM_KNIGHT: 420,
    cshogi.PROM_SILVER: 500,
    cshogi.PROM_BISHOP: 950,
    cshogi.PROM_ROOK: 1150,
}


def _piece_value(piece_type: int) -> int:
    return QUIET_POSITION_PIECE_VALUES.get(piece_type, 0)


def _least_attacker_value(board: cshogi.Board, color: int, sq: int):
    least_value = None
    for attacker_sq in board.attackers_to(color, sq):
        piece_value = _piece_value(board.piece_type(attacker_sq))
        if piece_value <= 0:
            continue
        if least_value is None or piece_value < least_value:
            least_value = piece_value
    return least_value


def _has_see_like_capture(board: cshogi.Board) -> bool:
    side_to_move = board.turn
    opponent = 1 - side_to_move

    for move in board.legal_moves:
        captured_piece_type = cshogi.move_cap(move)
        if captured_piece_type == 0:
            continue

        to_sq = cshogi.move_to(move)
        captured_value = _piece_value(captured_piece_type)
        if captured_value <= 0:
            continue

        board.push(move)
        try:
            occupied_value = _piece_value(board.piece_type(to_sq))
            opp_recapture_value = _least_attacker_value(board, board.turn, to_sq)
            if opp_recapture_value is None:
                return True
            if captured_value >= occupied_value:
                return True

            our_rerecapture_value = _least_attacker_value(board, opponent, to_sq)
            if our_rerecapture_value is None:
                continue
            if captured_value + our_rerecapture_value >= occupied_value + opp_recapture_value:
                return True
        finally:
            board.pop()

    return False


def _count_king_escape_routes(board: cshogi.Board, king_sq: int) -> int:
    king_escape_routes = 0
    y, x = divmod(king_sq, 9)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < 9 and 0 <= nx < 9:
                move = board.move(king_sq, ny * 9 + nx, False)
                if board.is_legal(move):
                    king_escape_routes += 1
    return king_escape_routes


def _king_zone_squares(king_sq: int):
    squares = {king_sq}
    y, x = divmod(king_sq, 9)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < 9 and 0 <= nx < 9:
                squares.add(ny * 9 + nx)
    return squares


def _analyze_board_tactical_state(board: cshogi.Board, include_king_safety: bool = False) -> dict:
    capture_moves = 0
    check_moves = 0
    promotion_moves = 0
    legal_moves = 0
    king_zone_tactical_moves = 0

    opp_king_sq = board.king_square(1 - board.turn)
    opp_king_zone = _king_zone_squares(opp_king_sq)

    for move in board.legal_moves:
        legal_moves += 1
        to_sq = cshogi.move_to(move)
        is_capture = cshogi.move_cap(move) != 0
        is_promotion = cshogi.move_is_promotion(move)

        if is_capture:
            capture_moves += 1
        if is_promotion:
            promotion_moves += 1
        if to_sq in opp_king_zone and (is_capture or is_promotion):
            king_zone_tactical_moves += 1

        board.push(move)
        if board.is_check():
            check_moves += 1
            if to_sq in opp_king_zone:
                king_zone_tactical_moves += 1
        board.pop()

    result = {
        "legal_moves": legal_moves,
        "capture_moves": capture_moves,
        "check_moves": check_moves,
        "promotion_moves": promotion_moves,
        "king_zone_tactical_moves": king_zone_tactical_moves,
    }

    if include_king_safety:
        my_king_sq = board.king_square(board.turn)
        my_king_zone = _king_zone_squares(my_king_sq)
        result["my_king_attackers"] = 1 if board.is_check() else 0
        result["king_escape_routes"] = _count_king_escape_routes(board, my_king_sq)
        result["opp_king_zone_pressure"] = sum(board.attackers_to_count(board.turn, sq) for sq in opp_king_zone)
        result["my_king_zone_pressure"] = sum(board.attackers_to_count(1 - board.turn, sq) for sq in my_king_zone)

    return result


def _classify_quiet_rejection_reason(board: cshogi.Board, quiet_level: str):
    if quiet_level == "none":
        return None
    if board.is_game_over():
        return "game_over"
    if board.is_nyugyoku():
        return "nyugyoku"
    if board.is_draw() != cshogi.NOT_REPETITION:
        return "draw"
    if board.is_check():
        return "in_check"
    if quiet_level == "1":
        return None

    analysis = _analyze_board_tactical_state(board, include_king_safety=(quiet_level == "3"))
    if board.mate_move_in_1ply():
        return "mate_in_1_available"
    if quiet_level == "2":
        return None

    if _has_see_like_capture(board):
        return "favorable_capture_available"
    if analysis["check_moves"] > 0:
        return "checking_move_available"
    if analysis["promotion_moves"] > 0:
        return "promotion_tactic_available"
    if analysis["king_zone_tactical_moves"] > 0:
        return "king_zone_tactic_available"
    if analysis["my_king_attackers"] > 0:
        return "king_under_attack"
    if analysis["opp_king_zone_pressure"] >= 3:
        return "opp_king_zone_pressure_high"
    if analysis["my_king_zone_pressure"] >= 4:
        return "my_king_zone_pressure_high"
    if analysis["king_escape_routes"] <= 1:
        return "few_king_escape_routes"

    return None


def _is_quiet_position(board: cshogi.Board, quiet_level: str) -> bool:
    return _classify_quiet_rejection_reason(board, quiet_level) is None


class PackedSfenWriter:
    def __init__(self, output_path: Path, shuffle_on_close: bool = False, seed: int = 0, score_clip: int = 32000):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.shuffle_on_close = shuffle_on_close
        self.seed = seed
        self.score_clip = score_clip
        self.board = cshogi.Board()
        self.psv = np.zeros(1, dtype=cshogi.PackedSfenValue)
        self.int16_info = np.iinfo(np.int16)
        self.rows_written = 0
        self.skipped_out_of_range_scores = 0
        self.clipped_scores = 0
        self.write_path = output_path.with_suffix(output_path.suffix + ".tmp") if shuffle_on_close else output_path
        self.f_out = self.write_path.open("wb")

    def write(self, sfen: str, ply: int, eval_score_cp: int, game_result: int) -> bool:
        if self.score_clip > 0 and abs(eval_score_cp) > self.score_clip:
            eval_score_cp = self.score_clip if eval_score_cp > 0 else -self.score_clip
            self.clipped_scores += 1

        if not (self.int16_info.min <= eval_score_cp <= self.int16_info.max):
            self.skipped_out_of_range_scores += 1
            return False

        self.board.set_sfen(sfen)
        self.board.to_psfen(self.psv)
        self.psv[0]["score"] = np.int16(eval_score_cp)
        self.psv[0]["move"] = np.uint16(0)
        self.psv[0]["gamePly"] = np.uint16(ply)
        self.psv[0]["game_result"] = np.int8(1 if game_result == 1 else -1 if game_result == 2 else 0)
        self.psv.tofile(self.f_out)
        self.rows_written += 1
        return True

    def close(self) -> None:
        self.f_out.close()
        if self.shuffle_on_close:
            self._shuffle_written_records()
            os.replace(self.write_path, self.output_path)

    def _shuffle_written_records(self) -> None:
        if self.rows_written <= 1:
            return

        rng = random.Random(self.seed)
        records = np.memmap(
            self.write_path,
            dtype=cshogi.PackedSfenValue,
            mode="r+",
            shape=(self.rows_written,),
        )
        for idx in range(self.rows_written - 1, 0, -1):
            swap_idx = rng.randrange(idx + 1)
            if swap_idx == idx:
                continue
            tmp = np.array(records[idx], copy=True)
            records[idx] = records[swap_idx]
            records[swap_idx] = tmp
        records.flush()
        del records


def iter_jsonl_records(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"JSON 解析エラー: {path}:{line_no} ({exc})", file=sys.stderr)


def _parse_position_line(position_line: str) -> tuple[str, list[str]]:
    parts = position_line.split()
    if not parts:
        raise ValueError("空の position 行です")

    if parts[0] == "startpos":
        if len(parts) == 1:
            return "startpos", []
        if len(parts) >= 2 and parts[1] == "moves":
            return "startpos", parts[2:]
        raise ValueError(f"startpos 行を解釈できません: {position_line[:120]}")

    if parts[0] == "sfen":
        if "moves" in parts:
            moves_index = parts.index("moves")
            return " ".join(parts[1:moves_index]), parts[moves_index + 1:]
        return " ".join(parts[1:]), []

    raise ValueError(f"position 行を解釈できません: {position_line[:120]}")


def iter_sfen_games(paths: Iterable[Path]) -> Iterator[tuple[Path, int, str, list[str], list[int]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            line_no = 0
            while True:
                position_line = f.readline()
                if not position_line:
                    break
                line_no += 1
                position_line = position_line.strip()
                if not position_line:
                    continue

                eval_line = f.readline()
                if not eval_line:
                    print(f"評価値行がありません: {path}:{line_no}", file=sys.stderr)
                    break
                line_no += 1
                eval_line = eval_line.strip()

                try:
                    initial_position, moves = _parse_position_line(position_line)
                    eval_values = [int(token) for token in eval_line.split()]
                except ValueError as exc:
                    print(f"sfen 解析エラー: {path}:{line_no - 1} ({exc})", file=sys.stderr)
                    continue

                yield path, line_no - 1, initial_position, moves, eval_values


def _select_writer(writers: list[PackedSfenWriter], rng: random.Random, val_split: float) -> PackedSfenWriter:
    if len(writers) == 2 and rng.random() < val_split:
        return writers[1]
    return writers[0]


def _should_keep_position(board: cshogi.Board, quiet_level: str) -> bool:
    if quiet_level == "none":
        return True
    return _is_quiet_position(board, quiet_level)


def _new_stats() -> dict:
    return {
        "games": 0,
        "skipped_games": 0,
        "positions_seen": 0,
        "positions_written": 0,
        "skipped_positions": 0,
        "mismatched_lengths": 0,
    }


def convert_sfen_files(args: argparse.Namespace, input_paths: list[Path], writers: list[PackedSfenWriter]) -> dict:
    rng = random.Random(args.seed)
    board = cshogi.Board()
    stats = _new_stats()

    for path, line_no, initial_position, moves, eval_values in tqdm(iter_sfen_games(input_paths), desc="Converting sfen games"):
        stats["games"] += 1
        if not moves or not eval_values:
            stats["skipped_games"] += 1
            continue
        if len(moves) != len(eval_values):
            stats["mismatched_lengths"] += 1

        try:
            set_initial_position(board, initial_position)
        except Exception as exc:
            print(f"初期局面の設定に失敗: {path}:{line_no} ({exc})", file=sys.stderr)
            stats["skipped_games"] += 1
            continue

        usable_len = min(len(moves), len(eval_values))
        for ply_index in range(usable_len):
            ply = ply_index + 1
            stats["positions_seen"] += 1

            try:
                eval_score_cp = int(eval_values[ply_index])
                current_sfen = board.sfen()
            except Exception as exc:
                print(f"局面/評価値取得エラー: {path}:{line_no} ply={ply} ({exc})", file=sys.stderr)
                stats["skipped_positions"] += 1
                break

            if args.min_ply <= ply <= args.max_ply and _should_keep_position(board, args.quiet_level):
                writer = _select_writer(writers, rng, args.val_split)
                if writer.write(current_sfen, ply, eval_score_cp, 0):
                    stats["positions_written"] += 1
                else:
                    stats["skipped_positions"] += 1
            else:
                stats["skipped_positions"] += 1

            try:
                board.push_usi(moves[ply_index])
            except Exception as exc:
                print(f"指し手適用エラー: {path}:{line_no} ply={ply} move={moves[ply_index]} ({exc})", file=sys.stderr)
                stats["skipped_positions"] += usable_len - ply
                break

    return stats


def convert_jsonl_files(args: argparse.Namespace, input_paths: list[Path], writers: list[PackedSfenWriter]) -> dict:
    rng = random.Random(args.seed)
    board = cshogi.Board()
    stats = _new_stats()

    for path, line_no, record in tqdm(iter_jsonl_records(input_paths), desc="Converting jsonl games"):
        stats["games"] += 1

        moves = record.get("moves") or []
        eval_values = record.get("eval_values") or []
        initial_position = record.get("initial_position", "startpos")
        game_result = parse_game_result(record.get("result", ""))

        if not moves or not eval_values:
            stats["skipped_games"] += 1
            continue
        if len(moves) != len(eval_values):
            stats["mismatched_lengths"] += 1

        try:
            set_initial_position(board, initial_position)
        except Exception as exc:
            print(f"初期局面の設定に失敗: {path}:{line_no} ({exc})", file=sys.stderr)
            stats["skipped_games"] += 1
            continue

        usable_len = min(len(moves), len(eval_values))
        for ply_index in range(usable_len):
            ply = ply_index + 1
            stats["positions_seen"] += 1

            try:
                eval_score_cp = int(eval_values[ply_index])
                current_sfen = board.sfen()
            except Exception as exc:
                print(f"局面/評価値取得エラー: {path}:{line_no} ply={ply} ({exc})", file=sys.stderr)
                stats["skipped_positions"] += 1
                break

            if args.min_ply <= ply <= args.max_ply and _should_keep_position(board, args.quiet_level):
                writer = _select_writer(writers, rng, args.val_split)
                if writer.write(current_sfen, ply, eval_score_cp, game_result):
                    stats["positions_written"] += 1
                else:
                    stats["skipped_positions"] += 1
            else:
                stats["skipped_positions"] += 1

            try:
                board.push_usi(moves[ply_index])
            except Exception as exc:
                print(f"指し手適用エラー: {path}:{line_no} ply={ply} move={moves[ply_index]} ({exc})", file=sys.stderr)
                stats["skipped_positions"] += usable_len - ply
                break

    return stats


def _merge_stats(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        base[key] += value
    return base


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(args.input_pattern, args.input_format)

    if args.output:
        writers = [PackedSfenWriter(Path(args.output), score_clip=args.score_clip)]
    else:
        output_dir = Path(args.output_dir)
        writers = [
            PackedSfenWriter(output_dir / "train.bin", shuffle_on_close=True, seed=args.seed, score_clip=args.score_clip),
            PackedSfenWriter(output_dir / "val.bin", score_clip=args.score_clip),
        ]

    try:
        if args.input_format == "jsonl" or all(path.suffix == ".jsonl" for path in input_paths):
            stats = convert_jsonl_files(args, input_paths, writers)
        elif args.input_format == "sfen" or all(path.suffix == ".sfen" for path in input_paths):
            stats = convert_sfen_files(args, input_paths, writers)
        else:
            stats = _new_stats()
            sfen_paths = [path for path in input_paths if path.suffix == ".sfen"]
            jsonl_paths = [path for path in input_paths if path.suffix == ".jsonl"]
            if sfen_paths:
                _merge_stats(stats, convert_sfen_files(args, sfen_paths, writers))
            if jsonl_paths:
                _merge_stats(stats, convert_jsonl_files(args, jsonl_paths, writers))
    finally:
        for writer in writers:
            writer.close()

    print("変換完了")
    print(f"入力: {len(input_paths)} ファイル")
    print(f"静止局面フィルタ: quiet-level={args.quiet_level}")
    print(f"処理対局数: {stats['games']:,}")
    print(f"スキップ対局数: {stats['skipped_games']:,}")
    print(f"参照局面数: {stats['positions_seen']:,}")
    print(f"出力局面数: {stats['positions_written']:,}")
    print(f"除外局面数: {stats['skipped_positions']:,}")
    if stats["mismatched_lengths"]:
        print(f"moves/eval_values 長不一致対局数: {stats['mismatched_lengths']:,}")
    for writer in writers:
        print(f"出力: {writer.output_path} ({writer.rows_written:,} 局面)")
        if writer.clipped_scores:
            print(f"  score を +/-{writer.score_clip} にクリップ: {writer.clipped_scores:,}")
        if writer.skipped_out_of_range_scores:
            print(
                f"  score が int16 範囲外のため除外: "
                f"{writer.skipped_out_of_range_scores:,}"
            )


if __name__ == "__main__":
    main()
