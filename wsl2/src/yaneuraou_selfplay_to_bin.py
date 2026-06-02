#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YaneuraOu の自己対局 jsonl から、create_dataset.py 互換の PackedSfenValue .bin を生成する。

使い方例:
  python yaneuraou_selfplay_to_bin.py \
    --input-pattern /home/akio/GitHub/YaneuraOu/selfplay/20260523202833T16_n200000* \
    --output /home/akio/GitHub/shogi_ai/wsl2/output_data_selfplay/20260523202833T16_n200000.bin

  python yaneuraou_selfplay_to_bin.py \
    --input-pattern /home/akio/GitHub/YaneuraOu/selfplay/20260523202833T16_n200000* \
    --output-dir /home/akio/GitHub/shogi_ai/wsl2/output_data_selfplay/20260523202833T16_n200000 \
    --val-split 0.1 \
    --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path
from typing import Iterable

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YaneuraOu selfplay jsonl を PackedSfenValue .bin へ変換する。"
    )
    parser.add_argument(
        "--input-pattern",
        required=True,
        help="入力 jsonl を探す glob。'.jsonl' を含まないパターンでも可。",
    )
    parser.add_argument(
        "--output",
        help="単一の .bin 出力先。--output-dir とは同時指定不可。",
    )
    parser.add_argument(
        "--output-dir",
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
    args = parser.parse_args()

    if bool(args.output) == bool(args.output_dir):
        parser.error("--output か --output-dir のどちらか一方を指定してください。")
    if not (0.0 <= args.val_split <= 1.0):
        parser.error("--val-split は 0.0 以上 1.0 以下で指定してください。")
    if args.min_ply <= 0:
        parser.error("--min-ply は 1 以上で指定してください。")
    if args.min_ply > args.max_ply:
        parser.error("--min-ply は --max-ply 以下である必要があります。")
    return args


def resolve_input_jsonl_paths(input_pattern: str) -> list[Path]:
    raw_matches = sorted(glob.glob(input_pattern))
    if not raw_matches and not input_pattern.endswith(".jsonl"):
        raw_matches = sorted(glob.glob(f"{input_pattern}.jsonl"))

    paths = [Path(p) for p in raw_matches if Path(p).suffix == ".jsonl"]
    if not paths:
        sys.exit(f"エラー: 入力 jsonl が見つかりません: {input_pattern}")
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


class PackedSfenWriter:
    def __init__(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.board = cshogi.Board()
        self.psv = np.zeros(1, dtype=cshogi.PackedSfenValue)
        self.int16_info = np.iinfo(np.int16)
        self.rows_written = 0
        self.skipped_out_of_range_scores = 0
        self.f_out = output_path.open("wb")

    def write(self, sfen: str, ply: int, eval_score_cp: int, game_result: int) -> bool:
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


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_jsonl_paths(args.input_pattern)

    if args.output:
        writers = [PackedSfenWriter(Path(args.output))]
    else:
        output_dir = Path(args.output_dir)
        writers = [
            PackedSfenWriter(output_dir / "train.bin"),
            PackedSfenWriter(output_dir / "val.bin"),
        ]

    rng = random.Random(args.seed)
    board = cshogi.Board()

    total_games = 0
    total_positions_seen = 0
    total_positions_written = 0
    skipped_games = 0
    skipped_positions = 0
    mismatched_lengths = 0

    try:
        for path, line_no, record in tqdm(iter_jsonl_records(input_paths), desc="Converting games"):
            total_games += 1

            moves = record.get("moves") or []
            eval_values = record.get("eval_values") or []
            initial_position = record.get("initial_position", "startpos")
            game_result = parse_game_result(record.get("result", ""))

            if not moves or not eval_values:
                skipped_games += 1
                continue

            if len(moves) != len(eval_values):
                mismatched_lengths += 1

            try:
                set_initial_position(board, initial_position)
            except Exception as exc:
                print(f"初期局面の設定に失敗: {path}:{line_no} ({exc})", file=sys.stderr)
                skipped_games += 1
                continue

            usable_len = min(len(moves), len(eval_values))
            for ply_index in range(usable_len):
                ply = ply_index + 1
                total_positions_seen += 1

                if not (args.min_ply <= ply <= args.max_ply):
                    try:
                        board.push_usi(moves[ply_index])
                    except Exception as exc:
                        print(f"指し手適用エラー: {path}:{line_no} ply={ply} ({exc})", file=sys.stderr)
                        skipped_positions += 1
                        break
                    continue

                try:
                    eval_score_cp = int(eval_values[ply_index])
                    current_sfen = board.sfen()
                except Exception as exc:
                    print(f"局面/評価値取得エラー: {path}:{line_no} ply={ply} ({exc})", file=sys.stderr)
                    skipped_positions += 1
                    break

                writer = writers[0]
                if len(writers) == 2 and rng.random() < args.val_split:
                    writer = writers[1]

                if writer.write(current_sfen, ply, eval_score_cp, game_result):
                    total_positions_written += 1
                else:
                    skipped_positions += 1

                try:
                    board.push_usi(moves[ply_index])
                except Exception as exc:
                    print(f"指し手適用エラー: {path}:{line_no} ply={ply} ({exc})", file=sys.stderr)
                    skipped_positions += 1
                    break
    finally:
        for writer in writers:
            writer.close()

    print("変換完了")
    print(f"入力 jsonl: {len(input_paths)} ファイル")
    print(f"処理対局数: {total_games:,}")
    print(f"スキップ対局数: {skipped_games:,}")
    print(f"参照局面数: {total_positions_seen:,}")
    print(f"出力局面数: {total_positions_written:,}")
    print(f"除外局面数: {skipped_positions:,}")
    if mismatched_lengths:
        print(f"moves/eval_values 長不一致対局数: {mismatched_lengths:,}")
    for writer in writers:
        print(f"出力: {writer.output_path} ({writer.rows_written:,} 局面)")
        if writer.skipped_out_of_range_scores:
            print(
                f"  score が int16 範囲外のため除外: "
                f"{writer.skipped_out_of_range_scores:,}"
            )


if __name__ == "__main__":
    main()
