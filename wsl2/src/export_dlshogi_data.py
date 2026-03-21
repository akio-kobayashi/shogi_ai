#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cshogi
import numpy as np


HuffmanCodedPosAndEval3 = np.dtype([
    ("hcp", cshogi.dtypeHcp),
    ("moveNum", np.uint16),
    ("result", np.uint8),
    ("opponent", np.uint8),
])
MoveInfo = np.dtype([
    ("selectedMove16", cshogi.dtypeMove16),
    ("eval", cshogi.dtypeEval),
    ("candidateNum", np.uint16),
])
MoveVisits = np.dtype([
    ("move16", cshogi.dtypeMove16),
    ("visitNum", np.uint16),
])


def _load_rows_from_csvs(input_csv: str = None, input_csvs: str = None):
    input_paths = []
    if input_csv:
        input_paths.append(Path(input_csv))
    if input_csvs:
        input_paths.extend(Path(p.strip()) for p in input_csvs.split(",") if p.strip())

    if not input_paths:
        sys.exit("エラー: 入力CSVが指定されていません。")

    rows = []
    header = None
    for path in input_paths:
        if not path.exists():
            sys.exit(f"エラー: 入力CSVが見つかりません: {path}")
        with open(path, "r", newline="", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            current_header = reader.fieldnames
            if header is None:
                header = current_header
            rows.extend(reader)
    return input_paths, header or [], rows


def _load_eval_map_from_csvs(input_csv: str = None, input_csvs: str = None):
    _, _, rows = _load_rows_from_csvs(input_csv, input_csvs)
    eval_map = {}
    for row in rows:
        sfen = row.get("sfen")
        if not sfen:
            continue
        try:
            eval_map[sfen] = int(row["eval_score_cp"])
        except (KeyError, TypeError, ValueError):
            continue
    return eval_map


def _load_positions(args):
    direct_input = bool(args.input_csv or args.input_csvs)
    joined_input = bool(args.positions_csv or args.positions_csvs or args.eval_sfen_csv or args.eval_sfen_csvs)

    if direct_input and joined_input:
        sys.exit("エラー: direct入力と join入力を同時に指定できません。")
    if not direct_input and not joined_input:
        sys.exit("エラー: direct入力または join入力のどちらかが必要です。")

    required = {"file_path", "kif_index", "ply", "game_result", "sfen", "eval_score_cp"}

    if direct_input:
        _, header, rows = _load_rows_from_csvs(args.input_csv, args.input_csvs)
        missing = required - set(header)
        if missing:
            sys.exit(f"エラー: 入力CSVに必要な列が不足しています: {', '.join(sorted(missing))}")
        return rows

    if bool(args.positions_csv or args.positions_csvs) != bool(args.eval_sfen_csv or args.eval_sfen_csvs):
        sys.exit("エラー: join入力では positions-csv 系と eval-sfen-csv 系を両方指定してください。")

    _, header, rows = _load_rows_from_csvs(args.positions_csv, args.positions_csvs)
    required_positions = {"file_path", "kif_index", "ply", "game_result", "sfen"}
    missing = required_positions - set(header)
    if missing:
        sys.exit(f"エラー: positions CSV に必要な列が不足しています: {', '.join(sorted(missing))}")

    eval_map = _load_eval_map_from_csvs(args.eval_sfen_csv, args.eval_sfen_csvs)
    joined_rows = []
    missing_eval = 0
    for row in rows:
        sfen = row.get("sfen")
        if not sfen or sfen not in eval_map:
            missing_eval += 1
            continue
        joined = row.copy()
        joined["eval_score_cp"] = eval_map[sfen]
        joined_rows.append(joined)

    if not joined_rows:
        sys.exit("エラー: 評価値を join できた局面がありませんでした。")

    print(f"join完了: 採用局面数={len(joined_rows):,}, 評価値なし除外={missing_eval:,}")
    return joined_rows


def _to_black_eval(score_cp: int, board_turn: int) -> int:
    return score_cp if board_turn == cshogi.BLACK else -score_cp


def _parse_game_result(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"game_result を解釈できません: {value}") from exc


def _iter_grouped_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["file_path"], int(row["kif_index"]))].append(row)
    for key in sorted(grouped.keys()):
        yield key, sorted(grouped[key], key=lambda r: int(r["ply"]))


def write_hcpe(args, rows):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hcpes = np.zeros(513, cshogi.HuffmanCodedPosAndEval)
    board = cshogi.Board()
    game_count = 0
    position_count = 0

    with open(output_path, "wb") as f_out:
        for (file_path, kif_index), game_rows in _iter_grouped_rows(rows):
            try:
                kifs = cshogi.CSA.Parser.parse_file(file_path)
                if not kifs:
                    continue
                kif = kifs[kif_index]
                ply_to_row = {int(row["ply"]): row for row in game_rows}

                board.set_sfen(kif.sfen)
                write_count = 0
                for ply, move in enumerate(kif.moves, 1):
                    row = ply_to_row.get(ply)
                    if row is None:
                        board.push(move)
                        continue

                    hcpe = hcpes[write_count]
                    board.to_hcp(hcpe["hcp"])
                    score_cp = int(row["eval_score_cp"])
                    eval_for_black = _to_black_eval(score_cp, board.turn)
                    hcpe["eval"] = np.int16(max(-32767, min(32767, eval_for_black)))
                    hcpe["bestMove16"] = cshogi.move16(move)
                    hcpe["gameResult"] = np.uint8(_parse_game_result(row["game_result"]))
                    write_count += 1
                    board.push(move)

                if write_count == 0:
                    continue
                hcpes[:write_count].tofile(f_out)
                game_count += 1
                position_count += write_count
            except Exception as exc:
                print(f"hcpe 変換エラー: {file_path}:{kif_index} ({exc})", file=sys.stderr)

    print(f"hcpe 出力完了: games={game_count:,}, positions={position_count:,}, output={output_path}")


def write_hcpe3(args, rows):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = np.zeros(1, HuffmanCodedPosAndEval3)
    move_info_vec = np.zeros(513, MoveInfo)
    move_visits_vec = np.zeros(513, MoveVisits)
    move_visits_vec["visitNum"] = 1
    board = cshogi.Board()

    game_count = 0
    position_count = 0

    with open(output_path, "wb") as f_out:
        for (file_path, kif_index), game_rows in _iter_grouped_rows(rows):
            try:
                kifs = cshogi.CSA.Parser.parse_file(file_path)
                if not kifs:
                    continue
                kif = kifs[kif_index]
                ply_to_row = {int(row["ply"]): row for row in game_rows}
                last_ply = max(ply_to_row.keys())
                if last_ply <= 0:
                    continue

                move_info_vec[:last_ply]["candidateNum"] = 0
                move_info_vec[:last_ply]["eval"] = 0
                move_info_vec[:last_ply]["selectedMove16"] = 0
                move_visits_vec[:last_ply]["move16"] = 0
                move_visits_vec[:last_ply]["visitNum"] = 1

                header["result"] = np.uint8(_parse_game_result(game_rows[0]["game_result"]))
                header["opponent"] = np.uint8(0)
                header["moveNum"] = np.uint16(last_ply)

                board.set_sfen(kif.sfen)
                board.to_hcp(header["hcp"])

                for ply, move in enumerate(kif.moves, 1):
                    if ply > last_ply:
                        break
                    move_info = move_info_vec[ply - 1]
                    move_info["selectedMove16"] = cshogi.move16(move)
                    row = ply_to_row.get(ply)
                    if row is not None:
                        score_cp = int(row["eval_score_cp"])
                        eval_for_black = _to_black_eval(score_cp, board.turn)
                        move_info["eval"] = np.int16(max(-32767, min(32767, eval_for_black)))
                        move_info["candidateNum"] = 1
                        move_visits_vec[ply - 1]["move16"] = cshogi.move16(move)
                        position_count += 1
                    board.push(move)

                header.tofile(f_out)
                for move_info, move_visits in zip(move_info_vec[:last_ply], move_visits_vec[:last_ply]):
                    move_info.tofile(f_out)
                    if move_info["candidateNum"] > 0:
                        move_visits.tofile(f_out)

                game_count += 1
            except Exception as exc:
                print(f"hcpe3 変換エラー: {file_path}:{kif_index} ({exc})", file=sys.stderr)

    print(f"hcpe3 出力完了: games={game_count:,}, positions={position_count:,}, output={output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="shogi_ai の評価済みCSVから DeepLearningShogi 用 hcpe/hcpe3 を生成する。"
    )
    parser.add_argument("--input-csv", help="eval_score_cp を含む局面CSV")
    parser.add_argument("--input-csvs", help="eval_score_cp を含む局面CSVのカンマ区切り")
    parser.add_argument("--positions-csv", help="局面CSV。eval-sfen と join する場合に指定")
    parser.add_argument("--positions-csvs", help="局面CSVのカンマ区切り。eval-sfen と join する場合に指定")
    parser.add_argument("--eval-sfen-csv", help="evaluate-sfen / merge-eval-sfen 出力CSV")
    parser.add_argument("--eval-sfen-csvs", help="evaluate-sfen / merge-eval-sfen 出力CSVのカンマ区切り")
    parser.add_argument("--output", required=True, help="出力ファイルパス")
    parser.add_argument("--format", required=True, choices=["hcpe", "hcpe3"], help="出力形式")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    rows = _load_positions(args)
    if args.format == "hcpe":
        write_hcpe(args, rows)
    else:
        write_hcpe3(args, rows)


if __name__ == "__main__":
    main()
