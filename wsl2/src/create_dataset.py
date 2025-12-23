# -*- coding: utf-8 -*-
"""
概要:
CSA棋譜ファイルから、nodchip/nnue-pytorch形式の学習データおよび検証データを生成するスクリプト。

処理フロー:
1. extract: CSAファイルから棋譜のメタデータを抽出し、CSVを生成する。
2. filter: メタデータCSVを条件でフィルタリングする。
3. evaluate: フィルタリング済みの棋譜の各局面をエンジンで評価し、評価値とSFEN付きのCSVを生成する。
4. generate: 評価値付きCSVから、最終的な学習データ(.bin or .h5)を生成する。
"""
import os
import csv
import random
import argparse
import sys
from collections import defaultdict
from pathlib import Path
import yaml

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm がインストールされていません。pip install tqdm を実行してください。", file=sys.stderr)
    def tqdm(iterable, **kwargs):
        return iterable

import cshogi
import numpy as np


# ================================
# データ生成ロジック
# ================================

def extract_metadata(args: argparse.Namespace) -> None:
    """
    CSAファイルをスキャンし、棋譜のメタデータをCSVに書き出す。
    """
    csa_root = args.csa_dir
    csv_path = args.metadata_csv

    print(f"フェーズ1: メタデータ抽出を開始します。出力先: {csv_path}")
    
    csa_files = list(Path(csa_root).rglob('*.csa')) + list(Path(csa_root).rglob('*.CSA'))
    if not csa_files:
        sys.exit(f"エラー: '{csa_root}' 内にCSAファイルが見つかりません。")

    print(f"{len(csa_files)}個のCSAファイルをスキャンします...")

    header = ['file_path', 'kif_index', 'black_player', 'white_player', 'rating_b', 'rating_w', 'game_result', 'total_moves']
    parser = cshogi.Parser()

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        with tqdm(csa_files, unit="file") as pbar:
            for csa_path in pbar:
                pbar.set_description(f"Processing {csa_path.name}")
                try:
                    kifs = parser.parse_file(str(csa_path))
                    for i, kif in enumerate(kifs):
                        if not kif.ratings or len(kif.ratings) < 2:
                            continue
                        
                        rating_b, rating_w = kif.ratings
                        writer.writerow([
                            str(csa_path), i, kif.black_player, kif.white_player,
                            rating_b, rating_w, kif.win, len(kif.moves)
                        ])
                except Exception as e:
                    print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)
    print("フェーズ1: メタデータ抽出が完了しました。")

def run_filter_metadata(args: argparse.Namespace) -> None:
    """
    メタデータCSVをフィルタリングし、結果を新しいCSVファイルに書き出す。
    """
    if not Path(args.metadata_csv).exists():
        sys.exit(f"エラー: 入力メタデータファイル '{args.metadata_csv}' が見つかりません。")

    print(f"--- メタデータのフィルタリングを開始 ---")
    print(f"入力ファイル: {args.metadata_csv}")
    print(f"出力ファイル: {args.output_csv}")

    with open(args.metadata_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_kifs = list(reader)
        header = reader.fieldnames

    print(f"フィルタリング前 - 合計棋譜数: {len(all_kifs)}")

    result_map = {'win': 1, 'lose': 2, 'draw': 0}
    allowed_results_int = {result_map[res.strip()] for res in args.allowed_results.split(',')}

    filtered_kifs = []
    for kif in tqdm(all_kifs, desc="フィルタリング中"):
        try:
            rating_b, rating_w = int(kif['rating_b']), int(kif['rating_w'])
            total_moves, game_result = int(kif['total_moves']), int(kif['game_result'])

            if not (args.min_rating <= rating_b <= args.max_rating and args.min_rating <= rating_w <= args.max_rating):
                continue
            if abs(rating_b - rating_w) > args.max_rating_diff:
                continue
            if not (args.min_moves <= total_moves <= args.max_moves):
                continue
            if game_result not in allowed_results_int:
                continue
            if args.filter_by_rating_outcome:
                if (rating_b > rating_w and game_result != 1) or (rating_w > rating_b and game_result != 2):
                    continue
            filtered_kifs.append(kif)
        except (ValueError, KeyError):
            continue

    print(f"フィルタリング後 - 合計棋譜数: {len(filtered_kifs)}")

    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(filtered_kifs)
        print("フィルタリング処理が完了しました。")
    except IOError as e:
        sys.exit(f"エラー: ファイルの書き込みに失敗しました: {e}")

def evaluate_metadata_logic(args: argparse.Namespace) -> None:
    from usi import UsiEngine
    if not Path(args.metadata_csv).exists():
        sys.exit(f"エラー: 入力メタデータファイル '{args.metadata_csv}' が見つかりません。")
    if not Path(args.engine_path).exists():
        sys.exit(f"エラー: エンジン実行ファイルが見つかりません: {args.engine_path}")

    print(f"--- 局面評価を開始 ---")
    try:
        engine = UsiEngine(str(args.engine_path))
        print("USIエンジン準備完了。")
    except Exception as e:
        sys.exit(f"エラー: USIエンジンの初期化に失敗しました: {e}")

    with open(args.metadata_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        all_kifs_meta = list(reader)
        header = reader.fieldnames

    output_csv_path = Path(args.output_csv)
    output_header = header + ['ply', 'eval_score_cp', 'sfen']
    print(f"評価結果を '{output_csv_path}' に書き込みます。")

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_header)
        writer.writeheader()
        kifs_by_file = defaultdict(list)
        for meta in all_kifs_meta:
            kifs_by_file[meta['file_path']].append(meta)

        with tqdm(kifs_by_file.items(), unit="file") as pbar:
            for csa_path, metas in pbar:
                pbar.set_description(f"Evaluating {Path(csa_path).name}")
                try:
                    all_kifs_in_file = list(cshogi.CSA.Parser.parse_file(csa_path))
                    for meta in metas:
                        kif = all_kifs_in_file[int(meta['kif_index'])]
                        board = cshogi.Board(kif.sfen)
                        for ply, move in enumerate(kif.moves, 1):
                            if ply > args.max_ply: break
                            if ply >= args.min_ply:
                                try:
                                    sfen = board.sfen()
                                    score_type, score_value = engine.evaluate_sfen(sfen, args.depth)
                                    eval_score_cp = score_value if score_type == "cp" else (32000 if score_value > 0 else -32000)
                                    meta_with_eval = meta.copy()
                                    meta_with_eval.update({'ply': ply, 'eval_score_cp': eval_score_cp, 'sfen': sfen})
                                    writer.writerow(meta_with_eval)
                                except Exception as e:
                                    print(f"\n評価エラー: 棋譜{meta['kif_index']} 手数{ply} ({e})", file=sys.stderr)
                            board.push(move)
                except Exception as e:
                    print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)
    engine.quit()
    print("局面評価が完了しました。")

def write_bin_file(positions: list, output_path: str):
    print(f"データセット '{output_path}' の生成を開始 (対象局面数: {len(positions)})")
    board = cshogi.Board()
    psv = np.zeros(1, dtype=cshogi.PackedSfenValue)
    with open(output_path, "wb") as f_out:
        for pos in tqdm(positions, desc=f"Writing {Path(output_path).name}"):
            try:
                board.set_sfen(pos['sfen'])
                board.to_psfen(psv)
                cshogi_result = int(pos['game_result'])
                write_result = 1 if cshogi_result == 1 else -1 if cshogi_result == 2 else 0
                psv[0]["score"] = np.int16(pos['eval_score_cp'])
                psv[0]["move"] = np.uint16(0)
                psv[0]["gamePly"] = np.uint16(pos['ply'])
                psv[0]["game_result"] = np.int8(write_result)
                psv.tofile(f_out)
            except Exception as e:
                print(f"\nデータ書き込みエラー: {pos} ({e})", file=sys.stderr)

def generate_datasets_logic(args: argparse.Namespace) -> None:
    if not Path(args.input_csv).exists():
        sys.exit(f"エラー: 入力ファイル '{args.input_csv}' が見つかりません。")

    print(f"--- データセット生成を開始 ---")
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f:
        all_positions = list(csv.DictReader(f))
    
    if not all_positions:
        sys.exit("エラー: 入力ファイルにデータがありません。")

    print(f"読み込み完了。総局面数: {len(all_positions)}")
    random.shuffle(all_positions)
    
    val_size = int(len(all_positions) * args.val_split)
    train_positions, val_positions = all_positions[val_size:], all_positions[:val_size]
    print(f"分割結果 - 訓練: {len(train_positions)}局面, 検証: {len(val_positions)}局面")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == 'bin':
        write_bin_file(train_positions, str(output_dir / "train.bin"))
        write_bin_file(val_positions, str(output_dir / "val.bin"))
    elif args.format == 'hdf5':
        print("HDF5の書き込み処理は現在実装されていません。")

    print("\nすべての処理が完了しました。")

# ================================
# main
# ================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CSA棋譜から学習データを生成するスクリプト。", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", help="設定YAMLファイルのパス。")
    subparsers = parser.add_subparsers(dest="command", required=True, help="利用可能なコマンド")

    # --- 'extract' コマンド ---
    extract_parser = subparsers.add_parser("extract", help="CSAファイルから棋譜のメタデータを抽出します。")
    extract_parser.add_argument("--csa-dir", help="CSAファイルが格納されているルートディレクトリ。")
    extract_parser.add_argument("--output-dir", help="生成されたメタデータ(.csv)を保存するディレクトリ。")
    extract_parser.set_defaults(func=extract_metadata)

    # --- 'filter' コマンド ---
    filter_parser = subparsers.add_parser("filter", help="メタデータCSVをフィルタリングします。")
    filter_parser.add_argument("--metadata-csv", help="入力となるメタデータCSVのパス。")
    filter_parser.add_argument("--output-csv", help="フィルタリング結果を保存するCSVのパス。")
    filter_parser.add_argument("--min-rating", type=int, default=0)
    filter_parser.add_argument("--max-rating", type=int, default=9999)
    filter_parser.add_argument("--max-rating-diff", type=int, default=9999)
    filter_parser.add_argument("--min-moves", type=int, default=0)
    filter_parser.add_argument("--max-moves", type=int, default=999)
    filter_parser.add_argument("--allowed-results", type=str, default="win,lose,draw")
    filter_parser.add_argument("--filter-by-rating-outcome", action='store_true')
    filter_parser.set_defaults(func=run_filter_metadata)

    # --- 'evaluate' コマンド ---
    evaluate_parser = subparsers.add_parser("evaluate", help="フィルタリング済みCSVの局面を評価します。")
    evaluate_parser.add_argument("--metadata-csv", help="入力となるフィルタリング済みCSVのパス。")
    evaluate_parser.add_argument("--engine-path", help="USIエンジンの実行ファイルのパス。")
    evaluate_parser.add_argument("--output-csv", help="評価値付きCSVの出力パス。")
    evaluate_parser.add_argument("--depth", type=int, default=10)
    evaluate_parser.add_argument("--min-ply", type=int, default=0)
    evaluate_parser.add_argument("--max-ply", type=int, default=999)
    evaluate_parser.set_defaults(func=evaluate_metadata_logic)

    # --- 'generate' コマンド ---
    generate_parser = subparsers.add_parser("generate", help="評価値付きCSVから学習データを生成します。")
    generate_parser.add_argument("--input-csv", help="入力となる評価値付きCSVのパス。")
    generate_parser.add_argument("--output-dir", help="生成されたデータセットを保存するディレクトリ。")
    generate_parser.add_argument("--format", choices=['bin', 'hdf5'], default='bin')
    generate_parser.add_argument("--val-split", type=float, default=0.1)
    generate_parser.set_defaults(func=generate_datasets_logic)

    # --- 引数のパースと設定の上書き ---
    temp_args, _ = parser.parse_known_args()
    config = {}
    if temp_args.config and Path(temp_args.config).exists():
        print(f"設定ファイル '{temp_args.config}' を読み込みます。")
        with open(temp_args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    if temp_args.command:
        command_config = config.get(temp_args.command, {})
        # subparsersのデフォルト値を上書き
        subparsers.choices[temp_args.command].set_defaults(**command_config)

    args = parser.parse_args()

    # --- パスの自動設定と必須引数のチェック ---
    if args.command == "extract":
        args.output_dir = args.output_dir or "output_data"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.metadata_csv = str(Path(args.output_dir) / "metadata.csv")
        if not args.csa_dir: sys.exit("エラー: --csa-dir は必須です。")
    elif args.command in ["filter", "evaluate", "generate"]:
        # config.yamlで設定されることを期待し、未設定の場合のみエラー
        if args.command == "filter" and not all([args.metadata_csv, args.output_csv]):
             sys.exit("エラー: --metadata-csv と --output-csv は必須です。")
        if args.command == "evaluate" and not all([args.metadata_csv, args.engine_path, args.output_csv]):
             sys.exit("エラー: --metadata-csv, --engine-path, --output-csv は必須です。")
        if args.command == "generate" and not all([args.input_csv, args.output_dir]):
             sys.exit("エラー: --input-csv と --output-dir は必須です。")
        
    args.func(args)

if __name__ == "__main__":
    main()
