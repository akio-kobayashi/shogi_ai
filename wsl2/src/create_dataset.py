# -*- coding: utf-8 -*-
"""
概要:
CSA棋譜ファイルから、nodchip/nnue-pytorch形式の学習データおよび検証データを生成するスクリプト。

処理フロー:
1. メタデータ抽出 (フェーズ1):
   指定されたCSAディレクトリを再帰的にスキャンし、各棋譜のメタデータ
   (ファイルパス, 棋譜インデックス, レーティング, 勝敗) をCSVファイルに書き出す。
   この処理は時間がかかるため、一度実行すればスキップ可能。

2. データセット生成 (フェーズ2):
   メタデータCSVを読み込み、レーティング等でフィルタリングを実行。
   フィルタリング後の棋譜リストを訓練用と検証用に分割し、
   それぞれについてバイナリ形式のデータセットファイル (.bin) を生成する。
"""
import os
import csv
import random
import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm がインストールされていません。pip install tqdm を実行してください。", file=sys.stderr)
    def tqdm(iterable, **kwargs):
        return iterable

import cshogi
import numpy as np


# ================================
# フェーズ1: メタデータ抽出
# ================================

def extract_metadata(csa_root: str, csv_path: str) -> None:
    """
    CSAファイルをスキャンし、棋譜のメタデータをCSVに書き出す。
    """
    print(f"フェーズ1: メタデータ抽出を開始します。出力先: {csv_path}")
    
    csa_files = [p for p in Path(csa_root).rglob('*.csa') if p.is_file()]
    csa_files += [p for p in Path(csa_root).rglob('*.CSA') if p.is_file()]

    if not csa_files:
        print(f"エラー: '{csa_root}' 内にCSAファイルが見つかりません。", file=sys.stderr)
        sys.exit(1)

    print(f"{len(csa_files)}個のCSAファイルをスキャンします...")

    header = ['file_path', 'kif_index', 'rating_b', 'rating_w', 'game_result', 'total_moves']
    
    parser = cshogi.Parser()

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with tqdm(csa_files, unit="file") as pbar:
            for csa_path in pbar:
                pbar.set_description(f"Processing {csa_path.name}")
                try:
                    # parse_fileはジェネレータを返す
                    kifs = parser.parse_file(str(csa_path))
                    for i, kif in enumerate(kifs):
                        # レーティング情報がない場合はスキップ
                        if not kif.ratings or len(kif.ratings) < 2:
                            continue
                        
                        rating_b, rating_w = kif.ratings
                        
                        # kif.win -> 1:先手勝ち, 2:後手勝ち, 0:引き分け
                        game_result = kif.win
                        total_moves = len(kif.moves)

                        writer.writerow([str(csa_path), i, rating_b, rating_w, game_result, total_moves])

                except Exception as e:
                    print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)

    print("フェーズ1: メタデータ抽出が完了しました。")


# ================================
# フェーズ2: データセット生成
# ================================

def result_to_score(win_flag: int, win_value: int) -> int:
    """勝敗フラグを先手視点の評価値に変換する。"""
    if win_flag == 1:
        return win_value
    elif win_flag == -1:
        return -win_value
    else:
        return 0

def process_kifs(
    kif_metadata_list: list,
    output_path: str,
    win_value: int,
    min_ply: int,
    max_ply: int,
    evaluated_scores: dict = None # 新しい引数
) -> None:
    """
    与えられた棋譜メタデータリストに基づき、データセット(.bin)を生成する。
    """
    print(f"データセット '{output_path}' の生成を開始します。対象棋譜数: {len(kif_metadata_list)}")

    # ファイルパスでグループ化し、ファイルI/Oを最小限に抑える
    kifs_by_file = defaultdict(list)
    for meta in kif_metadata_list:
        kifs_by_file[meta['file_path']].append(meta)

    with open(output_path, "wb") as f_out:
        with tqdm(kifs_by_file.items(), unit="file") as pbar:
            for csa_path, metas in pbar:
                pbar.set_description(f"Writing {Path(csa_path).name}")
                try:
                    all_kifs_in_file = list(cshogi.CSA.Parser.parse_file(csa_path))
                    
                    for meta in metas:
                        kif_index = int(meta['kif_index'])
                        kif = all_kifs_in_file[kif_index]
                        
                        board = cshogi.Board(kif.sfen)
                        
                        # 評価値の決定: evaluated_scoresがあればそれを使用、なければゲーム結果から生成
                        current_score = 0
                        if evaluated_scores:
                            # 評価値付きCSVのキーは (file_path, kif_index, ply)
                            # ここではまだplyが確定していないので、各局面でルックアップする
                            pass # 各局面でルックアップするため、ここでは初期化のみ
                        else:
                            current_score = result_to_score(int(meta['game_result']), win_value)

                        for ply, move in enumerate(kif.moves, 1):
                            if ply > max_ply:
                                break
                            
                            if ply >= min_ply:
                                if evaluated_scores:
                                    key = (meta['file_path'], int(meta['kif_index']), ply)
                                    if key in evaluated_scores:
                                        current_score = evaluated_scores[key]
                                    else:
                                        # 評価値が見つからない場合はスキップするか、デフォルト値を使用
                                        # ここではスキップする
                                        board.push(move)
                                        continue
                                
                                # PackedSfenValue形式で書き出し
                                psv = np.zeros(1, dtype=cshogi.PackedSfenValue)
                                board.to_psfen(psv)
                                psv[0]["score"] = np.int16(current_score)
                                psv[0]["move"] = np.uint16(0) # 教師手は使わない
                                psv[0]["gamePly"] = np.uint16(ply)
                                psv[0]["game_result"] = np.int8(int(meta['game_result']))
                                psv.tofile(f_out)

                            board.push(move)

                except Exception as e:
                    print(f"\nデータ生成エラー: {csa_path} ({e})", file=sys.stderr)

    print(f"データセット '{output_path}' の生成が完了しました。")


def generate_datasets_logic(args: argparse.Namespace) -> None:
    """
    メタデータCSVを読み込み、フィルタリング、分割、データセット生成を行う。
    """
    # --- メタデータCSVの存在確認 ---
    if not Path(args.metadata_csv).exists():
        print(f"エラー: メタデータファイル '{args.metadata_csv}' が見つかりません。", file=sys.stderr)
        print("先に 'extract' コマンドでメタデータを生成してください。", file=sys.stderr)
        sys.exit(1)
    
    print(f"フェーズ2: 既存のメタデータファイル '{args.metadata_csv}' を使用し、棋譜のフィルタリングと分割を開始します。")

    # --- 評価値付きメタデータCSVの読み込み (オプション) ---
    evaluated_scores = {}
    if args.evaluated_metadata_csv and Path(args.evaluated_metadata_csv).exists():
        print(f"評価値付きメタデータファイル '{args.evaluated_metadata_csv}' を読み込みます。")
        with open(args.evaluated_metadata_csv, 'r', newline='', encoding='utf-8') as f_eval:
            reader_eval = csv.DictReader(f_eval)
            for row in reader_eval:
                # ユニークなキーを作成 (file_path, kif_index, ply)
                key = (row['file_path'], int(row['kif_index']), int(row['ply']))
                evaluated_scores[key] = int(row['eval_score_cp'])
        print(f"{len(evaluated_scores)}個の評価値が読み込まれました。")
    elif args.evaluated_metadata_csv and not Path(args.evaluated_metadata_csv).exists():
        print(f"警告: 評価値付きメタデータファイル '{args.evaluated_metadata_csv}' が見つかりません。ゲーム結果から評価値を生成します。", file=sys.stderr)
    
    # --- フィルタリング ---
    print("棋譜のフィルタリングと分割を開始します。")
    
    with open(args.metadata_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_kifs = list(reader)

    print(f"フィルタリング前 - 合計棋譜数: {len(all_kifs)}")

    filtered_kifs = []
    for kif in all_kifs:
        try:
            rating_b = int(kif['rating_b'])
            rating_w = int(kif['rating_w'])
            
            # 最低レーティングフィルタ
            if rating_b < args.min_rating or rating_w < args.min_rating:
                continue
            
            # レーティング差フィルタ
            if abs(rating_b - rating_w) > args.max_rating_diff:
                continue
            
            filtered_kifs.append(kif)
        except (ValueError, KeyError):
            continue # ヘッダや不正な行をスキップ

    print(f"フィルタリング後 - 合計棋譜数: {len(filtered_kifs)}")

    if not filtered_kifs:
        print("エラー: フィルタリング条件を満たす棋譜がありません。", file=sys.stderr)
        sys.exit(1)

    # --- 分割 ---
    random.shuffle(filtered_kifs)
    
    val_size = int(len(filtered_kifs) * args.val_split)
    train_kifs = filtered_kifs[val_size:]
    val_kifs = filtered_kifs[:val_size]

    print(f"分割結果 - 訓練データ: {len(train_kifs)}棋譜, 検証データ: {len(val_kifs)}棋譜")

    # --- データセット生成 ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 訓練データ
    process_kifs(
        train_kifs,
        str(output_dir / "train.bin"),
        args.win_value,
        args.min_ply,
        args.max_ply,
        evaluated_scores # 評価値付きスコアを渡す
    )
    
    # 検証データ
    process_kifs(
        val_kifs,
        str(output_dir / "val.bin"),
        args.win_value,
        args.min_ply,
        args.max_ply,
        evaluated_scores # 評価値付きスコアを渡す
    )

    print("\nすべての処理が完了しました。")


def evaluate_metadata_logic(args: argparse.Namespace) -> None:
    """
    メタデータCSVを読み込み、USIエンジンで局面を評価し、評価値付きのCSVを生成する。
    """
    from usi import UsiEngine # UsiEngineをインポート

    # --- 入力CSVの存在確認 ---
    if not Path(args.metadata_csv).exists():
        print(f"エラー: 入力メタデータファイル '{args.metadata_csv}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    # --- エンジンパスの確認 ---
    engine_path = Path(args.engine_path)
    if not engine_path.exists():
        print(f"エラー: エンジン実行ファイルが見つかりません: {engine_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"フェーズ2: メタデータファイル '{args.metadata_csv}' を使用し、USIエンジンで局面評価を開始します。")

    # --- UsiEngineの初期化 ---
    try:
        engine = UsiEngine(str(engine_path))
        print("USIエンジン準備完了。")
    except Exception as e:
        print(f"エラー: USIエンジンの初期化に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    # --- メタデータの読み込み ---
    with open(args.metadata_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        all_kifs_meta = list(reader)

    # --- 出力CSVの準備 ---
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_header = reader.fieldnames + ['eval_score_cp'] # 評価値カラムを追加

    print(f"評価結果を '{output_csv_path}' に書き込みます。")

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_header)
        writer.writeheader()

        # ファイルパスでグループ化し、ファイルI/Oを最小限に抑える
        kifs_by_file = defaultdict(list)
        for meta in all_kifs_meta:
            kifs_by_file[meta['file_path']].append(meta)

        with tqdm(kifs_by_file.items(), unit="file") as pbar:
            for csa_path, metas in pbar:
                pbar.set_description(f"Evaluating {Path(csa_path).name}")
                try:
                    all_kifs_in_file = list(cshogi.CSA.Parser.parse_file(csa_path))
                    
                    for meta in metas:
                        kif_index = int(meta['kif_index'])
                        kif = all_kifs_in_file[kif_index]
                        
                        board = cshogi.Board(kif.sfen)
                        
                        # 棋譜を再生し、各局面を評価
                        for ply, move in enumerate(kif.moves, 1):
                            if ply > args.max_ply: # max_plyは評価対象の局面を絞るため
                                break
                            
                            if ply >= args.min_ply: # min_plyは評価対象の局面を絞るため
                                try:
                                    sfen = board.sfen()
                                    score_type, score_value = engine.evaluate_sfen(sfen, args.depth)
                                    
                                    # センチポーンに変換 (詰みの場合もCP_MAXで表現)
                                    eval_score_cp = score_value
                                    if score_type == "mate":
                                        eval_score_cp = 32000 if score_value > 0 else -32000
                                    
                                    # メタデータに評価値を追加して書き出し
                                    meta_with_eval = meta.copy()
                                    meta_with_eval['eval_score_cp'] = eval_score_cp
                                    writer.writerow(meta_with_eval)

                                except Exception as e:
                                    print(f"\n評価エラー: {csa_path} 棋譜{kif_index} 手数{ply} ({e})", file=sys.stderr)
                                    # エラーが発生した局面はスキップし、次の局面へ
                            
                            board.push(move)

                except Exception as e:
                    print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)
    
    engine.quit()
    print("フェーズ2: 局面評価が完了し、評価値付きメタデータCSVが生成されました。")


def run_evaluate_metadata(args: argparse.Namespace) -> None:
    evaluate_metadata_logic(args)

    # --- フィルタリング ---
    print("フェーズ2: 棋譜のフィルタリングと分割を開始します。")
    
    with open(args.metadata_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_kifs = list(reader)

    print(f"フィルタリング前 - 合計棋譜数: {len(all_kifs)}")

    filtered_kifs = []
    for kif in all_kifs:
        try:
            rating_b = int(kif['rating_b'])
            rating_w = int(kif['rating_w'])
            
            # 最低レーティングフィルタ
            if rating_b < args.min_rating or rating_w < args.min_rating:
                continue
            
            # レーティング差フィルタ
            if abs(rating_b - rating_w) > args.max_rating_diff:
                continue
            
            filtered_kifs.append(kif)
        except (ValueError, KeyError):
            continue # ヘッダや不正な行をスキップ

    print(f"フィルタリング後 - 合計棋譜数: {len(filtered_kifs)}")

    if not filtered_kifs:
        print("エラー: フィルタリング条件を満たす棋譜がありません。", file=sys.stderr)
        sys.exit(1)

    # --- 分割 ---
    random.shuffle(filtered_kifs)
    
    val_size = int(len(filtered_kifs) * args.val_split)
    train_kifs = filtered_kifs[val_size:]
    val_kifs = filtered_kifs[:val_size]

    print(f"分割結果 - 訓練データ: {len(train_kifs)}棋譜, 検証データ: {len(val_kifs)}棋譜")

    # --- データセット生成 ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 訓練データ
    process_kifs(
        train_kifs,
        str(output_dir / "train.bin"),
        args.win_value,
        args.min_ply,
        args.max_ply,
        evaluated_scores # 評価値付きスコアを渡す
    )
    
    # 検証データ
    process_kifs(
        val_kifs,
        str(output_dir / "val.bin"),
        args.win_value,
        args.min_ply,
        args.max_ply,
        evaluated_scores # 評価値付きスコアを渡す
    )

    print("\nすべての処理が完了しました。")


# ================================
# main
# ================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSA棋譜からnodchip/nnue-pytorch形式の学習データを生成するスクリプト。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")

    # --- 'extract' コマンドのパーサー ---
    extract_parser = subparsers.add_parser(
        "extract", help="CSAファイルから棋譜のメタデータを抽出し、CSVファイルに保存します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    extract_parser.add_argument("--csa-dir", required=True, help="CSAファイルが格納されているルートディレクトリ。")
    extract_parser.add_argument("--output-dir", default="output_data", help="生成されたメタデータ(.csv)を保存するディレクトリ。")
    extract_parser.add_argument("--metadata-csv", default=None, help="メタデータCSVのパス。指定しない場合は output-dir/metadata.csv になります。")
    extract_parser.set_defaults(func=run_extract_metadata)

    # --- 'evaluate' コマンドのパーサー ---
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="メタデータCSVの棋譜局面をUSIエンジンで評価し、評価値付きCSVを生成します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    evaluate_parser.add_argument("--metadata-csv", required=True, help="入力となるメタデータCSVのパス。")
    evaluate_parser.add_argument("--engine-path", required=True, help="USIエンジンの実行ファイルのパス。")
    evaluate_parser.add_argument("--output-csv", default=None, help="評価値付きメタデータCSVの出力パス。指定しない場合は output-dir/evaluated_metadata.csv になります。")
    evaluate_parser.add_argument("--depth", type=int, default=10, help="USIエンジンの探索の深さ。")
    evaluate_parser.add_argument("--min-ply", type=int, default=20, help="評価を開始する最小手数。")
    evaluate_parser.add_argument("--max-ply", type=int, default=512, help="評価する最大手数。")
    evaluate_parser.set_defaults(func=run_evaluate_metadata)

    # --- 'generate' コマンドのパーサー ---
    generate_parser = subparsers.add_parser(
        "generate", help="メタデータCSVから学習データと検証データ(.bin)を生成します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    generate_parser.add_argument("--output-dir", default="output_data", help="生成されたデータセット(.bin)とメタデータ(.csv)を保存するディレクトリ。")
    generate_parser.add_argument("--metadata-csv", default=None, help="入力となるメタデータCSVのパス。指定しない場合は output-dir/metadata.csv になります。")
    generate_parser.add_argument("--evaluated-metadata-csv", default=None, help="評価値付きメタデータCSVのパス。指定した場合、このCSVの評価値を使用します。")
    
    # フィルタリング設定
    generate_parser.add_argument("--min-rating", type=int, default=3000, help="学習対象とする対局者の最低レーティング。")
    generate_parser.add_argument("--max-rating-diff", type=int, default=1000, help="学習対象とする対局者間のレーティング差の上限。")
    
    # データ生成設定
    generate_parser.add_argument("--win-value", type=int, default=600, help="勝敗から変換する評価値の絶対値。")
    generate_parser.add_argument("--min-ply", type=int, default=20, help="この手数未満の局面は学習データにしない。")
    generate_parser.add_argument("--max-ply", type=int, default=512, help="安全のための手数の上限。")

    # 実行制御
    generate_parser.add_argument("--val-split", type=float, default=0.1, help="検証データとして分割する割合 (0.0-1.0)。")
    generate_parser.set_defaults(func=run_generate_datasets)

    args = parser.parse_args()

    if args.command:
        # output_dirが指定されている場合のみ作成
        if hasattr(args, 'output_dir') and args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # metadata_csvまたはoutput_csvのデフォルト値を設定
        if args.command == "extract" and args.metadata_csv is None:
            args.metadata_csv = str(Path(args.output_dir) / "metadata.csv")
        elif args.command == "evaluate" and args.output_csv is None:
            args.output_csv = str(Path(args.output_dir) / "evaluated_metadata.csv")
        elif args.command == "generate":
            if args.metadata_csv is None:
                args.metadata_csv = str(Path(args.output_dir) / "metadata.csv")
            if args.evaluated_metadata_csv is None:
                args.evaluated_metadata_csv = str(Path(args.output_dir) / "evaluated_metadata.csv")
            
        args.func(args)
    else:
        parser.print_help()

def run_extract_metadata(args: argparse.Namespace) -> None:
    extract_metadata(args.csa_dir, args.metadata_csv)

def run_generate_datasets(args: argparse.Namespace) -> None:
    generate_datasets_logic(args)

if __name__ == "__main__":
    main()
