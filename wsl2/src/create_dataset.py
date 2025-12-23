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
                                
                                # game_resultを cshogi(1,2,0) -> 学習用(+1,-1,0) に変換
                                cshogi_result = int(meta['game_result'])
                                if cshogi_result == 1:      # 先手勝ち
                                    write_result = 1
                                elif cshogi_result == 2:    # 後手勝ち
                                    write_result = -1
                                else:                       # 引き分け(0) or その他
                                    write_result = 0
                                psv[0]["game_result"] = np.int8(write_result)
                                psv.tofile(f_out)

                            board.push(move)

                except Exception as e:
                    print(f"\nデータ生成エラー: {csa_path} ({e})", file=sys.stderr)

    print(f"データセット '{output_path}' の生成が完了しました。")


def write_bin_file(positions: list, output_path: str):
    """
    局面情報のリストから、PackedSfenValue形式の.binファイルを生成する。
    """
    print(f"データセット '{output_path}' の生成を開始します。対象局面数: {len(positions)}")
    board = cshogi.Board()
    psv = np.zeros(1, dtype=cshogi.PackedSfenValue)

    with open(output_path, "wb") as f_out:
        for pos in tqdm(positions, desc=f"Writing {Path(output_path).name}"):
            try:
                board.set_sfen(pos['sfen'])
                board.to_psfen(psv)

                # game_resultを cshogi(1,2,0) -> 学習用(+1,-1,0) に変換
                cshogi_result = int(pos['game_result'])
                if cshogi_result == 1:
                    write_result = 1
                elif cshogi_result == 2:
                    write_result = -1
                else:
                    write_result = 0

                psv[0]["score"] = np.int16(pos['eval_score_cp'])
                psv[0]["move"] = np.uint16(0)
                psv[0]["gamePly"] = np.uint16(pos['ply'])
                psv[0]["game_result"] = np.int8(write_result)
                psv.tofile(f_out)
            except Exception as e:
                print(f"\nデータ書き込みエラー: {pos} ({e})", file=sys.stderr)

def write_hdf5_file(positions: list, output_path: str):
    """
    局面情報のリストから、メタデータ付きのHDF5ファイルを生成する。
    """
    import h5py
    print(f"データセット '{output_path}' の生成を開始します。対象局面数: {len(positions)}")

    # 各データを格納するリストを準備
    packed_sfens = []
    scores = []
    game_plies = []
    game_results = []
    ratings_b = []
    ratings_w = []

    board = cshogi.Board()
    psv_buffer = np.zeros(1, dtype=cshogi.PackedSfenValue)

    for pos in tqdm(positions, desc=f"Preparing {Path(output_path).name}"):
        try:
            board.set_sfen(pos['sfen'])
            board.to_psfen(psv_buffer)
            packed_sfens.append(psv_buffer.tobytes())

            cshogi_result = int(pos['game_result'])
            if cshogi_result == 1:
                write_result = 1
            elif cshogi_result == 2:
                write_result = -1
            else:
                write_result = 0

            scores.append(np.int16(pos['eval_score_cp']))
            game_plies.append(np.uint16(pos['ply']))
            game_results.append(np.int8(write_result))
            ratings_b.append(np.int16(pos['rating_b']))
            ratings_w.append(np.int16(pos['rating_w']))
        except Exception as e:
            print(f"\nデータ準備エラー: {pos} ({e})", file=sys.stderr)

    # HDF5ファイルに書き出し
    with h5py.File(output_path, 'w') as f:
        # PackedSfenValueのdtypeを再現
        dt = np.dtype([('sfen', np.uint8, 32), ('score', '<i2'), ('move', '<u2'), ('gamePly', '<u2'), ('game_result', 'i1)])
        
        psfen_dataset = f.create_dataset('packed_sfens', (len(packed_sfens),), dtype=dt, compression='gzip')
        psfen_dataset[:] = np.frombuffer(b''.join(packed_sfens), dtype=dt)

        f.create_dataset('scores', data=np.array(scores, dtype=np.int16), compression='gzip')
        f.create_dataset('game_plies', data=np.array(game_plies, dtype=np.uint16), compression='gzip')
        f.create_dataset('game_results', data=np.array(game_results, dtype=np.int8), compression='gzip')
        f.create_dataset('ratings_b', data=np.array(ratings_b, dtype=np.int16), compression='gzip')
        f.create_dataset('ratings_w', data=np.array(ratings_w, dtype=np.int16), compression='gzip')
    
    print(f"データセット '{output_path}' の生成が完了しました。")


def generate_datasets_logic(args: argparse.Namespace) -> None:
    """
    評価値・SFEN付きCSVを読み込み、訓練データと検証データを生成する。
    """
    # --- 入力CSVの存在確認 ---
    if not Path(args.input_csv).exists():
        print(f"エラー: 入力ファイル '{args.input_csv}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    print(f"--- データセット生成を開始 ---")
    print(f"入力ファイル: {args.input_csv}")
    print(f"出力形式: {args.format}")

    # --- 評価値付きCSVの読み込み ---
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_positions = list(reader)
    
    if not all_positions:
        print("エラー: 入力ファイルにデータがありません。", file=sys.stderr)
        sys.exit(1)

    print(f"読み込み完了。総局面数: {len(all_positions)}")

    # --- 分割 ---
    random.shuffle(all_positions)
    
    val_size = int(len(all_positions) * args.val_split)
    train_positions = all_positions[val_size:]
    val_positions = all_positions[:val_size]

    print(f"分割結果 - 訓練データ: {len(train_positions)}局面, 検証データ: {len(val_positions)}局面")

    # --- データセット生成 ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == 'bin':
        write_bin_file(train_positions, str(output_dir / "train.bin"))
        write_bin_file(val_positions, str(output_dir / "val.bin"))
    elif args.format == 'hdf5':
        # h5pyのインポートチェック
        try:
            import h5py
        except ImportError:
            print("エラー: h5pyがインストールされていません。'pip install h5py' を実行してください。", file=sys.stderr)
            sys.exit(1)
        write_hdf5_file(train_positions, str(output_dir / "train.h5"))
        write_hdf5_file(val_positions, str(output_dir / "val.h5"))

    print("\nすべての処理が完了しました。")


def evaluate_metadata_logic(args: argparse.Namespace) -> None:
    """
    メタデータCSVを読み込み、USIエンジンで局面を評価し、評価値とSFEN付きのCSVを生成する。
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
    
    print(f"--- 局面評価を開始 ---")
    print(f"入力ファイル: {args.metadata_csv}")

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
        header = reader.fieldnames

    # --- 出力CSVの準備 ---
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_header = header + ['ply', 'eval_score_cp', 'sfen'] # ply, 評価値, SFENカラムを追加

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
                            if ply > args.max_ply:
                                break
                            
                            if ply >= args.min_ply:
                                try:
                                    sfen = board.sfen()
                                    score_type, score_value = engine.evaluate_sfen(sfen, args.depth)
                                    
                                    eval_score_cp = score_value
                                    if score_type == "mate":
                                        eval_score_cp = 32000 if score_value > 0 else -32000
                                    
                                    meta_with_eval = meta.copy()
                                    meta_with_eval['ply'] = ply
                                    meta_with_eval['eval_score_cp'] = eval_score_cp
                                    meta_with_eval['sfen'] = sfen
                                    writer.writerow(meta_with_eval)

                                except Exception as e:
                                    print(f"\n評価エラー: {csa_path} 棋譜{kif_index} 手数{ply} ({e})", file=sys.stderr)
                            
                            board.push(move)

                except Exception as e:
                    print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)
    
    engine.quit()
    print("局面評価が完了し、評価値・SFEN付きメタデータCSVが生成されました。")


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
    
    parser.add_argument("-c", "--config", help="設定YAMLファイルのパス。コマンドライン引数はYAMLの設定を上書きします。")
    
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
        "generate", help="評価値付きCSVから学習データと検証データ(.bin)を生成します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- 'filter' コマンドのパーサー ---
    filter_parser = subparsers.add_parser(
        "filter", help="メタデータCSVをフィルタリングし、新しいCSVファイルを出力します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    filter_parser.add_argument("--metadata-csv", required=True, help="入力となるメタデータCSVのパス。")
    filter_parser.add_argument("--output-csv", required=True, help="フィルタリング結果を保存するCSVのパス。")
    
    # フィルタリング設定
    filter_parser.add_argument("--min-rating", type=int, default=3000, help="学習対象とする対局者の最低レーティング。")
    filter_parser.add_argument("--max-rating", type=int, default=9999, help="学習対象とする対局者の最大レーティング。")
    filter_parser.add_argument("--max-rating-diff", type=int, default=1000, help="学習対象とする対局者間のレーティング差の上限。")
    filter_parser.add_argument("--min-moves", type=int, default=0, help="学習対象とする棋譜の最小手数。")
    filter_parser.add_argument("--max-moves", type=int, default=999, help="学習対象とする棋譜の最大手数。")
    filter_parser.add_argument("--allowed-results", type=str, default="win,lose,draw", help="含める勝敗結果をカンマ区切りで指定 (win,lose,draw)。")
    filter_parser.add_argument("--filter-by-rating-outcome", action='store_true', help="レーティングが高い方が勝った棋譜のみを対象とする。")
    filter_parser.set_defaults(func=run_filter_metadata)


    generate_parser.add_argument("--input-csv", required=True, help="入力となる評価値付きメタデータCSVのパス。")
    generate_parser.add_argument("--output-dir", required=True, help="生成されたデータセット(.bin or .h5)を保存するディレクトリ。")
    generate_parser.add_argument("--format", choices=['bin', 'hdf5'], default='bin', help="出力フォーマットを選択します。")

    # 実行制御
    generate_parser.add_argument("--val-split", type=float, default=0.1, help="検証データとして分割する割合 (0.0-1.0)。")
    generate_parser.set_defaults(func=run_generate_datasets)

    # --- 引数のパースと設定の上書き ---
    # 1. 部分的なパースでconfigファイルパスを取得
    temp_args, _ = parser.parse_known_args()

    # 2. YAML設定の読み込みとデフォルト値の上書き
    if temp_args.config and Path(temp_args.config).exists():
        print(f"設定ファイル '{temp_args.config}' を読み込みます。")
        with open(temp_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # サブコマンドごとの設定をデフォルト値として適用
        for command_name, command_args in yaml_config.items():
            if command_name == parser.prog: # トップレベルの引数は未対応
                continue
            parser.set_defaults(**command_args)

    # 3. 最終的なパース（コマンドライン引数がYAML設定を上書き）
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
            if args.evaluated_metadata_csv is None and Path(str(Path(args.output_dir) / "evaluated_metadata.csv")).exists():
                 args.evaluated_metadata_csv = str(Path(args.output_dir) / "evaluated_metadata.csv")
            
        args.func(args)
    else:
        parser.print_help()

def run_extract_metadata(args: argparse.Namespace) -> None:
    extract_metadata(args.csa_dir, args.metadata_csv)

def run_generate_datasets(args: argparse.Namespace) -> None:
    generate_datasets_logic(args)

def run_filter_metadata(args: argparse.Namespace) -> None:
    """
    メタデータCSVをフィルタリングし、結果を新しいCSVファイルに書き出す。
    """
    if not Path(args.metadata_csv).exists():
        print(f"エラー: 入力メタデータファイル '{args.metadata_csv}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    print(f"--- メタデータのフィルタリングを開始 ---")
    print(f"入力ファイル: {args.metadata_csv}")
    print(f"出力ファイル: {args.output_csv}")

    with open(args.metadata_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_kifs = list(reader)
        header = reader.fieldnames

    print(f"フィルタリング前 - 合計棋譜数: {len(all_kifs)}")

    # cshogi基準(1:先手勝ち, 2:後手勝ち, 0:引き分け)に合わせる
    result_map = {'win': 1, 'lose': 2, 'draw': 0}
    allowed_results_int = {result_map[res.strip()] for res in args.allowed_results.split(',')}

    filtered_kifs = []
    for kif in tqdm(all_kifs, desc="フィルタリング中"):
        try:
            rating_b = int(kif['rating_b'])
            rating_w = int(kif['rating_w'])
            total_moves = int(kif['total_moves'])
            game_result = int(kif['game_result'])

            if not (args.min_rating <= rating_b <= args.max_rating and
                    args.min_rating <= rating_w <= args.max_rating):
                continue
            
            if abs(rating_b - rating_w) > args.max_rating_diff:
                continue

            if not (args.min_moves <= total_moves <= args.max_moves):
                continue

            if game_result not in allowed_results_int:
                continue

            if args.filter_by_rating_outcome:
                is_black_stronger = rating_b > rating_w
                is_white_stronger = rating_w > rating_b
                
                if is_black_stronger and game_result != 1:
                    continue
                if is_white_stronger and game_result != 2:
                    continue

            filtered_kifs.append(kif)
        except (ValueError, KeyError):
            continue

    print(f"フィルタリング後 - 合計棋譜数: {len(filtered_kifs)}")

    # フィルタ結果をCSVに書き出し
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(filtered_kifs)
        print("フィルタリング処理が完了しました。")
    except IOError as e:
        print(f"エラー: ファイルの書き込みに失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
