# -*- coding: utf-8 -*-
"""
CSA棋譜ファイルから、AIの学習データセットを生成する多機能スクリプト。
(以下略)
"""
import os
import csv
import random
import argparse
import sys
from collections import defaultdict
from pathlib import Path
import yaml
import traceback
import sqlite3

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import cshogi
import numpy as np
from usi import UsiEngine
from extract_features import make_feature_dict

# ================================
# データベース管理
# ================================

class SfenDB:
    """SFENの出現頻度と評価値を管理するSQLiteクラス。"""
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute('PRAGMA journal_mode = WAL')
        self.cursor.execute('PRAGMA synchronous = NORMAL')
        
        # 出現頻度管理
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sfen_counts (
                sfen TEXT PRIMARY KEY,
                total_count INTEGER DEFAULT 0,
                output_count INTEGER DEFAULT 0
            )
        ''')
        
        # 評価値キャッシュ (sfen + 探索条件をキーにする)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sfen_cache (
                sfen TEXT,
                depth INTEGER,
                nodes INTEGER,
                movetime INTEGER,
                score_type TEXT,
                score_value INTEGER,
                PRIMARY KEY (sfen, depth, nodes, movetime)
            )
        ''')
        self.conn.commit()

    def increment_total_count_batch(self, sfens: list):
        self.cursor.executemany('''
            INSERT INTO sfen_counts (sfen, total_count) VALUES (?, 1)
            ON CONFLICT(sfen) DO UPDATE SET total_count = total_count + 1
        ''', [(s,) for s in sfens])

    def check_output_limit(self, sfen: str, max_count: int) -> bool:
        """出力回数が上限に達しているか確認。max_count=0は無制限。"""
        if max_count <= 0: return True
        self.cursor.execute('SELECT output_count FROM sfen_counts WHERE sfen = ?', (sfen,))
        res = self.cursor.fetchone()
        count = res['output_count'] if res else 0
        return count < max_count

    def increment_output_count(self, sfen: str):
        self.cursor.execute('''
            INSERT INTO sfen_counts (sfen, output_count) VALUES (?, 1)
            ON CONFLICT(sfen) DO UPDATE SET output_count = output_count + 1
        ''', (sfen,))

    def get_eval(self, sfen: str, depth: int, nodes: int, movetime: int):
        self.cursor.execute('''
            SELECT score_type, score_value FROM sfen_cache 
            WHERE sfen = ? AND depth IS ? AND nodes IS ? AND movetime IS ?
        ''', (sfen, depth, nodes, movetime))
        res = self.cursor.fetchone()
        return (res['score_type'], res['score_value']) if res else None

    def save_eval(self, sfen: str, depth: int, nodes: int, movetime: int, score_type: str, score_value: int):
        self.cursor.execute('''
            INSERT INTO sfen_cache (sfen, depth, nodes, movetime, score_type, score_value) 
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sfen, depth, nodes, movetime) DO UPDATE SET 
                score_type = excluded.score_type, 
                score_value = excluded.score_value
        ''', (sfen, depth, nodes, movetime, score_type, score_value))

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

# ================================
# データ生成ロジック
# ================================

def count_sfen_logic(args: argparse.Namespace) -> None:
    """
    [count-sfenコマンド] 全棋譜をスキャンしてSFENの出現頻度をカウントする。
    """
    if not Path(args.input_csv).exists(): sys.exit(f"エラー: 入力ファイル '{args.input_csv}' が見つかりません。")
    db = SfenDB(args.db_path)
    
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f:
        metas = list(csv.DictReader(f))
    
    print(f"--- SFEN頻度集計を開始 (対象対局数: {len(metas)}) ---")
    kifs_by_file = defaultdict(list)
    for m in metas: kifs_by_file[m['file_path']].append(m)
    
    with tqdm(kifs_by_file.items(), unit="file") as pbar:
        for csa_path, metas_in_file in pbar:
            pbar.set_description(f"Counting {Path(csa_path).name}")
            try:
                all_kifs = cshogi.Parser.parse_file(csa_path)
                if not all_kifs: continue
                sfens_to_update = []
                for meta in metas_in_file:
                    kif = all_kifs[int(meta['kif_index'])]
                    board = cshogi.Board(kif.sfen)
                    for ply, move in enumerate(kif.moves, 1):
                        if ply > args.max_ply: break
                        if ply >= args.min_ply:
                            sfens_to_update.append(board.sfen())
                        board.push(move)
                
                if sfens_to_update:
                    db.increment_total_count_batch(sfens_to_update)
                    # 適宜コミットしてメモリを節約
                    if len(sfens_to_update) > 1000: db.commit()
            except Exception as e:
                print(f"\nエラー: {csa_path} ({e})", file=sys.stderr)
    
    db.close()
    print("SFEN頻度集計が完了しました。")

def extract_metadata(csa_dir: str, output_csv: str) -> None:
    """
    [extractコマンド] CSAファイル群をスキャンし、棋譜のメタデータをCSVに書き出す。
    """
    print(f"フェーズ1: メタデータ抽出を開始します。出力先: {output_csv}")
    csa_files = list(Path(csa_dir).rglob('*.csa')) + list(Path(csa_dir).rglob('*.CSA'))
    if not csa_files: sys.exit(f"エラー: '{csa_dir}' 内にCSAファイルが見つかりません。")
    print(f"{len(csa_files)}個のCSAファイルをスキャンします...")
    header = ['file_path', 'kif_index', 'black_player', 'white_player', 'rating_b', 'rating_w', 'game_result', 'total_moves']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        with tqdm(csa_files, unit="file") as pbar:
            for csa_path in pbar:
                pbar.set_description(f"Processing {csa_path.name}")
                try:
                    list_of_games = cshogi.Parser.parse_file(str(csa_path))
                    if list_of_games is None: continue

                    for i, game in enumerate(list_of_games):
                        if not (game.names and len(game.names) >= 2 and game.ratings and len(game.ratings) >= 2):
                            continue

                        if game.win == cshogi.BLACK_WIN:
                            game_result = 1
                        elif game.win == cshogi.WHITE_WIN:
                            game_result = 2
                        else:
                            game_result = 0

                        writer.writerow([
                            str(csa_path), i, game.names[0], game.names[1],
                            game.ratings[0], game.ratings[1], game_result, len(game.moves)
                        ])
                except Exception as e:
                    print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)
    print("フェーズ1: メタデータ抽出が完了しました。")

def run_filter_metadata(args: argparse.Namespace) -> None:
    """
    [filterコマンド] メタデータCSVをフィルタリングし、新しいCSVファイルを出力する。
    """
    if not Path(args.input_csv).exists(): sys.exit(f"エラー: 入力メタデータファイル '{args.input_csv}' が見つかりません。")
    
    print("--- フィルタリング条件の確認 ---")
    print(f"入力ファイル: {args.input_csv}")
    print(f"出力ファイル: {args.output_csv}")
    print(f"レーティング範囲: {args.min_rating} ～ {args.max_rating}")
    print(f"最大レーティング差: {args.max_rating_diff}")
    print(f"手数範囲: {args.min_moves} ～ {args.max_moves}")
    print(f"引き分けを除外するか: {args.no_draws}")
    print(f"レーティング通りか: {args.filter_by_rating_outcome}")
    print("--------------------------")

    print(f"--- メタデータのフィルタリングを開始 ---")
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_kifs, header = list(reader), reader.fieldnames
    print(f"フィルタリング前 - 合計棋譜数: {len(all_kifs)}")
    
    filtered_kifs = []
    for kif in tqdm(all_kifs, desc="フィルタリング中"):
        try:
            rating_b, rating_w = float(kif['rating_b']), float(kif['rating_w'])
            total_moves, game_result = int(kif['total_moves']), int(kif['game_result'])
            
            if not (args.min_rating <= rating_b <= args.max_rating and args.min_rating <= rating_w <= args.max_rating): continue
            if abs(rating_b - rating_w) > args.max_rating_diff: continue
            if not (args.min_moves <= total_moves <= args.max_moves): continue
            
            if args.no_draws and game_result == 0: continue
            
            if args.filter_by_rating_outcome and ((rating_b > rating_w and game_result != 1) or (rating_w > rating_b and game_result != 2)): continue
            
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

def run_label(args: argparse.Namespace) -> None:
    """
    [labelコマンド] エンジンを使わず、対局結果から評価値を付与（ラベリング）する。
    DBの指定がある場合、同一局面の出力上限管理を行う。
    """
    if not Path(args.input_csv).exists(): sys.exit(f"エラー: 入力ファイル '{args.input_csv}' が見つかりません。")
    db = SfenDB(args.db_path) if args.db_path else None
    print(f"--- ラベリング処理を開始 (DB: {args.db_path}, 上限: {args.max_sfen_count}) ---")
    
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        all_kifs_meta, header = list(reader), reader.fieldnames
        
    output_csv_path = Path(args.output_csv)
    output_header = header + ['ply', 'eval_score_cp', 'sfen']
    print(f"ラベル付きデータを '{output_csv_path}' に書き込みます。")
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_header)
        writer.writeheader()
        kifs_by_file = defaultdict(list)
        for meta in all_kifs_meta: kifs_by_file[meta['file_path']].append(meta)
        
        with tqdm(kifs_by_file.items(), unit="file") as pbar:
            for csa_path, metas in pbar:
                pbar.set_description(f"Labeling {Path(csa_path).name}")
                try:
                    all_kifs_in_file = cshogi.Parser.parse_file(csa_path)
                    if all_kifs_in_file is None: continue
                    for meta in metas:
                        kif = all_kifs_in_file[int(meta['kif_index'])]
                        game_result = int(meta['game_result'])
                        board = cshogi.Board(kif.sfen)
                        for ply, move in enumerate(kif.moves, 1):
                            sfen = board.sfen()
                            # DB指定がある場合はキャップ判定
                            if db is None or db.check_output_limit(sfen, args.max_sfen_count):
                                current_turn = board.turn
                                score = 0
                                if game_result == 1:
                                    score = args.score_scale if current_turn == cshogi.BLACK else -args.score_scale
                                elif game_result == 2:
                                    score = -args.score_scale if current_turn == cshogi.BLACK else args.score_scale
                                
                                meta_with_eval = meta.copy()
                                meta_with_eval.update({'ply': ply, 'eval_score_cp': score, 'sfen': sfen})
                                writer.writerow(meta_with_eval)
                                
                                if db:
                                    db.increment_output_count(sfen)
                            board.push(move)
                    if db: db.commit()
                except Exception as e:
                    print(f"\nラベリング処理エラー: {csa_path} ({e})", file=sys.stderr)
    if db: db.close()
    print("ラベリング処理が完了しました。")

def get_search_params(args: argparse.Namespace, ply: int):
    """
    手数(ply)に応じて探索パラメータ(depth, nodes, movetime)を決定する。
    序盤(early_ply_threshold以下)の場合はearly_xxxの値を優先し、指定がなければ通常時(中終盤)の値を使用する。
    """
    if ply <= args.early_ply_threshold:
        d = args.early_depth if args.early_depth is not None else args.depth
        n = args.early_nodes if args.early_nodes is not None else args.nodes
        m = args.early_movetime if args.early_movetime is not None else args.movetime
    else:
        d = args.depth
        n = args.nodes
        m = args.movetime
    return d, n, m

def evaluate_metadata_logic(args: argparse.Namespace) -> None:
    """
    [evaluateコマンド] USIエンジンで各局面を評価し、評価値付きCSVを生成する。
    DBの指定がある場合、評価値の再利用と同一局面の出力上限管理を行う。
    """
    if not Path(args.input_csv).exists(): sys.exit(f"エラー: 入力メタデータファイル '{args.input_csv}' が見つかりません。")
    if not Path(args.engine_path).exists(): sys.exit(f"エラー: エンジン実行ファイルが見つかりません: {args.engine_path}")
    
    db = SfenDB(args.db_path) if args.db_path else None
    print(f"--- 局面評価を開始 (DB: {args.db_path}, 上限: {args.max_sfen_count}) ---")
    
    try:
        engine = UsiEngine(str(args.engine_path))
        print("USIエンジン準備完了。")
    except Exception as e:
        sys.exit(f"エラー: USIエンジンの初期化に失敗しました: {e}")
        
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        all_kifs_meta, header = list(reader), reader.fieldnames
        
    output_csv_path = Path(args.output_csv)
    output_header = header + ['ply', 'eval_score_cp', 'sfen']
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_header)
        writer.writeheader()
        kifs_by_file = defaultdict(list)
        for meta in all_kifs_meta: kifs_by_file[meta['file_path']].append(meta)
        
        with tqdm(kifs_by_file.items(), unit="file") as pbar:
            for csa_path, metas in pbar:
                pbar.set_description(f"Evaluating {Path(csa_path).name}")
                try:
                    all_kifs_in_file = cshogi.Parser.parse_file(csa_path)
                    if all_kifs_in_file is None: continue
                    for meta in metas:
                        kif = all_kifs_in_file[int(meta['kif_index'])]
                        board = cshogi.Board(kif.sfen)
                        for ply, move in enumerate(kif.moves, 1):
                            if ply > args.max_ply: break
                            if ply >= args.min_ply:
                                sfen = board.sfen()
                                # DB指定がある場合はキャップ判定
                                if db is None or db.check_output_limit(sfen, args.max_sfen_count):
                                    try:
                                        d, n, m = get_search_params(args, ply)
                                        score_type, score_value = None, None
                                        
                                        # DB指定がある場合はキャッシュ確認
                                        if db:
                                            cached_eval = db.get_eval(sfen, d, n, m)
                                            if cached_eval:
                                                score_type, score_value = cached_eval
                                        
                                        # キャッシュがなければエンジン評価
                                        if score_type is None:
                                            score_type, score_value = engine.evaluate_sfen(sfen, depth=d, nodes=n, movetime=m)
                                            if db:
                                                db.save_eval(sfen, d, n, m, score_type, score_value)
                                        
                                        eval_score_cp = score_value if score_type == "cp" else (32000 if score_value > 0 else -32000)
                                        meta_with_eval = meta.copy()
                                        meta_with_eval.update({'ply': ply, 'eval_score_cp': eval_score_cp, 'sfen': sfen})
                                        writer.writerow(meta_with_eval)
                                        
                                        if db:
                                            db.increment_output_count(sfen)
                                    except Exception as e:
                                        print(f"\n評価エラー: 棋譜{meta['kif_index']} 手数{ply} ({e})", file=sys.stderr)
                                        traceback.print_exc()
                            board.push(move)
                    if db: db.commit()
                except Exception as e:
                    print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)
                    traceback.print_exc()
    engine.quit()
    if db: db.close()
    print("局面評価が完了しました。")

def write_bin_file(positions: list, output_path: str):
    """
    局面情報のリストから、PackedSfenValue形式の.binファイルを生成する。
    """
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
    """
    [generateコマンド] 評価値付きCSVから、.bin形式の学習データを生成する。
    """
    if not Path(args.input_csv).exists(): sys.exit(f"エラー: 入力ファイル '{args.input_csv}' が見つかりません。")
    print(f"--- .bin データセット生成を開始 ---")
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f:
        all_positions = list(csv.DictReader(f))
    if not all_positions: sys.exit("エラー: 入力ファイルにデータがありません。")
    print(f"読み込み完了。総局面数: {len(all_positions)}")
    random.shuffle(all_positions)
    val_size = int(len(all_positions) * args.val_split)
    train_positions, val_positions = all_positions[val_size:], all_positions[:val_size]
    print(f"分割結果 - 訓練: {len(train_positions)}局面, 検証: {len(val_positions)}局面")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_bin_file(train_positions, str(output_dir / "train.bin"))
    write_bin_file(val_positions, str(output_dir / "val.bin"))
    print("\nすべての処理が完了しました。")

def run_build_h5(args: argparse.Namespace) -> None:
    """
    [build-h5コマンド] フィルタリング済みCSVから、階層的なHDF5データセットを生成する。
    DBの指定がある場合、評価値のキャッシュ（再利用）を行う。
    """
    try: import h5py
    except ImportError: sys.exit("エラー: h5pyがインストールされていません。'pip install h5py' を実行してください。")
    if not Path(args.input_csv).exists(): sys.exit(f"エラー: 入力ファイル '{args.input_csv}' が見つかりません。")
    if not Path(args.engine_path).exists(): sys.exit(f"エラー: エンジン実行ファイルが見つかりません: {args.engine_path}")
    
    db = SfenDB(args.db_path) if args.db_path else None
    print(f"--- HDF5データセット構築開始 (DB: {args.db_path}) ---")
    
    try:
        engine = UsiEngine(str(args.engine_path))
        print("USIエンジン準備完了。")
    except Exception as e:
        sys.exit(f"エラー: USIエンジンの初期化に失敗しました: {e}")
        
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f_in:
        games_to_process = list(csv.DictReader(f_in))
        
    candidate_dtype = np.dtype([('move', np.uint16), ('score', np.int16), ('is_mate', np.bool_)])
    # 特徴量用のデータ型定義
    hand_dtype = np.dtype([(f'hand_{p}', np.int8) for p in ['P', 'L', 'N', 'S', 'G', 'B', 'R']])
    feature_dtype = np.dtype([
        ('in_check', np.int8),
        ('is_mate', np.int8),
        ('capture_available', np.int8),
        ('give_check_available', np.int8),
        ('sente_hand', hand_dtype),
        ('gote_hand', hand_dtype)
    ])
    
    position_dtype = np.dtype([
        ('ply', np.uint16),
        ('psv', cshogi.PackedSfenValue),
        ('is_check', np.bool_),
        ('features', feature_dtype),
        ('candidates', h5py.vlen_dtype(candidate_dtype))
    ])
    
    with h5py.File(args.output_h5, 'w') as f_out:
        print(f"{len(games_to_process)}対局の処理を開始します。")
        for i, game_meta in enumerate(tqdm(games_to_process, desc="Processing games")):
            game_group = f_out.create_group(f"game_{i}")
            for key, value in game_meta.items(): game_group.attrs[key] = value
            try:
                list_of_games = cshogi.Parser.parse_file(game_meta['file_path'])
                if list_of_games is None: continue
                game = list_of_games[int(game_meta['kif_index'])]
                board = cshogi.Board(game.sfen)
                game_positions_data = []
                for ply, move in enumerate(game.moves, 1):
                    sfen = board.sfen()
                    feat_dict = make_feature_dict(board)
                    
                    d, n, m = get_search_params(args, ply)
                    
                    # キャッシュ確認 (MultiPVの場合は現状エンジンを叩く必要があるが、
                    # 将来的にはMultiPVの結果も保存するように拡張可能。今回は単一評価値のみDB化)
                    # ※ build-h5はMultiPV前提なので、DBに候補手リストを保存するロジックが必要。
                    # ここでは簡略化のため、build-h5のキャッシュはスキップするか、
                    # MultiPV用の別のキャッシュテーブルを検討する。
                    # 今回はMultiPVの結果をまるごと保存する仕組みがないため、そのまま実行。
                    
                    candidates_info = engine.get_multipv(sfen, depth=d, nodes=n, movetime=m, num_pv=args.num_pv)
                    candidates_list = [(cand['move'], cand['score'], cand['is_mate']) for cand in candidates_info]
                    
                    pos_struct = np.zeros(1, dtype=position_dtype)
                    pos_struct[0]['ply'] = ply
                    board.to_psfen(pos_struct[0]['psv'])
                    pos_struct[0]['is_check'] = board.is_check()
                    
                    f = pos_struct[0]['features']
                    f['in_check'] = feat_dict['in_check']
                    f['is_mate'] = feat_dict['is_mate']
                    f['capture_available'] = feat_dict['capture_available']
                    f['give_check_available'] = feat_dict['give_check_available']
                    for p in ['P', 'L', 'N', 'S', 'G', 'B', 'R']:
                        f['sente_hand'][f'hand_{p}'] = feat_dict[f'S_hand_{p}']
                        f['gote_hand'][f'hand_{p}'] = feat_dict[f'G_hand_{p}']
                        
                    pos_struct[0]['candidates'] = np.array(candidates_list, dtype=candidate_dtype)
                    game_positions_data.append(pos_struct[0])
                    board.push(move)
                if game_positions_data:
                    game_group.create_dataset('positions', data=np.array(game_positions_data, dtype=position_dtype), compression='gzip')
            except Exception as e:
                print(f"\n対局処理エラー: {game_meta.get('file_path')} ({e})", file=sys.stderr)
    engine.quit()
    if db: db.close()
    print("\nHDF5データセットの構築が完了しました。")

def main() -> None:
    """スクリプトのエントリポイント。引数をパースして各処理を実行する。"""
    parser = argparse.ArgumentParser(description="CSA棋譜から学習データを生成するスクリプト。", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", help="設定YAMLファイルのパス。")
    subparsers = parser.add_subparsers(dest="command", required=True, help="利用可能なコマンド")

    extract_parser = subparsers.add_parser("extract", help="CSAファイルから棋譜のメタデータを抽出します。")
    extract_parser.add_argument("--csa-dir", help="CSAファイルが格納されているルートディレクトリ。")
    extract_parser.add_argument("--output-csv", help="メタデータCSVの出力パス。")
    extract_parser.set_defaults(func=lambda args: extract_metadata(args.csa_dir, args.output_csv))

    filter_parser = subparsers.add_parser("filter", help="メタデータCSVをフィルタリングします。")
    filter_parser.add_argument("--input-csv", help="入力となるメタデータCSVのパス。")
    filter_parser.add_argument("--output-csv", help="フィルタリング結果を保存するCSVのパス。")
    filter_parser.add_argument("--min-rating", type=int, default=0)
    filter_parser.add_argument("--max-rating", type=int, default=9999)
    filter_parser.add_argument("--max-rating-diff", type=int, default=9999)
    filter_parser.add_argument("--min-moves", type=int, default=0)
    filter_parser.add_argument("--max-moves", type=int, default=999)
    filter_parser.add_argument("--no-draws", action='store_true', help="これを指定すると、引き分けの対局を除外します。")
    filter_parser.add_argument("--filter-by-rating-outcome", action='store_true', help="レーティングが高い方のプレイヤーが勝った対局のみを抽出します（番狂わせを除外）。")
    filter_parser.set_defaults(func=run_filter_metadata)

    count_sfen_parser = subparsers.add_parser("count-sfen", help="SFENの出現頻度をカウントしてDBに保存します。")
    count_sfen_parser.add_argument("--input-csv", help="入力となるフィルタリング済みCSVのパス。")
    count_sfen_parser.add_argument("--db-path", default="sfen_cache.db", help="SFEN頻度と評価値を保存するSQLite DBのパス。")
    count_sfen_parser.add_argument("--min-ply", type=int, default=0)
    count_sfen_parser.add_argument("--max-ply", type=int, default=999)
    count_sfen_parser.set_defaults(func=count_sfen_logic)

    label_parser = subparsers.add_parser("label", help="対局結果から評価値をラベリングします（エンジン不要）。")
    label_parser.add_argument("--input-csv", help="入力となるフィルタリング済みCSVのパス。")
    label_parser.add_argument("--output-csv", help="ラベリング結果を保存するCSVのパス。")
    label_parser.add_argument("--score-scale", type=int, default=600)
    label_parser.add_argument("--db-path", help="SFEN頻度を管理するSQLite DBのパス。指定すると上限管理が有効になります。")
    label_parser.add_argument("--max-sfen-count", type=int, default=0, help="同一SFENの最大出力回数。0は無制限。")
    label_parser.set_defaults(func=run_label)

    evaluate_parser = subparsers.add_parser("evaluate", help="フィルタリング済みCSVの局面を評価します。")
    evaluate_parser.add_argument("--input-csv", help="入力となるフィルタリング済みCSVのパス。")
    evaluate_parser.add_argument("--engine-path", help="USIエンジンの実行ファイルのパス。")
    evaluate_parser.add_argument("--output-csv", help="評価値付きCSVの出力パス。")
    evaluate_parser.add_argument("--db-path", help="SFEN頻度と評価値を管理するSQLite DBのパス。指定するとキャッシュと上限管理が有効になります。")
    evaluate_parser.add_argument("--max-sfen-count", type=int, default=0, help="同一SFENの最大出力回数。0は無制限。")
    evaluate_parser.add_argument("--depth", type=int, default=10)
    evaluate_parser.add_argument("--nodes", type=int, default=None)
    evaluate_parser.add_argument("--movetime", type=int, default=None)
    evaluate_parser.add_argument("--early-depth", type=int, default=None)
    evaluate_parser.add_argument("--early-nodes", type=int, default=None)
    evaluate_parser.add_argument("--early-movetime", type=int, default=None)
    evaluate_parser.add_argument("--early-ply-threshold", type=int, default=0, help="序盤とみなす最大手数（この手数以下でearlyパラメータを適用）。")
    evaluate_parser.add_argument("--min-ply", type=int, default=0)
    evaluate_parser.add_argument("--max-ply", type=int, default=999)
    evaluate_parser.set_defaults(func=evaluate_metadata_logic)

    generate_parser = subparsers.add_parser("generate", help="評価値付きCSVから学習データ(.bin)を生成します。")
    generate_parser.add_argument("--input-csv", help="入力となる評価値付きCSVのパス。")
    generate_parser.add_argument("--output-dir", help="生成されたデータセットを保存するディレクトリ。")
    generate_parser.add_argument("--val-split", type=float, default=0.1)
    generate_parser.set_defaults(func=generate_datasets_logic)

    build_h5_parser = subparsers.add_parser("build-h5", help="フィルタリング済みCSVから階層的なHDF5データセットを生成します。")
    build_h5_parser.add_argument("--input-csv", help="入力となるフィルタリング済みCSVのパス。")
    build_h5_parser.add_argument("--output-h5", help="出力するHDF5ファイルのパス。")
    build_h5_parser.add_argument("--engine-path", help="USIエンジンの実行ファイルのパス。")
    build_h5_parser.add_argument("--db-path", help="SFENキャッシュを管理するSQLite DBのパス。")
    build_h5_parser.add_argument("--depth", type=int, default=10)
    build_h5_parser.add_argument("--nodes", type=int, default=None)
    build_h5_parser.add_argument("--movetime", type=int, default=None)
    build_h5_parser.add_argument("--early-depth", type=int, default=None)
    build_h5_parser.add_argument("--early-nodes", type=int, default=None)
    build_h5_parser.add_argument("--early-movetime", type=int, default=None)
    build_h5_parser.add_argument("--early-ply-threshold", type=int, default=0)
    build_h5_parser.add_argument("--num-pv", type=int, default=5)
    build_h5_parser.set_defaults(func=run_build_h5)

    temp_args, _ = parser.parse_known_args()
    config = {}
    if temp_args.config and Path(temp_args.config).exists():
        with open(temp_args.config, 'r') as f:
            config = yaml.safe_load(f)
    if temp_args.command and temp_args.command in config:
        subparsers.choices[temp_args.command].set_defaults(**config.get(temp_args.command, {}))

    args = parser.parse_args()
    
    if args.command == "extract":
        if not (args.csa_dir and args.output_csv):
            sys.exit("エラー: extractコマンドには --csa-dir と --output-csv の指定が必須です。")
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    elif args.command == "filter":
        if not (args.input_csv and args.output_csv):
             sys.exit("エラー: filterコマンドには --input-csv と --output-csv の指定が必須です。")
    elif args.command == "label":
        if not (args.input_csv and args.output_csv):
             sys.exit("エラー: labelコマンドには --input-csv と --output-csv の指定が必須です。")
    elif args.command == "evaluate":
        if not (args.input_csv and args.engine_path and args.output_csv):
             sys.exit("エラー: evaluateコマンドには --input-csv, --engine-path, --output-csv の指定が必須です。")
    elif args.command == "generate":
        if not (args.input_csv and args.output_dir):
             sys.exit("エラー: generateコマンドには --input-csv と --output-dir の指定が必須です。")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    elif args.command == "build-h5":
        if not (args.input_csv and args.output_h5 and args.engine_path):
             sys.exit("エラー: build-h5コマンドには --input-csv, --output-h5, --engine-path の指定が必須です。")
        Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
        
    args.func(args)

if __name__ == "__main__":
    main()