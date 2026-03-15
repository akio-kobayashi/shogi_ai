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
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

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

from collections import defaultdict, Counter

# (中略)

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

    def update_total_counts_batch(self, sfen_counts: dict):
        """メモリで集計した頻度(dict)を一括でDBにマージする。"""
        self.cursor.executemany('''
            INSERT INTO sfen_counts (sfen, total_count) VALUES (?, ?)
            ON CONFLICT(sfen) DO UPDATE SET total_count = total_count + excluded.total_count
        ''', list(sfen_counts.items()))

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

    # (中略: get_eval, save_eval, commit, close はそのまま)

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

import zlib
from collections import defaultdict

# (中略)

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
        self.cursor.execute('PRAGMA cache_size = -1000000') # 1GBのキャッシュ
        
        # 出現頻度管理
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sfen_counts (
                sfen TEXT PRIMARY KEY,
                total_count INTEGER DEFAULT 0,
                output_count INTEGER DEFAULT 0
            )
        ''')
        
        # 評価値キャッシュ
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

    def increment_total_count_if_duplicate(self, sfen: str):
        """
        重複が検知された局面のカウントを更新。
        初めてDBに登録される場合は、ビットセットで1回見ているはずなので2から開始する。
        """
        self.cursor.execute('''
            INSERT INTO sfen_counts (sfen, total_count) VALUES (?, 2)
            ON CONFLICT(sfen) DO UPDATE SET total_count = total_count + 1
        ''', (sfen,))

    def check_output_limit(self, sfen: str, max_count: int) -> bool:
        """出力回数が上限に達しているか確認。"""
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

# (中略)

def count_sfen_logic(args: argparse.Namespace) -> None:
    """
    [count-sfenコマンド] 全棋譜をスキャンしてSFENの出現頻度をカウントし、CSVに保存する。
    外部メモリ方式: 1) バケット分割して一時ファイルへ書き出し 2) バケットごとに集計。
    """
    if not Path(args.input_csv).exists(): sys.exit(f"エラー: 入力ファイル '{args.input_csv}' が見つかりません。")
    if args.num_buckets <= 0: sys.exit("エラー: --num-buckets は1以上を指定してください。")
    if args.min_count <= 0: sys.exit("エラー: --min-count は1以上を指定してください。")

    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f:
        metas = list(csv.DictReader(f))

    print(f"--- SFEN頻度集計を開始 (対象対局数: {len(metas)}) ---")
    kifs_by_file = defaultdict(list)
    for m in metas: kifs_by_file[m['file_path']].append(m)

    output_path = Path(args.output_csv)
    temp_root = Path(args.temp_dir) if args.temp_dir else output_path.parent / f".count_sfen_tmp_{output_path.stem}"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    bucket_paths = [temp_root / f"bucket_{i:04d}.txt" for i in range(args.num_buckets)]
    buffer_by_bucket = defaultdict(list)
    buffered_rows = 0
    flush_every = 50000

    def flush_buffers():
        nonlocal buffered_rows
        for bucket_id, records in buffer_by_bucket.items():
            if not records:
                continue
            with open(bucket_paths[bucket_id], 'a', encoding='utf-8') as bf:
                bf.write("\n".join(records))
                bf.write("\n")
        buffer_by_bucket.clear()
        buffered_rows = 0

    total_positions = 0

    print(f"フェーズ1/2: 一時バケットへ書き出し中 (buckets={args.num_buckets})")
    with tqdm(kifs_by_file.items(), unit="file") as pbar:
        for csa_path, metas_in_file in pbar:
            pbar.set_description(f"Counting {Path(csa_path).name}")
            try:
                all_kifs = cshogi.Parser.parse_file(csa_path)
                if not all_kifs: continue
                for meta in metas_in_file:
                    kif = all_kifs[int(meta['kif_index'])]
                    is_black_win = 1 if int(meta.get('game_result', 0)) == 1 else 0
                    board = cshogi.Board(kif.sfen)
                    for ply, move in enumerate(kif.moves, 1):
                        if ply > args.max_ply: break
                        if ply >= args.min_ply:
                            sfen = board.sfen()
                            total_positions += 1
                            bucket_id = zlib.crc32(sfen.encode('utf-8')) % args.num_buckets
                            buffer_by_bucket[bucket_id].append(f"{sfen}\t{is_black_win}")
                            buffered_rows += 1
                            if buffered_rows >= flush_every:
                                flush_buffers()
                        board.push(move)
            except Exception as e:
                print(f"\nエラー: {csa_path} ({e})", file=sys.stderr)

    if buffered_rows:
        flush_buffers()

    unique_positions = 0
    output_rows = 0
    print("フェーズ2/2: バケットごとに集計してCSVへ出力中")
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sfen', 'total_count', 'black_win_count'])
        for bucket_path in tqdm(bucket_paths, unit="bucket"):
            if not bucket_path.exists():
                continue
            counts = {}
            with open(bucket_path, 'r', encoding='utf-8') as bf:
                for line in bf:
                    line = line.rstrip('\n')
                    if not line:
                        continue
                    sfen, is_black_win = line.rsplit('\t', 1)
                    if sfen:
                        if sfen not in counts:
                            counts[sfen] = [0, 0]  # total_count, black_win_count
                        counts[sfen][0] += 1
                        counts[sfen][1] += int(is_black_win)
            unique_positions += len(counts)
            rows = [(sfen, c[0], c[1]) for sfen, c in counts.items() if c[0] >= args.min_count]
            rows.sort(key=lambda x: (-x[1], x[0]))
            writer.writerows(rows)
            output_rows += len(rows)

    print(f"\n集計完了。")
    print(f"総局面数: {total_positions:,}")
    print(f"ユニーク局面数: {unique_positions:,}")
    print(f"出力行数(min-count={args.min_count}): {output_rows:,}")
    print(f"出力CSV: {args.output_csv}")
    if args.keep_temp:
        print(f"一時ファイルを保持しました: {temp_root}")
    else:
        shutil.rmtree(temp_root, ignore_errors=True)

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


def _partition_file_tasks(kifs_by_file: dict, num_workers: int) -> list:
    """評価対象を局面数ベースで大まかに均等分割する。"""
    buckets = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]
    items = sorted(kifs_by_file.items(), key=lambda kv: len(kv[1]), reverse=True)
    for item in items:
        idx = loads.index(min(loads))
        buckets[idx].append(item)
        loads[idx] += len(item[1])
    return buckets


def _resolve_search_params_for_worker(cfg: dict, ply: int):
    if ply <= cfg['early_ply_threshold']:
        d = cfg['early_depth'] if cfg['early_depth'] is not None else cfg['depth']
        n = cfg['early_nodes'] if cfg['early_nodes'] is not None else cfg['nodes']
        m = cfg['early_movetime'] if cfg['early_movetime'] is not None else cfg['movetime']
    else:
        d = cfg['depth']
        n = cfg['nodes']
        m = cfg['movetime']
    return d, n, m


def _normalize_eval_score(score_type: str, score_value: int) -> int:
    return score_value if score_type == "cp" else (32000 if score_value > 0 else -32000)


def _evaluate_worker(task: dict) -> dict:
    """
    並列evaluate用ワーカー。
    注意: DB共有はロック競合しやすいため、このワーカーではDBを使用しない。
    """
    cfg = task['cfg']
    output_header = task['output_header']
    worker_output_csv = task['worker_output_csv']
    file_tasks = task['file_tasks']

    rows_written = 0
    errors = 0
    files_done = 0

    engine = UsiEngine(cfg['engine_path'])
    try:
        with open(worker_output_csv, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_header)
            writer.writeheader()
            row_buffer = []
            flush_size = 1000

            for csa_path, metas in file_tasks:
                try:
                    all_kifs_in_file = cshogi.Parser.parse_file(csa_path)
                    if all_kifs_in_file is None:
                        files_done += 1
                        continue
                    for meta in metas:
                        kif = all_kifs_in_file[int(meta['kif_index'])]
                        engine.new_game()
                        board = cshogi.Board(kif.sfen)
                        for ply, move in enumerate(kif.moves, 1):
                            if ply > cfg['max_ply']:
                                break
                            if ply >= cfg['min_ply']:
                                sfen = board.sfen()
                                d, n, m = _resolve_search_params_for_worker(cfg, ply)
                                score_type, score_value = engine.evaluate_sfen(sfen, depth=d, nodes=n, movetime=m)
                                eval_score_cp = score_value if score_type == "cp" else (32000 if score_value > 0 else -32000)
                                meta_with_eval = meta.copy()
                                meta_with_eval.update({'ply': ply, 'eval_score_cp': eval_score_cp, 'sfen': sfen})
                                row_buffer.append(meta_with_eval)
                                if len(row_buffer) >= flush_size:
                                    writer.writerows(row_buffer)
                                    row_buffer.clear()
                                rows_written += 1
                            board.push(move)
                    files_done += 1
                except Exception:
                    errors += 1
                    traceback.print_exc()
            if row_buffer:
                writer.writerows(row_buffer)
    finally:
        engine.quit()

    return {
        'rows_written': rows_written,
        'errors': errors,
        'files_done': files_done,
        'worker_output_csv': worker_output_csv,
    }


def _evaluate_unique_mode(args: argparse.Namespace, kifs_by_file: dict, output_csv_path: Path, output_header: list) -> None:
    """
    ユニーク局面評価モード:
    1) 全候補局面を列挙し (sfen, depth, nodes, movetime) キーで一意化して評価
    2) 元の行へ評価値を展開してCSV出力
    """
    if args.db_path:
        sys.exit("エラー: evaluate-mode=unique では --db-path は使用できません。")
    if args.eval_workers > 1:
        sys.exit("エラー: evaluate-mode=unique は現在 --eval-workers=1 のみ対応です。")

    print("--- 局面評価を開始 (mode: unique) ---")
    records = []
    output_count_by_sfen = defaultdict(int)

    with tqdm(kifs_by_file.items(), unit="file") as pbar:
        for csa_path, metas in pbar:
            pbar.set_description(f"Collecting {Path(csa_path).name}")
            try:
                all_kifs_in_file = cshogi.Parser.parse_file(csa_path)
                if all_kifs_in_file is None:
                    continue
                for meta in metas:
                    kif = all_kifs_in_file[int(meta['kif_index'])]
                    board = cshogi.Board(kif.sfen)
                    for ply, move in enumerate(kif.moves, 1):
                        if ply > args.max_ply:
                            break
                        if ply >= args.min_ply:
                            sfen = board.sfen()
                            if args.max_sfen_count > 0 and output_count_by_sfen[sfen] >= args.max_sfen_count:
                                board.push(move)
                                continue
                            output_count_by_sfen[sfen] += 1
                            d, n, m = get_search_params(args, ply)
                            records.append((meta.copy(), ply, sfen, d, n, m))
                        board.push(move)
            except Exception as e:
                print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)
                traceback.print_exc()

    unique_keys = sorted(set((sfen, d, n, m) for _, _, sfen, d, n, m in records))
    print(f"展開対象行数: {len(records):,}, ユニーク評価キー数: {len(unique_keys):,}")

    try:
        engine = UsiEngine(str(args.engine_path))
        print("USIエンジン準備完了。")
    except Exception as e:
        sys.exit(f"エラー: USIエンジンの初期化に失敗しました: {e}")

    eval_map = {}
    try:
        for sfen, d, n, m in tqdm(unique_keys, unit="pos", desc="Evaluating unique keys"):
            score_type, score_value = engine.evaluate_sfen(sfen, depth=d, nodes=n, movetime=m)
            eval_map[(sfen, d, n, m)] = score_value if score_type == "cp" else (32000 if score_value > 0 else -32000)
    finally:
        engine.quit()

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_header)
        writer.writeheader()
        row_buffer = []
        flush_size = 1000
        for meta, ply, sfen, d, n, m in records:
            meta.update({'ply': ply, 'eval_score_cp': eval_map[(sfen, d, n, m)], 'sfen': sfen})
            row_buffer.append(meta)
            if len(row_buffer) >= flush_size:
                writer.writerows(row_buffer)
                row_buffer.clear()
        if row_buffer:
            writer.writerows(row_buffer)

    print("局面評価が完了しました。")

def evaluate_metadata_logic(args: argparse.Namespace) -> None:
    """
    [evaluateコマンド] USIエンジンで各局面を評価し、評価値付きCSVを生成する。
    DBの指定がある場合、評価値の再利用と同一局面の出力上限管理を行う。
    """
    if not Path(args.input_csv).exists(): sys.exit(f"エラー: 入力メタデータファイル '{args.input_csv}' が見つかりません。")
    if not Path(args.engine_path).exists(): sys.exit(f"エラー: エンジン実行ファイルが見つかりません: {args.engine_path}")
    
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        all_kifs_meta, header = list(reader), reader.fieldnames

    output_csv_path = Path(args.output_csv)
    output_header = header + ['ply', 'eval_score_cp', 'sfen']

    kifs_by_file = defaultdict(list)
    for meta in all_kifs_meta:
        kifs_by_file[meta['file_path']].append(meta)

    if args.eval_mode == "unique":
        _evaluate_unique_mode(args, kifs_by_file, output_csv_path, output_header)
        return

    if args.eval_workers > 1:
        if args.db_path or args.max_sfen_count > 0:
            sys.exit("エラー: 並列evaluate(--eval-workers > 1)では --db-path / --max-sfen-count は使用できません。")

        print(f"--- 局面評価を開始 (並列: {args.eval_workers} workers) ---")
        temp_dir = output_csv_path.parent / f".evaluate_tmp_{output_csv_path.stem}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            'engine_path': str(args.engine_path),
            'min_ply': args.min_ply,
            'max_ply': args.max_ply,
            'depth': args.depth,
            'nodes': args.nodes,
            'movetime': args.movetime,
            'early_depth': args.early_depth,
            'early_nodes': args.early_nodes,
            'early_movetime': args.early_movetime,
            'early_ply_threshold': args.early_ply_threshold,
        }
        buckets = _partition_file_tasks(kifs_by_file, args.eval_workers)
        tasks = []
        for idx, file_tasks in enumerate(buckets):
            if not file_tasks:
                continue
            tasks.append({
                'cfg': cfg,
                'output_header': output_header,
                'worker_output_csv': str(temp_dir / f"worker_{idx:03d}.csv"),
                'file_tasks': file_tasks,
            })

        results = []
        with ProcessPoolExecutor(max_workers=args.eval_workers) as executor:
            futures = [executor.submit(_evaluate_worker, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="workers"):
                results.append(fut.result())

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_header)
            writer.writeheader()
            for res in sorted(results, key=lambda r: r['worker_output_csv']):
                with open(res['worker_output_csv'], 'r', newline='', encoding='utf-8') as f_in:
                    reader = csv.DictReader(f_in)
                    for row in reader:
                        writer.writerow(row)

        total_rows = sum(r['rows_written'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"局面評価が完了しました。出力行数: {total_rows:,}, workerエラー件数: {total_errors}")
        return

    db = SfenDB(args.db_path) if args.db_path else None
    print(f"--- 局面評価を開始 (DB: {args.db_path}, 上限: {args.max_sfen_count}) ---")
    try:
        engine = UsiEngine(str(args.engine_path))
        print("USIエンジン準備完了。")
    except Exception as e:
        sys.exit(f"エラー: USIエンジンの初期化に失敗しました: {e}")

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_header)
        writer.writeheader()
        row_buffer = []
        flush_size = 1000

        with tqdm(kifs_by_file.items(), unit="file") as pbar:
            for csa_path, metas in pbar:
                pbar.set_description(f"Evaluating {Path(csa_path).name}")
                try:
                    all_kifs_in_file = cshogi.Parser.parse_file(csa_path)
                    if all_kifs_in_file is None:
                        continue
                    for meta in metas:
                        kif = all_kifs_in_file[int(meta['kif_index'])]
                        engine.new_game()
                        board = cshogi.Board(kif.sfen)
                        for ply, move in enumerate(kif.moves, 1):
                            if ply > args.max_ply:
                                break
                            if ply >= args.min_ply:
                                sfen = board.sfen()
                                if db is None or db.check_output_limit(sfen, args.max_sfen_count):
                                    try:
                                        d, n, m = get_search_params(args, ply)
                                        score_type, score_value = None, None
                                        if db:
                                            cached_eval = db.get_eval(sfen, d, n, m)
                                            if cached_eval:
                                                score_type, score_value = cached_eval
                                        if score_type is None:
                                            score_type, score_value = engine.evaluate_sfen(sfen, depth=d, nodes=n, movetime=m)
                                            if db:
                                                db.save_eval(sfen, d, n, m, score_type, score_value)

                                        eval_score_cp = _normalize_eval_score(score_type, score_value)
                                        meta_with_eval = meta.copy()
                                        meta_with_eval.update({'ply': ply, 'eval_score_cp': eval_score_cp, 'sfen': sfen})
                                        row_buffer.append(meta_with_eval)
                                        if len(row_buffer) >= flush_size:
                                            writer.writerows(row_buffer)
                                            row_buffer.clear()
                                        if db:
                                            db.increment_output_count(sfen)
                                    except Exception as e:
                                        print(f"\n評価エラー: 棋譜{meta['kif_index']} 手数{ply} ({e})", file=sys.stderr)
                                        traceback.print_exc()
                            board.push(move)
                    if row_buffer:
                        writer.writerows(row_buffer)
                        row_buffer.clear()
                    if db:
                        db.commit()
                except Exception as e:
                    print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)
                    traceback.print_exc()
    engine.quit()
    if db:
        db.close()
    print("局面評価が完了しました。")


def _evaluate_sfen_worker(task: dict) -> dict:
    rows = task['rows']
    output_header = task['output_header']
    worker_output_csv = task['worker_output_csv']

    rows_written = 0
    errors = 0
    engine = UsiEngine(task['engine_path'])
    try:
        with open(worker_output_csv, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_header)
            writer.writeheader()
            for row in rows:
                sfen = row.get('sfen')
                if not sfen:
                    continue
                try:
                    score_type, score_value = engine.evaluate_sfen(
                        sfen,
                        depth=task['depth'],
                        nodes=task['nodes'],
                        movetime=task['movetime'],
                    )
                    row = row.copy()
                    row['eval_score_cp'] = _normalize_eval_score(score_type, score_value)
                    writer.writerow(row)
                    rows_written += 1
                except Exception:
                    errors += 1
                    traceback.print_exc()
    finally:
        engine.quit()

    return {
        'rows_written': rows_written,
        'errors': errors,
        'worker_output_csv': worker_output_csv,
    }


def evaluate_sfen_logic(args: argparse.Namespace) -> None:
    """
    [evaluate-sfenコマンド] SFEN一覧CSVを入力として、各ユニーク局面に評価値を付与する。
    count-sfen の出力をそのまま入力できることを想定する。
    """
    if not Path(args.input_csv).exists():
        sys.exit(f"エラー: 入力ファイル '{args.input_csv}' が見つかりません。")
    if not Path(args.engine_path).exists():
        sys.exit(f"エラー: エンジン実行ファイルが見つかりません: {args.engine_path}")

    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        rows, header = list(reader), reader.fieldnames

    if not rows:
        sys.exit("エラー: 入力ファイルにデータがありません。")
    if not header or 'sfen' not in header:
        sys.exit("エラー: 入力CSVに 'sfen' 列が必要です。")

    output_header = list(header)
    if 'eval_score_cp' not in output_header:
        output_header.append('eval_score_cp')

    output_csv_path = Path(args.output_csv)

    if args.eval_workers > 1:
        if args.db_path:
            sys.exit("エラー: 並列evaluate-sfen(--eval-workers > 1)では --db-path は使用できません。")

        print(f"--- SFEN評価を開始 (並列: {args.eval_workers} workers) ---")
        temp_dir = output_csv_path.parent / f".evaluate_sfen_tmp_{output_csv_path.stem}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = math.ceil(len(rows) / args.eval_workers)
        tasks = []
        for idx in range(args.eval_workers):
            chunk = rows[idx * chunk_size:(idx + 1) * chunk_size]
            if not chunk:
                continue
            tasks.append({
                'engine_path': str(args.engine_path),
                'output_header': output_header,
                'worker_output_csv': str(temp_dir / f"worker_{idx:03d}.csv"),
                'rows': chunk,
                'depth': args.depth,
                'nodes': args.nodes,
                'movetime': args.movetime,
            })

        results = []
        with ProcessPoolExecutor(max_workers=args.eval_workers) as executor:
            futures = [executor.submit(_evaluate_sfen_worker, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="workers"):
                results.append(fut.result())

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_header)
            writer.writeheader()
            for res in sorted(results, key=lambda r: r['worker_output_csv']):
                with open(res['worker_output_csv'], 'r', newline='', encoding='utf-8') as f_in:
                    reader = csv.DictReader(f_in)
                    for row in reader:
                        writer.writerow(row)

        total_rows = sum(r['rows_written'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"SFEN評価が完了しました。出力行数: {total_rows:,}, workerエラー件数: {total_errors}")
        return

    db = SfenDB(args.db_path) if args.db_path else None
    print(f"--- SFEN評価を開始 (DB: {args.db_path}) ---")
    try:
        engine = UsiEngine(str(args.engine_path))
        print("USIエンジン準備完了。")
    except Exception as e:
        sys.exit(f"エラー: USIエンジンの初期化に失敗しました: {e}")

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_header)
            writer.writeheader()
            for row in tqdm(rows, desc="Evaluating SFENs"):
                sfen = row.get('sfen')
                if not sfen:
                    continue

                score_type, score_value = None, None
                if db:
                    cached_eval = db.get_eval(sfen, args.depth, args.nodes, args.movetime)
                    if cached_eval:
                        score_type, score_value = cached_eval

                if score_type is None:
                    score_type, score_value = engine.evaluate_sfen(
                        sfen,
                        depth=args.depth,
                        nodes=args.nodes,
                        movetime=args.movetime,
                    )
                    if db:
                        db.save_eval(sfen, args.depth, args.nodes, args.movetime, score_type, score_value)

                row['eval_score_cp'] = _normalize_eval_score(score_type, score_value)
                writer.writerow(row)

            if db:
                db.commit()
    finally:
        engine.quit()
        if db:
            db.close()

    print(f"SFEN評価が完了しました。出力CSV: {output_csv_path}")

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


def _compute_sampling_target(freq: int, mode: str, value: float) -> float:
    if mode == "fixed":
        return max(1.0, float(value))
    if mode == "sqrt":
        return max(1.0, math.sqrt(freq))
    if mode == "log10":
        if freq <= 1:
            return 1.0
        return max(1.0, math.log10(freq))
    return float(freq)


def _count_king_escape_routes(board: cshogi.Board, king_sq: int) -> int:
    king_escape_routes = 0
    y, x = divmod(king_sq, 9)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < 9 and 0 <= nx < 9:
                to_sq = ny * 9 + nx
                move = board.move(king_sq, to_sq, False)
                if board.is_legal(move):
                    king_escape_routes += 1
    return king_escape_routes


def _analyze_board_tactical_state(board: cshogi.Board, include_king_safety: bool = False) -> dict:
    capture_moves = 0
    check_moves = 0
    promotion_moves = 0
    legal_moves = 0

    for move in board.legal_moves:
        legal_moves += 1
        if cshogi.move_cap(move) != 0:
            capture_moves += 1
        if cshogi.move_is_promotion(move):
            promotion_moves += 1
        board.push(move)
        if board.is_check():
            check_moves += 1
        board.pop()

    result = {
        "legal_moves": legal_moves,
        "capture_moves": capture_moves,
        "check_moves": check_moves,
        "promotion_moves": promotion_moves,
    }

    if include_king_safety:
        my_king_sq = board.king_square(board.turn)
        result["my_king_attackers"] = 1 if board.is_check() else 0
        result["king_escape_routes"] = _count_king_escape_routes(board, my_king_sq)

    return result


def _classify_quiet_rejection_reason(board: cshogi.Board, quiet_level: str):
    if quiet_level == "none":
        return None, None

    if board.is_game_over():
        return "game_over", None
    if board.is_nyugyoku():
        return "nyugyoku", None
    if board.is_draw() != cshogi.NOT_REPETITION:
        return "draw", None
    if board.is_check():
        return "in_check", None

    if quiet_level == "1":
        return None, None

    analysis = _analyze_board_tactical_state(board, include_king_safety=(quiet_level == "3"))
    if analysis["capture_moves"] > 0:
        return "capture_available", analysis
    if board.mate_move_in_1ply():
        return "mate_in_1_available", analysis

    if quiet_level == "2":
        return None, analysis

    # Strict quiet positions: additionally require low king pressure and no forced king flight.
    if analysis["my_king_attackers"] > 0:
        return "king_under_attack", analysis
    if analysis["king_escape_routes"] <= 1:
        return "few_king_escape_routes", analysis

    return None, analysis


def _is_quiet_position(board: cshogi.Board, quiet_level: str) -> bool:
    reason, _ = _classify_quiet_rejection_reason(board, quiet_level)
    return reason is None


def classify_sfen_logic(args: argparse.Namespace) -> None:
    """
    [classify-sfenコマンド] SFEN一覧CSVを静止局面と非静止局面に分類する。
    evaluate-sfen の前段で使い、評価条件を分けることを目的とする。
    """
    if not Path(args.input_csv).exists():
        sys.exit(f"エラー: 入力ファイル '{args.input_csv}' が見つかりません。")
    if args.quiet_level not in ("1", "2", "3"):
        sys.exit("エラー: classify-sfen の --quiet-level は 1, 2, 3 のいずれかを指定してください。")

    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        header = reader.fieldnames

    if not rows:
        sys.exit("エラー: 入力ファイルにデータがありません。")
    if not header or 'sfen' not in header:
        sys.exit("エラー: 入力CSVに 'sfen' 列が必要です。")

    quiet_output = Path(args.output_quiet_csv)
    tactical_output = Path(args.output_tactical_csv)
    quiet_output.parent.mkdir(parents=True, exist_ok=True)
    tactical_output.parent.mkdir(parents=True, exist_ok=True)

    board = cshogi.Board()
    quiet_rows = []
    tactical_rows = []
    invalid_rows = 0
    rejection_counts = defaultdict(int)

    print(f"--- SFEN分類を開始 (quiet-level={args.quiet_level}) ---")
    for row in tqdm(rows, desc="Classifying SFENs"):
        sfen = row.get('sfen')
        if not sfen:
            invalid_rows += 1
            continue
        try:
            board.set_sfen(sfen)
        except Exception:
            tactical_rows.append(row)
            invalid_rows += 1
            continue

        rejection_reason, _ = _classify_quiet_rejection_reason(board, args.quiet_level)
        if rejection_reason is None:
            quiet_rows.append(row)
        else:
            tactical_rows.append(row)
            rejection_counts[rejection_reason] += 1

    with open(quiet_output, 'w', newline='', encoding='utf-8') as f_quiet:
        writer = csv.DictWriter(f_quiet, fieldnames=header)
        writer.writeheader()
        writer.writerows(quiet_rows)

    with open(tactical_output, 'w', newline='', encoding='utf-8') as f_tactical:
        writer = csv.DictWriter(f_tactical, fieldnames=header)
        writer.writeheader()
        writer.writerows(tactical_rows)

    print(
        f"SFEN分類が完了しました。静止局面: {len(quiet_rows):,}, "
        f"非静止局面: {len(tactical_rows):,}, 無効/解析失敗: {invalid_rows:,}"
    )
    if rejection_counts:
        summary = ", ".join(f"{key}={value:,}" for key, value in sorted(rejection_counts.items()))
        print(f"非静止判定の内訳: {summary}")


def merge_eval_sfen_logic(args: argparse.Namespace) -> None:
    """
    [merge-eval-sfenコマンド] 複数の評価済みSFEN CSVを1つにマージする。
    classify-sfen -> evaluate-sfen を複数条件で回した後、generate の前段で使用する。
    """
    input_paths = [Path(p.strip()) for p in args.input_csvs.split(',') if p.strip()]
    if not input_paths:
        sys.exit("エラー: merge-eval-sfen には --input-csvs を1つ以上指定してください。")

    for path in input_paths:
        if not path.exists():
            sys.exit(f"エラー: 入力ファイル '{path}' が見つかりません。")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_rows = 0
    header = None
    with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = None
        for path in input_paths:
            with open(path, 'r', newline='', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in)
                current_header = reader.fieldnames
                if not current_header:
                    continue
                if header is None:
                    header = current_header
                    writer = csv.DictWriter(f_out, fieldnames=header)
                    writer.writeheader()
                elif current_header != header:
                    sys.exit(f"エラー: ヘッダが一致しません: {path}")

                for row in reader:
                    writer.writerow(row)
                    merged_rows += 1

    print(f"評価済みSFEN CSVのマージが完了しました。入力数: {len(input_paths)}, 出力行数: {merged_rows:,}")


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

    if args.min_ply > args.max_ply:
        sys.exit("エラー: --min-ply は --max-ply 以下である必要があります。")

    if all('ply' in pos for pos in all_positions):
        filtered_positions = []
        skipped_invalid_ply = 0
        for pos in all_positions:
            try:
                ply = int(pos['ply'])
            except (TypeError, ValueError):
                skipped_invalid_ply += 1
                continue
            if args.min_ply <= ply <= args.max_ply:
                filtered_positions.append(pos)

        if skipped_invalid_ply:
            print(f"ply列を解釈できないため除外した局面数: {skipped_invalid_ply:,}")
        all_positions = filtered_positions
        if not all_positions:
            sys.exit("エラー: plyフィルタ適用後に局面が残りませんでした。")
        print(f"plyフィルタ適用後の局面数: {len(all_positions):,} (範囲: {args.min_ply}..{args.max_ply})")
    else:
        print("ply列が存在しないため、generate での ply フィルタはスキップします。")

    if args.quiet_level not in ("none", "1", "2", "3"):
        sys.exit("エラー: --quiet-level は none, 1, 2, 3 のいずれかを指定してください。")
    if args.quiet_level != "none":
        print("注意: generate での --quiet-level は後方互換用です。新フローでは classify-sfen を evaluate-sfen の前段で使用してください。")
        board = cshogi.Board()
        quiet_positions = []
        quiet_skipped = 0
        for pos in tqdm(all_positions, desc=f"Filtering quiet level {args.quiet_level}"):
            sfen = pos.get('sfen')
            if not sfen:
                quiet_skipped += 1
                continue
            try:
                board.set_sfen(sfen)
            except Exception:
                quiet_skipped += 1
                continue
            if _is_quiet_position(board, args.quiet_level):
                quiet_positions.append(pos)
            else:
                quiet_skipped += 1
        all_positions = quiet_positions
        if not all_positions:
            sys.exit("エラー: 静止局面フィルタ適用後に局面が残りませんでした。")
        print(
            f"静止局面フィルタ適用後の局面数: {len(all_positions):,} "
            f"(level={args.quiet_level}, 除外={quiet_skipped:,})"
        )

    if args.sfen_sampling_mode != "none":
        if not args.sfen_count_csv:
            sys.exit("エラー: SFEN頻度サンプリングを使う場合は --sfen-count-csv の指定が必須です。")
        if not Path(args.sfen_count_csv).exists():
            sys.exit(f"エラー: SFEN頻度CSV '{args.sfen_count_csv}' が見つかりません。")
        if args.sfen_sampling_mode == "fixed" and args.sfen_cutoff_value <= 0:
            sys.exit("エラー: --sfen-sampling-mode fixed の場合、--sfen-cutoff-value は1以上を指定してください。")
        if args.sfen_sampling_min_freq < 1:
            sys.exit("エラー: --sfen-sampling-min-freq は1以上を指定してください。")

        freq_map = {}
        with open(args.sfen_count_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    freq_map[row['sfen']] = int(row['total_count'])
                except (KeyError, ValueError):
                    continue

        by_sfen = defaultdict(list)
        for pos in all_positions:
            sfen = pos.get('sfen')
            if sfen:
                by_sfen[sfen].append(pos)

        sampled_positions = []
        applied_sfens = 0
        for sfen, items in by_sfen.items():
            freq = freq_map.get(sfen, len(items))
            if freq < args.sfen_sampling_min_freq:
                sampled_positions.extend(items)
                continue
            target = _compute_sampling_target(freq, args.sfen_sampling_mode, args.sfen_cutoff_value)
            keep_prob = min(1.0, target / max(1, freq))
            applied_sfens += 1
            for item in items:
                if random.random() < keep_prob:
                    sampled_positions.append(item)
        all_positions = sampled_positions
        print(
            f"SFEN頻度サンプリングを適用: mode={args.sfen_sampling_mode}, "
            f"min_freq={args.sfen_sampling_min_freq}, "
            f"局面数 {len(sampled_positions):,}, 適用SFEN数 {applied_sfens:,}"
        )

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
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("-c", "--config", help="設定YAMLファイルのパス。")
    eval_common_parent = argparse.ArgumentParser(add_help=False)
    eval_common_parent.add_argument("--input-csv", help="入力CSVのパス。")
    eval_common_parent.add_argument("--engine-path", help="USIエンジンの実行ファイルのパス。")
    eval_common_parent.add_argument("--output-csv", help="評価結果CSVの出力パス。")
    eval_common_parent.add_argument("--db-path", help="評価値キャッシュを管理するSQLite DBのパス。")
    eval_common_parent.add_argument("--depth", type=int, default=10)
    eval_common_parent.add_argument("--nodes", type=int, default=None)
    eval_common_parent.add_argument("--movetime", type=int, default=None)
    eval_common_parent.add_argument("--eval-workers", type=int, default=1, help="評価時の並列ワーカー数。2以上でプロセス並列。")

    parser = argparse.ArgumentParser(
        description="CSA棋譜から学習データを生成するスクリプト。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[config_parent],
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="利用可能なコマンド")

    extract_parser = subparsers.add_parser("extract", parents=[config_parent], help="CSAファイルから棋譜のメタデータを抽出します。")
    extract_parser.add_argument("--csa-dir", help="CSAファイルが格納されているルートディレクトリ。")
    extract_parser.add_argument("--output-csv", help="メタデータCSVの出力パス。")
    extract_parser.set_defaults(func=lambda args: extract_metadata(args.csa_dir, args.output_csv))

    filter_parser = subparsers.add_parser("filter", parents=[config_parent], help="メタデータCSVをフィルタリングします。")
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

    count_sfen_parser = subparsers.add_parser("count-sfen", parents=[config_parent], help="SFENの出現頻度をカウントしてCSVに保存します。")
    count_sfen_parser.add_argument("--input-csv", help="入力となるフィルタリング済みCSVのパス。")
    count_sfen_parser.add_argument("--output-csv", default="sfen_counts.csv", help="SFEN頻度CSVの出力パス。")
    count_sfen_parser.add_argument("--min-count", type=int, default=1, help="出力する最小出現回数。")
    count_sfen_parser.add_argument("--num-buckets", type=int, default=1024, help="外部メモリ集計で使用するバケット数。")
    count_sfen_parser.add_argument("--temp-dir", help="一時バケットファイルの出力先ディレクトリ。")
    count_sfen_parser.add_argument("--keep-temp", action='store_true', help="集計後も一時バケットファイルを削除せず保持する。")
    count_sfen_parser.add_argument("--min-ply", type=int, default=0)
    count_sfen_parser.add_argument("--max-ply", type=int, default=999)
    count_sfen_parser.set_defaults(func=count_sfen_logic)

    classify_sfen_parser = subparsers.add_parser("classify-sfen", parents=[config_parent], help="SFEN一覧CSVを静止局面と非静止局面に分類します。")
    classify_sfen_parser.add_argument("--input-csv", help="入力となるSFEN一覧CSVのパス。")
    classify_sfen_parser.add_argument("--output-quiet-csv", help="静止局面CSVの出力パス。")
    classify_sfen_parser.add_argument("--output-tactical-csv", help="非静止局面CSVの出力パス。")
    classify_sfen_parser.add_argument(
        "--quiet-level",
        choices=["1", "2", "3"],
        default="2",
        help="静止局面判定の強さ。1=終局/王手/反復除外, 2=取り/1手詰め筋も除外, 3=さらに玉の危険度も抑える。",
    )
    classify_sfen_parser.set_defaults(func=classify_sfen_logic)

    label_parser = subparsers.add_parser("label", parents=[config_parent], help="対局結果から評価値をラベリングします（エンジン不要）。")
    label_parser.add_argument("--input-csv", help="入力となるフィルタリング済みCSVのパス。")
    label_parser.add_argument("--output-csv", help="ラベリング結果を保存するCSVのパス。")
    label_parser.add_argument("--score-scale", type=int, default=600)
    label_parser.add_argument("--db-path", help="SFEN頻度を管理するSQLite DBのパス。指定すると上限管理が有効になります。")
    label_parser.add_argument("--max-sfen-count", type=int, default=0, help="同一SFENの最大出力回数。0は無制限。")
    label_parser.set_defaults(func=run_label)

    evaluate_parser = subparsers.add_parser("evaluate", parents=[config_parent, eval_common_parent], help="フィルタリング済みCSVの局面を評価します。")
    evaluate_parser.add_argument("--max-sfen-count", type=int, default=0, help="同一SFENの最大出力回数。0は無制限。")
    evaluate_parser.add_argument("--early-depth", type=int, default=None)
    evaluate_parser.add_argument("--early-nodes", type=int, default=None)
    evaluate_parser.add_argument("--early-movetime", type=int, default=None)
    evaluate_parser.add_argument("--early-ply-threshold", type=int, default=0, help="序盤とみなす最大手数（この手数以下でearlyパラメータを適用）。")
    evaluate_parser.add_argument("--min-ply", type=int, default=0)
    evaluate_parser.add_argument("--max-ply", type=int, default=999)
    evaluate_parser.add_argument("--eval-mode", choices=["stream", "unique"], default="stream", help="局面評価方式。stream=逐次、unique=ユニーク評価後に展開。")
    evaluate_parser.set_defaults(func=evaluate_metadata_logic)

    evaluate_sfen_parser = subparsers.add_parser("evaluate-sfen", parents=[config_parent, eval_common_parent], help="SFEN一覧CSVの各局面を評価します。")
    evaluate_sfen_parser.set_defaults(func=evaluate_sfen_logic)

    merge_eval_sfen_parser = subparsers.add_parser("merge-eval-sfen", parents=[config_parent], help="複数の評価済みSFEN CSVを1つにマージします。")
    merge_eval_sfen_parser.add_argument("--input-csvs", help="マージ対象CSVのカンマ区切りリスト。")
    merge_eval_sfen_parser.add_argument("--output-csv", help="マージ後CSVの出力パス。")
    merge_eval_sfen_parser.set_defaults(func=merge_eval_sfen_logic)

    generate_parser = subparsers.add_parser("generate", parents=[config_parent], help="評価値付きCSVから学習データ(.bin)を生成します。")
    generate_parser.add_argument("--input-csv", help="入力となる評価値付きCSVのパス。")
    generate_parser.add_argument("--output-dir", help="生成されたデータセットを保存するディレクトリ。")
    generate_parser.add_argument("--val-split", type=float, default=0.1)
    generate_parser.add_argument("--min-ply", type=int, default=0, help="生成対象とする最小手数。")
    generate_parser.add_argument("--max-ply", type=int, default=999, help="生成対象とする最大手数。")
    generate_parser.add_argument(
        "--quiet-level",
        choices=["none", "1", "2", "3"],
        default="none",
        help="静止局面フィルタの強さ。none=無効, 1=終局/王手/反復除外, 2=取り/1手詰め筋も除外, 3=さらに玉の危険度も抑える。",
    )
    generate_parser.add_argument("--sfen-count-csv", help="count-sfenで生成したSFEN頻度CSVのパス。")
    generate_parser.add_argument("--sfen-sampling-mode", choices=["none", "fixed", "sqrt", "log10"], default="none", help="SFEN頻度を使ったサンプリング上限方式。")
    generate_parser.add_argument("--sfen-cutoff-value", type=float, default=1.0, help="fixed方式の上限値。")
    generate_parser.add_argument("--sfen-sampling-min-freq", type=int, default=1, help="この頻度未満のSFENにはサンプリング上限を適用しない。")
    generate_parser.set_defaults(func=generate_datasets_logic)

    build_h5_parser = subparsers.add_parser("build-h5", parents=[config_parent], help="フィルタリング済みCSVから階層的なHDF5データセットを生成します。")
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
    elif args.command == "count-sfen":
        if not args.input_csv:
             sys.exit("エラー: count-sfenコマンドには --input-csv の指定が必須です。")
        if not args.output_csv:
             sys.exit("エラー: count-sfenコマンドには --output-csv の指定が必須です。")
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    elif args.command == "label":
        if not (args.input_csv and args.output_csv):
             sys.exit("エラー: labelコマンドには --input-csv と --output-csv の指定が必須です。")
    elif args.command == "evaluate":
        if not (args.input_csv and args.engine_path and args.output_csv):
             sys.exit("エラー: evaluateコマンドには --input-csv, --engine-path, --output-csv の指定が必須です。")
    elif args.command == "evaluate-sfen":
        if not (args.input_csv and args.engine_path and args.output_csv):
             sys.exit("エラー: evaluate-sfenコマンドには --input-csv, --engine-path, --output-csv の指定が必須です。")
    elif args.command == "merge-eval-sfen":
        if not (args.input_csvs and args.output_csv):
             sys.exit("エラー: merge-eval-sfenコマンドには --input-csvs と --output-csv の指定が必須です。")
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    elif args.command == "classify-sfen":
        if not (args.input_csv and args.output_quiet_csv and args.output_tactical_csv):
             sys.exit("エラー: classify-sfenコマンドには --input-csv, --output-quiet-csv, --output-tactical-csv の指定が必須です。")
        Path(args.output_quiet_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_tactical_csv).parent.mkdir(parents=True, exist_ok=True)
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
