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


def plot_sfen_histogram_logic(args: argparse.Namespace) -> None:
    """[plot-sfen-histogramコマンド] count-sfen 出力の total_count 分布をヒストグラム化する。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("エラー: matplotlib が必要です。`pip install matplotlib` を実行してください。")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        sys.exit(f"エラー: 入力CSV '{args.input_csv}' が見つかりません。")
    if args.bins <= 0:
        sys.exit("エラー: --bins は1以上を指定してください。")

    counts = []
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or args.count_column not in reader.fieldnames:
            sys.exit(f"エラー: 入力CSVに '{args.count_column}' 列が必要です。")
        for row in reader:
            try:
                value = int(row[args.count_column])
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            counts.append(value)

    if not counts:
        sys.exit("エラー: ヒストグラム化できる正の頻度データがありません。")

    max_count = max(counts)
    if args.max_count is not None:
        counts = [value for value in counts if value <= args.max_count]
        if not counts:
            sys.exit("エラー: --max-count 適用後にデータが残りませんでした。")
        max_count = max(counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    if args.log_x:
        bins = np.logspace(0, math.log10(max_count), args.bins + 1)
    else:
        bins = args.bins
    ax.hist(counts, bins=bins, color="#3B82F6", edgecolor="#1E3A8A", alpha=0.85)
    ax.set_title(args.title or "SFEN Frequency Histogram")
    ax.set_xlabel(args.count_column)
    ax.set_ylabel("Number of SFENs")
    if args.log_x:
        ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(
        f"ヒストグラムを出力しました: {output_path} "
        f"(SFEN数={len(counts):,}, max_count={max_count:,}, bins={args.bins})"
    )

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


def merge_extract_logic(args: argparse.Namespace) -> None:
    """
    [merge-extractコマンド] 複数の extract 出力CSVを1つにマージする。
    """
    input_paths = [Path(p.strip()) for p in args.input_csvs.split(',') if p.strip()]
    if not input_paths:
        sys.exit("エラー: merge-extract には --input-csvs を1つ以上指定してください。")

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

    print(f"extract CSV のマージが完了しました。入力CSV数: {len(input_paths)}, 出力行数: {merged_rows:,}, 出力: {output_path}")

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

def _compute_result_label_score(
    game_result: int,
    current_turn: int,
    ply: int,
    total_moves: int,
    max_score: int,
    min_score: int,
    curve: float,
) -> int:
    if game_result == 0:
        return 0

    total_moves = max(int(total_moves), 1)
    ply = min(max(int(ply), 1), total_moves)
    max_score = max(int(max_score), 1)
    min_score = max(0, min(int(min_score), max_score))
    curve = max(float(curve), 1e-6)

    progress = ply / float(total_moves)
    growth = (1.0 - math.exp(-curve * progress)) / (1.0 - math.exp(-curve))
    magnitude = int(round(min_score + (max_score - min_score) * growth))

    if game_result == 1:
        return magnitude if current_turn == cshogi.BLACK else -magnitude
    if game_result == 2:
        return -magnitude if current_turn == cshogi.BLACK else magnitude
    return 0


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
                        total_moves = len(kif.moves)
                        board = cshogi.Board(kif.sfen)
                        for ply, move in enumerate(kif.moves, 1):
                            sfen = board.sfen()
                            # DB指定がある場合はキャップ判定
                            if db is None or db.check_output_limit(sfen, args.max_sfen_count):
                                current_turn = board.turn
                                score = _compute_result_label_score(
                                    game_result=game_result,
                                    current_turn=current_turn,
                                    ply=ply,
                                    total_moves=total_moves,
                                    max_score=args.score_scale,
                                    min_score=args.label_min_score,
                                    curve=args.label_curve,
                                )
                                
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


def _parse_int_list_option(value: str | None) -> list[int]:
    """カンマ区切り整数列をパースする。未指定時は空配列を返す。"""
    if not value:
        return []
    values = []
    for token in value.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def get_candidate_search_params(args: argparse.Namespace, ply: int) -> list[tuple[int | None, int | None, int | None]]:
    """
    build-h5 用の候補探索条件を返す。

    - `candidate_depths` / `early_candidate_depths` が指定されていれば、その複数深さを使う
    - 未指定なら既存の `get_search_params()` と同じ単一条件を返す
    """
    if ply <= args.early_ply_threshold:
        depth_values = _parse_int_list_option(getattr(args, "early_candidate_depths", None))
    else:
        depth_values = _parse_int_list_option(getattr(args, "candidate_depths", None))

    if depth_values:
        return [(depth, None, None) for depth in depth_values]

    return [get_search_params(args, ply)]


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
    existing_eval_map = {}
    if args.existing_eval_csv or args.existing_eval_csvs:
        existing_paths, existing_eval_map = _load_eval_map_from_csvs(args.existing_eval_csv, args.existing_eval_csvs)
        print(f"既存の評価済みSFEN CSV を読み込み: {len(existing_paths)} files, {len(existing_eval_map):,} SFEN")

    rows_to_evaluate = [row for row in rows if row.get('sfen') not in existing_eval_map]
    reused_rows = len(rows) - len(rows_to_evaluate)

    if args.eval_workers > 1:
        if args.db_path:
            sys.exit("エラー: 並列evaluate-sfen(--eval-workers > 1)では --db-path は使用できません。")

        print(f"--- SFEN評価を開始 (並列: {args.eval_workers} workers) ---")
        temp_dir = output_csv_path.parent / f".evaluate_sfen_tmp_{output_csv_path.stem}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        if not rows_to_evaluate:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=output_header)
                writer.writeheader()
                for row in rows:
                    row = row.copy()
                    row['eval_score_cp'] = existing_eval_map[row['sfen']]
                    writer.writerow(row)
            print(f"SFEN評価が完了しました。既存評価のみを利用: {reused_rows:,} 行, 新規評価 0 行")
            return

        chunk_size = math.ceil(len(rows_to_evaluate) / args.eval_workers)
        tasks = []
        for idx in range(args.eval_workers):
            chunk = rows_to_evaluate[idx * chunk_size:(idx + 1) * chunk_size]
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

        new_eval_map = {}
        for res in sorted(results, key=lambda r: r['worker_output_csv']):
            with open(res['worker_output_csv'], 'r', newline='', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    sfen = row.get('sfen')
                    if not sfen:
                        continue
                    try:
                        new_eval_map[sfen] = int(row['eval_score_cp'])
                    except (TypeError, ValueError, KeyError):
                        continue

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_header)
            writer.writeheader()
            for row in rows:
                sfen = row.get('sfen')
                if not sfen:
                    continue
                if sfen in existing_eval_map:
                    out_row = row.copy()
                    out_row['eval_score_cp'] = existing_eval_map[sfen]
                    writer.writerow(out_row)
                elif sfen in new_eval_map:
                    out_row = row.copy()
                    out_row['eval_score_cp'] = new_eval_map[sfen]
                    writer.writerow(out_row)

        total_rows = sum(r['rows_written'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(
            f"SFEN評価が完了しました。出力行数: {reused_rows + total_rows:,}, "
            f"既存流用: {reused_rows:,}, 新規評価: {total_rows:,}, workerエラー件数: {total_errors}"
        )
        return

    db = SfenDB(args.db_path) if args.db_path else None
    print(f"--- SFEN評価を開始 (DB: {args.db_path}) ---")
    if not rows_to_evaluate and existing_eval_map:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_header)
            writer.writeheader()
            for row in rows:
                sfen = row.get('sfen')
                if not sfen:
                    continue
                row = row.copy()
                row['eval_score_cp'] = existing_eval_map[sfen]
                writer.writerow(row)
        print(f"SFEN評価が完了しました。既存評価のみを利用: {reused_rows:,} 行, 新規評価 0 行")
        return
    try:
        engine = UsiEngine(str(args.engine_path))
        print("USIエンジン準備完了。")
    except Exception as e:
        sys.exit(f"エラー: USIエンジンの初期化に失敗しました: {e}")

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_header)
            writer.writeheader()
            newly_evaluated = 0
            for row in tqdm(rows, desc="Evaluating SFENs"):
                sfen = row.get('sfen')
                if not sfen:
                    continue
                if sfen in existing_eval_map:
                    row = row.copy()
                    row['eval_score_cp'] = existing_eval_map[sfen]
                    writer.writerow(row)
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
                    newly_evaluated += 1

                row['eval_score_cp'] = _normalize_eval_score(score_type, score_value)
                writer.writerow(row)

            if db:
                db.commit()
    finally:
        engine.quit()
        if db:
            db.close()

    print(
        f"SFEN評価が完了しました。出力CSV: {output_csv_path}, "
        f"既存流用: {reused_rows:,}, 新規評価: {newly_evaluated:,}"
    )

def write_bin_file(positions: list, output_path: str):
    """
    局面情報のリストから、PackedSfenValue形式の.binファイルを生成する。
    """
    print(f"データセット '{output_path}' の生成を開始 (対象局面数: {len(positions)})")
    board = cshogi.Board()
    psv = np.zeros(1, dtype=cshogi.PackedSfenValue)
    int16_info = np.iinfo(np.int16)
    skipped_out_of_range_scores = 0
    with open(output_path, "wb") as f_out:
        for pos in tqdm(positions, desc=f"Writing {Path(output_path).name}"):
            try:
                board.set_sfen(pos['sfen'])
                board.to_psfen(psv)
                cshogi_result = int(pos['game_result'])
                write_result = 1 if cshogi_result == 1 else -1 if cshogi_result == 2 else 0
                score = int(pos['eval_score_cp'])
                if not (int16_info.min <= score <= int16_info.max):
                    skipped_out_of_range_scores += 1
                    continue
                psv[0]["score"] = np.int16(score)
                psv[0]["move"] = np.uint16(0)
                psv[0]["gamePly"] = np.uint16(pos['ply'])
                psv[0]["game_result"] = np.int8(write_result)
                psv.tofile(f_out)
            except Exception as e:
                print(f"\nデータ書き込みエラー: {pos} ({e})", file=sys.stderr)
    if skipped_out_of_range_scores:
        print(f"score が int16 範囲外のため除外した局面数: {skipped_out_of_range_scores:,}")


class PackedSfenBinWriter:
    """PackedSfenValue を逐次書き込むための軽量ライタ。"""
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.board = cshogi.Board()
        self.psv = np.zeros(1, dtype=cshogi.PackedSfenValue)
        self.int16_info = np.iinfo(np.int16)
        self.skipped_out_of_range_scores = 0
        self.rows_written = 0
        self.f_out = open(output_path, "wb")

    def write_position(self, pos: dict) -> bool:
        try:
            self.board.set_sfen(pos['sfen'])
            self.board.to_psfen(self.psv)
            cshogi_result = int(pos['game_result'])
            write_result = 1 if cshogi_result == 1 else -1 if cshogi_result == 2 else 0
            score = int(pos['eval_score_cp'])
            if not (self.int16_info.min <= score <= self.int16_info.max):
                self.skipped_out_of_range_scores += 1
                return False
            self.psv[0]["score"] = np.int16(score)
            self.psv[0]["move"] = np.uint16(0)
            self.psv[0]["gamePly"] = np.uint16(pos['ply'])
            self.psv[0]["game_result"] = np.int8(write_result)
            self.psv.tofile(self.f_out)
            self.rows_written += 1
            return True
        except Exception as e:
            print(f"\nデータ書き込みエラー: {pos} ({e})", file=sys.stderr)
            return False

    def close(self):
        self.f_out.close()
        if self.skipped_out_of_range_scores:
            print(
                f"データセット '{self.output_path}' で "
                f"score が int16 範囲外のため除外した局面数: {self.skipped_out_of_range_scores:,}"
            )


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


def _load_sfen_frequency_map(sfen_count_csv: str) -> dict:
    freq_map = {}
    with open(sfen_count_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                freq_map[row['sfen']] = int(row['total_count'])
            except (KeyError, ValueError):
                continue
    return freq_map


def _compute_sampling_target_count(freq: int, mode: str, cutoff_value: float, min_freq: int) -> float:
    if mode == "none" or freq < min_freq:
        return float(freq)
    target = _compute_sampling_target(freq, mode, cutoff_value)
    # Keep the target monotonic around min_freq. Otherwise freq=min_freq may be
    # compressed below freq=min_freq-1, which inverts the frequency ordering.
    return min(float(freq), max(float(min_freq), target))


def _compute_sampling_keep_probability(freq: int, mode: str, cutoff_value: float, min_freq: int) -> float:
    if mode == "none":
        return 1.0
    target = _compute_sampling_target_count(freq, mode, cutoff_value, min_freq)
    return min(1.0, target / max(1.0, float(freq)))


def _weighted_quantile(sorted_pairs, q: float) -> float:
    if not sorted_pairs:
        raise ValueError("重み付き分位点を計算するためのデータがありません。")
    if q <= 0.0:
        return float(sorted_pairs[0][0])
    if q >= 1.0:
        return float(sorted_pairs[-1][0])

    total_weight = sum(weight for _, weight in sorted_pairs)
    if total_weight <= 0.0:
        raise ValueError("重み付き分位点の総重みが0以下です。")

    threshold = total_weight * q
    cumulative = 0.0
    for value, weight in sorted_pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(sorted_pairs[-1][0])


def _cp_to_teacher_logit(value_cp: float, score_scaling: float, teacher_temperature: float) -> float:
    return float(value_cp) / (float(score_scaling) * float(teacher_temperature))


def _load_corn_threshold_positions(args: argparse.Namespace) -> tuple[list, list]:
    input_paths, _, all_positions = _load_positions_from_csvs(args.input_csv, args.input_csvs)
    if not all_positions:
        sys.exit("エラー: 入力ファイルにデータがありません。")
    return input_paths, all_positions


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
    """Return the least valuable on-board attacker for a square, if any."""
    least_value = None
    for attacker_sq in board.attackers_to(color, sq):
        piece_value = _piece_value(board.piece_type(attacker_sq))
        if piece_value <= 0:
            continue
        if least_value is None or piece_value < least_value:
            least_value = piece_value
    return least_value


def _has_see_like_capture(board: cshogi.Board) -> bool:
    """
    Approximate SEE for quiet-position filtering.

    This is intentionally not a full swap-list / gain-array SEE:
    it only checks whether a legal capture still looks materially favorable
    after at most one recapture and one re-recapture using least-value attackers.
    """
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

            # A cheap SEE-like proxy: keep captures that still look material-positive
            # after one recapture / re-recapture cycle with least-value attackers.
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
                to_sq = ny * 9 + nx
                move = board.move(king_sq, to_sq, False)
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
    if board.mate_move_in_1ply():
        return "mate_in_1_available", analysis

    if quiet_level == "2":
        return None, analysis

    # Strict quiet positions: approximate qsearch-relevant tactics and king danger.
    if _has_see_like_capture(board):
        return "favorable_capture_available", analysis
    if analysis["check_moves"] > 0:
        return "checking_move_available", analysis
    if analysis["promotion_moves"] > 0:
        return "promotion_tactic_available", analysis
    if analysis["king_zone_tactical_moves"] > 0:
        return "king_zone_tactic_available", analysis
    if analysis["my_king_attackers"] > 0:
        return "king_under_attack", analysis
    if analysis["opp_king_zone_pressure"] >= 3:
        return "opp_king_zone_pressure_high", analysis
    if analysis["my_king_zone_pressure"] >= 4:
        return "my_king_zone_pressure_high", analysis
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


def diff_sfen_logic(args: argparse.Namespace) -> None:
    """
    [diff-sfenコマンド] candidate CSV から、base CSV に存在する SFEN を除外する。
    low-only SFEN を作る用途を想定。
    """
    base_path = Path(args.base_csv)
    candidate_path = Path(args.candidate_csv)
    if not base_path.exists():
        sys.exit(f"エラー: base CSV '{base_path}' が見つかりません。")
    if not candidate_path.exists():
        sys.exit(f"エラー: candidate CSV '{candidate_path}' が見つかりません。")

    with open(base_path, 'r', newline='', encoding='utf-8') as f_base:
        base_reader = csv.DictReader(f_base)
        if not base_reader.fieldnames or 'sfen' not in base_reader.fieldnames:
            sys.exit("エラー: base CSV に 'sfen' 列が必要です。")
        base_sfens = {row['sfen'] for row in base_reader if row.get('sfen')}

    with open(candidate_path, 'r', newline='', encoding='utf-8') as f_candidate:
        candidate_reader = csv.DictReader(f_candidate)
        header = candidate_reader.fieldnames
        if not header or 'sfen' not in header:
            sys.exit("エラー: candidate CSV に 'sfen' 列が必要です。")
        candidate_rows = list(candidate_reader)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_rows = []
    skipped_rows = 0
    for row in candidate_rows:
        sfen = row.get('sfen')
        if not sfen:
            skipped_rows += 1
            continue
        if sfen in base_sfens:
            skipped_rows += 1
            continue
        output_rows.append(row)

    with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=header)
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"SFEN差分抽出が完了しました。base SFEN数: {len(base_sfens):,}, "
        f"candidate 行数: {len(candidate_rows):,}, 出力行数: {len(output_rows):,}, 除外行数: {skipped_rows:,}"
    )


def _adjust_eval_score(score: int, mode: str, scale: float, max_abs_cp: int) -> int:
    if mode == "zero":
        return 0
    if mode == "scale":
        return int(round(score * scale))
    if mode == "clip":
        if max_abs_cp <= 0:
            return score
        clipped = max(-max_abs_cp, min(max_abs_cp, score))
        return int(round(clipped * scale))
    return score


def adjust_eval_logic(args: argparse.Namespace) -> None:
    """
    [adjust-evalコマンド] 評価済みCSVの eval_score_cp を縮小・ゼロ寄せ・クリップする。
    low-only SFEN の教師を弱める用途を想定。
    """
    input_path = Path(args.input_csv)
    if not input_path.exists():
        sys.exit(f"エラー: 入力ファイル '{input_path}' が見つかりません。")

    with open(input_path, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        header = list(reader.fieldnames or [])

    if not header or 'eval_score_cp' not in header:
        sys.exit("エラー: 入力CSVに 'eval_score_cp' 列が必要です。")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if 'source_eval_score_cp' not in header:
        header.append('source_eval_score_cp')
    if 'eval_adjust_mode' not in header:
        header.append('eval_adjust_mode')
    if 'eval_adjust_param' not in header:
        header.append('eval_adjust_param')

    adjusted_rows = 0
    skipped_rows = 0
    adjusted = []
    for row in rows:
        try:
            original_score = int(row['eval_score_cp'])
        except (TypeError, ValueError):
            skipped_rows += 1
            adjusted.append(row)
            continue

        new_score = _adjust_eval_score(original_score, args.mode, args.scale, args.max_abs_cp)
        row['source_eval_score_cp'] = original_score
        row['eval_score_cp'] = new_score
        row['eval_adjust_mode'] = args.mode
        if args.mode == "clip":
            row['eval_adjust_param'] = f"scale={args.scale:g},max_abs_cp={args.max_abs_cp}"
        elif args.mode == "scale":
            row['eval_adjust_param'] = f"scale={args.scale:g}"
        else:
            row['eval_adjust_param'] = "zero"
        adjusted.append(row)
        adjusted_rows += 1

    with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=header)
        writer.writeheader()
        writer.writerows(adjusted)

    print(
        f"評価値調整が完了しました。mode={args.mode}, 行数: {len(rows):,}, "
        f"調整済み: {adjusted_rows:,}, スキップ: {skipped_rows:,}, 出力CSV: {output_path}"
    )


def _load_positions_from_csvs(input_csv: str = None, input_csvs: str = None):
    input_paths = []
    if input_csv:
        input_paths.append(Path(input_csv))
    if input_csvs:
        input_paths.extend(Path(p.strip()) for p in input_csvs.split(',') if p.strip())

    if not input_paths:
        sys.exit("エラー: 少なくとも1つの入力CSVが必要です。")

    rows = []
    header = None
    for path in input_paths:
        if not path.exists():
            sys.exit(f"エラー: 入力ファイル '{path}' が見つかりません。")
        with open(path, 'r', newline='', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            current_header = reader.fieldnames
            if not current_header:
                continue
            if header is None:
                header = current_header
            elif current_header != header:
                sys.exit(f"エラー: ヘッダが一致しません: {path}")
            rows.extend(reader)

    return input_paths, header, rows


def _load_eval_map_from_csvs(input_csv: str = None, input_csvs: str = None) -> tuple[list, dict]:
    input_paths, header, rows = _load_positions_from_csvs(input_csv, input_csvs)
    if not header or 'sfen' not in header or 'eval_score_cp' not in header:
        sys.exit("エラー: 評価済みSFEN CSV には 'sfen' と 'eval_score_cp' 列が必要です。")

    eval_map = {}
    duplicate_same_score = 0
    for row in rows:
        sfen = row.get('sfen')
        if not sfen:
            continue
        try:
            score = int(row['eval_score_cp'])
        except (TypeError, ValueError):
            continue
        if sfen in eval_map:
            if eval_map[sfen] != score:
                sys.exit(f"エラー: 同一SFENに対して異なる評価値が存在します: {sfen}")
            duplicate_same_score += 1
            continue
        eval_map[sfen] = score

    if not eval_map:
        sys.exit("エラー: 評価済みSFEN CSV から有効な評価値を読み取れませんでした。")

    if duplicate_same_score:
        print(f"重複SFEN（同一評価値）を統合: {duplicate_same_score:,}")
    return input_paths, eval_map


def _load_generate_positions(args: argparse.Namespace) -> tuple[list, list]:
    if args.positions_csv or args.positions_csvs:
        position_paths, header, rows = _load_positions_from_csvs(args.positions_csv, args.positions_csvs)
        if not header:
            sys.exit("エラー: 局面CSVのヘッダを読み取れません。")
        required = {'sfen', 'ply', 'game_result'}
        missing = [key for key in required if key not in header]
        if missing:
            sys.exit(f"エラー: 局面CSVに必要な列が不足しています: {', '.join(missing)}")

        _, eval_map = _load_eval_map_from_csvs(args.eval_sfen_csv, args.eval_sfen_csvs)
        joined_rows = []
        missing_eval = 0
        for row in rows:
            sfen = row.get('sfen')
            if not sfen:
                missing_eval += 1
                continue
            if sfen not in eval_map:
                missing_eval += 1
                continue
            joined = row.copy()
            joined['eval_score_cp'] = eval_map[sfen]
            joined_rows.append(joined)

        if not joined_rows:
            sys.exit("エラー: 評価値を join できた局面がありませんでした。")
        print(
            f"評価済みSFENを局面CSVへ結合: 入力局面数={len(rows):,}, "
            f"採用局面数={len(joined_rows):,}, 評価値なし除外={missing_eval:,}, "
            f"評価SFEN数={len(eval_map):,}"
        )
        return position_paths, joined_rows

    input_paths, header, rows = _load_positions_from_csvs(args.input_csv, args.input_csvs)
    if not header or 'eval_score_cp' not in header:
        sys.exit("エラー: generate の直接入力CSVには 'eval_score_cp' 列が必要です。")
    return input_paths, rows


def _iter_csv_paths(input_csv: str = None, input_csvs: str = None) -> list[Path]:
    input_paths = []
    if input_csv:
        input_paths.append(Path(input_csv))
    if input_csvs:
        input_paths.extend(Path(p.strip()) for p in input_csvs.split(',') if p.strip())
    return input_paths


def _build_eval_lookup_db(eval_csv: str = None, eval_csvs: str = None, db_path: Path = None) -> tuple[list, Path]:
    input_paths = _iter_csv_paths(eval_csv, eval_csvs)
    if not input_paths:
        sys.exit("エラー: 評価済みSFEN CSV が指定されていません。")

    if db_path is None:
        db_path = Path(".generate_eval_lookup.sqlite3")

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('PRAGMA journal_mode = WAL')
    cursor.execute('PRAGMA synchronous = NORMAL')
    cursor.execute('CREATE TABLE eval_map (sfen TEXT PRIMARY KEY, eval_score_cp INTEGER NOT NULL)')

    inserted = 0
    duplicate_same_score = 0
    for path in input_paths:
        if not path.exists():
            conn.close()
            sys.exit(f"エラー: 評価済みSFEN CSV '{path}' が見つかりません。")
        with open(path, 'r', newline='', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            header = reader.fieldnames or []
            if 'sfen' not in header or 'eval_score_cp' not in header:
                conn.close()
                sys.exit("エラー: 評価済みSFEN CSV には 'sfen' と 'eval_score_cp' 列が必要です。")
            for row in reader:
                sfen = row.get('sfen')
                if not sfen:
                    continue
                try:
                    score = int(row['eval_score_cp'])
                except (TypeError, ValueError):
                    continue
                cursor.execute('SELECT eval_score_cp FROM eval_map WHERE sfen = ?', (sfen,))
                existing = cursor.fetchone()
                if existing is not None:
                    if existing[0] != score:
                        conn.close()
                        sys.exit(f"エラー: 同一SFENに対して異なる評価値が存在します: {sfen}")
                    duplicate_same_score += 1
                    continue
                cursor.execute('INSERT INTO eval_map (sfen, eval_score_cp) VALUES (?, ?)', (sfen, score))
                inserted += 1
                if inserted % 100000 == 0:
                    conn.commit()
    conn.commit()
    conn.close()

    if inserted == 0:
        sys.exit("エラー: 評価済みSFEN CSV から有効な評価値を読み取れませんでした。")
    if duplicate_same_score:
        print(f"重複SFEN（同一評価値）を統合: {duplicate_same_score:,}")
    return input_paths, db_path


def _stream_generate_join_mode(args: argparse.Namespace) -> bool:
    """
    join入力かつ追加の全体保持が不要な場合は、CSVを逐次joinして直接.binへ書く。
    戻り値がTrueならこの経路で generate を完了済み。
    """
    if not (args.positions_csv or args.positions_csvs):
        return False
    if args.sfen_sampling_mode != "none":
        return False
    if args.quiet_level != "none":
        return False

    position_paths = _iter_csv_paths(args.positions_csv, args.positions_csvs)
    if not position_paths:
        return False

    print("大規模 join 入力向けのストリーミング generate を使用します。")

    output_dir = Path(args.output_dir)
    lookup_db_path = output_dir / ".generate_eval_lookup.sqlite3"
    _, eval_db_path = _build_eval_lookup_db(args.eval_sfen_csv, args.eval_sfen_csvs, lookup_db_path)
    conn = sqlite3.connect(eval_db_path)
    cursor = conn.cursor()

    train_writer = PackedSfenBinWriter(str(output_dir / "train.bin"))
    val_writer = PackedSfenBinWriter(str(output_dir / "val.bin"))

    total_rows = 0
    train_rows = 0
    val_rows = 0
    missing_eval = 0
    skipped_invalid_ply = 0

    try:
        for path in position_paths:
            if not path.exists():
                sys.exit(f"エラー: 局面CSV '{path}' が見つかりません。")
            with open(path, 'r', newline='', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in)
                header = reader.fieldnames or []
                required = {'sfen', 'ply', 'game_result'}
                missing = [key for key in required if key not in header]
                if missing:
                    sys.exit(f"エラー: 局面CSVに必要な列が不足しています: {', '.join(missing)}")

                for row in tqdm(reader, desc=f"Joining {path.name}"):
                    total_rows += 1
                    sfen = row.get('sfen')
                    if not sfen:
                        missing_eval += 1
                        continue
                    try:
                        ply = int(row['ply'])
                    except (TypeError, ValueError):
                        skipped_invalid_ply += 1
                        continue
                    if not (args.min_ply <= ply <= args.max_ply):
                        continue

                    cursor.execute('SELECT eval_score_cp FROM eval_map WHERE sfen = ?', (sfen,))
                    res = cursor.fetchone()
                    if res is None:
                        missing_eval += 1
                        continue

                    joined = row.copy()
                    joined['ply'] = ply
                    joined['eval_score_cp'] = res[0]
                    if random.random() < args.val_split:
                        if val_writer.write_position(joined):
                            val_rows += 1
                    else:
                        if train_writer.write_position(joined):
                            train_rows += 1
    finally:
        train_writer.close()
        val_writer.close()
        conn.close()
        try:
            eval_db_path.unlink()
        except OSError:
            pass

    if train_rows + val_rows == 0:
        sys.exit("エラー: フィルタ適用後に局面が残りませんでした。")

    print(
        f"評価済みSFENを局面CSVへストリーミング結合: 入力局面数={total_rows:,}, "
        f"採用局面数={train_rows + val_rows:,}, 評価値なし/不正除外={missing_eval + skipped_invalid_ply:,}"
    )
    if skipped_invalid_ply:
        print(f"ply列を解釈できないため除外した局面数: {skipped_invalid_ply:,}")
    print(f"分割結果 - 訓練: {train_rows}局面, 検証: {val_rows}局面")
    print("\nすべての処理が完了しました。")
    return True


def generate_datasets_logic(args: argparse.Namespace) -> None:
    """
    [generateコマンド] 評価値付きCSVから、.bin形式の学習データを生成する。
    """
    print(f"--- .bin データセット生成を開始 ---")
    if _stream_generate_join_mode(args):
        return
    input_paths, all_positions = _load_generate_positions(args)
    if not all_positions: sys.exit("エラー: 入力ファイルにデータがありません。")
    if len(input_paths) == 1:
        print(f"読み込み完了。入力CSV: {input_paths[0]}, 総局面数: {len(all_positions)}")
    else:
        print(f"読み込み完了。入力CSV数: {len(input_paths)}, 総局面数: {len(all_positions)}")

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

        freq_map = _load_sfen_frequency_map(args.sfen_count_csv)

        by_sfen = defaultdict(list)
        for pos in all_positions:
            sfen = pos.get('sfen')
            if sfen:
                by_sfen[sfen].append(pos)

        sampled_positions = []
        applied_sfens = 0
        expected_output_positions = 0.0
        for sfen, items in by_sfen.items():
            freq = freq_map.get(sfen, len(items))
            target_count = _compute_sampling_target_count(
                freq,
                args.sfen_sampling_mode,
                args.sfen_cutoff_value,
                args.sfen_sampling_min_freq,
            )
            keep_prob = _compute_sampling_keep_probability(
                freq,
                args.sfen_sampling_mode,
                args.sfen_cutoff_value,
                args.sfen_sampling_min_freq,
            )
            expected_output_positions += target_count
            if keep_prob >= 1.0:
                sampled_positions.extend(items)
                continue
            applied_sfens += 1
            for item in items:
                if random.random() < keep_prob:
                    sampled_positions.append(item)
        all_positions = sampled_positions
        print(
            f"SFEN頻度サンプリングを適用: mode={args.sfen_sampling_mode}, "
            f"min_freq={args.sfen_sampling_min_freq}, "
            f"局面数 {len(sampled_positions):,}, "
            f"期待局面数 {expected_output_positions:.2f}, "
            f"適用SFEN数 {applied_sfens:,}"
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


def build_corn_thresholds_logic(args: argparse.Namespace) -> None:
    """
    [corn-thresholdsコマンド]
    generate と同じ頻度補正の期待分布から、CORN 用の cp/logit 閾値を構築する。
    """
    if args.num_thresholds <= 0:
        sys.exit("エラー: --num-thresholds は1以上を指定してください。")
    if args.score_scaling <= 0.0:
        sys.exit("エラー: --score-scaling は正の値である必要があります。")
    if args.teacher_temperature <= 0.0:
        sys.exit("エラー: --teacher-temperature は正の値である必要があります。")
    if args.sfen_sampling_mode != "none":
        if not args.sfen_count_csv:
            sys.exit("エラー: SFEN頻度補正を使う場合は --sfen-count-csv の指定が必須です。")
        if not Path(args.sfen_count_csv).exists():
            sys.exit(f"エラー: SFEN頻度CSV '{args.sfen_count_csv}' が見つかりません。")
        if args.sfen_sampling_mode == "fixed" and args.sfen_cutoff_value <= 0:
            sys.exit("エラー: --sfen-sampling-mode fixed の場合、--sfen-cutoff-value は1以上を指定してください。")
        if args.sfen_sampling_min_freq < 1:
            sys.exit("エラー: --sfen-sampling-min-freq は1以上を指定してください。")

    input_paths, all_positions = _load_corn_threshold_positions(args)
    print("--- CORN閾値の構築を開始 ---")
    if len(input_paths) == 1:
        print(f"読み込み完了。入力CSV: {input_paths[0]}, 総局面数: {len(all_positions):,}")
    else:
        print(f"読み込み完了。入力CSV数: {len(input_paths)}, 総局面数: {len(all_positions):,}")

    if args.min_ply > args.max_ply:
        sys.exit("エラー: --min-ply は --max-ply 以下である必要があります。")

    filtered_positions = []
    skipped_invalid_ply = 0
    skipped_invalid_score = 0
    has_any_ply = False
    for pos in all_positions:
        ply_raw = pos.get('ply')
        if ply_raw not in (None, ""):
            try:
                ply = int(ply_raw)
            except (TypeError, ValueError):
                skipped_invalid_ply += 1
                continue
            has_any_ply = True
            if not (args.min_ply <= ply <= args.max_ply):
                continue
        try:
            pos['_eval_score_cp_float'] = float(pos['eval_score_cp'])
        except (KeyError, TypeError, ValueError):
            skipped_invalid_score += 1
            continue
        filtered_positions.append(pos)

    if skipped_invalid_ply:
        print(f"ply列を解釈できないため除外した局面数: {skipped_invalid_ply:,}")
    if skipped_invalid_score:
        print(f"eval_score_cp列を解釈できないため除外した局面数: {skipped_invalid_score:,}")
    if not filtered_positions:
        sys.exit("エラー: フィルタ適用後に局面が残りませんでした。")
    if has_any_ply:
        print(f"plyフィルタ適用後の局面数: {len(filtered_positions):,} (範囲: {args.min_ply}..{args.max_ply})")
    else:
        print("ply列が無いため、plyフィルタは適用せず全局面を使用します。")
        print(f"評価値として使用する局面数: {len(filtered_positions):,}")

    freq_map = {}
    if args.sfen_sampling_mode != "none":
        freq_map = _load_sfen_frequency_map(args.sfen_count_csv)

    weighted_scores = []
    by_sfen = defaultdict(list)
    for pos in filtered_positions:
        sfen = pos.get('sfen')
        if not sfen:
            continue
        by_sfen[sfen].append(pos)

    applied_sfens = 0
    for sfen, items in by_sfen.items():
        freq = freq_map.get(sfen, len(items))
        target_count = _compute_sampling_target_count(
            freq,
            args.sfen_sampling_mode,
            args.sfen_cutoff_value,
            args.sfen_sampling_min_freq,
        )
        weight_per_item = target_count / max(1, len(items))
        if target_count != len(items):
            applied_sfens += 1
        for item in items:
            weighted_scores.append((item['_eval_score_cp_float'], weight_per_item))

    weighted_scores.sort(key=lambda x: x[0])
    total_weight = sum(weight for _, weight in weighted_scores)
    if total_weight <= 0.0:
        sys.exit("エラー: 有効な重み付き評価値分布を構築できませんでした。")

    cp_thresholds = []
    for i in range(1, args.num_thresholds + 1):
        q = i / (args.num_thresholds + 1)
        cp_thresholds.append(round(_weighted_quantile(weighted_scores, q), 6))

    deduped_cp_thresholds = []
    last = None
    for value in cp_thresholds:
        if last is None or value != last:
            deduped_cp_thresholds.append(value)
            last = value

    logit_thresholds = [
        round(_cp_to_teacher_logit(value, args.score_scaling, args.teacher_temperature), 6)
        for value in deduped_cp_thresholds
    ]

    print(
        f"重み付き分布を構築: 局面数={len(weighted_scores):,}, "
        f"期待総重み={total_weight:.2f}, "
        f"sampling_mode={args.sfen_sampling_mode}, "
        f"補正適用SFEN数={applied_sfens:,}"
    )
    print("")
    print("derived_from_cp_thresholds:")
    for value in deduped_cp_thresholds:
        print(f"  - {value:g}")
    print("")
    print("corn_aux_thresholds (logit space):")
    for value in logit_thresholds:
        print(f"  - {value:g}")
    print("")
    print(
        "logit conversion: cp / (score_scaling * teacher_temperature)"
        f" = cp / ({args.score_scaling:g} * {args.teacher_temperature:g})"
    )
    if args.corn_aux_weight is not None:
        print(f"corn_aux_weight: {args.corn_aux_weight:g}")
    print("")
    cli_values = ",".join(f"{v:g}" for v in logit_thresholds)
    print("CLI example:")
    suffix = f" --model.corn_aux_weight={args.corn_aux_weight:g}" if args.corn_aux_weight is not None else ""
    print(f"  --model.corn_aux_thresholds=[{cli_values}]{suffix}")

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
        
    candidate_dtype = np.dtype([
        ('search_depth', np.int16),
        ('search_nodes', np.int32),
        ('search_movetime', np.int32),
        ('multipv', np.int16),
        ('move', np.uint16),
        ('score', np.int16),
        ('is_mate', np.bool_),
    ])
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
        ('actual_move', np.uint16),
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
                    
                    search_param_sets = get_candidate_search_params(args, ply)
                    candidates_list = []
                    for d, n, m in search_param_sets:
                        candidates_info = engine.get_multipv(sfen, depth=d, nodes=n, movetime=m, num_pv=args.num_pv)
                        depth_value = -1 if d is None else d
                        nodes_value = -1 if n is None else n
                        movetime_value = -1 if m is None else m
                        for cand in candidates_info:
                            candidates_list.append((
                                depth_value,
                                nodes_value,
                                movetime_value,
                                cand['multipv'],
                                cand['move'],
                                cand['score'],
                                cand['is_mate'],
                            ))
                    
                    pos_struct = np.zeros(1, dtype=position_dtype)
                    pos_struct[0]['ply'] = ply
                    board.to_psfen(pos_struct[0]['psv'])
                    pos_struct[0]['actual_move'] = np.uint16(move)
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

    merge_extract_parser = subparsers.add_parser("merge-extract", parents=[config_parent], help="複数の extract 出力CSVを1つにマージします。")
    merge_extract_parser.add_argument("--input-csvs", help="マージ対象CSVのカンマ区切りリスト。")
    merge_extract_parser.add_argument("--output-csv", help="マージ後CSVの出力パス。")
    merge_extract_parser.set_defaults(func=merge_extract_logic)

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

    plot_sfen_hist_parser = subparsers.add_parser("plot-sfen-histogram", parents=[config_parent], help="count-sfen の total_count 分布をヒストグラム画像として出力します。")
    plot_sfen_hist_parser.add_argument("--input-csv", help="count-sfen で生成した入力CSVのパス。")
    plot_sfen_hist_parser.add_argument("--output-png", help="出力するヒストグラム画像のパス。")
    plot_sfen_hist_parser.add_argument("--count-column", default="total_count", help="ヒストグラム化する頻度列名。")
    plot_sfen_hist_parser.add_argument("--bins", type=int, default=100, help="ヒストグラムのビン数。")
    plot_sfen_hist_parser.add_argument("--max-count", type=int, default=None, help="この値を超える頻度を描画から除外する。")
    plot_sfen_hist_parser.add_argument("--log-x", action='store_true', help="x軸を対数にする。")
    plot_sfen_hist_parser.add_argument("--log-y", action='store_true', help="y軸を対数にする。")
    plot_sfen_hist_parser.add_argument("--dpi", type=int, default=150, help="出力画像のDPI。")
    plot_sfen_hist_parser.add_argument("--title", help="グラフタイトル。")
    plot_sfen_hist_parser.set_defaults(func=plot_sfen_histogram_logic)

    classify_sfen_parser = subparsers.add_parser("classify-sfen", parents=[config_parent], help="SFEN一覧CSVを静止局面と非静止局面に分類します。")
    classify_sfen_parser.add_argument("--input-csv", help="入力となるSFEN一覧CSVのパス。")
    classify_sfen_parser.add_argument("--output-quiet-csv", help="静止局面CSVの出力パス。")
    classify_sfen_parser.add_argument("--output-tactical-csv", help="非静止局面CSVの出力パス。")
    classify_sfen_parser.add_argument(
        "--quiet-level",
        choices=["1", "2", "3"],
        default="2",
        help="静止局面判定の強さ。1=終局/王手/反復除外, 2=1手詰め筋も除外, 3=SEE風に得な取り・王手候補・成り筋・玉周辺の危険も抑える。",
    )
    classify_sfen_parser.set_defaults(func=classify_sfen_logic)

    label_parser = subparsers.add_parser("label", parents=[config_parent], help="対局結果から評価値をラベリングします（エンジン不要）。")
    label_parser.add_argument("--input-csv", help="入力となるフィルタリング済みCSVのパス。")
    label_parser.add_argument("--output-csv", help="ラベリング結果を保存するCSVのパス。")
    label_parser.add_argument("--score-scale", type=int, default=600, help="終局付近で到達する評価値の最大絶対値。")
    label_parser.add_argument("--label-min-score", type=int, default=100, help="序盤側で使う評価値の最小絶対値。")
    label_parser.add_argument("--label-curve", type=float, default=4.0, help="終局に近づくほど評価値を増やす曲線の強さ。大きいほど終盤寄りで立ち上がる。")
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
    evaluate_sfen_parser.add_argument("--existing-eval-csv", help="既存の評価済みSFEN CSV。ここにあるSFENは再評価せず流用する。")
    evaluate_sfen_parser.add_argument("--existing-eval-csvs", help="既存の評価済みSFEN CSV のカンマ区切りリスト。")
    evaluate_sfen_parser.set_defaults(func=evaluate_sfen_logic)

    merge_eval_sfen_parser = subparsers.add_parser("merge-eval-sfen", parents=[config_parent], help="複数の評価済みSFEN CSVを1つにマージします。")
    merge_eval_sfen_parser.add_argument("--input-csvs", help="マージ対象CSVのカンマ区切りリスト。")
    merge_eval_sfen_parser.add_argument("--output-csv", help="マージ後CSVの出力パス。")
    merge_eval_sfen_parser.set_defaults(func=merge_eval_sfen_logic)

    diff_sfen_parser = subparsers.add_parser("diff-sfen", parents=[config_parent], help="candidate CSV から、base CSV に存在する SFEN を除外します。")
    diff_sfen_parser.add_argument("--base-csv", help="差分の基準となるCSV。")
    diff_sfen_parser.add_argument("--candidate-csv", help="差分抽出対象のCSV。")
    diff_sfen_parser.add_argument("--output-csv", help="差分抽出後CSVの出力パス。")
    diff_sfen_parser.set_defaults(func=diff_sfen_logic)

    adjust_eval_parser = subparsers.add_parser("adjust-eval", parents=[config_parent], help="評価済みCSVの eval_score_cp を縮小・ゼロ寄せ・クリップします。")
    adjust_eval_parser.add_argument("--input-csv", help="入力となる評価済みCSVのパス。")
    adjust_eval_parser.add_argument("--output-csv", help="調整後CSVの出力パス。")
    adjust_eval_parser.add_argument("--mode", choices=["scale", "zero", "clip"], default="scale", help="評価値調整方式。")
    adjust_eval_parser.add_argument("--scale", type=float, default=0.5, help="scale/clip時に掛ける係数。")
    adjust_eval_parser.add_argument("--max-abs-cp", type=int, default=1200, help="clip時の絶対値上限。")
    adjust_eval_parser.set_defaults(func=adjust_eval_logic)

    corn_thresholds_parser = subparsers.add_parser("corn-thresholds", parents=[config_parent], help="generate と同じ頻度補正を考慮した CORN 閾値を構築します。")
    corn_thresholds_parser.add_argument("--input-csv", help="入力となる評価値付きCSVのパス。")
    corn_thresholds_parser.add_argument("--input-csvs", help="マージして扱う評価値付きCSVのカンマ区切りリスト。")
    corn_thresholds_parser.add_argument("--min-ply", type=int, default=0, help="閾値計算対象とする最小手数。")
    corn_thresholds_parser.add_argument("--max-ply", type=int, default=999, help="閾値計算対象とする最大手数。")
    corn_thresholds_parser.add_argument("--sfen-count-csv", help="count-sfenで生成したSFEN頻度CSVのパス。")
    corn_thresholds_parser.add_argument("--sfen-sampling-mode", choices=["none", "fixed", "sqrt", "log10"], default="none", help="generate と同じ SFEN 頻度補正方式。")
    corn_thresholds_parser.add_argument("--sfen-cutoff-value", type=float, default=1.0, help="fixed方式の上限値。")
    corn_thresholds_parser.add_argument("--sfen-sampling-min-freq", type=int, default=1, help="この頻度未満のSFENには頻度補正を適用しない。")
    corn_thresholds_parser.add_argument("--num-thresholds", type=int, default=4, help="生成する閾値数。K個の閾値で K+1 クラス。")
    corn_thresholds_parser.add_argument("--score-scaling", type=float, default=361.0, help="cp から teacher-logit 空間へ変換する際の score_scaling。")
    corn_thresholds_parser.add_argument("--teacher-temperature", type=float, default=1.0, help="teacher-logit 空間へ変換する際の teacher_temperature。")
    corn_thresholds_parser.add_argument("--corn-aux-weight", type=float, default=None, help="出力例に含める corn_aux_weight。")
    corn_thresholds_parser.set_defaults(func=build_corn_thresholds_logic)

    generate_parser = subparsers.add_parser("generate", parents=[config_parent], help="評価値付きCSVから学習データ(.bin)を生成します。")
    generate_parser.add_argument("--input-csv", help="入力となる評価値付きCSVのパス。")
    generate_parser.add_argument("--input-csvs", help="マージして扱う評価値付きCSVのカンマ区切りリスト。")
    generate_parser.add_argument("--positions-csv", help="label 出力などの局面展開済みCSV。内部で eval-sfen 結果を join する場合に使用。")
    generate_parser.add_argument("--positions-csvs", help="マージして扱う局面展開済みCSVのカンマ区切りリスト。")
    generate_parser.add_argument("--eval-sfen-csv", help="evaluate-sfen で生成した評価済みSFEN CSV。")
    generate_parser.add_argument("--eval-sfen-csvs", help="マージして扱う評価済みSFEN CSVのカンマ区切りリスト。")
    generate_parser.add_argument("--output-dir", help="生成されたデータセットを保存するディレクトリ。")
    generate_parser.add_argument("--val-split", type=float, default=0.1)
    generate_parser.add_argument("--min-ply", type=int, default=0, help="生成対象とする最小手数。")
    generate_parser.add_argument("--max-ply", type=int, default=999, help="生成対象とする最大手数。")
    generate_parser.add_argument(
        "--quiet-level",
        choices=["none", "1", "2", "3"],
        default="none",
        help="静止局面フィルタの強さ。none=無効, 1=終局/王手/反復除外, 2=1手詰め筋も除外, 3=SEE風に得な取り・王手候補・成り筋・玉周辺の危険も抑える。",
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
    build_h5_parser.add_argument("--candidate-depths", help="中終盤で保存する候補探索深さのカンマ区切り。指定時は depth/nodes/movetime の単一条件指定より優先。")
    build_h5_parser.add_argument("--early-candidate-depths", help="序盤で保存する候補探索深さのカンマ区切り。指定時は early-depth/early-nodes/early-movetime より優先。")
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

    elif args.command == "merge-extract":
        if not (args.input_csvs and args.output_csv):
             sys.exit("エラー: merge-extractコマンドには --input-csvs と --output-csv の指定が必須です。")
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    elif args.command == "filter":
        if not (args.input_csv and args.output_csv):
             sys.exit("エラー: filterコマンドには --input-csv と --output-csv の指定が必須です。")
    elif args.command == "count-sfen":
        if not args.input_csv:
             sys.exit("エラー: count-sfenコマンドには --input-csv の指定が必須です。")
        if not args.output_csv:
             sys.exit("エラー: count-sfenコマンドには --output-csv の指定が必須です。")

    elif args.command == "plot-sfen-histogram":
        if not args.input_csv:
             sys.exit("エラー: plot-sfen-histogramコマンドには --input-csv の指定が必須です。")
        if not args.output_png:
             sys.exit("エラー: plot-sfen-histogramコマンドには --output-png の指定が必須です。")
        Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    elif args.command == "label":
        if not (args.input_csv and args.output_csv):
             sys.exit("エラー: labelコマンドには --input-csv と --output-csv の指定が必須です。")
    elif args.command == "evaluate":
        if not (args.input_csv and args.engine_path and args.output_csv):
             sys.exit("エラー: evaluateコマンドには --input-csv, --engine-path, --output-csv の指定が必須です。")
    elif args.command == "evaluate-sfen":
        if not (args.input_csv and args.engine_path and args.output_csv):
             sys.exit("エラー: evaluate-sfenコマンドには --input-csv, --engine-path, --output-csv の指定が必須です。")
        if args.existing_eval_csv and args.existing_eval_csvs:
             sys.exit("エラー: evaluate-sfenコマンドでは --existing-eval-csv と --existing-eval-csvs を同時に指定できません。")
    elif args.command == "merge-eval-sfen":
        if not (args.input_csvs and args.output_csv):
             sys.exit("エラー: merge-eval-sfenコマンドには --input-csvs と --output-csv の指定が必須です。")
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    elif args.command == "diff-sfen":
        if not (args.base_csv and args.candidate_csv and args.output_csv):
             sys.exit("エラー: diff-sfenコマンドには --base-csv, --candidate-csv, --output-csv の指定が必須です。")
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    elif args.command == "adjust-eval":
        if not (args.input_csv and args.output_csv):
             sys.exit("エラー: adjust-evalコマンドには --input-csv と --output-csv の指定が必須です。")
        if args.mode == "clip" and args.max_abs_cp <= 0:
             sys.exit("エラー: adjust-eval --mode clip では --max-abs-cp は1以上を指定してください。")
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    elif args.command == "corn-thresholds":
        if not (args.input_csv or args.input_csvs):
             sys.exit("エラー: corn-thresholdsコマンドには --input-csv または --input-csvs の指定が必須です。")
        if args.input_csv and args.input_csvs:
             sys.exit("エラー: corn-thresholdsコマンドでは --input-csv と --input-csvs を同時に指定できません。")
    elif args.command == "classify-sfen":
        if not (args.input_csv and args.output_quiet_csv and args.output_tactical_csv):
             sys.exit("エラー: classify-sfenコマンドには --input-csv, --output-quiet-csv, --output-tactical-csv の指定が必須です。")
        Path(args.output_quiet_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_tactical_csv).parent.mkdir(parents=True, exist_ok=True)
    elif args.command == "generate":
        direct_input = bool(args.input_csv or args.input_csvs)
        joined_input = bool(args.positions_csv or args.positions_csvs or args.eval_sfen_csv or args.eval_sfen_csvs)
        if not args.output_dir:
             sys.exit("エラー: generateコマンドには --output-dir の指定が必須です。")
        if direct_input and joined_input:
             sys.exit("エラー: generateコマンドでは direct入力(--input-csv/--input-csvs) と join入力(--positions-csv/--eval-sfen-csv 系) を同時に指定できません。")
        if not direct_input and not joined_input:
             sys.exit("エラー: generateコマンドには direct入力または join入力のどちらかが必要です。")
        if args.input_csv and args.input_csvs:
             sys.exit("エラー: generateコマンドでは --input-csv と --input-csvs を同時に指定できません。")
        if args.positions_csv and args.positions_csvs:
             sys.exit("エラー: generateコマンドでは --positions-csv と --positions-csvs を同時に指定できません。")
        if args.eval_sfen_csv and args.eval_sfen_csvs:
             sys.exit("エラー: generateコマンドでは --eval-sfen-csv と --eval-sfen-csvs を同時に指定できません。")
        if bool(args.positions_csv or args.positions_csvs) != bool(args.eval_sfen_csv or args.eval_sfen_csvs):
             sys.exit("エラー: join入力では --positions-csv 系と --eval-sfen-csv 系を両方指定してください。")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    elif args.command == "build-h5":
        if not (args.input_csv and args.output_h5 and args.engine_path):
             sys.exit("エラー: build-h5コマンドには --input-csv, --output-h5, --engine-path の指定が必須です。")
        Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
        
    args.func(args)

if __name__ == "__main__":
    main()
