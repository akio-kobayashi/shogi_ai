#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将棋の棋譜ファイル（CSA/KIF/KI2形式）から、客観的な特徴量を抽出するスクリプト。

【機能】
- metadata.csv に記載された棋譜ファイル群を読み込み、1局面=1行の形式で特徴量テーブルを生成します。
- 先手・後手両方の持ち駒を含め、視点に依存しない客観的な特徴量を記録します。
- 視点の選択（先手モデル用/後手モデル用など）は、このデータを利用する下流の学習スクリプト等で行うことを想定しています。
- 評価関数や探索は一切使用せず、局面から計数可能な統計量のみを特徴量として利用します。
- 出力はCSVまたはParquet形式に対応しています。

【依存パッケージ】
- cshogi
- pandas
- numpy
- tqdm
- python-snappy (Parquetでsnappy圧縮を利用する場合)
- fastparquet (Parquet書き出しエンジン)
"""

import os
import argparse
from typing import List, Dict, Optional, Any

import cshogi
import pandas as pd
import numpy as np
from tqdm import tqdm


# --- 定数定義 ---

# 持ち駒として特徴量にする駒種
PIECE_NAMES = ['P', 'L', 'N', 'S', 'G', 'B', 'R']
# cshogiの駒定数との対応 (歩=1, ..., 飛=7)
CSHOGI_PIECE_TO_NAME = {i + 1: name for i, name in enumerate(PIECE_NAMES)}

# 持ち駒特徴量のカラム名リスト
SENTE_HAND_COLS = [f'S_hand_{name}' for name in PIECE_NAMES]
GOTE_HAND_COLS = [f'G_hand_{name}' for name in PIECE_NAMES]

# 出力データフレームのカラムとデータ型定義
# メモリ効率を考慮し、なるべく小さい型を指定する
COLUMN_DTYPES = {
    'game_id': 'object',  # 文字列や混合型
    'ply': 'int16',
    'sente_elo': 'float32',  # 欠損値(NaN)を許容するためfloat
    'gote_elo': 'float32',
    'result_sente_win': 'float32', # 引分(0.5)を許容するためfloat
    'captured_after': 'int8',
    'capture_available': 'int8',
    'in_check': 'int8',
    'give_check_available': 'int8',
    'ply_feature': 'int16',
    'remaining_plies_to_end': 'int16',
    'is_mate': 'int8',
    **{col: 'int8' for col in SENTE_HAND_COLS},
    **{col: 'int8' for col in GOTE_HAND_COLS}
}


def get_winner_info(kifu: Any) -> float:
    """棋譜情報から先手の勝敗を 1.0, 0.0, 0.5 で返す"""
    if kifu.win == cshogi.BLACK_WIN:
        return 1.0
    elif kifu.win == cshogi.WHITE_WIN:
        return 0.0
    else:  # 引き分け、中断など
        return 0.5


def make_feature_dict(board: cshogi.Board) -> Dict[str, Any]:
    """
    特定の局面(board)から、統計的な特徴量を抽出して辞書形式で返す。
    評価関数や探索は使用しない。
    """
    is_in_check = 1 if board.is_check() else 0
    is_mate = 1 if board.is_game_over() else 0

    is_capture_available = 0
    is_give_check_available = 0
    
    legal_moves = board.legal_moves
    for m in legal_moves:
        if board.is_capture(m):
            is_capture_available = 1
        
        board.push(m)
        if board.is_check():
            is_give_check_available = 1
        board.pop()

        if is_capture_available and is_give_check_available:
            break
    
    features = {
        'in_check': is_in_check,
        'is_mate': is_mate,
        'capture_available': is_capture_available,
        'give_check_available': is_give_check_available,
    }

    # 先手・後手両方の持ち駒を記録
    hands = board.pieces_in_hand
    for turn, hand_counts in enumerate(hands):
        prefix = 'S' if turn == 0 else 'G'
        for count, pt_name in zip(hand_counts, PIECE_NAMES):
            features[f'{prefix}_hand_{pt_name}'] = count
            
    return features


def extract_features_for_game(metadata_row: pd.Series, path_column: str) -> Optional[pd.DataFrame]:
    """
    単一の棋譜ファイルから特徴量を抽出し、DataFrameとして返す。
    """
    kifu_path = metadata_row[path_column]
    game_id = metadata_row.get('game_id', os.path.splitext(os.path.basename(kifu_path))[0])

    if not os.path.exists(kifu_path):
        print(f"Warning: File not found: {kifu_path}. Skipping.")
        return None

    try:
        games = cshogi.Parser.parse_file(kifu_path)
        if not games: return None
        kif_idx = int(metadata_row.get('kif_index', 0))
        kifu = games[kif_idx]
    except Exception as e:
        print(f"Warning: Failed to parse {kifu_path}. Reason: {e}")
        return None

    sente_elo = metadata_row.get('rating_b', kifu.ratings[0] if kifu.ratings else None)
    gote_elo = metadata_row.get('rating_w', kifu.ratings[1] if kifu.ratings else None)
    result_sente_win = get_winner_info(kifu)
    
    moves = kifu.moves
    total_plies = len(moves)
    
    if total_plies == 0:
        return None

    board = cshogi.Board()
    game_features: List[Dict[str, Any]] = []

    captured_after_flag = 0

    for i, move in enumerate(moves):
        ply = i + 1

        # 共通ロジック make_feature_dict を呼び出し
        features = make_feature_dict(board)
        
        # メタデータ特有の情報を追加
        features.update({
            'game_id': game_id,
            'ply': ply,
            'sente_elo': sente_elo,
            'gote_elo': gote_elo,
            'result_sente_win': result_sente_win,
            'captured_after': captured_after_flag,
            'ply_feature': ply,
            'remaining_plies_to_end': total_plies - ply,
        })

        game_features.append(features)

        captured_after_flag = 1 if board.is_capture(move) else 0
        board.push(move)

    if not game_features:
        return None

    if not game_features:
        return None

    df = pd.DataFrame(game_features)
    # カラムの順序を定義
    ordered_columns = list(COLUMN_DTYPES.keys())
    # 存在しないカラムがあった場合のエラーを防ぐ
    df = df.reindex(columns=[col for col in ordered_columns if col in df.columns])
    df = df.astype({k: v for k, v in COLUMN_DTYPES.items() if k in df.columns})
    
    return df


def create_dummy_csa_files():
    """テスト用のダミーCSAファイルをカレントディレクトリに作成する"""
    print("Creating dummy CSA files for testing...")
    
    csa_content_1 = """V2.2
N+PLAYER1
N-PLAYER2
$EVENT:Test Game 1
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA * 
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  * 
P5 *  *  *  *  *  *  *  *  * 
P6 *  *  *  *  *  *  *  *  * 
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI * 
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
-3334FU
+8822UM
-3122GI
%TORYO,T10
-"""
    with open("test_game_1.csa", "w", encoding="utf-8") as f:
        f.write(csa_content_1)

    csa_content_2 = """V2.2
N+Sente
N-Gote
PI
+
+2726FU
-3334FU
+0022KA
-3122GI
%TORYO,T0
+"""
    with open("test_game_2.csa", "w", encoding="utf-8") as f:
        f.write(csa_content_2)
        
    print("Dummy files created: test_game_1.csa, test_game_2.csa")

def create_dummy_metadata_csv(path: str):
    """テスト用のダミーmetadata.csvを作成する"""
    print(f"Creating dummy metadata file: {path}")
    metadata = {
        'game_id': ['game_1', 'game_2'],
        'file_path': ['test_game_1.csa', 'test_game_2.csa'],
        'rating_b': [2800, 2600],
        'rating_w': [2750, 2650],
        'kif_index': [0, 0]
    }
    pd.DataFrame(metadata).to_csv(path, index=False)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="metadata.csvを元に棋譜ファイルから特徴量を抽出し、CSVまたはParquet形式で保存します。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="出力ファイルパス。拡張子に応じてフォーマットが自動で決まります (.csv or .parquet)。\n例: 'features.csv', 'features.parquet'"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="metadata.csv",
        help="棋譜のパスやメタ情報を含むCSVファイルのパス (デフォルト: metadata.csv)"
    )
    parser.add_argument(
        "--path_column",
        type=str,
        default="file_path",
        help="metadata.csv内で棋譜ファイルのパスが格納されているカラム名 (デフォルト: file_path)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="一度にメモリに読み込んでからファイルに書き出す対局数 (デフォルト: 100)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="このフラグを立てると、テスト用のダミー棋譜とメタデータを生成して処理を実行します。"
    )
    args = parser.parse_args()

    if args.test:
        dummy_metadata_path = "dummy_metadata.csv"
        create_dummy_csa_files()
        create_dummy_metadata_csv(dummy_metadata_path)
        metadata_path = dummy_metadata_path
        print(f"\n--- Running in TEST mode ---")
    else:
        metadata_path = args.metadata_path

    output_ext = os.path.splitext(args.output_path)[1].lower()
    if output_ext not in ['.csv', '.parquet']:
        raise ValueError("output_pathの拡張子は .csv または .parquet である必要があります。")

    try:
        metadata_df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at: {metadata_path}")
        return

    print(f"Loaded {len(metadata_df)} games from {metadata_path}.")
    print(f"Output will be saved to: {args.output_path} (Format: {output_ext})")
    print(f"Batch size: {args.batch_size}")

    all_dfs: List[pd.DataFrame] = []
    is_first_batch = True

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing games"):
        df = extract_features_for_game(row, args.path_column)
        if df is not None and not df.empty:
            all_dfs.append(df)
        
        if len(all_dfs) >= args.batch_size:
            batch_df = pd.concat(all_dfs, ignore_index=True)
            
            if output_ext == '.csv':
                batch_df.to_csv(
                    args.output_path,
                    mode='a' if not is_first_batch else 'w',
                    header=is_first_batch,
                    index=False
                )
            elif output_ext == '.parquet':
                batch_df.to_parquet(
                    args.output_path,
                    engine='fastparquet',
                    compression='snappy',
                    append=not is_first_batch,
                    write_index=False
                )

            all_dfs = []
            is_first_batch = False

    if all_dfs:
        final_batch_df = pd.concat(all_dfs, ignore_index=True)
        if output_ext == '.csv':
            final_batch_df.to_csv(
                args.output_path,
                mode='a' if not is_first_batch else 'w',
                header=is_first_batch,
                index=False
            )
        elif output_ext == '.parquet':
            final_batch_df.to_parquet(
                args.output_path,
                engine='fastparquet',
                compression='snappy',
                append=not is_first_batch,
                write_index=False
            )

    print(f"\nProcessing finished. Features saved to {args.output_path}")

    if args.test:
        print("\nCleaning up dummy files...")
        os.remove("test_game_1.csa")
        os.remove("test_game_2.csa")
        os.remove(dummy_metadata_path)
        print("Cleanup complete.")


if __name__ == '__main__':
    main()
