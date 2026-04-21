#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
metadata.csvを集計し、データセットの概要（プレイヤー数、レーティング分布、対局数など）を表示するスクリプト。
プレイヤーごとのサマリーは、オプションでCSVファイルに保存できます。
"""

import pandas as pd
import argparse
import sys

def summarize_metadata(metadata_path: str, output_path: str = None):
    """
    metadata.csvを読み込み、集計結果を出力する。
    
    Args:
        metadata_path (str): metadata.csvファイルのパス。
        output_path (str, optional): プレイヤーサマリーの出力先CSVファイルパス。
    """
    try:
        print(f"Loading metadata from: {metadata_path}")
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"エラー: 指定されたファイルが見つかりません: {metadata_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"エラー: ファイルの読み込み中に問題が発生しました: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 必須カラムの確認 ---
    required_columns = ['kif_index', 'black_player', 'white_player', 'rating_b', 'rating_w']
    if not all(col in df.columns for col in required_columns):
        print(f"エラー: metadata.csvに必要なカラム {required_columns} のいずれか、または全てが見つかりません。", file=sys.stderr)
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"見つからないカラム: {', '.join(missing_cols)}", file=sys.stderr)
        sys.exit(1)

    # --- 1. 全体サマリー ---
    total_games = len(df)
    all_players = pd.concat([df['black_player'], df['white_player']]).unique()
    num_unique_players = len(all_players)

    print("\n" + "="*30)
    print("      DATASET SUMMARY")
    print("="*30)
    print(f"Total Games Analyzed: {total_games}")
    print(f"Unique Players Found: {num_unique_players}")
    print("-"*30)

    # --- 2. レーティングの分布 ---
    # レーティングが0.0のものを除外してから分布を計算
    all_ratings = pd.concat([df['rating_b'], df['rating_w']]).dropna()
    all_ratings = all_ratings[all_ratings != 0.0] # 0.0を除外
    
    print("\n📊 Rating Distribution Summary:")
    if not all_ratings.empty:
        print(all_ratings.describe().round(2).to_string())
    else:
        print("（レーティングデータがありません、または全て0.0でした）")


    # --- 3. プレイヤーごとの統計情報 ---
    
    # 3-1. プレイヤーごとのゲーム数
    black_counts = df['black_player'].value_counts().rename('black_games')
    white_counts = df['white_player'].value_counts().rename('white_games')
    player_stats = pd.concat([black_counts, white_counts], axis=1)
    player_stats = player_stats.fillna(0).astype(int)
    player_stats['total_games'] = player_stats['black_games'] + player_stats['white_games']

    # 3-2. プレイヤーごとのレーティング統計
    df_black = df[['black_player', 'rating_b']].rename(columns={'black_player': 'player', 'rating_b': 'rating'})
    df_white = df[['white_player', 'rating_w']].rename(columns={'white_player': 'player', 'rating_w': 'rating'})
    all_player_ratings = pd.concat([df_black, df_white]).dropna(subset=['player', 'rating'])

    # レーティングが0.0の行を除外
    all_player_ratings = all_player_ratings[all_player_ratings['rating'] != 0.0]

    player_rating_summary = all_player_ratings.groupby('player')['rating'].agg(['mean', 'std', 'max', 'min']).round(0)
    player_rating_summary.columns = ['rating_mean', 'rating_std', 'rating_max', 'rating_min']

    # 標準偏差(std)がNaNになるのは対局数が1回の場合。その場合、ばらつきは0なのでfillna(0)で埋める。
    if 'rating_std' in player_rating_summary.columns:
        player_rating_summary['rating_std'] = player_rating_summary['rating_std'].fillna(0)

    # 3-3. ゲーム数とレーティング統計を結合
    player_summary_df = player_stats.join(player_rating_summary)
    
    # 3-4. 表示と保存のためのカラム順序を整理
    display_cols = [
        'total_games', 'black_games', 'white_games', 
        'rating_mean', 'rating_std', 'rating_max', 'rating_min'
    ]
    display_cols_exist = [col for col in display_cols if col in player_summary_df.columns]
    player_summary_df = player_summary_df[display_cols_exist]

    # 合計対局数が多い順にソート
    player_summary_df = player_summary_df.sort_values(by='total_games', ascending=False)
    
    print("\n" + "👥 Player Summary (Top 30):")
    # レーティング統計のNaNはそのまま表示
    print(player_summary_df.head(30).to_string())

    # --- 4. プレイヤーサマリーをCSVに保存 ---
    if output_path:
        try:
            player_summary_df.to_csv(output_path, index=True)
            print(f"\n💾 Player summary successfully saved to: {output_path}")
        except Exception as e:
            print(f"\nエラー: プレイヤーサマリーの保存中に問題が発生しました: {e}", file=sys.stderr)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="metadata.csvを集計し、データセットの概要を表示します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "metadata_path",
        nargs='?',
        default="metadata.csv",
        help="集計対象のmetadata.csvファイルのパス。"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="プレイヤーごとのサマリーを保存するCSVファイルへのパス。"
    )
    args = parser.parse_args()
    
    summarize_metadata(args.metadata_path, args.output)


if __name__ == '__main__':
    main()
