# -*- coding: utf-8 -*-
"""
SFEN頻度DBの内容を分析し、ヒストグラムを出力するスクリプト。
"""
import sqlite3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="SFEN頻度DBの統計情報を分析・可視化します。")
    parser.add_argument("--db-path", default="sfen_cache.db", help="分析対象のSQLite DBパス。")
    parser.add_argument("--output-img", default="sfen_hist.png", help="出力するヒストグラム画像のパス。")
    parser.add_argument("--top-n", type=int, default=20, help="頻度が高い局面を上位いくつ表示するか。")
    args = parser.parse_args()

    if not Path(args.db_path).exists():
        print(f"エラー: DBファイルが見つかりません: {args.db_path}")
        return

    print(f"DB接続中: {args.db_path}")
    conn = sqlite3.connect(args.db_path)
    
    # データを読み込み
    print("データ取得中...")
    query = "SELECT total_count, output_count FROM sfen_counts"
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("データが空です。count-sfenを先に実行してください。")
        conn.close()
        return

    # 統計情報の表示
    print("
--- 統計情報 ---")
    print(f"総ユニーク局面数: {len(df):,}")
    print(f"総出現回数（延べ）: {df['total_count'].sum():,}")
    print(f"平均出現頻度: {df['total_count'].mean():.2f}")
    print(f"最大出現頻度: {df['total_count'].max():,}")
    
    # 上位局面の表示
    print(f"
--- 出現頻度上位 {args.top_n} 局面 ---")
    top_df = pd.read_sql_query(f"SELECT sfen, total_count FROM sfen_counts ORDER BY total_count DESC LIMIT {args.top_n}", conn)
    print(top_df)

    # ヒストグラムの描画
    print(f"
ヒストグラム作成中...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 全出現頻度の分布 (Total Count)
    ax[0].hist(df['total_count'], bins=50, color='skyblue', edgecolor='black', log=True)
    ax[0].set_title("局面出現頻度の分布 (Total Count)")
    ax[0].set_xlabel("出現回数")
    ax[0].set_ylabel("局面数 (Log Scale)")
    ax[0].grid(axis='y', alpha=0.3)

    # 2. 実際に出力された頻度の分布 (Output Count)
    ax[1].hist(df['output_count'], bins=50, color='salmon', edgecolor='black', log=True)
    ax[1].set_title("データセット出力頻度の分布 (Output Count)")
    ax[1].set_xlabel("出力回数")
    ax[1].set_ylabel("局面数 (Log Scale)")
    ax[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output_img)
    print(f"画像を保存しました: {args.output_img}")

    conn.close()

if __name__ == "__main__":
    main()
