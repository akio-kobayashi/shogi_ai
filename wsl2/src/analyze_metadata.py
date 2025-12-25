# -*- coding: utf-8 -*-
"""
棋譜のメタデータCSVファイルを分析するためのユーティリティスクリプト。

`create_dataset.py`の`extract`や`filter`コマンドで生成されたCSVファイルを対象に、
統計情報の表示、データの可視化、フィルタリング条件のシミュレーションなどを行います。
データセットの特性を把握し、フィルタリング条件を調整するのに役立ちます。

サブコマンド:
- stats: 基本統計量を表示します。
- plot: 分布をグラフで可視化します。
- simulate: フィルタリング条件を試行し、結果の棋譜数を表示します。
"""
import argparse
import sys
from pathlib import Path
import yaml

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Matplotlibの日本語設定
try:
    import japanize_matplotlib
    japanize_matplotlib.japanize()
except ImportError:
    print("japanize_matplotlib が見つかりません。'pip install japanize-matplotlib' の実行を推奨します。", file=sys.stderr)


def run_stats(args: argparse.Namespace) -> None:
    """[statsコマンド] データセットの基本統計量を計算して表示する。"""
    print("--- 基本統計量の計算 ---")
    df = pd.read_csv(args.input_csv)

    print(f"総棋譜数: {len(df)}")
    print("\n--- レーティングと手数の統計 ---")
    print(df[['rating_b', 'rating_w', 'total_moves']].describe())

    print("\n--- 勝敗結果の分布 ---")
    result_map = {1: '先手勝ち', 2: '後手勝ち', 0: '引き分け'}
    df['game_result'] = pd.to_numeric(df['game_result'])
    result_counts = df['game_result'].value_counts().rename(index=result_map)
    print(result_counts)
    print("\n--- 勝敗結果の割合 ---")
    print(df['game_result'].value_counts(normalize=True).rename(index=result_map))


def run_plot(args: argparse.Namespace) -> None:
    """[plotコマンド] データの分布をプロットし、画像として保存する。"""
    print("--- 分布の可視化 ---")
    df = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # レーティングのヒストグラム
    print("レーティングのヒストグラムを生成中...")
    plt.figure(figsize=(12, 6))
    plt.hist([df['rating_b'].dropna(), df['rating_w'].dropna()], bins=50, label=['先手', '後手'], range=(1000, 4500))
    plt.title('レーティング分布')
    plt.xlabel('レーティング')
    plt.ylabel('棋譜数')
    plt.legend()
    plt.grid(True)
    rating_hist_path = output_dir / "rating_histogram.png"
    plt.savefig(rating_hist_path)
    plt.close()
    print(f"保存しました: {rating_hist_path}")

    # 手数のヒストグラム
    print("手数のヒストグラムを生成中...")
    plt.figure(figsize=(12, 6))
    plt.hist(df['total_moves'].dropna(), bins=50, range=(0, 500))
    plt.title('手数分布')
    plt.xlabel('手数')
    plt.ylabel('棋譜数')
    plt.grid(True)
    moves_hist_path = output_dir / "total_moves_histogram.png"
    plt.savefig(moves_hist_path)
    plt.close()
    print(f"保存しました: {moves_hist_path}")


def run_simulate(args: argparse.Namespace) -> None:
    """[simulateコマンド] フィルタリング条件を適用した結果をシミュレーションする。"""
    print("--- フィルタリングシミュレーション ---")
    df = pd.read_csv(args.input_csv)
    
    print(f"フィルタリング前の総棋譜数: {len(df)}")

    result_map = {'win': 1, 'lose': 2, 'draw': 0}
    allowed_results_int = {result_map[res.strip()] for res in args.allowed_results.split(',')}

    queries = []
    queries.append(f"({args.min_rating} <= rating_b <= {args.max_rating})")
    queries.append(f"({args.min_rating} <= rating_w <= {args.max_rating})")
    queries.append(f"abs(rating_b - rating_w) <= {args.max_rating_diff}")
    queries.append(f"({args.min_moves} <= total_moves <= {args.max_moves})")
    queries.append(f"game_result in {list(allowed_results_int)}")

    if args.filter_by_rating_outcome:
        queries.append("((rating_b > rating_w) and (game_result == 1)) or "
                       "((rating_w > rating_b) and (game_result == 2)) or "
                       "(rating_b == rating_w)")

    final_query = " & ".join(queries)
    print("\n適用するフィルタ条件:")
    print(final_query)
    
    filtered_df = df.query(final_query)

    print(f"\nフィルタリング後の総棋譜数: {len(filtered_df)}")
    
    remaining_ratio = len(filtered_df) / len(df) if len(df) > 0 else 0
    print(f"残存率: {remaining_ratio:.2%}")


def main() -> None:
    """スクリプトのエントリポイント。引数をパースして各処理を実行する。"""
    parser = argparse.ArgumentParser(
        description="メタデータCSVを分析するスクリプト。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-c", "--config", help="設定YAMLファイルのパス。")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="利用可能なコマンド")

    # --- 'stats' コマンド ---
    stats_parser = subparsers.add_parser("stats", help="データセットの基本統計量を表示します。")
    stats_parser.add_argument("--input-csv", help="分析対象のメタデータCSVファイル。")
    stats_parser.set_defaults(func=run_stats)

    # --- 'plot' コマンド ---
    plot_parser = subparsers.add_parser("plot", help="データの分布を可視化し、画像ファイルとして保存します。")
    plot_parser.add_argument("--input-csv", help="分析対象のメタデータCSVファイル。")
    plot_parser.add_argument("--output-dir", default="analysis_output", help="生成された画像を保存するディレクトリ。")
    plot_parser.set_defaults(func=run_plot)

    # --- 'simulate' コマンド ---
    simulate_parser = subparsers.add_parser("simulate", help="フィルタリング条件を適用した結果をシミュレーションします。")
    simulate_parser.add_argument("--input-csv", help="分析対象のメタデータCSVファイル。")
    
    # フィルタリング設定 (create_dataset.pyのfilterコマンドからコピー)
    simulate_parser.add_argument("--min-rating", type=int, default=3000, help="学習対象とする対局者の最低レーティング。")
    simulate_parser.add_argument("--max-rating", type=int, default=9999, help="学習対象とする対局者の最大レーティング。")
    simulate_parser.add_argument("--max-rating-diff", type=int, default=1000, help="学習対象とする対局者間のレーティング差の上限。")
    simulate_parser.add_argument("--min-moves", type=int, default=0, help="学習対象とする棋譜の最小手数。")
    simulate_parser.add_argument("--max-moves", type=int, default=999, help="学習対象とする棋譜の最大手数。")
    simulate_parser.add_argument("--allowed-results", type=str, default="win,lose,draw", help="含める勝敗結果をカンマ区切りで指定 (win,lose,draw)。")
    simulate_parser.add_argument("--filter-by-rating-outcome", action='store_true', help="レーティングが高い方が勝った棋譜のみを対象とする。")
    simulate_parser.set_defaults(func=run_simulate)

    # --- 引数のパースと設定の上書き ---
    temp_args, _ = parser.parse_known_args()
    config = {}
    if temp_args.config and Path(temp_args.config).exists():
        print(f"設定ファイル '{temp_args.config}' を読み込みます。")
        with open(temp_args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    if temp_args.command and temp_args.command in config:
        subparsers.choices[temp_args.command].set_defaults(**config.get(temp_args.command, {}))

    args = parser.parse_args()
    
    # --- 必須引数のチェック ---
    if not args.input_csv:
        sys.exit("エラー: --input-csv の指定が必須です。")
        
    args.func(args)


if __name__ == "__main__":
    main()