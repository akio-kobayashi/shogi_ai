# -*- coding: utf-8 -*-
"""
Floodgateの棋譜を解析し、評価値推移のグラフを出力するスクリプト。
"""
import os
import argparse
import sys
from pathlib import Path

import cshogi
import matplotlib.pyplot as plt
import japanize_matplotlib
from tqdm import tqdm

from usi import UsiEngine

def analyze_game(args):
    if not Path(args.csa_file).exists():
        sys.exit(f"エラー: 棋譜ファイルが見つかりません: {args.csa_file}")
    if not Path(args.engine_path).exists():
        sys.exit(f"エラー: エンジンが見つかりません: {args.engine_path}")

    # エンジンの起動
    engine = UsiEngine(args.engine_path)
    
    # 棋譜の読み込み
    parser = cshogi.Parser.parse_file(args.csa_file)
    if not parser:
        sys.exit("エラー: 棋譜のパースに失敗しました。")
    game = parser[0]
    
    # プレイヤーの特定
    black_name = game.names[0]
    white_name = game.names[1]
    
    if args.player:
        if args.player.lower() in ['black', 'sente', 'b'] or args.player.lower() in black_name.lower():
            target_side = cshogi.BLACK
        elif args.player.lower() in ['white', 'gote', 'w'] or args.player.lower() in white_name.lower():
            target_side = cshogi.WHITE
        else:
            sys.exit(f"エラー: プレイヤー名 '{args.player}' が見つかりません。 (先手: {black_name}, 後手: {white_name})")
    else:
        target_side = cshogi.BLACK

    target_name = black_name if target_side == cshogi.BLACK else white_name
    print(f"分析視点: {target_name} ({'先手' if target_side == cshogi.BLACK else '後手'})")

    board = cshogi.Board(game.sfen)
    eval_list = []
    my_mistakes = []      # 自分が評価値を下げた手
    opponent_mistakes = [] # 相手が評価値を下げた（自分にとってチャンスだった）手
    
    # 初期局面の評価 (0手目)
    score_type, score_value = engine.evaluate_sfen(board.sfen(), depth=args.depth, nodes=args.nodes, movetime=args.movetime)
    last_eval = score_value if score_type == "cp" else (30000 if score_value > 0 else -30000)
    
    # 常に target_side 視点の評価値に固定
    if target_side == cshogi.WHITE: last_eval = -last_eval
    eval_list.append(last_eval)

    for i, move in enumerate(tqdm(game.moves)):
        ply = i + 1
        current_turn = board.turn
        move_usi = board.move_to_usi(move)
        board.push(move)
        
        # エンジン評価
        score_type, score_value = engine.evaluate_sfen(board.sfen(), depth=args.depth, nodes=args.nodes, movetime=args.movetime)
        current_eval = score_value if score_type == "cp" else (30000 if score_value > 0 else -30000)
        
        # 先手視点に変換
        if board.turn == cshogi.WHITE:
            current_eval = -current_eval
        # ターゲット視点に変換
        if target_side == cshogi.WHITE:
            current_eval = -current_eval
        
        eval_diff = current_eval - last_eval
        
        if current_turn == target_side:
            # 自分の指し手: 評価値が下がったらミス
            if eval_diff < -args.blunder_threshold:
                my_mistakes.append((ply, current_eval, f"{ply}手目 {move_usi} (ミス: {eval_diff})"))
        else:
            # 相手の指し手: 評価値が上がったら相手のミス
            if eval_diff > args.blunder_threshold:
                opponent_mistakes.append((ply, current_eval, f"{ply}手目 {move_usi} (相手の失着: +{eval_diff})"))

        eval_list.append(current_eval)
        last_eval = current_eval

    engine.quit()

    # 結果の表示
    print(f"\n--- {target_name} の分析結果 ---")
    print(f"[自分のミス (下落幅 > {args.blunder_threshold})]")
    for m in my_mistakes: print(m[2])
    print(f"\n[相手の失着 (上昇幅 > {args.blunder_threshold})]")
    for m in opponent_mistakes: print(m[2])

    # グラフの作成
    plt.figure(figsize=(12, 6))
    plt.plot(eval_list, marker='o', markersize=3, color='gray', alpha=0.5, label=f'評価値 ({target_name}視点)')
    
    if my_mistakes:
        plt.scatter([m[0] for m in my_mistakes], [m[1] for m in my_mistakes], color='red', s=80, label='自分のミス', zorder=5)
    if opponent_mistakes:
        plt.scatter([m[0] for m in opponent_mistakes], [m[1] for m in opponent_mistakes], color='blue', s=80, label='相手の失着', zorder=5)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title(f"対局分析: {Path(args.csa_file).name}\n分析対象: {target_name}")
    plt.xlabel("手数")
    plt.ylabel("評価値 (cp)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(args.output_img)
    print(f"\nグラフを保存しました: {args.output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Floodgate棋譜解析・プレイヤー視点可視化ツール")
    parser.add_argument("csa_file", help="解析するCSAファイルのパス")
    parser.add_argument("--engine-path", required=True, help="USIエンジンのパス")
    parser.add_argument("--player", help="分析対象のプレイヤー名 (または black/white/sente/gote)")
    parser.add_argument("--output-img", default="analysis.png", help="グラフ画像の出力パス")
    parser.add_argument("--depth", type=int, default=10, help="探索深さ")
    parser.add_argument("--nodes", type=int, help="探索ノード数")
    parser.add_argument("--movetime", type=int, help="探索時間(ms)")
    parser.add_argument("--blunder-threshold", type=int, default=200, help="ミスとみなす評価値の変動幅")
    
    args = parser.parse_args()
    analyze_game(args)
