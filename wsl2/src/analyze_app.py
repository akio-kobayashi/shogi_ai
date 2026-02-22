# -*- coding: utf-8 -*-
import gradio as gr
import cshogi
import pandas as pd
import plotly.graph_objects as go
import os
from pathlib import Path
from usi import UsiEngine

# グローバル変数で解析結果を保持
class AnalysisState:
    def __init__(self):
        self.game = None
        self.evals = []
        self.sfens = []
        self.moves = []
        self.target_name = ""

state = AnalysisState()

def analyze_game(csa_file, engine_path, player_name, depth, blunder_threshold):
    if not csa_file or not engine_path:
        return "エラー: 棋譜ファイルとエンジンパスを指定してください。", None, ""

    try:
        engine = UsiEngine(engine_path)
        parser = cshogi.Parser.parse_file(csa_file.name)
        if not parser: return "エラー: 棋譜のパースに失敗", None, ""
        game = parser[0]
        
        black_name, white_name = game.names[0], game.names[1]
        if player_name:
            if player_name.lower() in ['black', 'sente', 'b'] or player_name.lower() in black_name.lower():
                target_side = cshogi.BLACK
            else:
                target_side = cshogi.WHITE
        else:
            target_side = cshogi.BLACK
        
        state.target_name = black_name if target_side == cshogi.BLACK else white_name
        state.game = game
        state.sfens = [game.sfen]
        state.evals = []
        state.moves = ["開始"]
        
        board = cshogi.Board(game.sfen)
        
        # 初期評価
        score_type, score_value = engine.evaluate_sfen(board.sfen(), depth=depth)
        v = score_value if score_type == "cp" else (30000 if score_value > 0 else -30000)
        if target_side == cshogi.WHITE: v = -v
        state.evals.append(v)

        blunders_text = f"### {state.target_name} の分析結果
"

        for i, move in enumerate(game.moves):
            current_turn = board.turn
            move_usi = board.move_to_usi(move)
            board.push(move)
            state.sfens.append(board.sfen())
            state.moves.append(move_usi)
            
            score_type, score_value = engine.evaluate_sfen(board.sfen(), depth=depth)
            v = score_value if score_type == "cp" else (30000 if score_value > 0 else -30000)
            
            if board.turn == cshogi.WHITE: v = -v # 先手視点
            if target_side == cshogi.WHITE: v = -v # ターゲット視点
            
            diff = v - state.evals[-1]
            if current_turn == target_side and diff < -blunder_threshold:
                blunders_text += f"- {i+1}手目 {move_usi} (ミス: {diff})
"
            elif current_turn != target_side and diff > blunder_threshold:
                blunders_text += f"- {i+1}手目 {move_usi} (相手の失着: +{diff})
"
                
            state.evals.append(v)
            
        engine.quit()
        
        # Plotlyグラフ作成
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(state.evals))), y=state.evals, mode='lines+markers', name='評価値'))
        fig.update_layout(title=f"評価値推移 ({state.target_name}視点)", xaxis_title="手数", yaxis_title="評価値 (cp)", hovermode="x unified")
        
        return "解析完了", fig, blunders_text

    except Exception as e:
        return f"エラー: {str(e)}", None, ""

def update_board(ply):
    if not state.sfens or ply >= len(state.sfens):
        return None
    board = cshogi.Board(state.sfens[int(ply)])
    return board.to_svg()

with gr.Blocks(title="Floodgate Analysis App") as demo:
    gr.Markdown("# Floodgate 棋譜分析アプリ")
    
    with gr.Row():
        with gr.Column(scale=1):
            csa_input = gr.File(label="CSA棋譜ファイル")
            engine_input = gr.Textbox(label="USIエンジンパス", placeholder="/path/to/engine")
            player_input = gr.Textbox(label="対象プレイヤー名", placeholder="自分（未指定なら先手）")
            depth_input = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="探索深さ")
            threshold_input = gr.Number(value=200, label="悪手閾値")
            analyze_btn = gr.Button("解析実行", variant="primary")
            
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="評価値グラフ")
            ply_slider = gr.Slider(minimum=0, maximum=256, value=0, step=1, label="手数選択")
            board_svg = gr.HTML(label="局面図")

    with gr.Row():
        blunder_output = gr.Markdown(label="解析ログ")

    # イベント定義
    analyze_btn.click(
        analyze_game, 
        inputs=[csa_input, engine_input, player_input, depth_input, threshold_input], 
        outputs=[gr.Textbox(label="ステータス"), plot_output, blunder_output]
    )
    
    # スライダーが動いたら盤面更新
    ply_slider.change(update_board, inputs=[ply_slider], outputs=[board_svg])
    
    # 解析完了時にスライダーの最大値を更新する仕組みが必要だが、Gradioの制約上、
    # analyze_gameの戻り値でSliderの更新オブジェクトを返すように調整可能
    def on_analyze_end(status, fig, blunders):
        if state.sfens:
            return gr.update(maximum=len(state.sfens)-1, value=0)
        return gr.update()
    
    analyze_btn.click(on_analyze_end, None, [ply_slider], queue=False)

if __name__ == "__main__":
    # share=True に設定することで、外部からアクセス可能な一時URL (https://xxx.gradio.app) を発行します。
    # server_name="0.0.0.0" により、ローカルネットワーク内の他のデバイスからもIP指定でアクセス可能です。
    demo.launch(share=True, server_name="0.0.0.0")
