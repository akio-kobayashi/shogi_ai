# -*- coding: utf-8 -*-
"""
概要:
cshogiライブラリの基本的な動作テストを行うための一時的なスクリプト。
盤面の初期化、指し手の適用、盤面表示などの機能を確認する目的で使用されます。
"""

import cshogi

board = cshogi.Board()
print("--- 初期盤面 ---")
print(board)
print("\n--- 初手(7g7f)後 ---")
board.push_usi("7g7f") # 初手
print(board)

