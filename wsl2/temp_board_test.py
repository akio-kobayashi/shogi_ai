import cshogi

board = cshogi.Board()
print("--- 初期盤面 ---")
print(board)
print("\n--- 初手(7g7f)後 ---")
board.push_usi("7g7f") # 初手
print(board)

