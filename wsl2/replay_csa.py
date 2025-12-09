
import argparse
import subprocess
import sys
import time
import cshogi
from pathlib import Path
from typing import Optional, Tuple

# ================================
# USIエンジン制御クラス
# ================================

class UsiEngine:
    """USIプロトコルで将棋エンジンと通信するクラス。"""
    def __init__(self, engine_path: str) -> None:
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"エンジン実行ファイルが見つかりません: {engine_path}")

        self.proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # 行バッファ
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("エンジンの標準入出力の確保に失敗しました。")

        self._send("usi")
        self._wait_for("usiok")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str) -> None:
        """エンジンにコマンドを1行送信する。"""
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _readline(self, timeout: float = 60.0) -> Optional[str]:
        """エンジンから1行読み込む。タイムアウトした場合はNoneを返す。"""
        line = self.proc.stdout.readline()
        return line.strip() if line else None

    def _wait_for(self, keyword: str, timeout: float = 60.0) -> None:
        """指定したキーワードを含む行が来るまで待つ。"""
        start = time.time()
        while True:
            line = self._readline(timeout=timeout)
            if line is None or time.time() - start > timeout:
                raise TimeoutError(f"キーワード '{keyword}' を待機中にタイムアウトしました。")
            if keyword in line:
                return

    def evaluate_sfen(self, sfen: str, depth: int) -> Tuple[str, int]:
        """
        指定されたSFENの局面を評価する。
        戻り値: (score_type, score_value)
            - score_type: "cp" (センチポーン) または "mate" (詰み)
            - score_value: 評価値または詰み手数
        """
        self._send("usinewgame")
        self._send(f"position sfen {sfen.strip()}")
        self._send(f"go depth {depth}")

        last_score_type = "cp"
        last_score_value = 0

        while True:
            line = self._readline(timeout=300.0) # 探索には時間がかかるため長めのタイムアウト
            if line is None:
                raise TimeoutError("エンジンからの応答がタイムアウトしました。")

            if line.startswith("info"):
                s_type, s_val = self._parse_score_from_info(line)
                if s_type is not None:
                    last_score_type, last_score_value = s_type, s_val

            if line.startswith("bestmove"):
                break

        return last_score_type, last_score_value

    def _parse_score_from_info(self, line: str) -> Tuple[Optional[str], int]:
        """info行から評価値 (score) を抽出する。"""
        parts = line.split()
        try:
            if "score" in parts:
                idx = parts.index("score")
                score_type = parts[idx + 1]
                score_value = int(parts[idx + 2])
                if score_type in ("cp", "mate"):
                    return score_type, score_value
        except (ValueError, IndexError):
            pass
        return None, 0

    def quit(self) -> None:
        """エンジンを終了させる。"""
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ================================
# 盤面表示用ヘルパー
# ================================

SFEN_TO_CSA = {
    'P': 'FU', 'L': 'KY', 'N': 'KE', 'S': 'GI', 'G': 'KI', 'B': 'KA', 'R': 'HI', 'K': 'OU',
    '+P': 'TO', '+L': 'NY', '+N': 'NK', '+S': 'NG', '+B': 'UM', '+R': 'RY',
    'p': 'FU', 'l': 'KY', 'n': 'KE', 's': 'GI', 'g': 'KI', 'b': 'KA', 'r': 'HI', 'k': 'OU',
    '+p': 'TO', '+l': 'NY', '+n': 'NK', '+s': 'NG', '+b': 'UM', '+r': 'RY',
}

HAND_PIECES_CSA_ORDER = ['HI', 'KA', 'KI', 'GI', 'KE', 'KY', 'FU']
HAND_PIECES_SFEN_ORDER = ['R', 'B', 'G', 'S', 'N', 'L', 'P']

def format_board_csa(board: cshogi.Board) -> str:
    """cshogi.BoardオブジェクトをCSA形式の文字列に変換する。"""
    sfen = board.sfen()
    parts = sfen.split(' ')
    board_part = parts[0]
    turn_part = parts[1]
    hand_part = parts[2]

    lines = []

    # 盤面
    ranks = board_part.split('/')
    for i, rank in enumerate(ranks):
        line = f"P{i+1}"
        j = 0
        while j < len(rank):
            char = rank[j]
            if char.isdigit():
                for _ in range(int(char)):
                    line += " * "
                j += 1
            elif char == '+':
                promoted = True
                j += 1
                piece_char = rank[j]
                player = '+' if piece_char.isupper() else '-'
                csa_piece = SFEN_TO_CSA['+' + piece_char.lower()]
                line += f"{player}{csa_piece}"
                j += 1
            else:
                player = '+' if char.isupper() else '-'
                csa_piece = SFEN_TO_CSA[char]
                line += f"{player}{csa_piece}"
                j += 1
        lines.append(line)

    # 持ち駒
    black_hand_str = "P+"
    white_hand_str = "P-"
    
    if hand_part != '-':
        hand_counts = {}
        num_str = ""
        for char in hand_part:
            if char.isdigit():
                num_str += char
            else:
                count = int(num_str) if num_str else 1
                hand_counts[char] = count
                num_str = ""
        
        for sfen_p, csa_p in zip(HAND_PIECES_SFEN_ORDER, HAND_PIECES_CSA_ORDER):
            # 先手の持ち駒
            if sfen_p.upper() in hand_counts:
                for _ in range(hand_counts[sfen_p.upper()]):
                    black_hand_str += f"00{csa_p}"
            # 後手の持ち駒
            if sfen_p.lower() in hand_counts:
                 for _ in range(hand_counts[sfen_p.lower()]):
                    white_hand_str += f"00{csa_p}"

    if len(black_hand_str) == 2:
        black_hand_str += "00AL"
    if len(white_hand_str) == 2:
        white_hand_str += "00AL"

    lines.append(black_hand_str)
    lines.append(white_hand_str)
    
    # 手番
    lines.append('+' if turn_part == 'b' else '-')

    return "\n".join(lines)


# ================================
# メイン処理
# ================================

def replay_csa_file(
    csa_path: str,
    engine_path: str,
    depth: int,
    interval: float
) -> None:
    """CSAファイルを読み込み、対局を再現しながら評価値を表示する。"""
    print("エンジンを起動中...")
    engine = UsiEngine(engine_path)
    print("エンジン準備完了。")

    try:
        # cshogi.CSA.Parser.parse_file はジェネレータを返す
        kif_generator = cshogi.CSA.Parser.parse_file(csa_path)
        kif = next(kif_generator) # 最初の棋譜のみを対象とする
    except FileNotFoundError:
        print(f"エラー: CSAファイルが見つかりません: {csa_path}", file=sys.stderr)
        return
    except StopIteration:
        print(f"エラー: CSAファイルに棋譜データが含まれていません: {csa_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"エラー: CSAファイルの読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        return

    board = cshogi.Board()
    print("--- 対局開始 ---")
    print(format_board_csa(board))
    print("-" * 20)

    for i, move in enumerate(kif.moves):
        try:
            sfen = board.sfen()
            turn = '先手' if sfen.split(' ')[1] == 'b' else '後手'

            # 現在の局面を評価
            s_type, s_val = engine.evaluate_sfen(sfen, depth)
            
            # 情報を表示
            print(f"--- {i+1}手目 ({turn}) ---")
            print(f"指し手: {cshogi.move_to_usi(move)}")
            print(f"評価値: {s_type} {s_val}")
            
            # 盤面を更新して表示
            board.push(move)
            print(format_board_csa(board))
            print("-" * 20)

            time.sleep(interval)

        except TimeoutError as e:
            print(f"\n評価タイムアウト: ({e})", file=sys.stderr)
            continue
        except Exception as e:
            print(f"\n処理エラー: ({e})", file=sys.stderr)
            break
            
    print("--- 対局終了 ---")
    engine.quit()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSA棋譜ファイルを読み込み、USIエンジンで評価しながら対局を再現するスクリプト。"
    )
    parser.add_argument(
        "-f", "--csa-file", required=True,
        help="CSAファイルのパス。"
    )
    parser.add_argument(
        "-e", "--engine", required=True,
        help="USIエンジンの実行ファイルのパス。"
    )
    parser.add_argument(
        "--depth", type=int, default=10,
        help="USIエンジンの探索の深さ。(デフォルト: 10)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="各手の表示間隔（秒）。(デフォルト: 1.0)"
    )

    args = parser.parse_args()

    try:
        replay_csa_file(
            csa_path=args.csa_file,
            engine_path=args.engine,
            depth=args.depth,
            interval=args.interval
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
