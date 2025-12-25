# -*- coding: utf-8 -*-
"""
概要: USI (Universal Shogi Interface) プロトコルを介して将棋エンジンと通信するためのクラス。
機能:
- USIエンジンの起動と終了
- USIコマンドの送受信
- 特定の局面 (SFEN) の評価値を取得
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cshogi

class UsiEngine:
    """USIプロトコルで将棋エンジンと通信するクラス。"""
    def __init__(self, engine_path: str) -> None:
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"エンジン実行ファイルが見つかりません: {engine_path}")

        engine_dir = Path(engine_path).parent
        command_to_run = [engine_path]
        is_windows_target = '.exe' in engine_path.lower() or '.bat' in engine_path.lower()

        # Popenに渡す共通の引数
        popen_args = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "text": True,
            "encoding": "cp932",
            "bufsize": 1
        }

        # WSL環境でWindowsの実行ファイル(.exe or .bat)を扱う場合の特別処理
        if sys.platform == 'linux' and is_windows_target:
            try:
                # wslpathコマンドでWindows形式のパスに変換
                win_engine_path = subprocess.run(['wslpath', '-w', engine_path], capture_output=True, text=True, check=True).stdout.strip()
                
                # cmd.exe /c "C:\path\to\run.bat" のようにコマンドを構築
                command_to_run = ['cmd.exe', '/c', win_engine_path]
                print(f"Info: WSLでWindowsコマンドを実行します: {command_to_run}")
                self.proc = subprocess.Popen(command_to_run, **popen_args)

            except Exception as e:
                print(f"警告: cmd.exe経由でのエンジン起動に失敗しました。: {e}")
                # 失敗した場合は、元の方法で試行
                self.proc = subprocess.Popen([engine_path], **popen_args)
        else:
            # WSL以外、またはLinux実行ファイルの場合は従来のロジック
            self.proc = subprocess.Popen(command_to_run, cwd=engine_dir, **popen_args)

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
        # (Popenのstdoutはblockingなので、ここでは単純なreadlineを使う)
        # 実際にはselectや別スレッドでの読み込みが堅牢だが、今回は簡潔さを優先。
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

    def set_multipv(self, num_pv: int) -> None:
        """MultiPVの値を設定する。"""
        self._send(f"setoption name MultiPV value {num_pv}")

    def get_multipv(self, sfen: str, depth: int, num_pv: int) -> list:
        """
        指定されたSFENの局面で、複数の指し手候補と評価値を取得する。
        戻り値: 候補手の情報を含む辞書のリスト
            [
                {'multipv': 1, 'move': (指し手), 'score': (評価値), 'is_mate': (詰みフラグ)},
                {'multipv': 2, 'move': (指し手), 'score': (評価値), 'is_mate': (詰みフラグ)},
                ...
            ]
        """
        self._send("usinewgame")
        self.set_multipv(num_pv)
        self._send(f"position sfen {sfen.strip()}")
        self._send(f"go depth {depth}")

        results = {}
        board = cshogi.Board(sfen) # USI文字列をパースするためにBoardオブジェクトが必要

        while True:
            line = self._readline(timeout=300.0)
            if line is None:
                raise TimeoutError("エンジンからの応答がタイムアウトしました。")

            if line.startswith("info") and "multipv" in line:
                info = self._parse_multipv_info(line, board)
                if info:
                    # 同じmultipvの結果が複数来ることがあるので、常に上書きする
                    results[info['multipv']] = info

            if line.startswith("bestmove"):
                break
        
        # multipvの値でソートして返す
        sorted_results = sorted(results.values(), key=lambda x: x['multipv'])
        return sorted_results

    def _parse_multipv_info(self, line: str, board: cshogi.Board) -> Optional[dict]:
        """MultiPVのinfo行をパースする。"""
        parts = line.split()
        try:
            if "multipv" in parts and "score" in parts and "pv" in parts:
                pv_idx = parts.index("pv")
                if len(parts) <= pv_idx + 1:
                    return None # pvの後に指し手がない

                move_usi = parts[pv_idx + 1]
                move_int = board.move_from_usi(move_usi)

                mpv_idx = parts.index("multipv")
                multipv = int(parts[mpv_idx + 1])

                score_idx = parts.index("score")
                score_type = parts[score_idx + 1]
                score_value = int(parts[score_idx + 2])

                return {
                    'multipv': multipv,
                    'move': move_int,
                    'score': score_value,
                    'is_mate': score_type == 'mate'
                }
        except (ValueError, IndexError):
            pass
        return None

    def quit(self) -> None:
        """エンジンを終了させる。"""
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
