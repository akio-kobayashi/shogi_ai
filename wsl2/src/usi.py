# -*- coding: utf-8 -*-
"""
概要: USI (Universal Shogi Interface) プロトコルを介して将棋エンジンと通信するためのクラス。
機能:
- USIエンジンの起動と終了
- USIコマンドの送受信
- 特定の局面 (SFEN) の評価値を取得
"""

import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

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

    def quit(self) -> None:
        """エンジンを終了させる。"""
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
