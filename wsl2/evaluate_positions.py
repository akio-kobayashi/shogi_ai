
import argparse
import subprocess
import sys
import time
import cshogi
import struct
from pathlib import Path
from typing import Optional, Tuple, Dict, Iterator

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm がインストールされていません。pip install tqdm を実行してください。", file=sys.stderr)
    # tqdm がなくても動くようにダミーを定義
    def tqdm(iterable, **kwargs):
        return iterable

# ================================
# 定数
# ================================

# USIの評価値を16bit整数に変換する際のセンチポーンの最大値
# これを超える値は詰みと見なされる
CP_MAX = 32000


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


# ================================
# 補助関数
# ================================

def iter_csa_files(base_dir: str) -> Iterator[Path]:
    """指定ディレクトリ以下の *.csa, *.CSA ファイルを再帰的に列挙する。"""
    base_path = Path(base_dir)
    if not base_path.is_dir():
        raise NotADirectoryError(f"指定されたパスはディレクトリではありません: {base_dir}")
    
    yield from base_path.rglob("*.csa")
    yield from base_path.rglob("*.CSA")


def normalize_sfen(sfen: str) -> str:
    """SFENの手数部分を除外し、盤面、手番、持ち駒だけの文字列を返す。"""
    return " ".join(sfen.split()[:3])

def convert_score_to_int16(score_type: str, score_value: int) -> int:
    """USIの評価値を16ビット符号付き整数に変換する。"""
    if score_type == "cp":
        return max(-CP_MAX, min(CP_MAX, score_value))
    elif score_type == "mate":
        if score_value > 0:  # N手詰
            return 32767 - (score_value - 1)
        else:  # N手で詰まされる
            return -32767 - (score_value + 1)
    return 0 # "error"などの場合

# ================================
# メイン処理
# ================================

def process_csa_directory(
    csa_dir: str,
    engine_path: str,
    output_file: str,
    depth: int,
    min_ply: int,
    min_rating: int,
    max_rating: int
) -> None:
    """CSAファイルを処理し、局面を評価してファイルに保存する。"""
    print("エンジンを起動中...")
    engine = UsiEngine(engine_path)
    print("エンジン準備完了。")

    evaluated_sfens: Dict[str, int] = {}
    csa_files = list(iter_csa_files(csa_dir))
    
    if not csa_files:
        print(f"ディレクトリ '{csa_dir}' 内にCSAファイルが見つかりませんでした。", file=sys.stderr)
        engine.quit()
        return

    print(f"{len(csa_files)}個のCSAファイルを処理します。")

    with open(output_file, "wb") as f_out:
        with tqdm(csa_files, unit="file") as pbar:
            for csa_path in pbar:
                pbar.set_description(f"Processing {csa_path.name}")
                try:
                    for kif in cshogi.CSA.Parser.parse_file(str(csa_path)):
                        # レーティングフィルタ
                        if kif.ratings:
                            rating_b, rating_w = kif.ratings
                            if not (min_rating <= rating_b <= max_rating and
                                    min_rating <= rating_w <= max_rating):
                                continue  # 棋譜をスキップ

                        board = cshogi.Board()
                        for i, move in enumerate(kif.moves):
                            board.push(move)
                            ply = i + 1
                            if ply >= min_ply:
                                norm_sfen = normalize_sfen(board.sfen())
                                if norm_sfen not in evaluated_sfens:
                                    try:
                                        s_type, s_val = engine.evaluate_sfen(board.sfen(), depth)
                                        eval_score = convert_score_to_int16(s_type, s_val)
                                        
                                        packed_sfen = cshogi.PackedSfen.pack(board)
                                        record = struct.pack("<16sh", packed_sfen, eval_score)
                                        f_out.write(record)
                                        
                                        evaluated_sfens[norm_sfen] = eval_score
                                    except TimeoutError as e:
                                        print(f"\n評価タイムアウト: {norm_sfen} ({e})", file=sys.stderr)
                                    except Exception as e:
                                        print(f"\n評価エラー: {norm_sfen} ({e})", file=sys.stderr)

                except Exception as e:
                    print(f"\nファイル処理エラー: {csa_path} ({e})", file=sys.stderr)

    engine.quit()
    print(f"\n処理完了。{len(evaluated_sfens)}個のユニークな局面を評価し、'{output_file}'に保存しました。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSA棋譜ファイルを解析し、USIエンジンで局面を評価してPackedSfenValue形式で保存するスクリプト。"
    )
    parser.add_argument(
        "-d", "--csa-dir", required=True,
        help="CSAファイルが格納されているディレクトリのパス。"
    )
    parser.add_argument(
        "-e", "--engine", required=True,
        help="USIエンジンの実行ファイルのパス (*.exe)。"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="出力ファイル名 (PackedSfenValue形式)。"
    )
    parser.add_argument(
        "--depth", type=int, default=10,
        help="USIエンジンの探索の深さ。(デフォルト: 10)"
    )
    parser.add_argument(
        "--min-ply", type=int, default=40,
        help="評価を開始する最小手数。(デフォルト: 40)"
    )
    parser.add_argument(
        "--min-rating", type=int, default=0,
        help="処理対象とする棋譜のプレーヤーの最低レーティング。(デフォルト: 0)"
    )
    parser.add_argument(
        "--max-rating", type=int, default=9999,
        help="処理対象とする棋譜のプレーヤーの最高レーティング。(デフォルト: 9999)"
    )

    args = parser.parse_args()

    try:
        process_csa_directory(
            csa_dir=args.csa_dir,
            engine_path=args.engine,
            output_file=args.output,
            depth=args.depth,
            min_ply=args.min_ply,
            min_rating=args.min_rating,
            max_rating=args.max_rating
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
