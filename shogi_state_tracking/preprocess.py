"""開始局面を動的に選ぶための、PyTorch非依存の前処理。

実データを局面ごとに複製せず、元のinitial_sfenとUSI指し手列から必要な開始局面を
cshogiで再生する。学習workerからも、固定評価データ作成からも利用できる。
"""

import hashlib
from typing import Dict, List, Mapping, Optional

from create_dataset import encode_initial_state, import_cshogi


def candidate_start_plies(
    total_moves: int,
    candidate_count: int = 40,
    min_suffix_moves: int = 40,
) -> List[int]:
    """対局全体へほぼ等間隔に配置した開始候補を返す。

    start_plyは「開始局面までに既に指された手数」。0は元の開始局面を表す。
    各候補にはmin_suffix_moves以上の指し手が残る。候補可能数が40未満なら全て使う。
    """
    if total_moves < 0:
        raise ValueError("total_moves must be non-negative")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if min_suffix_moves < 1:
        raise ValueError("min_suffix_moves must be positive")
    max_start = total_moves - min_suffix_moves
    if max_start < 0:
        return []
    possible = max_start + 1
    if possible <= candidate_count:
        return list(range(possible))
    if candidate_count == 1:
        return [max_start // 2]

    # roundだけでは同じ整数を生む場合があるため、最後に重複を除く。
    candidates = [
        round(index * max_start / (candidate_count - 1))
        for index in range(candidate_count)
    ]
    return list(dict.fromkeys(candidates))


def choose_start_ply(
    game_id: str,
    candidates: List[int],
    seed: int,
    epoch: int,
    replica: int = 0,
) -> int:
    """Pythonのhash randomizationに依存しない開始候補選択。"""
    if not candidates:
        raise ValueError("start candidates are empty")
    key = "{}:{}:{}:{}".format(seed, epoch, replica, game_id).encode("utf-8")
    digest = hashlib.sha1(key).digest()
    index = int.from_bytes(digest[:8], byteorder="big") % len(candidates)
    return candidates[index]


def materialize_segment(
    record: Mapping[str, object],
    start_ply: int,
    max_suffix_moves: Optional[int] = None,
) -> Dict[str, object]:
    """JSONLの1対局を、指定開始局面からの1系列へ変換する。

    ``max_suffix_moves``を指定すると、開始局面以降をその手数で切る。
    Transformerの系列長を一定範囲へ収めるためのwindowingであり、局面の再生や
    指手の合法性検証は切る前に行う開始prefixに対して従来どおり実施する。
    """
    cshogi = import_cshogi()
    moves = list(record["move_tokens"])
    if not 0 <= start_ply <= len(moves):
        raise ValueError("start_ply is outside the game")
    if max_suffix_moves is not None and max_suffix_moves <= 0:
        raise ValueError("max_suffix_moves must be positive when specified")

    board = cshogi.Board(str(record["initial_sfen"]))
    for ply, move_usi in enumerate(moves[:start_ply], 1):
        try:
            move = board.move_from_usi(str(move_usi))
            if not board.is_legal(move):
                raise ValueError("illegal move")
            board.push(move)
        except Exception as exc:
            raise ValueError(
                "game {} cannot be replayed at ply {} ({})".format(
                    record.get("game_id", "?"), ply, move_usi
                )
            ) from exc

    return {
        "game_id": record["game_id"],
        "engine_scope": record["engine_scope"],
        "start_ply": start_ply,
        "start_sfen": board.sfen(),
        "initial_state_tokens": encode_initial_state(board, cshogi),
        "move_tokens": moves[
            start_ply:
            None
            if max_suffix_moves is None
            else start_ply + max_suffix_moves
        ],
    }
