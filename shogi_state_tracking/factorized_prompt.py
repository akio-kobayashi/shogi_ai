"""座標分解した指手語彙（factorized_v2）の定義。

開始局面promptは :mod:`new_prompt` と共有し，指手だけを複数tokenへ分解する。
ゲーム規則や合法手は語彙化・学習損失へ組み込まない。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from create_dataset import PIECE_NAMES, all_usi_move_tokens
from new_prompt import SPECIAL_TOKENS, count_token, piece_token, square_token, square_tokens


FACTORIZED_SCHEMA_VERSION = 2
EOM_TOKEN = "<EOM>"
PROMOTE_TOKEN = "<PROMOTE>"
DROP_PIECES = "PLNSGBR"
DROP_TOKENS = tuple("<DROP_{}>".format(piece) for piece in DROP_PIECES)


def factorize_usi(move_usi: str) -> List[str]:
    """USI指手を ``source/drop, destination, [promotion], EOM`` に分解する。"""
    move_usi = str(move_usi)
    if "*" in move_usi:
        if len(move_usi) != 4 or move_usi[0] not in DROP_PIECES or move_usi[1] != "*":
            raise ValueError("invalid USI drop: {}".format(move_usi))
        return ["<DROP_{}>".format(move_usi[0]), square_token(move_usi[2:4]), EOM_TOKEN]
    if len(move_usi) not in (4, 5) or (len(move_usi) == 5 and move_usi[4] != "+"):
        raise ValueError("invalid USI move: {}".format(move_usi))
    result = [square_token(move_usi[:2]), square_token(move_usi[2:4])]
    if len(move_usi) == 5:
        result.append(PROMOTE_TOKEN)
    result.append(EOM_TOKEN)
    return result


def unfactorize_usi(tokens: Sequence[str]) -> str:
    """1指手分のtoken列をUSIへ戻す。文法違反は拒否する。"""
    values = list(tokens)
    if not values or values[-1] != EOM_TOKEN:
        raise ValueError("factorized move must end with <EOM>")
    body = values[:-1]
    if len(body) == 2 and body[0] in DROP_TOKENS:
        return body[0][len("<DROP_") : -1] + "*" + _square(body[1])
    if len(body) not in (2, 3) or (len(body) == 3 and body[2] != PROMOTE_TOKEN):
        raise ValueError("invalid factorized move: {}".format(values))
    return _square(body[0]) + _square(body[1]) + ("+" if len(body) == 3 else "")


def _square(token: str) -> str:
    if not token.startswith("<SQ_") or not token.endswith(">"):
        raise ValueError("not a square token: {}".format(token))
    value = token[4:-1]
    square_token(value)
    return value


def factorized_vocabulary_tokens() -> List[str]:
    """状態prompt，RAP駒種，座標分解指手を含む固定語彙を返す。"""
    pieces = [piece_token(color, piece_type) for color in ("B", "W") for piece_type in range(1, 15)]
    tokens = (
        list(SPECIAL_TOKENS)
        + pieces
        + square_tokens()
        + [count_token(count) for count in range(1, 19)]
        + ["<TURN_BLACK>", "<TURN_WHITE>"]
        + list(DROP_TOKENS)
        + [PROMOTE_TOKEN, EOM_TOKEN]
    )
    if len(tokens) != len(set(tokens)):
        raise AssertionError("factorized vocabulary contains duplicates")
    return tokens


def write_factorized_vocabulary(path: str | Path) -> Dict[str, object]:
    tokens = factorized_vocabulary_tokens()
    payload: Dict[str, object] = {
        "schema_version": FACTORIZED_SCHEMA_VERSION,
        "format": "shogi_piece_coordinate_prompt_factorized_moves",
        "token_to_id": {token: index for index, token in enumerate(tokens)},
        "move_encoding": "factorized_v2",
        "move_grammar": "source_or_drop destination [PROMOTE] EOM",
        "syntactic_usi_actions": len(all_usi_move_tokens()),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload

