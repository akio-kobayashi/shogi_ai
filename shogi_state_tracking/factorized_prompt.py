"""125語の局面prompt・指手語彙（factorized_v3）の定義．

局面は所属別の駒・座標と持ち駒を列挙し，指手は移動先座標で終了する．
``<EOM>`` は用いず，成りは移動先の前へ置く．このモジュールは表記だけを
定義し，合法手規則をモデルへ与えない．
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from create_dataset import HAND_ORDER, all_usi_move_tokens
from new_prompt import count_token, square_token, square_tokens


FACTORIZED_SCHEMA_VERSION = 4
MOVE_ENCODING = "factorized_v3_no_eom"
TERMINAL_ENCODING = "eos_on_complete_decisive_game_v1"
DROP_TOKEN = "<DROP>"
PROMOTE_TOKEN = "<PROMOTE>"
DROP_PIECES = "PLNSGBR"

SPECIAL_TOKENS = (
    "<PAD>",
    "<BOS>",
    "<BOARD_BLACK>",
    "<BOARD_WHITE>",
    "<HAND_BLACK>",
    "<HAND_WHITE>",
    "<MOVES>",
    "<EOS>",
    DROP_TOKEN,
    PROMOTE_TOKEN,
    "<TURN_BLACK>",
    "<TURN_WHITE>",
)
PIECE_TOKENS = (
    "<P>", "<L>", "<N>", "<S>", "<G>", "<B>", "<R>", "<K>",
    "<PRO_P>", "<PRO_L>", "<PRO_N>", "<PRO_S>", "<HORSE>", "<DRAGON>",
)
BASIC_PIECE_TOKENS = tuple("<{}>".format(piece) for piece in DROP_PIECES)
PIECE_TYPE_TO_TOKEN = {
    1: "<P>", 2: "<L>", 3: "<N>", 4: "<S>", 5: "<B>", 6: "<R>",
    7: "<G>", 8: "<K>", 9: "<PRO_P>", 10: "<PRO_L>",
    11: "<PRO_N>", 12: "<PRO_S>", 13: "<HORSE>", 14: "<DRAGON>",
}
TOKEN_TO_PIECE_TYPE = {token: piece_type for piece_type, token in PIECE_TYPE_TO_TOKEN.items()}
HAND_NAME_TO_TOKEN = {name: "<{}>".format(name) for name in HAND_ORDER}


def piece_type_token(piece_type: int) -> str:
    try:
        return PIECE_TYPE_TO_TOKEN[int(piece_type)]
    except KeyError as exc:
        raise ValueError("unsupported piece type: {}".format(piece_type)) from exc


def annotation_piece_token(value: str) -> str:
    """旧artifactの色付き駒tokenも，新しい無色tokenへ正規化する．"""
    token = str(value)
    if token in PIECE_TOKENS:
        return token
    if token.startswith("<B_") or token.startswith("<W_"):
        name = token[3:-1]
        aliases = {"PP": "PRO_P", "PL": "PRO_L", "PN": "PRO_N", "PS": "PRO_S", "PB": "HORSE", "PR": "DRAGON"}
        normalized = "<{}>".format(aliases.get(name, name))
        if normalized in PIECE_TOKENS:
            return normalized
    raise ValueError("invalid piece annotation token: {}".format(value))


def factorize_usi(move_usi: str) -> List[str]:
    """USI指手を，新しい一意な指手文法へ分解する．"""
    move_usi = str(move_usi)
    if "*" in move_usi:
        if len(move_usi) != 4 or move_usi[0] not in DROP_PIECES or move_usi[1] != "*":
            raise ValueError("invalid USI drop: {}".format(move_usi))
        return [DROP_TOKEN, "<{}>".format(move_usi[0]), square_token(move_usi[2:4])]
    if len(move_usi) not in (4, 5) or (len(move_usi) == 5 and move_usi[4] != "+"):
        raise ValueError("invalid USI move: {}".format(move_usi))
    source = square_token(move_usi[:2])
    destination = square_token(move_usi[2:4])
    return [source, PROMOTE_TOKEN, destination] if len(move_usi) == 5 else [source, destination]


def unfactorize_usi(tokens: Sequence[str]) -> str:
    """1指手分のtoken列をUSIへ戻す．RAP tokenは含めない．"""
    values = list(tokens)
    if len(values) == 3 and values[0] == DROP_TOKEN and values[1] in BASIC_PIECE_TOKENS:
        return values[1][1:-1] + "*" + _square(values[2])
    if len(values) == 2:
        return _square(values[0]) + _square(values[1])
    if len(values) == 3 and values[1] == PROMOTE_TOKEN:
        return _square(values[0]) + _square(values[2]) + "+"
    raise ValueError("invalid factorized move: {}".format(values))


def _square(token: str) -> str:
    if not token.startswith("<SQ_") or not token.endswith(">"):
        raise ValueError("not a square token: {}".format(token))
    value = token[4:-1]
    square_token(value)
    return value


def _piece_token_from_cshogi_piece(piece: int) -> str:
    if int(piece) == 0:
        raise ValueError("empty square has no piece token")
    return piece_type_token(int(piece) % 16)


def _square_from_cshogi_index(index: int) -> str:
    if not 0 <= int(index) < 81:
        raise ValueError("board index must be in [0, 80]")
    return "{}{}".format(int(index) // 9 + 1, "abcdefghi"[int(index) % 9])


def encode_state_prompt(board, cshogi_module) -> List[str]:
    """盤面・持ち駒・手番を仕様どおりの正準token列へ符号化する．"""
    tokens: List[str] = [
        "<TURN_BLACK>" if board.turn == cshogi_module.BLACK else "<TURN_WHITE>"
    ]
    for color_index, header in ((cshogi_module.BLACK, "<BOARD_BLACK>"), (cshogi_module.WHITE, "<BOARD_WHITE>")):
        tokens.append(header)
        for index, piece in enumerate(board.pieces):
            piece = int(piece)
            if piece == 0 or (cshogi_module.WHITE if piece >= 16 else cshogi_module.BLACK) != color_index:
                continue
            tokens.extend((_piece_token_from_cshogi_piece(piece), square_token(_square_from_cshogi_index(index))))
    for color_index, header in ((cshogi_module.BLACK, "<HAND_BLACK>"), (cshogi_module.WHITE, "<HAND_WHITE>")):
        tokens.append(header)
        for hand_name, count in zip(HAND_ORDER, board.pieces_in_hand[color_index]):
            if int(count) > 0:
                tokens.extend((HAND_NAME_TO_TOKEN[hand_name], count_token(int(count))))
    validate_state_prompt_tokens(tokens)
    return tokens


def validate_state_prompt_tokens(tokens: Sequence[str]) -> None:
    """正準局面promptの構文，順序，重複を検査する．"""
    values = [str(value) for value in tokens]
    if not values or values[0] not in {"<TURN_BLACK>", "<TURN_WHITE>"}:
        raise ValueError("state prompt must start with a turn token")
    headers = ("<BOARD_BLACK>", "<BOARD_WHITE>", "<HAND_BLACK>", "<HAND_WHITE>")
    positions = []
    for header in headers:
        if values.count(header) != 1:
            raise ValueError("state prompt needs exactly one {}".format(header))
        positions.append(values.index(header))
    if positions != sorted(positions) or positions[0] != 1:
        raise ValueError("state prompt sections are not in canonical order")
    occupied = set()
    kings = []
    for section_index in range(2):
        begin = positions[section_index] + 1
        end = positions[section_index + 1]
        body = values[begin:end]
        if len(body) % 2:
            raise ValueError("board section must contain piece/square pairs")
        last_square = -1
        for piece, square in zip(body[0::2], body[1::2]):
            if piece not in PIECE_TOKENS or square not in set(square_tokens()):
                raise ValueError("invalid board pair: {} {}".format(piece, square))
            square_index = square_tokens().index(square)
            if square_index <= last_square or square in occupied:
                raise ValueError("board squares must be unique and canonically ordered")
            occupied.add(square)
            if piece == "<K>":
                kings.append(section_index)
            last_square = square_index
    if kings != [0, 1]:
        raise ValueError("each board section must contain exactly one king")
    hand_order = {token: index for index, token in enumerate(BASIC_PIECE_TOKENS)}
    for section_index in range(2, 4):
        begin = positions[section_index] + 1
        end = positions[section_index + 1] if section_index + 1 < 4 else len(values)
        body = values[begin:end]
        if len(body) % 2:
            raise ValueError("hand section must contain piece/count pairs")
        last_piece = -1
        for piece, count in zip(body[0::2], body[1::2]):
            if piece not in hand_order or count not in {count_token(value) for value in range(1, 19)}:
                raise ValueError("invalid hand pair: {} {}".format(piece, count))
            if hand_order[piece] <= last_piece:
                raise ValueError("hand pieces must be unique and canonically ordered")
            last_piece = hand_order[piece]


def factorized_vocabulary_tokens() -> List[str]:
    tokens = list(SPECIAL_TOKENS) + list(PIECE_TOKENS) + square_tokens() + [count_token(count) for count in range(1, 19)]
    if len(tokens) != 125 or len(tokens) != len(set(tokens)):
        raise AssertionError("factorized_v3 vocabulary must contain exactly 125 unique tokens")
    return tokens


def write_factorized_vocabulary(path: str | Path) -> Dict[str, object]:
    tokens = factorized_vocabulary_tokens()
    payload: Dict[str, object] = {
        "schema_version": FACTORIZED_SCHEMA_VERSION,
        "format": "shogi_canonical_state_prompt_factorized_moves",
        "token_to_id": {token: index for index, token in enumerate(tokens)},
        "move_encoding": MOVE_ENCODING,
        "terminal_encoding": TERMINAL_ENCODING,
        "move_grammar": "source destination | source PROMOTE destination | DROP piece destination",
        "probe_position": "h_pre_at_moves_or_previous_destination; h_post_at_current_destination",
        "syntactic_usi_actions": len(all_usi_move_tokens()),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
