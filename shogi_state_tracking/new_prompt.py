"""駒・座標状態promptと部分的行動教師の共通定義。

このモジュールは新実験用であり，既存の固定96トークン形式を変更しない。
CSAを読むデータセット構築環境ではcshogiを用いるが，生成済みartifactを読む
学習・評価環境はここに含まれる文字列トークンだけを扱う。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from create_dataset import HAND_ORDER, PIECE_NAMES, all_usi_move_tokens


NEW_PROMPT_SCHEMA_VERSION = 1

SPECIAL_TOKENS = (
    "<PAD>",
    "<BOS>",
    "<STATE>",
    "<BOARD>",
    "</BOARD>",
    "<HANDS>",
    "</HANDS>",
    "<MOVES>",
    "<EOS>",
)

RANKS = "abcdefghi"
FILES = tuple(range(1, 10))
USI_MOVE_SET = frozenset(all_usi_move_tokens())
SQUARE_TOKENS = tuple(
    "<SQ_{}{}>".format(file_index, rank)
    for file_index in FILES
    for rank in RANKS
)
SQUARE_TOKEN_SET = frozenset(SQUARE_TOKENS)
COUNT_TOKEN_SET = frozenset("<COUNT_{}>".format(number) for number in range(1, 19))
BOARD_PIECE_TOKEN_SET = frozenset(
    "<{}_{}>".format(color, PIECE_NAMES[piece_type])
    for color in ("B", "W")
    for piece_type in PIECE_NAMES
)
HAND_PIECE_TOKEN_SET = frozenset(
    "<{}_{}>".format(color, piece_name)
    for color in ("B", "W")
    for piece_name in HAND_ORDER
)
HAND_PIECE_TYPE = {name: piece_type for piece_type, name in PIECE_NAMES.items()}


def piece_token(color: str, piece_type: int) -> str:
    """先後と駒種から開始局面・教師共通の駒種トークンを返す。"""
    if color not in {"B", "W"}:
        raise ValueError("color must be B or W")
    if piece_type not in PIECE_NAMES:
        raise ValueError("unsupported piece type: {}".format(piece_type))
    return "<{}_{}>".format(color, PIECE_NAMES[piece_type])


def square_token(square: str) -> str:
    """USIのマス名を新prompt用座標トークンへ変換する。"""
    if len(square) != 2 or square[0] not in "123456789" or square[1] not in RANKS:
        raise ValueError("invalid USI square: {}".format(square))
    return "<SQ_{}>".format(square)


def square_tokens() -> List[str]:
    return list(SQUARE_TOKENS)


def count_token(count: int) -> str:
    if not 1 <= int(count) <= 18:
        raise ValueError("hand count must be in [1, 18]")
    return "<COUNT_{}>".format(int(count))


def move_token(move_usi: str) -> str:
    if move_usi not in USI_MOVE_SET:
        raise ValueError("unsupported USI move: {}".format(move_usi))
    return "<MOVE_{}>".format(move_usi)


def move_token_to_usi(token: str) -> str:
    if not token.startswith("<MOVE_") or not token.endswith(">"):
        raise ValueError("not an atomic move token: {}".format(token))
    move_usi = token[len("<MOVE_") : -1]
    # move_tokenで文法的USI語彙に含まれることを検査する。
    move_token(move_usi)
    return move_usi


def atomic_move_tokens() -> List[str]:
    return ["<MOVE_{}>".format(move) for move in all_usi_move_tokens()]


def new_prompt_vocabulary_tokens() -> List[str]:
    """新実験の固定語彙を決定論的順序で返す。"""
    piece_tokens = [
        piece_token(color, piece_type)
        for color in ("B", "W")
        for piece_type in range(1, 15)
    ]
    tokens = (
        list(SPECIAL_TOKENS)
        + piece_tokens
        + square_tokens()
        + [count_token(count) for count in range(1, 19)]
        + ["<TURN_BLACK>", "<TURN_WHITE>"]
        + atomic_move_tokens()
    )
    if len(tokens) != len(set(tokens)):
        raise AssertionError("new prompt vocabulary contains duplicate tokens")
    return tokens


def write_new_prompt_vocabulary(path: str | Path) -> Dict[str, object]:
    """新schemaの語彙と最小限のメタデータを書き出す。"""
    tokens = new_prompt_vocabulary_tokens()
    payload: Dict[str, object] = {
        "schema_version": NEW_PROMPT_SCHEMA_VERSION,
        "format": "shogi_piece_coordinate_prompt",
        "token_to_id": {token: index for index, token in enumerate(tokens)},
        "special_tokens": list(SPECIAL_TOKENS),
        "piece_tokens": [
            piece_token(color, piece_type)
            for color in ("B", "W")
            for piece_type in range(1, 15)
        ],
        "square_tokens": square_tokens(),
        "count_tokens": [count_token(count) for count in range(1, 19)],
        "move_encoding": "one atomic USI move per token",
        "move_tokens": atomic_move_tokens(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def _piece_token_from_cshogi_piece(piece: int) -> str:
    if piece == 0:
        raise ValueError("empty square has no piece token")
    piece_type = int(piece) % 16
    color = "W" if int(piece) >= 16 else "B"
    return piece_token(color, piece_type)


def _square_from_cshogi_index(index: int) -> str:
    """cshogiの ``1a,1b,...,9i`` 配列添字をUSI座標へ変換する。"""
    if not 0 <= index < 81:
        raise ValueError("board index must be in [0, 80]")
    return "{}{}".format(index // 9 + 1, RANKS[index % 9])


def _cshogi_index_from_square(square: str) -> int:
    if len(square) != 2 or square[0] not in "123456789" or square[1] not in RANKS:
        raise ValueError("invalid USI square: {}".format(square))
    return (int(square[0]) - 1) * 9 + RANKS.index(square[1])


def encode_state_prompt(board, cshogi_module) -> List[str]:
    """局面を駒・座標・持ち駒・手番からなる可変長promptへ変換する。"""
    tokens: List[str] = ["<STATE>", "<BOARD>"]
    for index, piece in enumerate(board.pieces):
        if int(piece) == 0:
            continue
        tokens.extend(
            (
                _piece_token_from_cshogi_piece(int(piece)),
                square_token(_square_from_cshogi_index(index)),
            )
        )
    tokens.append("</BOARD>")
    tokens.append("<HANDS>")
    for color_index, color in (
        (cshogi_module.BLACK, "B"),
        (cshogi_module.WHITE, "W"),
    ):
        for hand_piece, count in zip(HAND_ORDER, board.pieces_in_hand[color_index]):
            if int(count) <= 0:
                continue
            # HAND_ORDERとPIECE_NAMESの基本駒表記は同じである。
            piece_type = HAND_PIECE_TYPE[hand_piece]
            tokens.extend((piece_token(color, piece_type), count_token(int(count))))
    tokens.append("</HANDS>")
    tokens.append(
        "<TURN_BLACK>" if board.turn == cshogi_module.BLACK else "<TURN_WHITE>"
    )
    return tokens


def is_drop_usi(move_usi: str) -> bool:
    return "*" in move_usi


def source_square_from_usi(move_usi: str) -> str:
    if is_drop_usi(move_usi):
        raise ValueError("drops do not have a board source square")
    if len(move_usi) not in {4, 5}:
        raise ValueError("invalid normal USI move: {}".format(move_usi))
    return move_usi[:2]


def move_annotation(board, move_usi: str) -> Dict[str, object]:
    """指す前の局面とUSI指手から部分的行動教師を作る。"""
    if is_drop_usi(move_usi):
        return {"eligible": False}
    source = source_square_from_usi(move_usi)
    piece = int(board.pieces[_cshogi_index_from_square(source)])
    if piece == 0:
        raise ValueError("source square {} is empty before {}".format(source, move_usi))
    return {
        "eligible": True,
        "piece": _piece_token_from_cshogi_piece(piece),
        "source": square_token(source),
    }


def annotate_game_moves(initial_sfen: str, move_usis: Sequence[str], cshogi_module) -> List[Dict[str, object]]:
    """全指手に移動前駒種・開始位置注釈を付け，合法性も同時に検査する。"""
    board = cshogi_module.Board(str(initial_sfen))
    annotations: List[Dict[str, object]] = []
    for ply, move_usi in enumerate(move_usis, 1):
        move = board.move_from_usi(str(move_usi))
        if not board.is_legal(move):
            raise ValueError("illegal move at ply {}: {}".format(ply, move_usi))
        annotations.append(move_annotation(board, str(move_usi)))
        board.push(move)
    return annotations


def validate_state_prompt_tokens(tokens: Sequence[str]) -> None:
    """cshogi非依存で状態promptの文法と局面の基本不変条件を検査する。"""
    tokens = list(tokens)
    try:
        board_start = tokens.index("<BOARD>")
        board_end = tokens.index("</BOARD>")
        hands_start = tokens.index("<HANDS>")
        hands_end = tokens.index("</HANDS>")
    except ValueError as exc:
        raise ValueError("state prompt has missing section boundary") from exc
    if tokens[:board_start] != ["<STATE>"]:
        raise ValueError("state prompt must start with <STATE> <BOARD>")
    if not (board_start < board_end < hands_start < hands_end == len(tokens) - 2):
        raise ValueError("state prompt section order is invalid")
    if tokens[-1] not in {"<TURN_BLACK>", "<TURN_WHITE>"}:
        raise ValueError("state prompt must end with a turn token")

    board_body = tokens[board_start + 1 : board_end]
    if len(board_body) % 2:
        raise ValueError("board section must contain piece/square pairs")
    occupied_squares = set()
    king_tokens = []
    for piece, square in zip(board_body[::2], board_body[1::2]):
        if piece not in BOARD_PIECE_TOKEN_SET:
            raise ValueError("invalid board piece token: {}".format(piece))
        if square not in SQUARE_TOKEN_SET:
            raise ValueError("invalid board square token: {}".format(square))
        if square in occupied_squares:
            raise ValueError("duplicate board square: {}".format(square))
        occupied_squares.add(square)
        if piece in {"<B_K>", "<W_K>"}:
            king_tokens.append(piece)
    if sorted(king_tokens) != ["<B_K>", "<W_K>"]:
        raise ValueError("state prompt must contain one king for each side")

    hands_body = tokens[hands_start + 1 : hands_end]
    if len(hands_body) % 2:
        raise ValueError("hands section must contain piece/count pairs")
    hand_pieces = set()
    for piece, count in zip(hands_body[::2], hands_body[1::2]):
        if piece not in HAND_PIECE_TOKEN_SET:
            raise ValueError("invalid hand piece token: {}".format(piece))
        if count not in COUNT_TOKEN_SET:
            raise ValueError("invalid hand count token: {}".format(count))
        if piece in hand_pieces:
            raise ValueError("duplicate hand piece: {}".format(piece))
        hand_pieces.add(piece)


def validate_move_annotations(
    move_tokens: Sequence[str], annotations: Sequence[Mapping[str, object]]
) -> None:
    """cshogi非依存で指手と保存済み教師注釈の構文的整合性を検査する。"""
    if len(move_tokens) != len(annotations):
        raise ValueError("move_tokens and move_annotations length mismatch")
    for move, annotation in zip(move_tokens, annotations):
        move_usi = move_token_to_usi(str(move)) if str(move).startswith("<MOVE_") else str(move)
        eligible = bool(annotation.get("eligible", False))
        if is_drop_usi(move_usi):
            if eligible:
                raise ValueError("drop move must not have a partial action annotation")
            continue
        if not eligible:
            raise ValueError("normal move must have a partial action annotation")
        source = str(annotation.get("source", ""))
        piece = str(annotation.get("piece", ""))
        if source != square_token(source_square_from_usi(move_usi)):
            raise ValueError("annotation source does not match move source")
        if not (piece.startswith("<B_") or piece.startswith("<W_")):
            raise ValueError("annotation has invalid piece token")
