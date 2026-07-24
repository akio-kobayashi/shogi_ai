"""指し手境界表現から将棋局面を線形復号するための部品。"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch import nn

from create_dataset import PIECE_NAMES, import_cshogi


BOARD_CLASS_COUNT = 29
HAND_MAX_COUNTS = (18, 4, 4, 4, 4, 2, 2) * 2
HAND_NAMES = (
    "black_pawn",
    "black_lance",
    "black_knight",
    "black_silver",
    "black_gold",
    "black_bishop",
    "black_rook",
    "white_pawn",
    "white_lance",
    "white_knight",
    "white_silver",
    "white_gold",
    "white_bishop",
    "white_rook",
)
BOARD_NAMES = (
    ("empty",)
    + tuple("black_{}".format(PIECE_NAMES[index]) for index in range(1, 15))
    + tuple("white_{}".format(PIECE_NAMES[index]) for index in range(1, 15))
)


@dataclass
class ProbeTargets:
    board: torch.Tensor
    hands: torch.Tensor
    turn: torch.Tensor


@dataclass
class ProbeLogits:
    board: torch.Tensor
    hands: Tuple[torch.Tensor, ...]
    turn: torch.Tensor


def board_state_targets(board, cshogi_module) -> Tuple[List[int], List[int], int]:
    """cshogi局面を盤面29クラス、持ち駒14枚数、手番へ変換する。"""
    board_targets: List[int] = []
    for piece in board.pieces:
        if piece == 0:
            board_targets.append(0)
            continue
        piece_type = int(piece) % 16
        if not 1 <= piece_type <= 14:
            raise ValueError("unexpected cshogi piece value: {}".format(piece))
        color_offset = 14 if int(piece) >= 16 else 0
        board_targets.append(color_offset + piece_type)

    hand_targets: List[int] = []
    for color in (cshogi_module.BLACK, cshogi_module.WHITE):
        hand_targets.extend(int(count) for count in board.pieces_in_hand[color])

    if len(board_targets) != 81 or len(hand_targets) != 14:
        raise AssertionError("invalid shogi state target dimensions")
    turn = 0 if board.turn == cshogi_module.BLACK else 1
    return board_targets, hand_targets, turn


def replay_probe_targets(start_sfen: str, move_tokens: Sequence[str]) -> ProbeTargets:
    """開始局面と正解指し手列からstate_0..state_Tを作る。"""
    cshogi = import_cshogi()
    board = cshogi.Board(str(start_sfen))
    boards: List[List[int]] = []
    hands: List[List[int]] = []
    turns: List[int] = []

    def append_current() -> None:
        board_target, hand_target, turn_target = board_state_targets(board, cshogi)
        boards.append(board_target)
        hands.append(hand_target)
        turns.append(turn_target)

    append_current()
    for ply, move_usi in enumerate(move_tokens, 1):
        move = board.move_from_usi(str(move_usi))
        if not board.is_legal(move):
            raise ValueError(
                "illegal probe replay move at ply {}: {}".format(ply, move_usi)
            )
        board.push(move)
        append_current()

    return ProbeTargets(
        board=torch.tensor(boards, dtype=torch.long),
        hands=torch.tensor(hands, dtype=torch.long),
        turn=torch.tensor(turns, dtype=torch.long),
    )


class LinearStateProbe(nn.Module):
    """単一のd_modelベクトルから全局面を読む線形プローブ。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.board_head = nn.Linear(d_model, 81 * BOARD_CLASS_COUNT)
        self.hand_heads = nn.ModuleList(
            nn.Linear(d_model, maximum + 1) for maximum in HAND_MAX_COUNTS
        )
        self.turn_head = nn.Linear(d_model, 2)

    def forward(self, features: torch.Tensor) -> ProbeLogits:
        if features.ndim != 2 or features.shape[-1] != self.d_model:
            raise ValueError("features must have shape [samples, d_model]")
        board = self.board_head(features).view(
            features.shape[0], 81, BOARD_CLASS_COUNT
        )
        hands = tuple(head(features) for head in self.hand_heads)
        turn = self.turn_head(features)
        return ProbeLogits(board=board, hands=hands, turn=turn)


def linear_probe_loss(
    logits: ProbeLogits,
    targets: ProbeTargets,
    board_weight: float = 1.0,
    hand_weight: float = 1.0,
    turn_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    board_loss = nn.functional.cross_entropy(
        logits.board.reshape(-1, BOARD_CLASS_COUNT),
        targets.board.reshape(-1),
    )
    hand_losses = [
        nn.functional.cross_entropy(slot_logits, targets.hands[:, slot])
        for slot, slot_logits in enumerate(logits.hands)
    ]
    hand_loss = torch.stack(hand_losses).mean()
    turn_loss = nn.functional.cross_entropy(logits.turn, targets.turn)
    total = (
        float(board_weight) * board_loss
        + float(hand_weight) * hand_loss
        + float(turn_weight) * turn_loss
    )
    return total, {
        "board_loss": float(board_loss.detach()),
        "hand_loss": float(hand_loss.detach()),
        "turn_loss": float(turn_loss.detach()),
        "total_loss": float(total.detach()),
    }


def predictions_from_logits(
    logits: ProbeLogits,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    board = logits.board.argmax(dim=-1)
    hands = torch.stack([slot.argmax(dim=-1) for slot in logits.hands], dim=1)
    turn = logits.turn.argmax(dim=-1)
    return board, hands, turn


def _macro_f1(target: torch.Tensor, prediction: torch.Tensor, classes: int) -> float:
    scores = []
    for class_index in range(classes):
        truth = target == class_index
        if not bool(truth.any()):
            continue
        predicted = prediction == class_index
        true_positive = int((truth & predicted).sum())
        false_positive = int((~truth & predicted).sum())
        false_negative = int((truth & ~predicted).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(sum(scores) / len(scores)) if scores else 0.0


def state_metrics(
    targets: ProbeTargets,
    board_prediction: torch.Tensor,
    hand_prediction: torch.Tensor,
    turn_prediction: torch.Tensor,
) -> Dict[str, object]:
    """空マス・持ち駒なしへの偏りを含めて診断できる評価値を返す。"""
    if board_prediction.shape != targets.board.shape:
        raise ValueError("board prediction shape mismatch")
    if hand_prediction.shape != targets.hands.shape:
        raise ValueError("hand prediction shape mismatch")
    if turn_prediction.shape != targets.turn.shape:
        raise ValueError("turn prediction shape mismatch")

    board_correct = board_prediction == targets.board
    hand_correct = hand_prediction == targets.hands
    occupied = targets.board != 0
    nonzero_hand = targets.hands != 0

    occupied_total = int(occupied.sum())
    nonzero_hand_total = int(nonzero_hand.sum())
    board_exact = board_correct.all(dim=1)
    hand_exact = hand_correct.all(dim=1)
    turn_correct = turn_prediction == targets.turn
    full_exact = board_exact & hand_exact & turn_correct

    per_hand = {
        name: float(hand_correct[:, index].float().mean())
        for index, name in enumerate(HAND_NAMES)
    }
    per_board_class = {}
    for class_index, name in enumerate(BOARD_NAMES):
        class_mask = targets.board == class_index
        if bool(class_mask.any()):
            per_board_class[name] = float(
                board_correct[class_mask].float().mean()
            )
    return {
        "samples": int(targets.board.shape[0]),
        "board_exact_match": float(board_exact.float().mean()),
        "board_square_accuracy": float(board_correct.float().mean()),
        "board_occupied_accuracy": (
            float(board_correct[occupied].float().mean())
            if occupied_total
            else None
        ),
        "board_macro_f1": _macro_f1(
            targets.board.reshape(-1),
            board_prediction.reshape(-1),
            BOARD_CLASS_COUNT,
        ),
        "hand_exact_match": float(hand_exact.float().mean()),
        "hand_slot_accuracy": float(hand_correct.float().mean()),
        "hand_nonzero_accuracy": (
            float(hand_correct[nonzero_hand].float().mean())
            if nonzero_hand_total
            else None
        ),
        "hand_mae": float(
            (hand_prediction - targets.hands).abs().float().mean()
        ),
        "turn_accuracy": float(turn_correct.float().mean()),
        "full_state_exact_match": float(full_exact.float().mean()),
        "board_accuracy_by_class": per_board_class,
        "hand_accuracy_by_slot": per_hand,
    }


def subset_targets(targets: ProbeTargets, mask: torch.Tensor) -> ProbeTargets:
    return ProbeTargets(
        board=targets.board[mask],
        hands=targets.hands[mask],
        turn=targets.turn[mask],
    )


def majority_predictions(
    train_targets: ProbeTargets, evaluation_samples: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """学習集合の位置別・slot別最頻クラスによる対照条件。"""
    board_mode = train_targets.board.mode(dim=0).values
    hand_mode = train_targets.hands.mode(dim=0).values
    turn_mode = train_targets.turn.mode().values
    return (
        board_mode.unsqueeze(0).expand(evaluation_samples, -1).clone(),
        hand_mode.unsqueeze(0).expand(evaluation_samples, -1).clone(),
        turn_mode.expand(evaluation_samples).clone(),
    )


def distance_bin(distance: int) -> str:
    if distance == 0:
        return "0"
    if distance <= 10:
        return "1-10"
    if distance <= 20:
        return "11-20"
    if distance <= 40:
        return "21-40"
    return "41+"


def stratified_metrics(
    targets: ProbeTargets,
    board_prediction: torch.Tensor,
    hand_prediction: torch.Tensor,
    turn_prediction: torch.Tensor,
    distances: torch.Tensor,
    scopes: Sequence[str],
) -> Dict[str, Mapping[str, object]]:
    if len(scopes) != targets.board.shape[0]:
        raise ValueError("scope count does not match targets")
    result: Dict[str, Mapping[str, object]] = {}

    labels = [distance_bin(int(value)) for value in distances]
    tracking_mask = distances > 0
    if bool(tracking_mask.any()):
        result["state_tracking_1_plus"] = state_metrics(
            subset_targets(targets, tracking_mask),
            board_prediction[tracking_mask],
            hand_prediction[tracking_mask],
            turn_prediction[tracking_mask],
        )
    for bin_name in ("0", "1-10", "11-20", "21-40", "41+"):
        mask = torch.tensor(
            [label == bin_name for label in labels], dtype=torch.bool
        )
        if bool(mask.any()):
            result["distance_{}".format(bin_name)] = state_metrics(
                subset_targets(targets, mask),
                board_prediction[mask],
                hand_prediction[mask],
                turn_prediction[mask],
            )

    for scope in ("open", "mixed", "closed"):
        mask = torch.tensor([value == scope for value in scopes], dtype=torch.bool)
        if bool(mask.any()):
            result["scope_{}".format(scope)] = state_metrics(
                subset_targets(targets, mask),
                board_prediction[mask],
                hand_prediction[mask],
                turn_prediction[mask],
            )
    return result
