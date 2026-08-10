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
    # 旧artifactには保存されていないためoptionalにし，既存評価との互換性を保つ。
    in_check: torch.Tensor | None = None


@dataclass
class ProbeLogits:
    board: torch.Tensor
    hands: Tuple[torch.Tensor, ...]
    turn: torch.Tensor
    in_check: torch.Tensor


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
        self.in_check_head = nn.Linear(d_model, 2)

    def forward(self, features: torch.Tensor) -> ProbeLogits:
        if features.ndim != 2 or features.shape[-1] != self.d_model:
            raise ValueError("features must have shape [samples, d_model]")
        board = self.board_head(features).view(
            features.shape[0], 81, BOARD_CLASS_COUNT
        )
        hands = tuple(head(features) for head in self.hand_heads)
        turn = self.turn_head(features)
        return ProbeLogits(board=board, hands=hands, turn=turn, in_check=self.in_check_head(features))


def linear_probe_loss(
    logits: ProbeLogits,
    targets: ProbeTargets,
    board_weight: float = 1.0,
    hand_weight: float = 1.0,
    turn_weight: float = 1.0,
    check_weight: float = 1.0,
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
    check_loss = None
    if targets.in_check is not None:
        check_loss = nn.functional.cross_entropy(logits.in_check, targets.in_check)
    total = (
        float(board_weight) * board_loss
        + float(hand_weight) * hand_loss
        + float(turn_weight) * turn_loss
    )
    if check_loss is not None:
        total = total + float(check_weight) * check_loss
    return total, {
        "board_loss": float(board_loss.detach()),
        "hand_loss": float(hand_loss.detach()),
        "turn_loss": float(turn_loss.detach()),
        "check_loss": None if check_loss is None else float(check_loss.detach()),
        "total_loss": float(total.detach()),
    }


def predictions_from_logits(
    logits: ProbeLogits,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    board = logits.board.argmax(dim=-1)
    hands = torch.stack([slot.argmax(dim=-1) for slot in logits.hands], dim=1)
    turn = logits.turn.argmax(dim=-1)
    return board, hands, turn


def classification_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    class_indices: Sequence[int],
) -> Dict[str, float]:
    """正解集合に出現するクラスについてmacro指標を返す。

    balanced accuracyは各クラスrecallの単純平均である。未出現クラスを0点として
    混ぜるとsplitごとのクラス欠落に左右されるため，support>0のクラスだけを使う。
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for class_index in class_indices:
        truth = target == class_index
        if not bool(truth.any()):
            continue
        predicted = prediction == class_index
        true_positive = int((truth & predicted).sum())
        false_positive = int((~truth & predicted).sum())
        false_negative = int((truth & ~predicted).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    denominator = max(len(recall_scores), 1)
    return {
        "balanced_accuracy": float(sum(recall_scores) / denominator),
        "macro_precision": float(sum(precision_scores) / denominator),
        "macro_recall": float(sum(recall_scores) / denominator),
        "macro_f1": float(sum(f1_scores) / denominator),
        "supported_classes": len(recall_scores),
    }


def binary_classification_metrics(
    target: torch.Tensor, prediction: torch.Tensor
) -> Dict[str, float]:
    """1をpositiveとする二値precision／recall／F1とbalanced accuracy。"""
    target = target.reshape(-1).to(dtype=torch.long)
    prediction = prediction.reshape(-1).to(dtype=torch.long)
    positive = target == 1
    predicted_positive = prediction == 1
    tp = int((positive & predicted_positive).sum())
    fp = int((~positive & predicted_positive).sum())
    fn = int((positive & ~predicted_positive).sum())
    tn = int((~positive & ~predicted_positive).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    supported = classification_metrics(target, prediction, (0, 1))
    return {
        "accuracy": float((target == prediction).float().mean()),
        "balanced_accuracy": supported["balanced_accuracy"],
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


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
    predicted_occupied = board_prediction != 0
    occupancy_correct = predicted_occupied == occupied
    board_multiclass = classification_metrics(
        targets.board.reshape(-1), board_prediction.reshape(-1), range(BOARD_CLASS_COUNT)
    )
    occupied_piece_multiclass = classification_metrics(
        targets.board[occupied], board_prediction[occupied], range(1, BOARD_CLASS_COUNT)
    ) if occupied_total else None
    occupancy_binary = binary_classification_metrics(
        occupied.to(dtype=torch.long), predicted_occupied.to(dtype=torch.long)
    )
    hand_count_multiclass = classification_metrics(
        targets.hands.reshape(-1), hand_prediction.reshape(-1), range(max(HAND_MAX_COUNTS) + 1)
    )
    hand_nonzero_binary = binary_classification_metrics(
        nonzero_hand.to(dtype=torch.long), (hand_prediction != 0).to(dtype=torch.long)
    )
    turn_binary = binary_classification_metrics(targets.turn, turn_prediction)

    per_hand = {
        name: float(hand_correct[:, index].float().mean())
        for index, name in enumerate(HAND_NAMES)
    }
    # 駒種別の値は「正解がその駒であるマス」に条件付けた完全一致率である。
    # 大駒のように出現数が少ないクラスも比較できるよう，分母も併記する。
    per_board_class = {}
    per_board_class_samples = {}
    for class_index, name in enumerate(BOARD_NAMES):
        class_mask = targets.board == class_index
        if bool(class_mask.any()):
            per_board_class_samples[name] = int(class_mask.sum())
            per_board_class[name] = float(
                board_correct[class_mask].float().mean()
            )
    return {
        "samples": int(targets.board.shape[0]),
        "board_exact_match": float(board_exact.float().mean()),
        "board_square_accuracy": float(board_correct.float().mean()),
        # 全81マスについて，空／非空だけを判定する二値指標。
        "board_occupancy_accuracy": float(occupancy_correct.float().mean()),
        "board_occupancy_balanced_accuracy": occupancy_binary["balanced_accuracy"],
        "board_occupancy_precision": occupancy_binary["precision"],
        "board_occupancy_recall": occupancy_binary["recall"],
        "board_occupancy_f1": occupancy_binary["f1"],
        # 正解が駒のあるマスに限定し，駒種・所属まで一致した割合。
        # 旧名 board_occupied_accuracy は互換性のため残す。
        "board_piece_accuracy_on_occupied": (
            float(board_correct[occupied].float().mean())
            if occupied_total
            else None
        ),
        "board_occupied_accuracy": (
            float(board_correct[occupied].float().mean())
            if occupied_total
            else None
        ),
        "board_balanced_accuracy": board_multiclass["balanced_accuracy"],
        "board_macro_precision": board_multiclass["macro_precision"],
        "board_macro_recall": board_multiclass["macro_recall"],
        "board_macro_f1": board_multiclass["macro_f1"],
        "board_piece_on_occupied_balanced_accuracy": None if occupied_piece_multiclass is None else occupied_piece_multiclass["balanced_accuracy"],
        "board_piece_on_occupied_macro_precision": None if occupied_piece_multiclass is None else occupied_piece_multiclass["macro_precision"],
        "board_piece_on_occupied_macro_recall": None if occupied_piece_multiclass is None else occupied_piece_multiclass["macro_recall"],
        "board_piece_on_occupied_macro_f1": None if occupied_piece_multiclass is None else occupied_piece_multiclass["macro_f1"],
        "hand_exact_match": float(hand_exact.float().mean()),
        "hand_slot_accuracy": float(hand_correct.float().mean()),
        "hand_nonzero_accuracy": (
            float(hand_correct[nonzero_hand].float().mean())
            if nonzero_hand_total
            else None
        ),
        "hand_count_balanced_accuracy": hand_count_multiclass["balanced_accuracy"],
        "hand_count_macro_precision": hand_count_multiclass["macro_precision"],
        "hand_count_macro_recall": hand_count_multiclass["macro_recall"],
        "hand_count_macro_f1": hand_count_multiclass["macro_f1"],
        "hand_nonzero_balanced_accuracy": hand_nonzero_binary["balanced_accuracy"],
        "hand_nonzero_precision": hand_nonzero_binary["precision"],
        "hand_nonzero_recall": hand_nonzero_binary["recall"],
        "hand_nonzero_f1": hand_nonzero_binary["f1"],
        "hand_mae": float(
            (hand_prediction - targets.hands).abs().float().mean()
        ),
        "turn_accuracy": float(turn_correct.float().mean()),
        "turn_balanced_accuracy": turn_binary["balanced_accuracy"],
        "turn_precision": turn_binary["precision"],
        "turn_recall": turn_binary["recall"],
        "turn_f1": turn_binary["f1"],
        "full_state_exact_match": float(full_exact.float().mean()),
        "board_accuracy_by_class": per_board_class,
        "board_samples_by_class": per_board_class_samples,
        "hand_accuracy_by_slot": per_hand,
    }


def subset_targets(targets: ProbeTargets, mask: torch.Tensor) -> ProbeTargets:
    return ProbeTargets(
        board=targets.board[mask],
        hands=targets.hands[mask],
        turn=targets.turn[mask],
        in_check=None if targets.in_check is None else targets.in_check[mask],
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

    # player_scopeだけでなくposition_scope（seen/unseen/strict）も同じ集計器で
    # 扱えるよう、入力に実際に現れたラベルを層別化する。
    for scope in sorted(set(str(value) for value in scopes)):
        mask = torch.tensor([value == scope for value in scopes], dtype=torch.bool)
        if bool(mask.any()):
            result["scope_{}".format(scope)] = state_metrics(
                subset_targets(targets, mask),
                board_prediction[mask],
                hand_prediction[mask],
                turn_prediction[mask],
            )
    return result
