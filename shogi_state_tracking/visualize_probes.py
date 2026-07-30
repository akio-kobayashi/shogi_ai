#!/usr/bin/env python3
"""線形プローブの予測を将棋盤のSVGとして可視化する。

``evaluate_probes.py``が出力する``probe_predictions.pt``を入力とする。
外部描画ライブラリを使わず、論文原稿へ貼り込みやすいSVGを生成する。
"""

import argparse
import html
from pathlib import Path
from typing import Mapping, Sequence

import torch

from probes import BOARD_NAMES, HAND_NAMES


PIECE_GLYPHS = {
    "P": "歩",
    "L": "香",
    "N": "桂",
    "S": "銀",
    "G": "金",
    "B": "角",
    "R": "飛",
    "K": "玉",
    "PP": "と",
    "PL": "杏",
    "PN": "圭",
    "PS": "全",
    "PB": "馬",
    "PR": "龍",
}
HAND_GLYPHS = {
    "pawn": "歩",
    "lance": "香",
    "knight": "桂",
    "silver": "銀",
    "gold": "金",
    "bishop": "角",
    "rook": "飛",
}
RANK_LABELS = ("一", "二", "三", "四", "五", "六", "七", "八", "九")
CELL = 76
LEFT = 54
TOP = 76
BOARD_SIZE = CELL * 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="probe_predictions.ptから将棋盤SVGを生成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode", choices=("aggregate", "position", "difference")
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--predictions-b",
        help="difference modeで比較する第2モデルのprobe_predictions.pt",
    )
    parser.add_argument("--source", default="final")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metric",
        choices=("accuracy", "occupied-accuracy"),
        default="accuracy",
        help="aggregate modeの集約方法",
    )
    return parser.parse_args()


def load_predictions(path: str) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("prediction artifact must be a mapping")
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("unsupported prediction artifact format")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping) or not evaluation:
        raise ValueError("prediction artifact has no evaluation sources")
    return payload


def resolve_source(
    payload: Mapping[str, object], requested: str
) -> tuple[str, Mapping[str, object]]:
    evaluation = payload["evaluation"]
    assert isinstance(evaluation, Mapping)
    if requested in evaluation:
        source = requested
    elif requested == "final":
        layers = [
            key
            for key in evaluation
            if isinstance(key, str) and key.startswith("layer_")
        ]
        if not layers:
            raise ValueError("final source is unavailable")
        source = max(layers, key=lambda key: int(key.split("_", 1)[1]))
    else:
        raise ValueError(
            "source {!r} is unavailable; choices: {}".format(
                requested, ", ".join(str(key) for key in evaluation)
            )
        )
    value = evaluation[source]
    if not isinstance(value, Mapping):
        raise ValueError("source payload is not a mapping")
    return source, value


def tensor(payload: Mapping[str, object], name: str) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError("prediction payload is missing tensor {!r}".format(name))
    return value.cpu()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def heat_color(value: float) -> str:
    """0=red, 0.5=yellow, 1=green."""
    value = max(0.0, min(1.0, float(value)))
    if value < 0.5:
        amount = value * 2.0
        red, green = 239, int(92 + 130 * amount)
    else:
        amount = (value - 0.5) * 2.0
        red, green = int(239 - 130 * amount), 222
    blue = int(80 + 70 * (1.0 - abs(value - 0.5) * 2.0))
    return "rgb({},{},{})".format(red, green, blue)


def diverging_color(value: float) -> str:
    """-1=blue, 0=neutral, +1=red."""
    value = max(-1.0, min(1.0, float(value)))
    if value >= 0:
        amount = value
        return "rgb({},{},{})".format(
            245, int(245 - 130 * amount), int(245 - 130 * amount)
        )
    amount = -value
    return "rgb({},{},{})".format(
        int(245 - 130 * amount), int(245 - 130 * amount), 245
    )


def board_xy(index: int) -> tuple[int, int]:
    if not 0 <= index < 81:
        raise ValueError("board index must be in [0, 81)")
    # board_state_targets follows 1a,1b,...,9i. Display the usual 9-to-1
    # file order from left to right while retaining that target ordering.
    file_index, rank_index = divmod(index, 9)
    return LEFT + (8 - file_index) * CELL, TOP + rank_index * CELL


def svg_start(title: str, width: int, height: int) -> list[str]:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{}" '
        'width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            esc(title), width, height, width, height
        ),
        "<title>{}</title>".format(esc(title)),
        '<rect width="100%" height="100%" fill="#fffaf0"/>',
        '<style>text{font-family: sans-serif; fill:#332b22} '
        ".label{font-size:13px}.value{font-size:12px;font-weight:600}"
        ".piece{font-size:29px}.prediction{font-size:12px}"
        ".hand{font-size:18px}</style>",
    ]


def add_board_frame(lines: list[str]) -> None:
    lines.append(
        '<rect x="{}" y="{}" width="{}" height="{}" fill="#d8a85e" '
        'stroke="#332b22" stroke-width="2"/>'.format(
            LEFT, TOP, BOARD_SIZE, BOARD_SIZE
        )
    )
    for index in range(1, 9):
        x = LEFT + index * CELL
        y = TOP + index * CELL
        lines.append(
            '<path d="M {x} {y0} V {y1} M {x0} {y} H {x1}" '
            'stroke="#6b4f2b" stroke-width="1"/>'.format(
                x=x, y=y, y0=TOP, y1=TOP + BOARD_SIZE,
                x0=LEFT, x1=LEFT + BOARD_SIZE,
            )
        )
    for display_file in range(9, 0, -1):
        x = LEFT + (9 - display_file + 0.5) * CELL
        lines.append(
            '<text class="label" x="{}" y="{}" text-anchor="middle">{}</text>'.format(
                x, TOP - 14, display_file
            )
        )
    for rank_index, label in enumerate(RANK_LABELS):
        y = TOP + (rank_index + 0.5) * CELL + 5
        lines.append(
            '<text class="label" x="{}" y="{}" text-anchor="end">{}</text>'.format(
                LEFT - 12, y, label
            )
        )


def piece_label(class_index: int) -> tuple[str, str]:
    if class_index == 0:
        return "", "empty"
    side = "black" if class_index <= 14 else "white"
    name = BOARD_NAMES[class_index].split("_", 1)[1]
    return PIECE_GLYPHS.get(name, name), "{} {}".format(side, name)


def add_hand_text(lines: list[str], hands: Sequence[int], prefix: str, y: int) -> None:
    x = LEFT
    lines.append('<text class="label" x="{}" y="{}">{}</text>'.format(x, y, prefix))
    x += 64
    offset = 0 if prefix == "black" else 7
    for index in range(7):
        count = int(hands[offset + index])
        if count == 0:
            continue
        name = HAND_NAMES[offset + index].split("_", 1)[1]
        lines.append(
            '<text class="hand" x="{}" y="{}">{}{} </text>'.format(
                x, y, HAND_GLYPHS[name], count
            )
        )
        x += 54


def aggregate_svg(
    payload: Mapping[str, object], source: str, metric: str
) -> str:
    target = tensor(payload, "board_target")
    prediction = tensor(payload, "board_prediction")
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 81:
        raise ValueError("board tensors must have shape [samples, 81]")
    correct = target == prediction
    if metric == "occupied-accuracy":
        occupied = target != 0
        counts = occupied.sum(dim=0)
        values = torch.where(
            counts > 0,
            (correct & occupied).sum(dim=0).float() / counts.clamp_min(1),
            torch.zeros(81),
        )
        label = "occupied-square piece accuracy"
    else:
        values = correct.float().mean(dim=0)
        label = "all-square accuracy"

    lines = svg_start(
        "{}: {}".format(source, label), LEFT + BOARD_SIZE + 220, TOP + BOARD_SIZE + 120
    )
    add_board_frame(lines)
    for index, value in enumerate(values.tolist()):
        x, y = board_xy(index)
        lines.append(
            '<rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="0.72" '
            'stroke="#6b4f2b" stroke-width="1"/>'.format(
                x, y, CELL, CELL, heat_color(value)
            )
        )
        lines.append(
            '<text class="value" x="{}" y="{}" text-anchor="middle">{:.2f}</text>'.format(
                x + CELL / 2, y + CELL / 2 + 4, value
            )
        )
    occupied_count = int((target != 0).sum())
    lines.extend(
        [
            '<text class="label" x="{}" y="{}">source: {}</text>'.format(
                LEFT, 30, source
            ),
            '<text class="label" x="{}" y="{}">metric: {}</text>'.format(
                LEFT, 48, label
            ),
            '<text class="label" x="{}" y="{}">samples: {} / occupied labels: {}</text>'.format(
                LEFT, TOP + BOARD_SIZE + 34, target.shape[0], occupied_count
            ),
            '<rect x="{}" y="{}" width="{}" height="16" fill="url(#heat)"/>'.format(
                LEFT + BOARD_SIZE + 30, TOP + 30, 130
            ),
            '<defs><linearGradient id="heat">'
            '<stop offset="0%" stop-color="#ef5c50"/><stop offset="50%" stop-color="#efde50"/>'
            '<stop offset="100%" stop-color="#6cde50"/></linearGradient></defs>',
            '<text class="label" x="{}" y="{}">0.0</text>'.format(
                LEFT + BOARD_SIZE + 30, TOP + 64
            ),
            '<text class="label" x="{}" y="{}">1.0</text>'.format(
                LEFT + BOARD_SIZE + 145, TOP + 64
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines)


def position_svg(payload: Mapping[str, object], source: str, index: int) -> str:
    target = tensor(payload, "board_target")
    prediction = tensor(payload, "board_prediction")
    target_probability = tensor(payload, "board_target_probability")
    hands_target = tensor(payload, "hand_target")
    hands_prediction = tensor(payload, "hand_prediction")
    if not 0 <= index < target.shape[0]:
        raise IndexError("position index outside evaluation artifact")
    lines = svg_start(
        "{} position {}".format(source, index), LEFT + BOARD_SIZE + 220, TOP + BOARD_SIZE + 160
    )
    add_board_frame(lines)
    for square in range(81):
        x, y = board_xy(square)
        actual = int(target[index, square])
        predicted = int(prediction[index, square])
        probability = float(target_probability[index, square])
        glyph, actual_name = piece_label(actual)
        pred_glyph, pred_name = piece_label(predicted)
        correct = actual == predicted
        lines.append(
            '<rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="0.65" '
            'stroke="{}" stroke-width="{}"/>'.format(
                x, y, CELL, CELL, heat_color(probability),
                "#27833b" if correct else "#b52b2b", 2 if not correct else 1,
            )
        )
        if glyph:
            lines.append(
                '<text class="piece" x="{}" y="{}" text-anchor="middle">{}</text>'.format(
                    x + CELL / 2, y + 34, glyph
                )
            )
        if not correct:
            lines.append(
                '<text class="prediction" x="{}" y="{}" text-anchor="middle">→{}</text>'.format(
                    x + CELL / 2, y + 58, pred_glyph or "空"
                )
            )
        lines.append(
            '<title>square {}: {} / predicted {} / p={:.3f}</title>'.format(
                square, esc(actual_name), esc(pred_name), probability
            )
        )
    distance = int(tensor(payload, "distances")[index])
    game_ids = payload.get("game_ids", [])
    game_id = game_ids[index] if isinstance(game_ids, Sequence) else "unknown"
    lines.extend(
        [
            '<text class="label" x="{}" y="{}">source: {} / index: {} / distance: {} / game: {}</text>'.format(
                LEFT, 30, source, index, distance, esc(game_id)
            ),
        ]
    )
    add_hand_text(lines, hands_target[index].tolist(), "black", TOP + BOARD_SIZE + 40)
    add_hand_text(lines, hands_prediction[index].tolist(), "predicted", TOP + BOARD_SIZE + 68)
    lines.append(
        '<text class="label" x="{}" y="{}">green: correct, red: incorrect, background: target-class probability</text>'.format(
            LEFT, TOP + BOARD_SIZE + 102
        )
    )
    lines.append("</svg>")
    return "\n".join(lines)


def difference_svg(
    payload_a: Mapping[str, object],
    payload_b: Mapping[str, object],
    source: str,
) -> str:
    _, data_a = resolve_source(payload_a, source)
    _, data_b = resolve_source(payload_b, source)
    target_a = tensor(data_a, "board_target")
    target_b = tensor(data_b, "board_target")
    pred_a = tensor(data_a, "board_prediction")
    pred_b = tensor(data_b, "board_prediction")
    if target_a.shape != target_b.shape or not torch.equal(target_a, target_b):
        raise ValueError("comparison artifacts do not share the same targets")
    values = (pred_a == target_a).float().mean(0) - (pred_b == target_b).float().mean(0)
    lines = svg_start("difference: {} minus second model".format(source), LEFT + BOARD_SIZE + 220, TOP + BOARD_SIZE + 120)
    add_board_frame(lines)
    for index, value in enumerate(values.tolist()):
        x, y = board_xy(index)
        lines.append(
            '<rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="0.8"/>'.format(
                x, y, CELL, CELL, diverging_color(value)
            )
        )
        lines.append(
            '<text class="value" x="{}" y="{}" text-anchor="middle">{:+.2f}</text>'.format(
                x + CELL / 2, y + CELL / 2 + 4, value
            )
        )
    lines.extend(
        [
            '<text class="label" x="{}" y="{}">positive: first artifact better / negative: second artifact better</text>'.format(
                LEFT, 30
            ),
            '<text class="label" x="{}" y="{}">samples: {}</text>'.format(
                LEFT, TOP + BOARD_SIZE + 34, target_a.shape[0]
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    payload = load_predictions(args.predictions)
    source, source_payload = resolve_source(payload, args.source)
    if args.mode == "aggregate":
        content = aggregate_svg(source_payload, source, args.metric)
    elif args.mode == "position":
        content = position_svg(source_payload, source, args.index)
    else:
        if not args.predictions_b:
            raise ValueError("difference mode requires --predictions-b")
        payload_b = load_predictions(args.predictions_b)
        content = difference_svg(payload, payload_b, source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
