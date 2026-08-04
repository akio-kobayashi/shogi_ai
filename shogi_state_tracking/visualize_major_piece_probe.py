#!/usr/bin/env python3
"""指定局面における重要駒クラスの線形probe確率を盤面へ重ねてSVG化する。

全評価局面・全クラスの確率を``probe_predictions.pt``へ保存するとファイルが巨大になる。
このスクリプトは必要な対局・手数だけを再計算し，cshogiで再生した正解局面の上へ
指定クラスの確率を描く。
"""

import argparse
from pathlib import Path
from typing import Mapping

import torch

from data import FIXED_SEQUENCE_OVERHEAD, FixedStartSequenceDataset, load_vocabulary
from evaluate_probes import load_backbone, resolve_device
from probes import BOARD_NAMES, LinearStateProbe, board_state_targets
from visualize_probes import (
    BOARD_SIZE,
    CELL,
    LEFT,
    TOP,
    add_board_frame,
    add_hand_text,
    add_piece,
    board_xy,
    esc,
    heat_color,
    piece_label,
    svg_start,
)
from create_dataset import import_cshogi


TRACKED_PIECE_CLASSES = (
    "black_B", "black_R", "black_PB", "black_PR",
    "white_B", "white_R", "white_PB", "white_PR",
    "black_K", "white_K",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="指定した大駒・玉クラスのprobe確率をcshogi盤面に重ねる",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--probes", required=True, help="linear_probes.pt")
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--ply", required=True, type=int, help="開始局面を0とする局面番号")
    parser.add_argument("--piece", required=True, choices=TRACKED_PIECE_CLASSES)
    parser.add_argument("--source", default="final", help="layer_0等。finalは最終層")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve_probe_source(saved: Mapping[str, object], requested: str) -> str:
    sources = saved.get("sources", [])
    if not isinstance(sources, (list, tuple)):
        raise ValueError("linear probes artifact has invalid sources")
    source_names = [str(value) for value in sources]
    if requested in source_names:
        return requested
    if requested == "final":
        layers = [name for name in source_names if name.startswith("layer_")]
        if layers:
            return max(layers, key=lambda name: int(name.split("_", 1)[1]))
    raise ValueError("probe source is unavailable: {}".format(requested))


def find_example(dataset: FixedStartSequenceDataset, game_id: str):
    for index, record in enumerate(dataset.records):
        if str(record.get("game_id")) == game_id:
            return dataset[index]
    raise ValueError("game_id is absent from evaluation JSONL: {}".format(game_id))


def feature_at_ply(model, model_type: str, example, source: str, ply: int, device):
    input_ids = example["input_ids"].unsqueeze(0).to(device)
    if ply < 0 or ply >= input_ids.shape[1] - FIXED_SEQUENCE_OVERHEAD + 1:
        raise ValueError("ply is outside the replayed sequence")
    recurrent_mask = example["recurrent_mask"].unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    exact = model_type in {"t2mlr", "t^2mlr", "t²mlr"}
    with torch.inference_mode():
        output = model(
            input_ids,
            attention_mask=None if exact else attention_mask,
            recurrent_mask=recurrent_mask,
            exact_recurrence=exact,
        )
        position = 1 + 96 + ply
        if source == "token_embedding":
            return model.token_embedding(input_ids)[0, position].detach().cpu()
        if source == "recurrent":
            if output.recurrent_states is None:
                raise ValueError("checkpoint has no recurrent state")
            return output.recurrent_states[0, position].detach().cpu()
        layer = int(source.split("_", 1)[1])
        return output.hidden_states[layer][0, position].detach().cpu()


def replay_board(example, ply: int):
    cshogi = import_cshogi()
    board = cshogi.Board(str(example["start_sfen"]))
    move_ids = example["input_ids"][1 + 96 + 1 : -1].tolist()
    # token IDs are resolved by the caller to avoid relying on an inverse vocabulary here.
    return board, cshogi, move_ids


def board_for_ply(example, id_to_token: Mapping[int, str], ply: int):
    board, cshogi, move_ids = replay_board(example, ply)
    if ply > len(move_ids):
        raise ValueError("ply is outside the replayed sequence")
    for token_id in move_ids[:ply]:
        move_usi = id_to_token[int(token_id)]
        move = board.move_from_usi(move_usi)
        if not board.is_legal(move):
            raise ValueError("illegal teacher move while replaying: {}".format(move_usi))
        board.push(move)
    return board, cshogi


def heatmap_svg(board, cshogi, probabilities: torch.Tensor, class_index: int, source: str, game_id: str, ply: int) -> str:
    target, hands, turn = board_state_targets(board, cshogi)
    title = "{} probability at {} ply {}".format(BOARD_NAMES[class_index], game_id, ply)
    lines = svg_start(title, LEFT + BOARD_SIZE + 230, TOP + BOARD_SIZE + 160)
    add_board_frame(lines, heatmap=True)
    for square, probability in enumerate(probabilities.tolist()):
        x, y = board_xy(square)
        actual = int(target[square])
        _, actual_name = piece_label(actual)
        is_target = actual == class_index
        lines.append(
            '<rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="0.70" stroke="{}" stroke-width="{}"/>'.format(
                x, y, CELL, CELL, heat_color(float(probability)),
                "#147a35" if is_target else "#6b4f2b", 3 if is_target else 1,
            )
        )
        add_piece(lines, x, y, actual)
        lines.append(
            '<text class="prediction" x="{}" y="{}" text-anchor="middle">{:.2f}</text>'.format(
                x + CELL / 2, y + 60, float(probability)
            )
        )
        lines.append('<title>{}: {} / p={:.4f}</title>'.format(esc(actual_name), BOARD_NAMES[class_index], float(probability)))
    lines.extend([
        '<text class="label" x="{}" y="{}">source: {} / game: {} / ply: {}</text>'.format(LEFT, 30, esc(source), esc(game_id), ply),
        '<text class="label" x="{}" y="{}">heat: P({} at square); green border: ground-truth target</text>'.format(LEFT, 48, BOARD_NAMES[class_index]),
    ])
    add_hand_text(lines, hands, "black", TOP + BOARD_SIZE + 40)
    add_hand_text(lines, hands, "white", TOP + BOARD_SIZE + 68)
    lines.append('<text class="label" x="{}" y="{}">turn: {}</text>'.format(LEFT, TOP + BOARD_SIZE + 102, "black" if turn == 0 else "white"))
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    vocabulary = load_vocabulary(args.vocab)
    id_to_token = {index: token for token, index in vocabulary.items()}
    model, model_type, config = load_backbone(args.checkpoint, device, False)
    max_moves = int(config.max_seq_len) - FIXED_SEQUENCE_OVERHEAD
    dataset = FixedStartSequenceDataset(args.evaluation_jsonl, vocabulary, max_moves)
    example = find_example(dataset, args.game_id)

    saved = torch.load(args.probes, map_location="cpu")
    if not isinstance(saved, Mapping):
        raise ValueError("linear probes artifact must be a mapping")
    source = resolve_probe_source(saved, args.source)
    states = saved.get("probe_state_dicts")
    if not isinstance(states, Mapping) or source not in states:
        raise ValueError("linear probe weights are unavailable for source: {}".format(source))
    probe = LinearStateProbe(int(config.d_model))
    probe.load_state_dict(states[source])
    probe.to(device).eval()

    feature = feature_at_ply(model, model_type, example, source, args.ply, device)
    with torch.inference_mode():
        logits = probe(feature.unsqueeze(0).to(device)).board[0].softmax(dim=-1).cpu()
    class_index = BOARD_NAMES.index(args.piece)
    board, cshogi = board_for_ply(example, id_to_token, args.ply)
    content = heatmap_svg(board, cshogi, logits[:, class_index], class_index, source, args.game_id, args.ply)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
