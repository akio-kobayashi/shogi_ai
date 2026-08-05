#!/usr/bin/env python3
"""新prompt線形プローブの大駒・玉位置確率を，保存済み局面へ重ねてSVG化する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import load_vocabulary
from evaluate_new_prompt_probes import label_maps
from models import ModelConfig, build_model
from new_prompt import move_token
from probes import BOARD_NAMES, LinearStateProbe
from visualize_probes import BOARD_SIZE, CELL, LEFT, TOP, add_board_frame, add_hand_text, board_xy, heat_color, piece_label, svg_start


def parse_args():
    parser = argparse.ArgumentParser(description="新prompt probeを盤面ヒートマップとして出力する")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--vocab", required=True)
    parser.add_argument("--evaluation-jsonl", required=True); parser.add_argument("--probes", required=True)
    parser.add_argument("--game-id", required=True); parser.add_argument("--start-ply", type=int, required=True); parser.add_argument("--ply", type=int, required=True)
    parser.add_argument("--piece", required=True, choices=("black_B", "black_R", "black_PB", "black_PR", "white_B", "white_R", "white_PB", "white_PR", "black_K", "white_K"))
    parser.add_argument("--source", default="final"); parser.add_argument("--device", default="auto"); parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args(); device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    vocabulary = load_vocabulary(args.vocab); payload = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig(**payload["config"]); model = build_model(str(payload.get("model_type", "vanilla")), config).to(device); model.load_state_dict(payload["model_state_dict"]); model.eval()
    target = None
    with Path(args.evaluation_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("game_id") != args.game_id: continue
            for example in record.get("probe_examples", []):
                if int(example["start_ply"]) == args.start_ply and int(example["ply"]) == args.ply:
                    target = example; break
            if target: break
    if target is None: raise ValueError("matching probe example is absent")
    saved = torch.load(args.probes, map_location="cpu"); source = "layer_{}".format(config.n_layers) if args.source == "final" else args.source
    probe = LinearStateProbe(config.d_model); probe.load_state_dict(saved["probe_state_dicts"][source]); probe.to(device).eval()
    tokens = ["<BOS>"] + target["state_prompt_tokens"] + ["<MOVES>"] + [move_token(move) for move in target["history_moves"]]
    ids = torch.tensor([[vocabulary[token] for token in tokens]], device=device)
    with torch.inference_mode():
        output = model(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
        feature = output.hidden_states[int(source.split("_", 1)[1])][0, -1]
        logits = probe(feature.unsqueeze(0)).board[0].softmax(-1).cpu()
    board_map, hand_names = label_maps(); actual = [board_map[label] for label in target["probe_targets"]["board_labels_cshogi_order"]]
    cls = BOARD_NAMES.index(args.piece); probabilities = logits[:, cls].tolist()
    lines = svg_start("{} probability".format(args.piece), LEFT + BOARD_SIZE + 240, TOP + BOARD_SIZE + 160); add_board_frame(lines)
    for square, probability in enumerate(probabilities):
        x, y = board_xy(square); glyph, name = piece_label(actual[square]); truth = actual[square] == cls
        lines.append('<rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity=".70" stroke="{}" stroke-width="{}"/>'.format(x,y,CELL,CELL,heat_color(probability),"#147a35" if truth else "#6b4f2b",3 if truth else 1))
        if glyph: lines.append('<text class="piece" x="{}" y="{}" text-anchor="middle">{}</text>'.format(x+CELL/2,y+31,glyph))
        lines.append('<text class="prediction" x="{}" y="{}" text-anchor="middle">{:.2f}</text>'.format(x+CELL/2,y+60,probability))
    hands = [int(target["probe_targets"]["hands"].get(name,0)) for name in hand_names]
    add_hand_text(lines, hands[:7], "black", TOP + BOARD_SIZE + 40); add_hand_text(lines, hands[7:], "white", TOP + BOARD_SIZE + 68)
    lines.append('<text class="label" x="{}" y="{}">{} / game {} / start {} / ply {} / {}</text>'.format(LEFT,30,args.piece,args.game_id,args.start_ply,args.ply,source)); lines.append("</svg>")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(lines)+"\n", encoding="utf-8"); print(output)


if __name__ == "__main__": main()
