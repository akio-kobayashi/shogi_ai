#!/usr/bin/env python3
"""部分的行動教師のStart-Actual／Start-Otherトークンプローブ。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import load_vocabulary
from models import ModelConfig, build_model
from new_prompt import square_tokens, move_token


def parse_args():
    parser = argparse.ArgumentParser(description="部分的行動教師モデルの開始位置トークンプローブ")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--vocab", required=True)
    parser.add_argument("--evaluation-jsonl", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--max-games", type=int, default=5000); parser.add_argument("--candidates-per-game", type=int, default=3)
    parser.add_argument("--max-moves", type=int); parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args(); vocabulary = load_vocabulary(args.vocab)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig(**payload["config"])
    if args.max_moves is None: args.max_moves = int(payload.get("new_prompt", {}).get("max_moves", 512))
    model = build_model(str(payload.get("model_type", "vanilla")), config).to(device)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    square_ids = [vocabulary[token] for token in square_tokens()]
    square_index = {token_id: index for index, token_id in enumerate(square_ids)}
    totals = {"queries": 0, "actual_top1": 0, "actual_top5": 0, "other_top1": 0, "other_top5": 0, "other_probability_mass": 0.0}
    games = candidates = 0
    with Path(args.evaluation_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or games >= args.max_games: continue
            record = json.loads(line); steps = {int(step["ply"]): step for step in record.get("evaluation_steps", [])}
            selected = list(record.get("start_candidates", []))[:args.candidates_per_game]
            if not selected: continue
            games += 1
            for candidate in selected:
                candidates += 1; start = int(candidate["start_ply"]); end = min(len(record["move_tokens"]), start + args.max_moves)
                for ply in range(start, end):
                    annotation = record["move_annotations"][ply]
                    if not annotation.get("eligible", False): continue
                    step = steps.get(ply)
                    if step is None: continue
                    tokens = ["<BOS>"] + candidate["state_prompt_tokens"] + ["<MOVES>"] + [move_token(move) for move in record["move_tokens"][start:ply]] + [annotation["piece"]]
                    if len(tokens) > config.max_seq_len: break
                    ids = torch.tensor([[vocabulary[token] for token in tokens]], device=device)
                    with torch.inference_mode(): logits = model(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool)).logits[0, -1, square_ids]
                    probability = logits.float().softmax(-1); rank = logits.argsort(descending=True); top = [square_ids[int(value)] for value in rank[:5]]
                    actual = vocabulary[annotation["source"]]
                    legal_sources = [vocabulary[token] for token in step["legal_sources_by_piece"].get(annotation["piece"], [])]
                    if not legal_sources: continue
                    totals["queries"] += 1; totals["actual_top1"] += int(top[0] == actual); totals["actual_top5"] += int(actual in top)
                    totals["other_top1"] += int(top[0] in legal_sources); totals["other_top5"] += int(any(value in legal_sources for value in top))
                    totals["other_probability_mass"] += float(probability[[square_index[value] for value in legal_sources]].sum())
    n = totals["queries"]
    if not n: raise ValueError("no eligible token-probe query")
    metrics = {"queries": n, "start_actual_top1": totals["actual_top1"]/n, "start_actual_top5": totals["actual_top5"]/n, "start_other_top1": totals["other_top1"]/n, "start_other_top5": totals["other_top5"]/n, "start_other_probability_mass": totals["other_probability_mass"]/n}
    output = {"checkpoint": args.checkpoint, "settings": vars(args), "metrics": metrics, "note": "This is an in-distribution diagnostic only for partial-action training."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"); print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
