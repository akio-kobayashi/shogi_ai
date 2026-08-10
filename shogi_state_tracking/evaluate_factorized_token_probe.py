#!/usr/bin/env python3
"""factorized_v2のRAP駒種tokenから開始升を読む高速token probe。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data import load_vocabulary
from evaluate_factorized_moves import padded_forward, parse_distances, resolve_device
from factorized_prompt import factorize_usi
from models import ModelConfig, build_model
from new_prompt import square_tokens


def iter_query_batches(args, vocabulary, config, state_prompt_mode, start_selection, distances, statistics):
    """評価queryを全件materializeせず，固定batchで逐次返す。"""
    batch = []
    with Path(args.evaluation_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if statistics["games"] >= args.max_games or statistics["queries"] >= args.max_queries:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            selected = list(record.get("start_candidates", []))
            if start_selection == "fixed_initial":
                selected = [value for value in selected if int(value.get("start_ply", -1)) == 0]
            selected = selected[: args.candidates_per_game]
            if not selected:
                continue
            statistics["games"] += 1
            steps = {int(step["ply"]): step for step in record.get("evaluation_steps", [])}
            for candidate in selected:
                start = int(candidate["start_ply"])
                history = []
                state = [] if state_prompt_mode == "implicit_initial" else list(candidate["state_prompt_tokens"])
                base = ["<BOS>", *state, "<MOVES>"]
                for distance in range(max(distances) + 1):
                    ply = start + distance
                    if ply >= len(record["move_tokens"]):
                        break
                    annotation = record["move_annotations"][ply]
                    step = steps.get(ply)
                    if distance in distances and annotation.get("eligible", False) and step is not None:
                        piece = str(annotation["piece"])
                        tokens = base + history + [piece]
                        if len(tokens) <= config.max_seq_len:
                            target_parts = factorize_usi(str(record["move_tokens"][ply]))
                            source_token, destination_token = target_parts[:2]
                            legal_destinations = sorted({
                                factorize_usi(move)[1]
                                for move in step["legal_moves"]
                                if "*" not in move and factorize_usi(move)[0] == source_token
                            })
                            batch.append({
                                "start_ids": [vocabulary[token] for token in tokens],
                                "end_ids": [vocabulary[token] for token in base + history + [source_token]],
                                "actual": vocabulary[str(annotation["source"])],
                                "legal": [vocabulary[token] for token in step["legal_sources_by_piece"].get(piece, [])],
                                "actual_destination": vocabulary[destination_token],
                                "legal_destinations": [vocabulary[token] for token in legal_destinations],
                                "distance": distance,
                            })
                            statistics["queries"] += 1
                            if len(batch) >= args.batch_size:
                                yield batch
                                batch = []
                            if statistics["queries"] >= args.max_queries:
                                break
                    history.extend(factorize_usi(str(record["move_tokens"][ply])))
                if statistics["queries"] >= args.max_queries:
                    break
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description="factorized_v2 RAP token probe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-distances", default="8,32")
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--candidates-per-game", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    distances = parse_distances(args.history_distances)
    vocabulary = load_vocabulary(args.vocab)
    payload = torch.load(args.checkpoint, map_location="cpu")
    if payload.get("new_prompt", {}).get("move_encoding") != "factorized_v2":
        raise ValueError("checkpoint is not factorized_v2")
    config = ModelConfig(**payload["config"])
    checkpoint_settings = payload.get("new_prompt", {})
    state_prompt_mode = str(checkpoint_settings.get("state_prompt_mode", "explicit"))
    start_selection = str(checkpoint_settings.get("start_selection", "random_candidates"))
    device = resolve_device(args.device)
    model = build_model(str(payload.get("model_type", "vanilla")), config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    del payload
    model.eval()
    square_ids = [vocabulary[token] for token in square_tokens()]
    square_index = {token_id: index for index, token_id in enumerate(square_ids)}
    totals = {
        "queries": 0,
        "start_actual_top1": 0, "start_actual_top5": 0,
        "start_other_top1": 0, "start_other_top5": 0, "start_other_probability_mass": 0.0,
        "end_actual_top1": 0, "end_actual_top5": 0,
        "end_other_top1": 0, "end_other_top5": 0, "end_other_probability_mass": 0.0,
    }
    by_distance = {}
    started = time.perf_counter()
    statistics = {"games": 0, "queries": 0}
    with torch.inference_mode():
        for batch in iter_query_batches(
            args, vocabulary, config, state_prompt_mode, start_selection,
            distances, statistics,
        ):
            logits, lengths = padded_forward(model, [query["start_ids"] for query in batch], vocabulary["<PAD>"], device)
            rows = torch.arange(len(batch), device=device)
            values = logits[rows, lengths - 1][:, square_ids].float()
            probabilities = values.softmax(-1)
            top = torch.topk(values, 5, dim=-1).indices
            end_logits, end_lengths = padded_forward(model, [query["end_ids"] for query in batch], vocabulary["<PAD>"], device)
            end_values = end_logits[rows, end_lengths - 1][:, square_ids].float()
            end_probabilities = end_values.softmax(-1)
            end_top = torch.topk(end_values, 5, dim=-1).indices
            for row, query in enumerate(batch):
                predicted = [square_ids[int(index)] for index in top[row]]
                legal = query["legal"]
                predicted_destinations = [square_ids[int(index)] for index in end_top[row]]
                legal_destinations = query["legal_destinations"]
                if not legal or not legal_destinations:
                    continue
                for group in (totals, by_distance.setdefault(str(query["distance"]), {key: 0 for key in totals})):
                    group["queries"] += 1
                    group["start_actual_top1"] += int(predicted[0] == query["actual"])
                    group["start_actual_top5"] += int(query["actual"] in predicted)
                    group["start_other_top1"] += int(predicted[0] in legal)
                    group["start_other_top5"] += int(any(value in legal for value in predicted))
                    group["start_other_probability_mass"] += float(probabilities[row, [square_index[value] for value in legal]].sum())
                    group["end_actual_top1"] += int(predicted_destinations[0] == query["actual_destination"])
                    group["end_actual_top5"] += int(query["actual_destination"] in predicted_destinations)
                    group["end_other_top1"] += int(predicted_destinations[0] in legal_destinations)
                    group["end_other_top5"] += int(any(value in legal_destinations for value in predicted_destinations))
                    group["end_other_probability_mass"] += float(end_probabilities[row, [square_index[value] for value in legal_destinations]].sum())
            done = statistics["queries"]
            completed = totals["queries"]
            if args.progress_every and completed // args.progress_every != max(0, completed - len(batch)) // args.progress_every:
                print(json.dumps({"event": "token_probe_progress", "queries": completed, "generated_queries": done, "max_queries": args.max_queries, "elapsed_sec": round(time.perf_counter() - started, 1)}), flush=True)

    if not statistics["queries"]:
        raise ValueError("no token-probe queries")

    def summary(value):
        n = value["queries"]
        return {"queries": n, **{key: number / n for key, number in value.items() if key != "queries"}}

    output = {
        "format_version": 2,
        "checkpoint": args.checkpoint,
        "state_prompt_mode": state_prompt_mode,
        "start_selection": start_selection,
        "settings": vars(args),
        "metrics": summary(totals),
        "by_history_distance": {key: summary(value) for key, value in by_distance.items()},
        "note": "Start inserts a piece token and is in-distribution only for RAP-trained conditions. End supplies the actual source coordinate and is available for every factorized model.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "token_probe_complete", "output": str(path), "queries": totals["queries"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
