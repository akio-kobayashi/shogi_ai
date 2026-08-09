#!/usr/bin/env python3
"""新prompt artifactだけで行う，制約付き原子的USI指手評価。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from data import load_vocabulary
from models import ModelConfig, build_model
from new_prompt import atomic_move_tokens, move_token


def parse_args():
    parser = argparse.ArgumentParser(description="新prompt checkpointを指手・合法性で評価する")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--candidates-per-game", type=int, default=3)
    parser.add_argument(
        "--primary-history-distances",
        default="8,32",
        help="主指標へ含める開始局面からの指手距離。0は開始promptをそのまま読めるため既定では除外する。",
    )
    parser.add_argument(
        "--report-history-distances",
        default="0,8,32",
        help="距離別内訳として保存する開始局面からの指手距離。",
    )
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def device_for(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_history_distances(value):
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        distance = int(item)
        if distance < 0:
            raise ValueError("history distance must be nonnegative")
        if distance not in result:
            result.append(distance)
    if not result:
        raise ValueError("at least one history distance is required")
    return tuple(result)


def _empty_totals():
    return {"targets": 0, "cross_entropy": 0.0, "top1": 0, "top5": 0, "legal_top1": 0, "legal_top5": 0, "legal_mass": 0.0}


def _add_metrics(total, target_id, top, legal_ids, log_prob, legal_prob):
    total["targets"] += 1; total["cross_entropy"] -= float(log_prob)
    total["top1"] += int(top[0] == target_id); total["top5"] += int(target_id in top)
    total["legal_top1"] += int(top[0] in legal_ids); total["legal_top5"] += int(any(item in legal_ids for item in top))
    total["legal_mass"] += float(legal_prob)


def summarize(total):
    n = total["targets"]
    if not n: return None
    return {"move_targets": n, "cross_entropy": total["cross_entropy"] / n, "perplexity": math.exp(total["cross_entropy"] / n), "top1_accuracy": total["top1"] / n, "top5_accuracy": total["top5"] / n, "legality": {"top1_legal_rate": total["legal_top1"] / n, "top5_contains_legal_rate": total["legal_top5"] / n, "mean_legal_probability_mass": total["legal_mass"] / n}}


def position_scope_at_ply(record, candidate, ply):
    """対象指手の直前局面に付いたscopeを返す。

    candidateのscopeは開始局面だけの情報である。開始後の指手評価を同じscopeへ
    誤って集約しないため，plyごとの注釈を優先する。
    """
    values = record.get("position_scope_by_ply")
    if isinstance(values, list) and 0 <= ply < len(values):
        return str(values[ply])
    # schema v1のartifactは全plyのscopeを保存していない。ただし距離0, 8, 32の
    # probe例には正しい対象plyのscopeがあるため，今回の主評価ではそこを利用できる。
    start = int(candidate["start_ply"])
    for example in record.get("probe_examples", []):
        if int(example.get("start_ply", -1)) == start and int(example.get("ply", -1)) == ply:
            return str(example.get("position_scope", "unknown_position_scope"))
    return str(candidate.get("position_scope", record.get("position_scope", "unknown_position_scope")))


def evaluate_candidate(
    model, record, candidate, vocabulary, action_ids, action_index, args, device,
    all_totals, primary_totals, scopes,
):
    start = int(candidate["start_ply"])
    steps = record.get("evaluation_steps", [])
    by_ply = {int(step["ply"]): step for step in steps}
    end = min(len(record["move_tokens"]), start + args.max_moves)
    tokens = ["<BOS>"] + candidate["state_prompt_tokens"] + ["<MOVES>"]
    moves_position = len(tokens) - 1
    move_usis = [str(record["move_tokens"][ply]) for ply in range(start, end)]
    tokens += [move_token(move) for move in move_usis]
    if len(tokens) > args.max_seq_len:
        return 0
    ids = torch.tensor([[vocabulary[token] for token in tokens]], device=device)
    attention_mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.inference_mode():
        logits = model(ids, attention_mask=attention_mask).logits[0]
    action_logits = logits[:, action_ids]
    log_probs = torch.log_softmax(action_logits.float(), dim=-1)
    probs = log_probs.exp()
    for offset, ply in enumerate(range(start, end)):
        step = by_ply.get(ply)
        if step is None:
            continue
        position = moves_position + offset
        target_id = vocabulary[move_token(str(step["target_move"]))]
        target_action_index = action_index[target_id]
        ranking = torch.argsort(action_logits[position], descending=True)
        top = [action_ids[int(index)] for index in ranking[:5]]
        legal_ids = [vocabulary[move_token(move)] for move in step["legal_moves"]]
        legal_action_indices = [action_index[item] for item in legal_ids]
        top1 = top[0]
        legal_probability = probs[position, legal_action_indices].sum()
        log_probability = log_probs[position, target_action_index]
        _add_metrics(all_totals, target_id, top, legal_ids, log_probability, legal_probability)
        history_distance = offset
        position_scope = position_scope_at_ply(record, candidate, ply)
        trajectory_scope = str(record.get("trajectory_scope", "unknown_position_scope"))
        if history_distance in args.primary_history_distances:
            _add_metrics(primary_totals, target_id, top, legal_ids, log_probability, legal_probability)
            _add_metrics(scopes["primary_position"].setdefault(position_scope, _empty_totals()), target_id, top, legal_ids, log_probability, legal_probability)
            _add_metrics(scopes["primary_trajectory"].setdefault(trajectory_scope, _empty_totals()), target_id, top, legal_ids, log_probability, legal_probability)
        if history_distance in args.report_history_distances:
            key = str(history_distance)
            _add_metrics(scopes["distance"].setdefault(key, _empty_totals()), target_id, top, legal_ids, log_probability, legal_probability)
            _add_metrics(scopes["distance_position"].setdefault(key, {}).setdefault(position_scope, _empty_totals()), target_id, top, legal_ids, log_probability, legal_probability)
            _add_metrics(scopes["distance_trajectory"].setdefault(key, {}).setdefault(trajectory_scope, _empty_totals()), target_id, top, legal_ids, log_probability, legal_probability)
    return 1


def main():
    args = parse_args()
    args.primary_history_distances = parse_history_distances(args.primary_history_distances)
    args.report_history_distances = parse_history_distances(args.report_history_distances)
    vocabulary = load_vocabulary(args.vocab)
    action_ids = [vocabulary[token] for token in atomic_move_tokens()]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("move vocabulary has duplicate ids")
    action_index = {token_id: index for index, token_id in enumerate(action_ids)}
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig(**checkpoint["config"])
    checkpoint_data = checkpoint.get("new_prompt", {})
    if args.max_seq_len is None:
        args.max_seq_len = config.max_seq_len
    if args.max_moves is None:
        args.max_moves = int(checkpoint_data.get("max_moves", 512))
    if config.vocab_size != len(vocabulary):
        raise ValueError("checkpoint and vocabulary sizes differ")
    device = device_for(args.device)
    model_type = str(checkpoint.get("model_type", "vanilla"))
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    totals = {"games": 0, "candidates": 0}
    all_totals = _empty_totals()
    primary_totals = _empty_totals()
    scopes = {
        "primary_position": {}, "primary_trajectory": {}, "distance": {},
        "distance_position": {}, "distance_trajectory": {},
    }
    with Path(args.evaluation_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or totals["games"] >= args.max_games:
                continue
            record = json.loads(line)
            candidates = list(record.get("start_candidates", []))[: args.candidates_per_game]
            if not candidates:
                continue
            totals["games"] += 1
            for candidate in candidates:
                totals["candidates"] += evaluate_candidate(model, record, candidate, vocabulary, action_ids, action_index, args, device, all_totals, primary_totals, scopes)
    if not all_totals["targets"]:
        raise ValueError("no evaluable move targets")
    if not primary_totals["targets"]:
        raise ValueError("no evaluable move targets at primary history distances")
    metrics = {
        "games": totals["games"],
        "candidates": totals["candidates"],
        "primary_history_distances": list(args.primary_history_distances),
        **summarize(primary_totals),
        "all_history_distances": summarize(all_totals),
        "by_position_scope": {key: summarize(value) for key, value in scopes["primary_position"].items()},
        "by_trajectory_scope": {key: summarize(value) for key, value in scopes["primary_trajectory"].items()},
        "by_history_distance": {key: summarize(value) for key, value in scopes["distance"].items()},
        "by_history_distance_and_position_scope": {
            distance: {scope: summarize(value) for scope, value in groups.items()}
            for distance, groups in scopes["distance_position"].items()
        },
        "by_history_distance_and_trajectory_scope": {
            distance: {scope: summarize(value) for scope, value in groups.items()}
            for distance, groups in scopes["distance_trajectory"].items()
        },
    }
    output = {"checkpoint": args.checkpoint, "model_type": model_type, "settings": vars(args), "metrics": metrics}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
