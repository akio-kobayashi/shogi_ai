#!/usr/bin/env python3
"""Toshniwal et al. (2021/2022) の chess probing に対応する将棋評価．

主集合は，平手初期局面から51--100 plyの接頭辞を与え，非歩の盤上駒による
非成り通常移動を本譜とする．Start課題では駒種，End課題では移動元をprompt
として与える．全語彙上の次token順位から ExM，LgM accuracy，LgM R-Precision
を求めるため，座標語彙だけへの事後的な制限は行わない．
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import time
from pathlib import Path

import torch

from data import load_vocabulary
from evaluate_factorized_moves import padded_forward, resolve_device
from factorized_prompt import (
    MOVE_ENCODING,
    TERMINAL_ENCODING,
    annotation_piece_token,
    factorize_history_move,
    factorize_usi,
)
from models import ModelConfig, build_model
from new_prompt import square_tokens
from train_model import amp_context, resolve_amp


EXCLUDED_PAWN_TYPES = {"<P>", "<PRO_P>"}


def parse_args():
    parser = argparse.ArgumentParser(description="Chess-compatible Start/End probing for factorized shogi models")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-prefix-plies", type=int, default=51)
    parser.add_argument("--max-prefix-plies", type=int, default=100)
    parser.add_argument("--max-instances", type=int, default=1000)
    parser.add_argument("--max-games", type=int, default=0, help="0なら全評価棋譜を走査")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--length-bucket-pool-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def stable_rank(seed, game_id, ply):
    value = "{}\0{}\0{}".format(seed, game_id, ply).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def legal_destinations_by_source(step):
    result = {}
    for move in step.get("legal_moves", []):
        move = str(move)
        if "*" in move:
            continue
        parts = factorize_usi(move)
        result.setdefault(parts[0], set()).add(parts[-1])
    return {source: sorted(values) for source, values in result.items()}


def choose_other(options, seed, game_id, ply, label):
    values = sorted(options)
    if not values:
        return None
    rank = stable_rank(seed, "{}:{}".format(game_id, label), ply)
    return values[rank % len(values)]


def make_instance(record, ply, history, evaluation_annotation_mode, seed):
    move = str(record["move_tokens"][ply])
    annotation = dict(record["move_annotations"][ply])
    if "*" in move or not bool(annotation.get("eligible", False)) or move.endswith("+"):
        return None
    piece = annotation_piece_token(str(annotation["piece"]))
    if piece in EXCLUDED_PAWN_TYPES:
        return None
    parts = factorize_usi(move)
    source, destination = parts[0], parts[-1]
    step = record["evaluation_steps"][ply]
    if int(step.get("ply", -1)) != ply or str(step.get("target_move")) != move:
        raise ValueError("evaluation_steps do not align at game={} ply={}".format(record.get("game_id"), ply))

    by_source = legal_destinations_by_source(step)
    # 成り分岐が存在するとsource直後の正しい次tokenが座標または<PROMOTE>に
    # 分かれ，チェスの二token UCI End課題と一致しないため主集合から除外する．
    if any(
        "*" not in str(legal_move)
        and factorize_usi(str(legal_move))[0] == source
        and str(legal_move).endswith("+")
        for legal_move in step.get("legal_moves", [])
    ):
        return None
    actual_legal_destinations = by_source.get(source, [])
    legal_sources = [str(value) for value in step.get("legal_sources_by_piece", {}).get(piece, [])]
    if source not in legal_sources or destination not in actual_legal_destinations:
        raise ValueError("saved legal sets do not contain target at game={} ply={}".format(record.get("game_id"), ply))

    other_piece_options = []
    for candidate_piece, sources in step.get("legal_sources_by_piece", {}).items():
        normalized = annotation_piece_token(str(candidate_piece))
        if normalized == piece or normalized in EXCLUDED_PAWN_TYPES or not sources:
            continue
        other_piece_options.append(normalized)
    other_piece = choose_other(other_piece_options, seed, record.get("game_id", ""), ply, "start_other")

    source_piece = {}
    for candidate_piece, sources in step.get("legal_sources_by_piece", {}).items():
        normalized = annotation_piece_token(str(candidate_piece))
        for candidate_source in sources:
            source_piece[str(candidate_source)] = normalized
    other_source_options = [
        candidate_source for candidate_source in by_source
        if candidate_source != source and source_piece.get(candidate_source) not in EXCLUDED_PAWN_TYPES
    ]
    other_source = choose_other(other_source_options, seed, record.get("game_id", ""), ply, "end_other")

    base = ["<BOS>", "<MOVES>", *history]
    if evaluation_annotation_mode == "ap":
        # APは全通常移動に駒種を残すoracle表記であり，チェス論文のUCI+APに対応する．
        end_actual_prompt = [*base, piece, source]
    else:
        end_actual_prompt = [*base, source]
    tasks = {
        "start_actual": {"prompt": [*base, piece], "exact": source, "legal": legal_sources},
        "end_actual": {"prompt": end_actual_prompt, "exact": destination, "legal": actual_legal_destinations},
    }
    if other_piece is not None:
        tasks["start_other"] = {
            "prompt": [*base, other_piece],
            "exact": None,
            "legal": [str(value) for value in step["legal_sources_by_piece"][other_piece]],
        }
    if other_source is not None:
        prompt = [*base, source_piece[other_source], other_source] if evaluation_annotation_mode == "ap" else [*base, other_source]
        tasks["end_other"] = {
            "prompt": prompt, "exact": None, "legal": by_source[other_source],
        }
    return {
        "game_id": str(record.get("game_id", "")),
        "ply": ply,
        "piece": piece,
        "tasks": tasks,
        "token_count": max(len(task["prompt"]) for task in tasks.values()),
    }


def select_instances(args, evaluation_annotation_mode, max_seq_len):
    heap = []
    games = candidates = 0
    with Path(args.evaluation_jsonl).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if args.max_games and games >= args.max_games:
                break
            record = json.loads(line)
            games += 1
            moves = list(record.get("move_tokens", []))
            annotations = list(record.get("move_annotations", []))
            steps = list(record.get("evaluation_steps", []))
            if not (len(moves) == len(annotations) == len(steps)):
                raise ValueError("{}:{} move/annotation/evaluation step length mismatch".format(args.evaluation_jsonl, line_number))
            history = []
            for ply, (move, annotation) in enumerate(zip(moves, annotations)):
                prefix_plies = ply
                if args.min_prefix_plies <= prefix_plies <= args.max_prefix_plies:
                    instance = make_instance(record, ply, history, evaluation_annotation_mode, args.seed)
                    if instance is not None and instance["token_count"] <= max_seq_len:
                        candidates += 1
                        rank = stable_rank(args.seed, instance["game_id"], ply)
                        entry = (-rank, instance["game_id"], ply, instance)
                        if len(heap) < args.max_instances:
                            heapq.heappush(heap, entry)
                        elif rank < -heap[0][0]:
                            heapq.heapreplace(heap, entry)
                history.extend(factorize_history_move(str(move), annotation, evaluation_annotation_mode))
                if ply >= args.max_prefix_plies:
                    break
    selected = [entry[3] for entry in sorted(heap, key=lambda value: (-value[0], value[1], value[2]))]
    identity = "\n".join("{}:{}".format(value["game_id"], value["ply"]) for value in selected)
    task_counts = {
        task: sum(task in value["tasks"] for value in selected)
        for task in ("start_actual", "start_other", "end_actual", "end_other")
    }
    return selected, {
        "games_scanned": games,
        "eligible_candidates": candidates,
        "selected_instances": len(selected),
        "selected_instance_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "task_counts": task_counts,
    }


def iter_task_batches(instances, vocabulary, batch_size, pool_batches):
    tasks = []
    for instance in instances:
        for task_name, task in instance["tasks"].items():
            tasks.append({
                "task": task_name,
                "prompt_ids": [vocabulary[token] for token in task["prompt"]],
                "exact_id": vocabulary[task["exact"]] if task["exact"] is not None else None,
                "legal_ids": [vocabulary[token] for token in task["legal"]],
            })
    pool_size = max(1, batch_size * max(1, pool_batches))
    for pool_start in range(0, len(tasks), pool_size):
        pool = tasks[pool_start : pool_start + pool_size]
        pool.sort(key=lambda item: len(item["prompt_ids"]))
        for start in range(0, len(pool), batch_size):
            yield pool[start : start + batch_size]


def empty_metrics():
    return {"queries": 0, "exact_move_correct": 0, "legal_move_correct": 0, "legal_r_precision": 0.0, "square_top1": 0}


def summarize(values):
    n = values["queries"]
    if not n:
        return {"queries": 0}
    result = {
        "queries": n,
        "legal_move_accuracy": values["legal_move_correct"] / n,
        "legal_r_precision": values["legal_r_precision"] / n,
        "square_top1_rate": values["square_top1"] / n,
    }
    if values.get("exact_queries", 0):
        result["exact_move_queries"] = values["exact_queries"]
        result["exact_move_accuracy"] = values["exact_move_correct"] / values["exact_queries"]
    return result


def score_next_token(values, exact_id, legal_ids, square_id_set):
    """全語彙上の順位を採点する．非座標tokenを事前にmaskしてはならない．"""
    predicted = int(values.argmax())
    top_r = torch.topk(values, min(len(legal_ids), values.numel())).indices.tolist()
    return {
        "legal_move_correct": int(predicted in legal_ids),
        "square_top1": int(predicted in square_id_set),
        "exact_move_correct": int(exact_id is not None and predicted == exact_id),
        "legal_r_precision": len(set(top_r) & set(legal_ids)) / len(legal_ids),
    }


def main():
    args = parse_args()
    if args.min_prefix_plies < 0 or args.max_prefix_plies < args.min_prefix_plies:
        raise ValueError("invalid prefix-ply range")
    if args.max_instances <= 0:
        raise ValueError("max-instances must be positive")
    vocabulary = load_vocabulary(args.vocab)
    payload = torch.load(args.checkpoint, map_location="cpu")
    settings = payload.get("new_prompt", {})
    if settings.get("move_encoding") != MOVE_ENCODING or settings.get("terminal_encoding") != TERMINAL_ENCODING:
        raise ValueError("checkpoint is not the current factorized_v3 decisive-game model")
    if settings.get("state_prompt_mode") != "implicit_initial" or settings.get("start_selection") != "fixed_initial":
        raise ValueError("chess-compatible protocol requires implicit fixed standard initial position")
    evaluation_annotation_mode = "ap" if settings.get("annotation_mode") == "ap" else "vanilla"
    config = ModelConfig(**payload["config"])
    instances, selection = select_instances(args, evaluation_annotation_mode, config.max_seq_len)
    if not instances:
        raise ValueError("no chess-compatible evaluation instances were selected")

    device = resolve_device(args.device)
    amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    model_type = str(payload.get("model_type", "vanilla"))
    model = build_model(model_type, config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    del payload
    model.eval()
    square_id_set = {vocabulary[token] for token in square_tokens()}
    totals = {task: empty_metrics() for task in ("start_actual", "start_other", "end_actual", "end_other")}
    completed = 0
    started = time.perf_counter()
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for batch in iter_task_batches(instances, vocabulary, args.batch_size, args.length_bucket_pool_batches):
            logits, lengths = padded_forward(model, [item["prompt_ids"] for item in batch], vocabulary["<PAD>"], device)
            rows = torch.arange(len(batch), device=device)
            values = logits[rows, lengths - 1].float()
            for row, item in enumerate(batch):
                metrics = totals[item["task"]]
                legal = item["legal_ids"]
                if not legal:
                    continue
                scores = score_next_token(values[row], item["exact_id"], legal, square_id_set)
                metrics["queries"] += 1
                metrics["legal_move_correct"] += scores["legal_move_correct"]
                metrics["square_top1"] += scores["square_top1"]
                if item["exact_id"] is not None:
                    metrics["exact_queries"] = metrics.get("exact_queries", 0) + 1
                    metrics["exact_move_correct"] += scores["exact_move_correct"]
                metrics["legal_r_precision"] += scores["legal_r_precision"]
            completed += len(batch)
            if args.progress_every and completed // args.progress_every != (completed - len(batch)) // args.progress_every:
                print(json.dumps({"event": "chess_protocol_progress", "task_queries": completed, "elapsed_sec": round(time.perf_counter() - started, 1)}), flush=True)

    output = {
        "format_version": 1,
        "protocol": "toshniwal_chess_probe_shogi_adaptation_v1",
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "evaluation_input_annotation_mode": evaluation_annotation_mode,
        "settings": vars(args),
        "selection": selection,
        "primary_filter": {
            "standard_initial_position": True,
            "prefix_plies_inclusive": [args.min_prefix_plies, args.max_prefix_plies],
            "actual_move": "non-drop board move whose source has no legal promotion branch",
            "actual_piece": "non-pawn (P and PRO_P excluded)",
            "ranking_space": "full vocabulary (not square-only masked)",
        },
        "metrics": {task: summarize(values) for task, values in totals.items()},
        "amp": amp_name,
        "limitations": [
            "This is a shogi adaptation, not an identical chess task: the board has 81 squares and shogi-specific rules.",
            "Promoting target moves and drops are excluded from the primary set to preserve the source-to-destination token form.",
            "Start tasks are in-distribution only for RAP/AP-trained models; vanilla Start results are not a fair comparison.",
            "AP retains oracle piece tokens in history and before End prompts, matching its annotated inference notation.",
        ],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "chess_protocol_complete", "output": str(path), **selection}, ensure_ascii=False))


if __name__ == "__main__":
    main()
