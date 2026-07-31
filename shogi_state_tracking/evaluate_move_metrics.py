#!/usr/bin/env python3
"""指手予測と合法手指標だけを計算する軽量評価。"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

import torch

from create_dataset import all_usi_move_tokens, import_cshogi
from data import FIXED_SEQUENCE_OVERHEAD, FixedStartSequenceDataset, IGNORE_INDEX
from data import load_vocabulary
from evaluate_probes import load_backbone, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="評価データ上の指手予測・合法手指標だけを計算する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    started_at = time.perf_counter()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    vocabulary = load_vocabulary(args.vocab)
    id_to_token = {index: token for token, index in vocabulary.items()}
    model, model_type, config = load_backbone(
        args.checkpoint, device, untrained=False
    )
    max_suffix_moves = config.max_seq_len - FIXED_SEQUENCE_OVERHEAD
    if max_suffix_moves <= 0:
        raise ValueError(
            "checkpoint max_seq_len is too short for the fixed state prefix"
        )
    dataset = FixedStartSequenceDataset(
        args.evaluation_jsonl,
        vocabulary,
        max_suffix_moves=max_suffix_moves,
    )
    limit = len(dataset) if args.max_examples <= 0 else min(args.max_examples, len(dataset))
    cshogi = import_cshogi()
    token_to_id = dict(vocabulary)
    syntactic_moves = set(all_usi_move_tokens())
    eos_id = token_to_id.get("<EOS>")

    move_loss_sum = 0.0
    move_targets = 0
    move_top1 = 0
    move_top5 = 0
    legal_positions = 0
    legal_top1 = 0
    legal_top5 = 0
    syntactic_top1 = 0
    legal_probability_mass = 0.0
    legal_vocabulary_coverage = 0.0
    total_moves = 0

    print(
        "move_evaluation_start examples={} device={} start_mode=fixed_start_ply_0".format(
            limit, device
        ),
        flush=True,
    )
    with torch.inference_mode():
        for example_index in range(limit):
            example = dataset[example_index]
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            recurrent_mask = example["recurrent_mask"].unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            exact = model_type in {"t2mlr", "t^2mlr", "t²mlr"}
            output = model(
                input_ids,
                attention_mask=None if exact else attention_mask,
                recurrent_mask=recurrent_mask,
                exact_recurrence=exact,
            )

            labels = example["labels"].to(device)
            supervised = labels != IGNORE_INDEX
            move_supervised = supervised
            if eos_id is not None:
                move_supervised = supervised & (labels != eos_id)
            move_logits = output.logits[0, move_supervised]
            move_labels = labels[move_supervised]
            if move_labels.numel():
                move_loss_sum += float(
                    torch.nn.functional.cross_entropy(
                        move_logits, move_labels, reduction="sum"
                    )
                )
                move_targets += int(move_labels.numel())
                move_top1 += int(
                    (move_logits.argmax(dim=-1) == move_labels).sum()
                )
                top_k = min(5, move_logits.shape[-1])
                move_top5 += int(
                    (
                        move_logits.topk(top_k, dim=-1).indices
                        == move_labels[:, None]
                    )
                    .any(dim=1)
                    .sum()
                )

            moves_marker = 1 + 96
            move_ids = input_ids[0, moves_marker + 1 : -1].tolist()
            move_tokens = [id_to_token[int(token_id)] for token_id in move_ids]
            replay_board = cshogi.Board(str(example["start_sfen"]))
            for move_index, target_move in enumerate(move_tokens):
                prediction_position = moves_marker + move_index
                logits = output.logits[0, prediction_position]
                legal_moves = [
                    cshogi.move_to_usi(move) for move in replay_board.legal_moves
                ]
                legal_ids = [
                    token_to_id[move]
                    for move in legal_moves
                    if move in token_to_id
                ]
                legal_id_set = set(legal_ids)
                top_ids = logits.topk(min(5, logits.shape[-1])).indices.tolist()
                top_token = id_to_token[int(top_ids[0])]
                legal_positions += 1
                legal_top1 += int(int(top_ids[0]) in legal_id_set)
                legal_top5 += int(
                    any(int(value) in legal_id_set for value in top_ids)
                )
                syntactic_top1 += int(top_token in syntactic_moves)
                legal_vocabulary_coverage += len(legal_ids) / max(
                    len(legal_moves), 1
                )
                if legal_ids:
                    probabilities = torch.softmax(logits, dim=-1)
                    legal_index = torch.tensor(
                        legal_ids, dtype=torch.long, device=device
                    )
                    legal_probability_mass += float(
                        probabilities.index_select(0, legal_index).sum()
                    )

                target = replay_board.move_from_usi(str(target_move))
                if not replay_board.is_legal(target):
                    raise ValueError(
                        "ground-truth move is illegal in game {} at local ply {}: {}".format(
                            example["game_id"], move_index + 1, target_move
                        )
                    )
                replay_board.push(target)
            total_moves += len(move_tokens)

            processed = example_index + 1
            if args.progress_every > 0 and (
                processed == 1 or processed % args.progress_every == 0
            ):
                elapsed = time.perf_counter() - started_at
                print(
                    "move_evaluation_progress examples={}/{} elapsed_sec={:.1f}".format(
                        processed, limit, elapsed
                    ),
                    flush=True,
                )

    mean_loss = move_loss_sum / max(move_targets, 1)
    return {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "model_type": model_type,
        "evaluation_jsonl": str(args.evaluation_jsonl),
        "device": str(device),
        "settings": {
            "examples": limit,
            "max_seq_len": config.max_seq_len,
            "max_suffix_moves": max_suffix_moves,
            "seed": args.seed,
            "start_mode": "fixed_start_ply_0",
        },
        "metrics": {
            "games": limit,
            "move_targets": move_targets,
            "total_replayed_moves": total_moves,
            "cross_entropy": mean_loss,
            "perplexity": math.exp(min(mean_loss, 20.0)),
            "top1_accuracy": move_top1 / max(move_targets, 1),
            "top5_accuracy": move_top5 / max(move_targets, 1),
            "legality": {
                "move_positions": legal_positions,
                "top1_legal_rate": legal_top1 / max(legal_positions, 1),
                "top5_contains_legal_rate": legal_top5
                / max(legal_positions, 1),
                "top1_syntactic_move_rate": syntactic_top1
                / max(legal_positions, 1),
                "mean_legal_probability_mass": legal_probability_mass
                / max(legal_positions, 1),
                "mean_legal_move_vocabulary_coverage": legal_vocabulary_coverage
                / max(legal_positions, 1),
            },
        },
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "move_metrics.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("move_evaluation_complete output={}".format(path), flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
