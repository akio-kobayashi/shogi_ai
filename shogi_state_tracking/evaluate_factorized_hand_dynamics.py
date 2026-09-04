#!/usr/bin/env python3
"""Evaluate shogi hand-state dynamics and the validity of predicted drops.

This evaluator is deliberately separate from the ordinary move and token-probe
evaluators.  It uses already fitted linear state probes to ask whether a
capture increments, and a drop decrements, the appropriate hand counter.  It
also evaluates a model's drop distribution against the *true* hand and legal
move set.  No rule information is fed to the model; cshogi-derived labels are
used only by this evaluator.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch

from data import load_vocabulary
from evaluate_factorized_moves import beam_batch_cached, padded_forward
from evaluate_new_prompt_probes import label_maps
from factorized_prompt import (
    BASIC_PIECE_TOKENS,
    DROP_TOKEN,
    MOVE_ENCODING,
    TERMINAL_ENCODING,
    factorize_history_move,
    unfactorize_usi,
)
from models import ModelConfig, build_model
from new_prompt import square_tokens
from probes import LinearStateProbe, predictions_from_logits
from train_model import amp_context, resolve_amp
from provenance import write_metrics_json


def parse_args():
    parser = argparse.ArgumentParser(description="factorized_v3持ち駒遷移・駒打ち評価")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--linear-probes", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources", default="available", help="available,final,layer_0,...")
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--max-drop-queries", type=int, default=5000)
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beam-micro-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    return torch.device(value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def _hand_vector(probe_targets: Mapping[str, object], hand_names: Sequence[str]) -> list[int]:
    hands = dict(probe_targets["hands"])
    return [int(hands.get(name, 0)) for name in hand_names]


def _reservoir_add(values: list, item: dict, seen: int, limit: int, rng: random.Random) -> int:
    seen += 1
    if limit <= 0:
        return seen
    if len(values) < limit:
        values.append(item)
    else:
        index = rng.randrange(seen)
        if index < limit:
            values[index] = item
    return seen


def _base_tokens(record: Mapping[str, object], state_prompt_mode: str) -> list[str]:
    candidates = [value for value in record.get("start_candidates", []) if int(value.get("start_ply", -1)) == 0]
    if len(candidates) != 1:
        raise ValueError("evaluation record must have exactly one ply-0 candidate")
    state = [] if state_prompt_mode == "implicit_initial" else [str(value) for value in candidates[0]["state_prompt_tokens"]]
    return ["<BOS>", *state, "<MOVES>"]


def read_queries(args, state_prompt_mode: str, evaluation_annotation_mode: str, max_seq_len: int):
    """Reservoir-sample capture/drop transitions and actual-drop decisions."""
    rng = random.Random(args.seed)
    events: list[dict] = []
    drops: list[dict] = []
    event_seen = drop_seen = games = 0
    _, hand_names = label_maps()
    with Path(args.evaluation_jsonl).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if games >= args.max_games:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            moves = [str(value) for value in record.get("move_tokens", [])]
            annotations = [dict(value) for value in record.get("move_annotations", [])]
            steps = list(record.get("evaluation_steps", []))
            if len(steps) != len(moves) or len(annotations) != len(moves):
                raise ValueError("{}:{} evaluation_steps/move_annotations do not align".format(args.evaluation_jsonl, line_number))
            base = _base_tokens(record, state_prompt_mode)
            history: list[str] = []
            games += 1
            for ply, move in enumerate(moves):
                before = _hand_vector(steps[ply]["probe_targets"], hand_names)
                after = _hand_vector(steps[ply + 1]["probe_targets"], hand_names) if ply + 1 < len(steps) else None
                current_tokens = base + history
                move_tokens = factorize_history_move(move, annotations[ply], evaluation_annotation_mode)
                after_tokens = current_tokens + move_tokens
                if after is not None and len(after_tokens) <= max_seq_len:
                    delta = [right - left for left, right in zip(before, after)]
                    nonzero = [index for index, value in enumerate(delta) if value]
                    event_type = "drop" if "*" in move else ("capture" if any(value > 0 for value in delta) else None)
                    if event_type is not None:
                        expected_sign = -1 if event_type == "drop" else 1
                        if len(nonzero) != 1 or delta[nonzero[0]] != expected_sign:
                            raise ValueError("{}:{} unexpected hand transition at ply {}: {}".format(args.evaluation_jsonl, line_number, ply, delta))
                        event_seen = _reservoir_add(events, {
                            "game_id": str(record.get("game_id", "")), "ply": ply,
                            "event_type": event_type, "changed_slot": nonzero[0],
                            "before_tokens": current_tokens, "after_tokens": after_tokens,
                            "before_hands": before, "after_hands": after,
                        }, event_seen, args.max_events, rng)
                # For AP this remains a native query: an actual drop has no
                # oracle piece token before <DROP>.
                if "*" in move and len(current_tokens) + 3 <= max_seq_len:
                    drop_seen = _reservoir_add(drops, {
                        "game_id": str(record.get("game_id", "")), "ply": ply,
                        "prefix_tokens": current_tokens, "target": move,
                        "side": 0 if steps[ply]["probe_targets"]["turn"] == "<TURN_BLACK>" else 1,
                        "hands": before, "legal_moves": [str(value) for value in steps[ply]["legal_moves"]],
                    }, drop_seen, args.max_drop_queries, rng)
                history.extend(move_tokens)
    return events, drops, {"games": games, "eligible_events": event_seen, "eligible_actual_drops": drop_seen}


def _pad_token_batch(batch: Sequence[Sequence[str]], vocabulary: Mapping[str, int], device: torch.device):
    lengths_cpu = torch.tensor([len(tokens) for tokens in batch], dtype=torch.long)
    width = int(lengths_cpu.max())
    ids_cpu = torch.full((len(batch), width), int(vocabulary["<PAD>"]), dtype=torch.long)
    for row, tokens in enumerate(batch):
        ids_cpu[row, : len(tokens)] = torch.tensor([vocabulary[token] for token in tokens], dtype=torch.long)
    mask_cpu = torch.arange(width)[None, :] < lengths_cpu[:, None]
    non_blocking = device.type == "cuda"
    return ids_cpu.to(device, non_blocking=non_blocking), mask_cpu.to(device, non_blocking=non_blocking), lengths_cpu.to(device, non_blocking=non_blocking)


def extract_event_features(model, events, vocabulary, sources, device, amp_dtype, batch_size, progress_every):
    result = {source: {"before": [], "after": []} for source in sources}
    started = time.perf_counter()
    ordered = sorted(events, key=lambda item: max(len(item["before_tokens"]), len(item["after_tokens"])))
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            sequences = [item[key] for item in batch for key in ("before_tokens", "after_tokens")]
            ids, attention, lengths = _pad_token_batch(sequences, vocabulary, device)
            recurrent = torch.zeros_like(attention)
            for row, tokens in enumerate(sequences):
                recurrent[row, tokens.index("<MOVES>") + 1 : len(tokens)] = True
            output = model(ids, attention_mask=attention, recurrent_mask=recurrent, output_hidden_states=True)
            rows = torch.arange(len(sequences), device=device)
            positions = lengths - 1
            for source in sources:
                layer = int(source.split("_", 1)[1])
                selected = output.hidden_states[layer][rows, positions].float().cpu()
                result[source]["before"].append(selected[0::2])
                result[source]["after"].append(selected[1::2])
            if progress_every and start + len(batch) >= progress_every and (start + len(batch)) // progress_every != start // progress_every:
                print(json.dumps({"event": "hand_feature_progress", "events": start + len(batch), "total": len(ordered), "elapsed_sec": round(time.perf_counter() - started, 1)}), flush=True)
    return {source: {side: torch.cat(chunks) for side, chunks in pair.items()} for source, pair in result.items()}, ordered


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if not denominator else float(numerator) / float(denominator)


def hand_transition_metrics(
    before_prediction: torch.Tensor,
    after_prediction: torch.Tensor,
    events: Sequence[dict],
    hand_names: Sequence[str] | None = None,
) -> dict:
    before_target = torch.tensor([item["before_hands"] for item in events], dtype=torch.long)
    after_target = torch.tensor([item["after_hands"] for item in events], dtype=torch.long)
    predicted_delta = after_prediction - before_prediction
    target_delta = after_target - before_target
    def evaluate_indices(indices: list[int]) -> dict:
        if not indices:
            return {"events": 0}
        idx = torch.tensor(indices, dtype=torch.long)
        slots = torch.tensor([events[index]["changed_slot"] for index in indices], dtype=torch.long)
        rows = torch.arange(len(indices))
        unrelated = torch.ones((len(indices), 14), dtype=torch.bool)
        unrelated[rows, slots] = False
        pred_delta = predicted_delta[idx]
        true_delta = target_delta[idx]
        before = before_prediction[idx]
        after = after_prediction[idx]
        before_true = before_target[idx]
        after_true = after_target[idx]
        return {
            "events": len(indices),
            "before_hand_exact_match": float((before == before_true).all(dim=1).float().mean()),
            "after_hand_exact_match": float((after == after_true).all(dim=1).float().mean()),
            "before_slot_accuracy": float((before == before_true).float().mean()),
            "after_slot_accuracy": float((after == after_true).float().mean()),
            "changed_slot_before_accuracy": float((before[rows, slots] == before_true[rows, slots]).float().mean()),
            "changed_slot_after_accuracy": float((after[rows, slots] == after_true[rows, slots]).float().mean()),
            "changed_slot_delta_accuracy": float((pred_delta[rows, slots] == true_delta[rows, slots]).float().mean()),
            "unrelated_slots_unchanged_rate": float((pred_delta[unrelated] == 0).float().mean()),
            "full_hand_delta_exact_match": float((pred_delta == true_delta).all(dim=1).float().mean()),
        }
    result = {
        group: evaluate_indices([
            index for index, item in enumerate(events)
            if group == "all" or item["event_type"] == group
        ])
        for group in ("all", "capture", "drop")
    }
    names = list(hand_names or ["slot_{}".format(index) for index in range(14)])
    result["by_changed_slot"] = {
        names[slot]: evaluate_indices([
            index for index, item in enumerate(events)
            if int(item["changed_slot"]) == slot
        ])
        for slot in range(14)
        if any(int(item["changed_slot"]) == slot for item in events)
    }
    def ply_bucket(ply: int) -> str:
        if ply < 16:
            return "0-15"
        if ply < 32:
            return "16-31"
        if ply < 64:
            return "32-63"
        return "64+"
    result["by_ply_bucket"] = {
        bucket: evaluate_indices([
            index for index, item in enumerate(events)
            if ply_bucket(int(item["ply"])) == bucket
        ])
        for bucket in ("0-15", "16-31", "32-63", "64+")
        if any(ply_bucket(int(item["ply"])) == bucket for item in events)
    }
    return result


def evaluate_drop_queries(model, queries, vocabulary, device, amp_dtype, batch_size, beam_micro_batch_size, progress_every):
    piece_ids = [int(vocabulary[token]) for token in BASIC_PIECE_TOKENS]
    square_ids = [int(vocabulary[token]) for token in square_tokens()]
    drop_id = int(vocabulary[DROP_TOKEN])
    counters = defaultdict(float)
    started = time.perf_counter()
    ordered = sorted(queries, key=lambda item: len(item["prefix_tokens"]))
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            prefix_ids = [[vocabulary[token] for token in item["prefix_tokens"]] for item in batch]
            generated = beam_batch_cached(model, prefix_ids, vocabulary, device, micro_batch_size=beam_micro_batch_size)
            logits, lengths = padded_forward(model, prefix_ids, vocabulary["<PAD>"], device)
            rows = torch.arange(len(batch), device=device)
            first = logits[rows, lengths - 1].float()
            first_allowed = square_ids + [drop_id]
            first_prob = torch.softmax(first[:, first_allowed], dim=-1)
            counters["drop_first_probability_sum"] += float(first_prob[:, -1].sum())

            drop_prefixes = [values + [drop_id] for values in prefix_ids]
            piece_logits, piece_lengths = padded_forward(model, drop_prefixes, vocabulary["<PAD>"], device)
            piece_prob = torch.softmax(piece_logits[rows, piece_lengths - 1].float()[:, piece_ids], dim=-1)
            top_piece_local = piece_prob.argmax(dim=-1).tolist()
            selected_piece_ids = [piece_ids[index] for index in top_piece_local]

            destination_prefixes = [values + [piece_id] for values, piece_id in zip(drop_prefixes, selected_piece_ids)]
            destination_logits, destination_lengths = padded_forward(model, destination_prefixes, vocabulary["<PAD>"], device)
            destination_prob = torch.softmax(destination_logits[rows, destination_lengths - 1].float()[:, square_ids], dim=-1)
            top_destination_local = destination_prob.argmax(dim=-1).tolist()

            for row, item in enumerate(batch):
                counters["queries"] += 1
                side = int(item["side"])
                hand = item["hands"][side * 7 : (side + 1) * 7]
                held = [index for index, count in enumerate(hand) if count > 0]
                counters["held_piece_probability_mass_sum"] += float(piece_prob[row, held].sum()) if held else 0.0
                piece_index = top_piece_local[row]
                piece_token = BASIC_PIECE_TOKENS[piece_index]
                counters["top_piece_in_hand"] += int(hand[piece_index] > 0)
                legal_moves = set(item["legal_moves"])
                legal_destinations = []
                for square_index, square_id in enumerate(square_ids):
                    move = unfactorize_usi([DROP_TOKEN, piece_token, square_tokens()[square_index]])
                    if move in legal_moves:
                        legal_destinations.append(square_index)
                counters["selected_piece_has_legal_destination"] += int(bool(legal_destinations))
                counters["legal_destination_probability_mass_sum"] += float(destination_prob[row, legal_destinations].sum()) if legal_destinations else 0.0
                predicted_drop = unfactorize_usi([DROP_TOKEN, piece_token, square_tokens()[top_destination_local[row]]])
                counters["forced_drop_top1_legal"] += int(predicted_drop in legal_moves)
                counters["forced_drop_top1_target"] += int(predicted_drop == item["target"])
                greedy_move = generated[row][0][0] if generated[row] else None
                counters["free_greedy_is_drop"] += int(greedy_move is not None and "*" in greedy_move)
                counters["free_greedy_legal"] += int(greedy_move in legal_moves)
                counters["free_greedy_target"] += int(greedy_move == item["target"])
            done = start + len(batch)
            if progress_every and done >= progress_every and done // progress_every != start // progress_every:
                print(json.dumps({"event": "drop_evaluation_progress", "queries": done, "total": len(ordered), "elapsed_sec": round(time.perf_counter() - started, 1)}), flush=True)
    n = int(counters["queries"])
    return {
        "queries": n,
        "actual_next_move_is_drop": True,
        "mean_drop_first_probability": _safe_ratio(counters["drop_first_probability_sum"], n),
        "mean_probability_mass_on_held_pieces_given_drop": _safe_ratio(counters["held_piece_probability_mass_sum"], n),
        "top_piece_in_hand_rate_given_drop": _safe_ratio(counters["top_piece_in_hand"], n),
        "selected_piece_has_legal_destination_rate": _safe_ratio(counters["selected_piece_has_legal_destination"], n),
        "mean_legal_destination_mass_for_top_piece": _safe_ratio(counters["legal_destination_probability_mass_sum"], n),
        "forced_drop_top1_legal_rate": _safe_ratio(counters["forced_drop_top1_legal"], n),
        "forced_drop_top1_target_accuracy": _safe_ratio(counters["forced_drop_top1_target"], n),
        "free_greedy_drop_rate": _safe_ratio(counters["free_greedy_is_drop"], n),
        "free_greedy_legal_rate": _safe_ratio(counters["free_greedy_legal"], n),
        "free_greedy_target_accuracy": _safe_ratio(counters["free_greedy_target"], n),
    }


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    vocabulary = load_vocabulary(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    settings = checkpoint.get("new_prompt", {})
    if settings.get("move_encoding") != MOVE_ENCODING or settings.get("terminal_encoding") != TERMINAL_ENCODING:
        raise ValueError("checkpoint is not the current factorized_v3 experiment")
    state_prompt_mode = str(settings.get("state_prompt_mode", "implicit_initial"))
    evaluation_annotation_mode = "ap" if settings.get("annotation_mode") == "ap" else "vanilla"
    config = ModelConfig(**checkpoint["config"])
    if config.vocab_size != len(vocabulary):
        raise ValueError("checkpoint and vocabulary sizes differ")
    probe_artifact = torch.load(args.linear_probes, map_location="cpu")
    available_sources = list(probe_artifact.get("sources", []))
    if args.sources == "available":
        sources = available_sources
    elif args.sources == "final":
        sources = ["layer_{}".format(config.n_layers)]
    else:
        sources = list(dict.fromkeys(value.strip() for value in args.sources.split(",") if value.strip()))
    missing = [source for source in sources if source not in probe_artifact.get("probe_state_dicts", {})]
    if not sources or missing:
        raise ValueError("linear probe artifact lacks requested sources: {}".format(", ".join(missing)))

    device = _resolve_device(args.device)
    amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    model = build_model(str(checkpoint.get("model_type", "vanilla")), config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    events, drops, sampling = read_queries(args, state_prompt_mode, evaluation_annotation_mode, config.max_seq_len)
    if not events:
        raise ValueError("no capture/drop transitions fit the model context")
    features, ordered_events = extract_event_features(model, events, vocabulary, sources, device, amp_dtype, args.batch_size, args.progress_every)
    transition_results = {}
    for source in sources:
        probe = LinearStateProbe(config.d_model).to(device)
        probe.load_state_dict(probe_artifact["probe_state_dicts"][source]); probe.eval()
        with torch.inference_mode():
            before_logits = probe(features[source]["before"].to(device))
            after_logits = probe(features[source]["after"].to(device))
            _, before_hands, _ = predictions_from_logits(before_logits)
            _, after_hands, _ = predictions_from_logits(after_logits)
        transition_results[source] = hand_transition_metrics(
            before_hands.cpu(), after_hands.cpu(), ordered_events,
            probe_artifact.get("hand_names"),
        )
        del probe
    drop_results = evaluate_drop_queries(model, drops, vocabulary, device, amp_dtype, args.batch_size, args.beam_micro_batch_size, args.progress_every) if drops else {"queries": 0, "status": "unavailable"}
    result = {
        "format_version": 1,
        "checkpoint": args.checkpoint,
        "linear_probes": args.linear_probes,
        "evaluation_jsonl": args.evaluation_jsonl,
        "model_type": str(checkpoint.get("model_type", "vanilla")),
        "annotation_mode": str(settings.get("annotation_mode", "vanilla")),
        "evaluation_input_annotation_mode": evaluation_annotation_mode,
        "oracle_piece_conditioned": evaluation_annotation_mode == "ap",
        "amp": amp_name,
        "settings": vars(args),
        "sampling": {**sampling, "sampled_events": len(events), "sampled_actual_drops": len(drops)},
        "hand_transition_probe": transition_results,
        "drop_validity": drop_results,
        "definitions": {
            "hand_transition_probe": "same fitted linear probe is applied immediately before and after a true capture/drop",
            "top_piece_in_hand_rate_given_drop": "top piece under P(piece | prefix, DROP) has positive true hand count",
            "forced_drop_top1_legal_rate": "top destination for the top forced-drop piece is in the cshogi legal move set",
            "ap_caveat": "AP is evaluated on its native history annotations; it is an oracle condition, not a fair no-hint competitor",
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_json(output, result)
    print(json.dumps({"event": "hand_dynamics_evaluation_complete", "output": str(output), "events": len(events), "drops": len(drops)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
