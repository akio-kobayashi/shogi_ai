#!/usr/bin/env python3
"""同一prefixへ候補行動tokenを与えたときの持ち駒状態表現を評価する。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from data import load_vocabulary
from evaluate_factorized_drop_relevance import (
    add_observation,
    calibration_temperatures,
    clustered_bootstrap_interval,
    finish,
    pad_batch,
    resolve_device,
    resolve_sources,
    stable_source_seed,
)
from evaluate_new_prompt_probes import label_maps
from factorized_drop_relevance import (
    choose_irrelevant_hand_slot,
    choose_normal_branch,
    matching_balance,
    read_positions,
    rebind_pairs,
    select_anchors_and_controls,
    selected_keys_for_pairs,
)
from factorized_prompt import BASIC_PIECE_TOKENS, DROP_TOKEN, MOVE_ENCODING, TERMINAL_ENCODING
from models import ModelConfig, build_model
from probes import LinearStateProbe
from train_model import amp_context, resolve_amp
from provenance import write_metrics_json


def parse_args():
    parser = argparse.ArgumentParser(description="同一prefixのDROP／通常移動分岐による状態表現評価")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--linear-probes", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--calibration-jsonl")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-input-mode", choices=("auto", "native", "no-annotation"), default="auto")
    parser.add_argument("--sources", default="available")
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--max-calibration-examples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def input_protocol(metadata: dict, requested: str) -> dict:
    is_ap = str(metadata.get("annotation_mode", "vanilla")) == "ap"
    if requested == "auto":
        requested = "native" if is_ap else "no-annotation"
    history_mode = "ap" if is_ap and requested == "native" else "vanilla"
    return {
        "requested": requested,
        "history_annotation_mode": history_mode,
        "normal_branch_token": "piece_annotation" if history_mode == "ap" else "source_square",
        "is_ap_checkpoint": is_ap,
        "primary_comparable": not is_ap and history_mode == "vanilla",
        "interpretation": (
            "AP oracle: annotated history and annotated normal-action prefix"
            if is_ap and history_mode == "ap"
            else "AP sensitivity analysis under annotation-free out-of-distribution history"
            if is_ap
            else "primary annotation-free comparison"
        ),
    }


def load_probes(artifact, sources, d_model, device):
    _, hand_names = label_maps()
    if list(artifact.get("hand_names", [])) != hand_names:
        raise ValueError("linear probe hand labels do not match the evaluator")
    probes = {}
    for source in sources:
        probe = LinearStateProbe(d_model).to(device)
        probe.load_state_dict(artifact["probe_state_dicts"][source])
        probe.eval()
        probes[source] = probe
    return probes, hand_names


def branch_samples(pairs, protocol, seed, max_seq_len):
    samples = []
    used_pairs = []
    summary = defaultdict(int)
    for pair_index, pair in enumerate(pairs):
        piece = int(pair["piece"])
        group_items = (("actual_drop", pair["anchor"]), ("actual_normal", pair["control"]))
        selected = []
        for group, item in group_items:
            normal = choose_normal_branch(item, seed, BASIC_PIECE_TOKENS[piece])
            if normal is None or len(item["prefix_tokens"]) + 1 > max_seq_len:
                selected = []
                summary["pair_excluded_without_two_valid_branches"] += 1
                break
            selected.append((group, item, normal))
        if len(selected) != 2:
            continue
        used_pairs.append(pair)
        summary["complete_pairs"] += 1
        for group, item, normal in selected:
            normal_token = normal["piece"] if protocol["normal_branch_token"] == "piece_annotation" else normal["source"]
            relevant_slot = int(item["side"]) * 7 + piece
            irrelevant_slot = choose_irrelevant_hand_slot(item, relevant_slot, seed)
            base = {
                "game_id": str(item["game_id"]),
                "ply": int(item["ply"]),
                "hands": list(item["hands"]),
                "instance_id": "{}:{}".format(pair_index, group),
                "pair_index": pair_index,
                "actual_group": group,
                "tracked_piece": piece,
                "relevant_slot": relevant_slot,
                "irrelevant_slot": irrelevant_slot,
                "normal_branch_piece": normal["piece"],
                "normal_branch_source": normal["source"],
                "normal_branch_token": normal_token,
                "normal_branch_same_piece_as_drop": normal["piece"] == BASIC_PIECE_TOKENS[piece],
            }
            for branch, tokens in (
                ("pre", list(item["prefix_tokens"])),
                ("drop", [*item["prefix_tokens"], DROP_TOKEN]),
                ("normal", [*item["prefix_tokens"], normal_token]),
            ):
                samples.append({**base, "branch": branch, "prefix_tokens": tokens})
            summary["instances"] += 1
            summary["with_irrelevant_nonzero_slot"] += int(irrelevant_slot is not None)
            summary["normal_branch_same_piece_as_drop"] += int(
                normal["piece"] == BASIC_PIECE_TOKENS[piece]
            )
    summary["samples"] = len(samples)
    return samples, dict(summary), used_pairs


def summarize_differences(records, source, seed):
    result = {}
    for group in ("all", "actual_drop", "actual_normal"):
        selected = records if group == "all" else [value for value in records if value["actual_group"] == group]
        entry = {"instances": len(selected)}
        for field in (
            "relevant_count_drop_minus_normal",
            "relevant_held_drop_minus_normal",
            "irrelevant_count_drop_minus_normal",
            "irrelevant_held_drop_minus_normal",
            "selective_count_difference_in_differences",
            "selective_held_difference_in_differences",
        ):
            values = [value for value in selected if value.get(field) is not None]
            entry[field] = {
                "samples": len(values),
                "mean": None if not values else sum(float(value[field]) for value in values) / len(values),
                "clustered_95ci": clustered_bootstrap_interval(
                    values, field, seed + stable_source_seed(source, len(result) + len(field))
                ),
            }
        result[group] = entry
    return result


def evaluate(model, probes, samples, vocabulary, device, amp_dtype, batch_size, progress_every, temperatures, seed):
    accum = {
        source: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        for source in probes
    }
    # new_accumulatorを遅延importせず，最初の観測時に明示的に生成する。
    from evaluate_factorized_drop_relevance import new_accumulator

    raw = {source: defaultdict(dict) for source in probes}
    ordered = sorted(samples, key=lambda item: len(item["prefix_tokens"]))
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            ids, mask, lengths = pad_batch(batch, vocabulary, device)
            output = model(ids, attention_mask=mask, output_hidden_states=True)
            rows = torch.arange(len(batch), device=device)
            positions = lengths - 1
            for source, probe in probes.items():
                layer = int(source.split("_", 1)[1])
                features = output.hidden_states[layer][rows, positions]
                hand_logits = tuple(head(features) for head in probe.hand_heads)
                for row, item in enumerate(batch):
                    for relevance, slot in (
                        ("relevant", item["relevant_slot"]),
                        ("irrelevant", item["irrelevant_slot"]),
                    ):
                        if slot is None:
                            continue
                        slot = int(slot)
                        target = int(item["hands"][slot])
                        branch = str(item["branch"])
                        group = str(item["actual_group"])
                        store = accum[source][group][branch].get(relevance)
                        if store is None:
                            store = new_accumulator()
                            accum[source][group][branch][relevance] = store
                        count_probability, held_probability = add_observation(
                            store, hand_logits[slot][row], target, temperatures.get(source, 1.0)
                        )
                        raw[source][str(item["instance_id"])][(branch, relevance)] = {
                            "count_probability": count_probability,
                            "held_probability": held_probability,
                            "game_id": str(item["game_id"]),
                            "actual_group": group,
                        }
            done = start + len(batch)
            if progress_every and done // progress_every != start // progress_every:
                print(json.dumps({"event": "action_condition_progress", "samples": done, "total": len(ordered)}), flush=True)

    result = {}
    for source in probes:
        aggregate = {}
        for group, branches in accum[source].items():
            aggregate[group] = {
                branch: {relevance: finish(store) for relevance, store in relevance_values.items() if store is not None}
                for branch, relevance_values in branches.items()
            }
        records = []
        for instance_id, values in raw[source].items():
            relevant_drop = values.get(("drop", "relevant"))
            relevant_normal = values.get(("normal", "relevant"))
            if relevant_drop is None or relevant_normal is None:
                continue
            irrelevant_drop = values.get(("drop", "irrelevant"))
            irrelevant_normal = values.get(("normal", "irrelevant"))
            record = {
                "instance_id": instance_id,
                "anchor_game_id": relevant_drop["game_id"],
                "actual_group": relevant_drop["actual_group"],
                "relevant_count_drop_minus_normal": relevant_drop["count_probability"] - relevant_normal["count_probability"],
                "relevant_held_drop_minus_normal": relevant_drop["held_probability"] - relevant_normal["held_probability"],
                "irrelevant_count_drop_minus_normal": None,
                "irrelevant_held_drop_minus_normal": None,
                "selective_count_difference_in_differences": None,
                "selective_held_difference_in_differences": None,
            }
            if irrelevant_drop is not None and irrelevant_normal is not None:
                record["irrelevant_count_drop_minus_normal"] = irrelevant_drop["count_probability"] - irrelevant_normal["count_probability"]
                record["irrelevant_held_drop_minus_normal"] = irrelevant_drop["held_probability"] - irrelevant_normal["held_probability"]
                record["selective_count_difference_in_differences"] = (
                    record["relevant_count_drop_minus_normal"] - record["irrelevant_count_drop_minus_normal"]
                )
                record["selective_held_difference_in_differences"] = (
                    record["relevant_held_drop_minus_normal"] - record["irrelevant_held_drop_minus_normal"]
                )
            records.append(record)
        result[source] = {
            "branch_metrics": aggregate,
            "within_prefix_contrasts": summarize_differences(records, source, seed),
        }
    return result


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    vocabulary = load_vocabulary(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = dict(checkpoint.get("new_prompt", {}))
    if metadata.get("move_encoding") != MOVE_ENCODING or metadata.get("terminal_encoding") != TERMINAL_ENCODING:
        raise ValueError("checkpoint is not the current factorized experiment")
    if metadata.get("state_prompt_mode") != "implicit_initial" or metadata.get("start_selection") != "fixed_initial":
        raise ValueError("action-condition evaluation requires the implicit fixed-initial experiment")
    protocol = input_protocol(metadata, args.evaluation_input_mode)
    config = ModelConfig(**checkpoint["config"])
    model_type = str(checkpoint.get("model_type", "vanilla"))
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    del checkpoint

    artifact = torch.load(args.linear_probes, map_location="cpu")
    sources = resolve_sources(args.sources, artifact, config.n_layers)
    if not sources:
        raise ValueError("linear probe artifact contains no requested layer sources")
    probes, hand_names = load_probes(artifact, sources, config.d_model, device)
    del artifact

    print(json.dumps({"event": "action_condition_scan_start", "protocol": protocol}), flush=True)
    lightweight, census = read_positions(
        args.evaluation_jsonl, metadata["state_prompt_mode"], protocol["history_annotation_mode"],
        hand_names, config.max_seq_len - 1,
    )
    lightweight_pairs, matching = select_anchors_and_controls(lightweight, args.max_pairs, args.seed)
    keys = selected_keys_for_pairs(lightweight_pairs)
    materialized, materialization_census = read_positions(
        args.evaluation_jsonl, metadata["state_prompt_mode"], protocol["history_annotation_mode"],
        hand_names, config.max_seq_len - 1, selected_keys=keys, materialize=True,
    )
    pairs = rebind_pairs(lightweight_pairs, materialized)
    samples, branch_summary, used_pairs = branch_samples(
        pairs, protocol, args.seed, config.max_seq_len
    )
    if not samples:
        raise ValueError("no same-prefix DROP/normal branches were constructed")
    print(json.dumps({
        "event": "action_condition_branches_complete", "pairs": len(pairs),
        "instances": branch_summary.get("instances", 0), "samples": len(samples),
    }), flush=True)

    temperatures = {source: 1.0 for source in sources}
    calibration = {"enabled": False, "examples": 0}
    if args.calibration_jsonl:
        calibration_light, calibration_census = read_positions(
            args.calibration_jsonl, metadata["state_prompt_mode"], protocol["history_annotation_mode"],
            hand_names, config.max_seq_len,
        )
        calibration_light = [item for item in calibration_light if int(item["ply"]) > 0]
        calibration_light.sort(key=lambda item: random.Random(
            "{}:{}:{}".format(args.seed, item["game_id"], item["ply"])
        ).random())
        if args.max_calibration_examples > 0:
            calibration_light = calibration_light[: args.max_calibration_examples]
        calibration_keys = {(str(item["game_id"]), int(item["ply"])) for item in calibration_light}
        calibration_samples, _ = read_positions(
            args.calibration_jsonl, metadata["state_prompt_mode"], protocol["history_annotation_mode"],
            hand_names, config.max_seq_len, selected_keys=calibration_keys, materialize=True,
        )
        if not calibration_samples:
            raise ValueError("calibration split contains no eligible positions")
        temperatures = calibration_temperatures(
            model, probes, calibration_samples, vocabulary, device, amp_dtype, args.batch_size
        )
        calibration = {
            "enabled": True, "examples": len(calibration_samples),
            "census": calibration_census, "temperature_by_source": temperatures,
        }

    metrics = evaluate(
        model, probes, samples, vocabulary, device, amp_dtype, args.batch_size,
        args.progress_every, temperatures, args.seed,
    )
    result = {
        "format_version": 1,
        "checkpoint": args.checkpoint,
        "linear_probes": args.linear_probes,
        "evaluation_jsonl": args.evaluation_jsonl,
        "model_type": model_type,
        "checkpoint_annotation_mode": str(metadata.get("annotation_mode", "vanilla")),
        "protocol": protocol,
        "amp": amp_name,
        "settings": vars(args),
        "census": census,
        "materialization_census": materialization_census,
        "matching": matching,
        "matching_balance_before_branch_filter": matching_balance(lightweight_pairs),
        "matching_balance": matching_balance(used_pairs),
        "branch_summary": branch_summary,
        "calibration": calibration,
        "metrics": metrics,
        "primary_estimand": (
            "[(relevant true-count probability after <DROP>) - (after normal-action prefix)] "
            "- [(irrelevant held-piece probability after <DROP>) - (after normal-action prefix)]"
        ),
        "definitions": {
            "pre": "same complete history before the current action's first token",
            "drop": "the same prefix followed by exactly one <DROP> token",
            "normal": "the same prefix followed by exactly one legal normal-action prefix token",
            "actual_drop": "the recorded next move was a drop",
            "actual_normal": "a different-game matched position whose recorded next move was a normal move",
        },
        "limitations": [
            "The intervention supplies an action-prefix token; it does not prove that the unprompted model internally selected that action.",
            "Linear-probe confidence measures decodability through a fitted observer, not the model's own subjective confidence.",
            "AP-native and AP-no-annotation are oracle and distribution-shift analyses and are not pooled with the primary conditions.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_json(output, result)
    print(json.dumps({
        "event": "action_condition_complete", "output": str(output),
        "instances": branch_summary.get("instances", 0), "protocol": protocol["interpretation"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
