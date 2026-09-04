#!/usr/bin/env python3
"""同一prefix行動条件実験の位置頑健性と行動予測を検証する。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from data import load_vocabulary
from evaluate_factorized_action_condition import input_protocol
from evaluate_factorized_drop_relevance import (
    clustered_bootstrap_interval,
    pad_batch,
    resolve_device,
    resolve_sources,
    verify_prefix_full_consistency,
)
from evaluate_new_prompt_probes import label_maps
from factorized_drop_relevance import (
    action_condition_game_partition,
    choose_irrelevant_hand_slot,
    read_positions,
    rebind_pairs,
    select_anchors_and_controls,
    selected_keys_for_pairs,
    stable_number,
)
from factorized_prompt import BASIC_PIECE_TOKENS, DROP_TOKEN, MOVE_ENCODING, TERMINAL_ENCODING
from models import ModelConfig, build_model
from probes import HAND_MAX_COUNTS
from train_model import amp_context, resolve_amp
from provenance import with_provenance, write_metrics_json


BRANCHES = ("pre", "drop", "normal")
PROBE_FAMILIES = (*BRANCHES, "pooled")


class HandProbe(nn.Module):
    """盤面headを持たない，持ち駒14項目専用の線形プローブ。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(d_model, maximum + 1) for maximum in HAND_MAX_COUNTS)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(head(features) for head in self.heads)


def parse_args():
    parser = argparse.ArgumentParser(description="同一prefix行動条件実験の頑健性検証")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--linear-probes", required=True, help="source層の定義と因果監査に用いる既存artifact")
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-output", required=True)
    parser.add_argument("--evaluation-input-mode", choices=("auto", "native", "no-annotation"), default="auto")
    parser.add_argument("--sources", default="middle,late,final")
    parser.add_argument("--max-probe-pairs", type=int, default=1500)
    parser.add_argument("--max-calibration-pairs", type=int, default=500)
    parser.add_argument("--max-evaluation-pairs", type=int, default=2000)
    parser.add_argument("--normal-branches", type=int, default=3)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-patience", type=int, default=3)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def split_positions(positions: list[dict], seed: int) -> dict[str, list[dict]]:
    result = {name: [] for name in ("probe_train", "calibration", "evaluation")}
    for item in positions:
        result[action_condition_game_partition(str(item["game_id"]), seed)].append(item)
    return result


def select_normal_branches(item: dict, count: int, seed: int, avoid_piece: str) -> list[dict]:
    values = [dict(value) for value in item.get("normal_branches", [])]
    values.sort(key=lambda value: (
        value["piece"] == avoid_piece,
        stable_number(seed, item["game_id"], item["ply"], value["piece"], value["source"]),
    ))
    result = []
    seen = set()
    for value in values:
        key = (value["piece"], value["source"])
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= count:
            break
    return result


def make_samples(pairs: list[dict], protocol: dict, normal_count: int, max_seq_len: int, seed: int):
    samples = []
    used_pairs = []
    excluded = 0
    for pair_index, pair in enumerate(pairs):
        piece = int(pair["piece"])
        staged = []
        for actual_group, item in (("actual_drop", pair["anchor"]), ("actual_normal", pair["control"])):
            normals = select_normal_branches(item, normal_count, seed, BASIC_PIECE_TOKENS[piece])
            if not normals or len(item["prefix_tokens"]) + 1 > max_seq_len:
                staged = []
                excluded += 1
                break
            staged.append((actual_group, item, normals))
        if not staged:
            continue
        used_pairs.append(pair)
        for actual_group, item, normals in staged:
            relevant_slot = int(item["side"]) * 7 + piece
            irrelevant_slot = choose_irrelevant_hand_slot(item, relevant_slot, seed)
            base = {
                "game_id": str(item["game_id"]), "ply": int(item["ply"]),
                "instance_id": "{}:{}".format(pair_index, actual_group),
                "actual_group": actual_group, "tracked_piece": piece,
                "relevant_slot": relevant_slot, "irrelevant_slot": irrelevant_slot,
                "hands": list(item["hands"]), "move": str(item["move"]),
            }
            samples.append({**base, "branch": "pre", "branch_index": 0,
                            "prefix_tokens": list(item["prefix_tokens"])})
            samples.append({**base, "branch": "drop", "branch_index": 0,
                            "prefix_tokens": [*item["prefix_tokens"], DROP_TOKEN]})
            for index, normal in enumerate(normals):
                token = normal["piece"] if protocol["normal_branch_token"] == "piece_annotation" else normal["source"]
                samples.append({**base, "branch": "normal", "branch_index": index,
                                "normal_piece": normal["piece"], "normal_source": normal["source"],
                                "prefix_tokens": [*item["prefix_tokens"], token]})
    return samples, used_pairs, {"excluded_pairs": excluded, "samples": len(samples), "pairs": len(used_pairs)}


def balanced_family_indices(branches: list[str], family: str, seed: int) -> torch.Tensor:
    by_branch = {branch: [i for i, value in enumerate(branches) if value == branch] for branch in BRANCHES}
    if family != "pooled":
        return torch.tensor(by_branch[family], dtype=torch.long)
    size = min(len(values) for values in by_branch.values())
    selected = []
    for branch, values in by_branch.items():
        values = sorted(values, key=lambda index: stable_number(seed, branch, index))
        selected.extend(values[:size])
    return torch.tensor(selected, dtype=torch.long)


def extract_source(model, samples, source: str, vocabulary, device, amp_dtype, batch_size, piece_ids):
    features, hands, relevant_slots, metadata = [], [], [], []
    behavior = defaultdict(lambda: {"samples": 0, "correct": 0, "probability": 0.0, "rank": 0.0})
    layer = int(source.split("_", 1)[1])
    ordered = sorted(samples, key=lambda item: len(item["prefix_tokens"]))
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start:start + batch_size]
            ids, mask, lengths = pad_batch(batch, vocabulary, device)
            output = model(ids, attention_mask=mask, output_hidden_states=True)
            rows = torch.arange(len(batch), device=device); positions = lengths - 1
            features.append(output.hidden_states[layer][rows, positions].float().cpu())
            hands.append(torch.tensor([item["hands"] for item in batch], dtype=torch.long))
            relevant_slots.extend(int(item["relevant_slot"]) for item in batch)
            metadata.extend({key: item[key] for key in (
                "game_id", "instance_id", "actual_group", "branch", "branch_index",
                "tracked_piece", "irrelevant_slot",
            )} for item in batch)
            restricted = output.logits[rows, positions][:, piece_ids].float().softmax(dim=-1)
            for row, item in enumerate(batch):
                if item["branch"] != "drop":
                    continue
                target = int(item["tracked_piece"])
                rank = int((restricted[row] > restricted[row, target]).sum()) + 1
                store = behavior[str(item["actual_group"])]
                store["samples"] += 1
                store["correct"] += int(rank == 1)
                store["probability"] += float(restricted[row, target])
                store["rank"] += rank
    return {
        "features": torch.cat(features), "hands": torch.cat(hands),
        "relevant_slots": torch.tensor(relevant_slots), "metadata": metadata,
        "branches": [item["branch"] for item in metadata],
        "behavior": {
            group: {
                "samples": value["samples"],
                "piece_top1_accuracy_after_drop": value["correct"] / value["samples"],
                "mean_correct_piece_probability_after_drop": value["probability"] / value["samples"],
                "mean_correct_piece_rank_after_drop": value["rank"] / value["samples"],
            } for group, value in behavior.items() if value["samples"]
        },
    }


def hand_loss(probe: HandProbe, features: torch.Tensor, hands: torch.Tensor) -> torch.Tensor:
    logits = probe(features)
    return torch.stack([
        nn.functional.cross_entropy(slot_logits, hands[:, slot])
        for slot, slot_logits in enumerate(logits)
    ]).mean()


def validation_loss(probe, data, indices, batch_size, device):
    total = 0.0; samples = 0
    probe.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            loss = hand_loss(probe, data["features"][selected].to(device), data["hands"][selected].to(device))
            total += float(loss) * len(selected); samples += len(selected)
    return total / max(samples, 1)


def train_probe(train, calibration, family, d_model, args, device, source_seed):
    train_indices = balanced_family_indices(train["branches"], family, source_seed)
    calibration_indices = balanced_family_indices(calibration["branches"], family, source_seed + 1)
    if not len(train_indices) or not len(calibration_indices):
        raise ValueError("empty branch-probe split for {}".format(family))
    torch.manual_seed(source_seed)
    probe = HandProbe(d_model).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_learning_rate,
                                  weight_decay=args.probe_weight_decay)
    generator = torch.Generator().manual_seed(source_seed)
    best_state = None; best_loss = float("inf"); wait = 0; history = []
    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = hand_loss(probe, train["features"][selected].to(device), train["hands"][selected].to(device))
            loss.backward(); optimizer.step()
        current = validation_loss(probe, calibration, calibration_indices, args.batch_size, device)
        history.append({"epoch": epoch, "validation_loss": current})
        if current < best_loss - 1e-7:
            best_loss = current; best_state = copy.deepcopy(probe.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= args.probe_patience:
                break
    probe.load_state_dict(best_state); probe.eval()
    return probe, {"best_validation_loss": best_loss, "best_epoch": history[-1]["epoch"] - wait,
                   "train_samples": len(train_indices), "calibration_samples": len(calibration_indices),
                   "history": history}


def fit_temperature(probe, calibration, indices, batch_size, device):
    logits_by_slot = [[] for _ in range(14)]; targets = [[] for _ in range(14)]
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            values = probe(calibration["features"][selected].to(device))
            for slot in range(14):
                logits_by_slot[slot].append(values[slot].float().cpu())
                targets[slot].append(calibration["hands"][selected, slot])
    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=50, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(); temperature = log_temperature.clamp(-3, 3).exp()
        loss = sum(nn.functional.cross_entropy(torch.cat(logits_by_slot[s]) / temperature,
                                               torch.cat(targets[s]), reduction="sum") for s in range(14))
        loss = loss / max(sum(len(torch.cat(targets[s])) for s in range(14)), 1)
        loss.backward(); return loss
    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-3, 3).exp())


def evaluate_probe(probe, data, branch, temperature, batch_size, device):
    indices = torch.tensor([i for i, value in enumerate(data["branches"]) if value == branch], dtype=torch.long)
    count_correct = held_correct = total_slots = 0
    relevant_correct = relevant_held = relevant_n = 0
    relevant_probability = relevant_nll = 0.0
    records = defaultdict(lambda: {"drop": None, "normals": []})
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            logits = probe(data["features"][selected].to(device))
            for local, data_index in enumerate(selected.tolist()):
                targets = data["hands"][data_index]
                for slot, slot_logits in enumerate(logits):
                    prediction = int(slot_logits[local].argmax())
                    target = int(targets[slot])
                    count_correct += int(prediction == target)
                    held_correct += int((prediction > 0) == (target > 0)); total_slots += 1
                slot = int(data["relevant_slots"][data_index]); target = int(targets[slot])
                probability = torch.softmax(logits[slot][local].float() / temperature, dim=-1)
                prediction = int(probability.argmax())
                relevant_correct += int(prediction == target)
                relevant_held += int((prediction > 0) == (target > 0)); relevant_n += 1
                p = float(probability[target]); relevant_probability += p
                relevant_nll += -math.log(max(p, 1e-12))
                meta = data["metadata"][data_index]
                if branch == "drop": records[meta["instance_id"]]["drop"] = p
                elif branch == "normal": records[meta["instance_id"]]["normals"].append(p)
    return {
        "samples": len(indices), "hand_slot_count_accuracy": count_correct / max(total_slots, 1),
        "hand_slot_held_accuracy": held_correct / max(total_slots, 1),
        "relevant_count_accuracy": relevant_correct / max(relevant_n, 1),
        "relevant_held_accuracy": relevant_held / max(relevant_n, 1),
        "mean_relevant_true_count_probability": relevant_probability / max(relevant_n, 1),
        "relevant_count_nll": relevant_nll / max(relevant_n, 1),
    }, records


def robustness_contrast(drop_records, normal_records, metadata, seed):
    rows = []
    meta_by_instance = {item["instance_id"]: item for item in metadata}
    for instance_id in set(drop_records) & set(normal_records):
        drop = drop_records[instance_id]["drop"]
        normals = normal_records[instance_id]["normals"]
        if drop is None or not normals:
            continue
        mean_normal = sum(normals) / len(normals)
        variance = sum((value - mean_normal) ** 2 for value in normals) / len(normals)
        meta = meta_by_instance[instance_id]
        rows.append({"anchor_game_id": meta["game_id"], "difference": drop - mean_normal,
                     "normal_sd": math.sqrt(variance), "normal_range": max(normals) - min(normals)})
    result = {"instances": len(rows)}
    for field in ("difference", "normal_sd", "normal_range"):
        result[field] = {
            "mean": None if not rows else sum(row[field] for row in rows) / len(rows),
            "clustered_95ci": clustered_bootstrap_interval(rows, field, seed),
        }
    return result


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = resolve_device(args.device); amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    vocabulary = load_vocabulary(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = dict(checkpoint.get("new_prompt", {}))
    if metadata.get("move_encoding") != MOVE_ENCODING or metadata.get("terminal_encoding") != TERMINAL_ENCODING:
        raise ValueError("checkpoint is not the current factorized experiment")
    protocol = input_protocol(metadata, args.evaluation_input_mode)
    config = ModelConfig(**checkpoint["config"]); model_type = str(checkpoint.get("model_type", "vanilla"))
    model = build_model(model_type, config).to(device); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    del checkpoint
    artifact = torch.load(args.linear_probes, map_location="cpu")
    sources = resolve_sources(args.sources, artifact, config.n_layers)
    _, hand_names = label_maps()
    if not sources: raise ValueError("no requested probe sources")

    lightweight, census = read_positions(args.evaluation_jsonl, metadata["state_prompt_mode"],
                                          protocol["history_annotation_mode"], hand_names,
                                          config.max_seq_len - 1)
    partitions = split_positions(lightweight, args.seed)
    maxima = {"probe_train": args.max_probe_pairs, "calibration": args.max_calibration_pairs,
              "evaluation": args.max_evaluation_pairs}
    light_pairs = {}; selected = set(); matching = {}
    for index, name in enumerate(("probe_train", "calibration", "evaluation")):
        light_pairs[name], matching[name] = select_anchors_and_controls(
            partitions[name], maxima[name], args.seed + index)
        selected.update(selected_keys_for_pairs(light_pairs[name]))
    materialized, materialization_census = read_positions(
        args.evaluation_jsonl, metadata["state_prompt_mode"], protocol["history_annotation_mode"],
        hand_names, config.max_seq_len - 1, selected_keys=selected, materialize=True)
    samples = {}; branch_summary = {}; used_pairs = {}
    for index, name in enumerate(("probe_train", "calibration", "evaluation")):
        rebound = rebind_pairs(light_pairs[name], materialized)
        samples[name], used_pairs[name], branch_summary[name] = make_samples(
            rebound, protocol, args.normal_branches, config.max_seq_len, args.seed + index)
        if not samples[name]: raise ValueError("{} contains no complete branch samples".format(name))

    causal_audit = verify_prefix_full_consistency(
        model, [item for item in samples["evaluation"] if item["branch"] == "pre"],
        sources, vocabulary, device, amp_dtype, config.max_seq_len)
    piece_ids = [int(vocabulary[token]) for token in BASIC_PIECE_TOKENS]
    all_metrics = {}; saved = {}
    for source_index, source in enumerate(sources):
        print(json.dumps({"event": "robustness_source_start", "source": source}), flush=True)
        extracted = {
            name: extract_source(model, samples[name], source, vocabulary, device, amp_dtype,
                                 args.batch_size, piece_ids)
            for name in ("probe_train", "calibration", "evaluation")
        }
        source_metrics = {"cross_position_generalization": {},
                          "behavior_after_drop": extracted["evaluation"]["behavior"]}
        saved[source] = {}
        pooled_records = None
        for family in PROBE_FAMILIES:
            family_seed = args.seed + source_index * 101 + PROBE_FAMILIES.index(family)
            probe, training = train_probe(extracted["probe_train"], extracted["calibration"],
                                          family, config.d_model, args, device, family_seed)
            calibration_indices = balanced_family_indices(extracted["calibration"]["branches"],
                                                           "pooled", family_seed)
            temperature = fit_temperature(probe, extracted["calibration"], calibration_indices,
                                          args.batch_size, device)
            matrix = {}; branch_records = {}
            for branch in BRANCHES:
                matrix[branch], branch_records[branch] = evaluate_probe(
                    probe, extracted["evaluation"], branch, temperature, args.batch_size, device)
            source_metrics["cross_position_generalization"][family] = {
                "training": training, "temperature": temperature, "tested_at": matrix}
            saved[source][family] = {key: value.detach().cpu() for key, value in probe.state_dict().items()}
            if family == "pooled": pooled_records = branch_records
        source_metrics["pooled_probe_within_prefix"] = robustness_contrast(
            pooled_records["drop"], pooled_records["normal"],
            extracted["evaluation"]["metadata"], args.seed + source_index)
        all_metrics[source] = source_metrics
        del extracted

    split_games = {name: sorted({str(item["game_id"]) for item in values}) for name, values in partitions.items()}
    overlap = {
        "train_calibration": len(set(split_games["probe_train"]) & set(split_games["calibration"])),
        "train_evaluation": len(set(split_games["probe_train"]) & set(split_games["evaluation"])),
        "calibration_evaluation": len(set(split_games["calibration"]) & set(split_games["evaluation"])),
    }
    if any(overlap.values()): raise AssertionError("game-disjoint split failed: {}".format(overlap))
    result = {
        "format_version": 2, "checkpoint": args.checkpoint, "model_type": model_type,
        "protocol": protocol, "settings": vars(args), "amp": amp_name,
        "census": census, "materialization_census": materialization_census,
        "split_audit": {"game_counts": {key: len(value) for key, value in split_games.items()},
                        "game_overlap_counts": overlap, "passed": True},
        "matching": matching, "branch_summary": branch_summary,
        "causal_prefix_full_audit": causal_audit, "metrics": all_metrics,
        "primary_result": "pooled branch-balanced probe: DROP minus mean of multiple legal normal branches",
        "interpretation_limits": [
            "The pooled probe removes train/test branch-position mismatch but remains a decodability measure.",
            "The cross-position matrix diagnoses token-position-specific rotations of the representation.",
            "The behavioral piece probability is the model output after <DROP>, not a probe output.",
            "Game-cluster bootstrap does not replace variation across independently trained model seeds.",
        ],
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_json(output, result)
    probe_output = Path(args.probe_output); probe_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(with_provenance({"format_version": 1, "checkpoint": args.checkpoint, "sources": sources,
                                "probe_families": PROBE_FAMILIES, "state_dicts": saved}), probe_output)
    print(json.dumps({"event": "action_condition_robustness_complete", "output": str(output)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
