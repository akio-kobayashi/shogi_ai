#!/usr/bin/env python3
"""referenceモデルの駒打ち前後で持ち駒情報の線形復号信頼度を測る。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from data import load_vocabulary
from evaluate_new_prompt_probes import label_maps
from factorized_drop_relevance import (
    action_condition_game_partition, matching_balance, read_positions, rebind_pairs,
    select_anchors_and_controls, selected_keys_for_pairs, trajectory_samples,
)
from factorized_prompt import MOVE_ENCODING, TERMINAL_ENCODING, factorize_history_move
from models import ModelConfig, build_model
from probes import LinearStateProbe
from train_model import amp_context, resolve_amp
from provenance import write_metrics_json


def parse_args():
    parser = argparse.ArgumentParser(description="駒打ち中心の持ち駒信頼度時系列評価")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--linear-probes", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument(
        "--calibration-jsonl",
        help="省略時は--evaluation-jsonlのcalibration区画を使う。"
             "evaluation_stepsを持つsplitだけを指定できる（validation.jsonlは持たない）。",
    )
    parser.add_argument(
        "--partition-seed", type=int, default=20260802,
        help="対局単位60/20/20分割の種。action-condition実験と同じ分割を使う。",
    )
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources", default="available")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--max-drops", type=int, default=5000)
    parser.add_argument("--max-calibration-examples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    return torch.device(value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def resolve_sources(value: str, artifact: dict, n_layers: int) -> list[str]:
    available = list(artifact.get("sources", []))
    if value == "available":
        return available
    aliases = {"middle": n_layers // 2, "late": (3 * n_layers) // 4, "final": n_layers}
    result = []
    for item in value.split(","):
        item = item.strip()
        if item in aliases:
            item = "layer_{}".format(aliases[item])
        if item not in available:
            raise ValueError("probe source is unavailable: {}".format(item))
        if item not in result:
            result.append(item)
    return result


def pad_batch(batch, vocabulary, device):
    lengths = torch.tensor([len(item["prefix_tokens"]) for item in batch], dtype=torch.long)
    width = int(lengths.max())
    ids = torch.full((len(batch), width), int(vocabulary["<PAD>"]), dtype=torch.long)
    for row, item in enumerate(batch):
        ids[row, : len(item["prefix_tokens"])] = torch.tensor(
            [vocabulary[token] for token in item["prefix_tokens"]], dtype=torch.long
        )
    mask = torch.arange(width)[None, :] < lengths[:, None]
    return ids.to(device), mask.to(device), lengths.to(device)


def new_accumulator():
    return {
        "samples": 0, "count_correct": 0, "held_correct": 0,
        "true_count_probability": 0.0, "true_held_probability": 0.0,
        "count_nll": 0.0, "count_brier": 0.0, "entropy": 0.0, "margin": 0.0,
        "confidence_bins": [[0, 0, 0.0] for _ in range(10)],
    }


def add_observation(store, logits: torch.Tensor, target: int, temperature: float = 1.0):
    probability = torch.softmax(logits.float() / float(temperature), dim=-1)
    prediction = int(probability.argmax())
    held_probability = float(1.0 - probability[0])
    true_count_probability = float(probability[target])
    true_held_probability = held_probability if target > 0 else 1.0 - held_probability
    top2 = probability.topk(min(2, probability.numel())).values
    margin = float(top2[0] - top2[1]) if probability.numel() > 1 else 1.0
    entropy = float(-(probability * probability.clamp_min(1e-12).log()).sum())
    one_hot = torch.zeros_like(probability); one_hot[target] = 1.0
    confidence = float(probability.max())
    bin_index = min(int(confidence * 10), 9)
    store["samples"] += 1
    store["count_correct"] += int(prediction == target)
    store["held_correct"] += int((prediction > 0) == (target > 0))
    store["true_count_probability"] += true_count_probability
    store["true_held_probability"] += true_held_probability
    store["count_nll"] += -math.log(max(true_count_probability, 1e-12))
    store["count_brier"] += float((probability - one_hot).square().sum())
    store["entropy"] += entropy
    store["margin"] += margin
    store["confidence_bins"][bin_index][0] += 1
    store["confidence_bins"][bin_index][1] += int(prediction == target)
    store["confidence_bins"][bin_index][2] += confidence
    return true_count_probability, true_held_probability


def finish(store):
    count = int(store["samples"])
    if not count:
        return {"samples": 0}
    ece = 0.0
    bins = []
    for lower, (n, correct, confidence_sum) in enumerate(store["confidence_bins"]):
        if not n:
            continue
        accuracy = correct / n; confidence = confidence_sum / n
        ece += n / count * abs(accuracy - confidence)
        bins.append({"lower": lower / 10, "upper": (lower + 1) / 10, "samples": n,
                     "accuracy": accuracy, "confidence": confidence})
    return {
        "samples": count,
        "count_accuracy": store["count_correct"] / count,
        "held_status_accuracy": store["held_correct"] / count,
        "mean_true_count_probability": store["true_count_probability"] / count,
        "mean_true_held_probability": store["true_held_probability"] / count,
        "count_nll": store["count_nll"] / count,
        "count_brier": store["count_brier"] / count,
        "mean_entropy": store["entropy"] / count,
        "mean_top2_margin": store["margin"] / count,
        "ece_10_bin": ece,
        "calibration_bins": bins,
    }


def calibration_temperatures(model, probes, samples, vocabulary, device, amp_dtype, batch_size):
    grouped = {source: {slot: {"logits": [], "targets": []} for slot in range(14)} for source in probes}
    ordered = sorted(samples, key=lambda item: len(item["prefix_tokens"]))
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            ids, mask, lengths = pad_batch(batch, vocabulary, device)
            output = model(ids, attention_mask=mask, output_hidden_states=True)
            rows = torch.arange(len(batch), device=device); positions = lengths - 1
            for source, probe in probes.items():
                layer = int(source.split("_", 1)[1]); features = output.hidden_states[layer][rows, positions]
                for slot, head in enumerate(probe.hand_heads):
                    grouped[source][slot]["logits"].append(head(features).float().cpu())
                    grouped[source][slot]["targets"].append(torch.tensor([int(item["hands"][slot]) for item in batch]))
    result = {}
    for source, slots in grouped.items():
        values = [
            (torch.cat(item["logits"]), torch.cat(item["targets"]).long())
            for item in slots.values() if item["logits"]
        ]
        log_temperature = torch.tensor(0.0, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=50, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad(); temperature = log_temperature.clamp(-3, 3).exp()
            loss = sum(torch.nn.functional.cross_entropy(logits / temperature, targets, reduction="sum") for logits, targets in values)
            loss = loss / max(sum(targets.numel() for _, targets in values), 1)
            loss.backward(); return loss
        optimizer.step(closure)
        result[source] = float(log_temperature.detach().clamp(-3, 3).exp())
    return result


def verify_prefix_full_consistency(model, samples, sources, vocabulary, device, amp_dtype, max_seq_len, max_samples=8):
    """未来tokenを追加してもh_preが変わらないことを実測する。"""
    selected = []
    used_games = set()
    for item in samples:
        suffix = factorize_history_move(str(item["move"]), {}, "vanilla")
        game_id = str(item["game_id"])
        if len(item["prefix_tokens"]) + len(suffix) <= max_seq_len and game_id not in used_games:
            selected.append((item, suffix)); used_games.add(game_id)
            if len(selected) >= max_samples:
                break
    if not selected:
        raise ValueError("no sample can be used for prefix/full-sequence causal consistency audit")
    tolerance = 2e-3 if amp_dtype is not None else 1e-4
    cases = []; max_logit_error = 0.0; max_hidden_error = 0.0
    for item, suffix in selected:
        full = token_ids_for_audit([*item["prefix_tokens"], *suffix], vocabulary, device)
        prefix_length = len(item["prefix_tokens"])
        query = prefix_length - 1
        # 短いprefixと長いfullを別形状でforwardすると，AMP/SDPAが異なる
        # kernelと丸め経路を選び，因果性とは無関係な差が生じ得る．両者を
        # 同じ形状・同じmaskにし，未来tokenの内容だけをPADへ置換して比較する．
        # query位置から未来位置はcausal maskで遮断されるため，正しい実装なら
        # queryのlogitと隠れ状態は未来tokenの内容に依存しない．
        prefix_padded = full.clone()
        prefix_padded[:, prefix_length:] = int(vocabulary["<PAD>"])
        full_mask = torch.ones_like(full, dtype=torch.bool)
        with torch.inference_mode(), amp_context(device, amp_dtype):
            prefix_output = model(
                prefix_padded, attention_mask=full_mask, output_hidden_states=True
            )
            full_output = model(
                full, attention_mask=full_mask, output_hidden_states=True
            )
        logit_error = float((prefix_output.logits[:, query] - full_output.logits[:, query]).abs().max())
        hidden_error = max(
            float((prefix_output.hidden_states[int(source.split("_", 1)[1])][:, query]
                   - full_output.hidden_states[int(source.split("_", 1)[1])][:, query]).abs().max())
            for source in sources
        )
        max_logit_error = max(max_logit_error, logit_error)
        max_hidden_error = max(max_hidden_error, hidden_error)
        cases.append({"game_id": str(item["game_id"]), "ply": int(item["ply"]),
                      "query_position": query, "future_tokens": len(suffix),
                      "comparison_shape": list(full.shape),
                      "future_token_content_replaced_with_pad": True,
                      "identical_attention_mask": True,
                      "max_absolute_logit_error": logit_error,
                      "max_absolute_hidden_state_error": hidden_error})
    if max(max_logit_error, max_hidden_error) > tolerance:
        raise RuntimeError("prefix/full-sequence causal consistency failed: logits={} hidden={}".format(
            max_logit_error, max_hidden_error))
    return {
        "samples": len(cases), "cases": cases,
        "max_absolute_logit_error": max_logit_error,
        "max_absolute_hidden_state_error": max_hidden_error,
        "tolerance": tolerance, "passed": True,
    }


def token_ids_for_audit(tokens, vocabulary, device):
    return torch.tensor([[vocabulary[token] for token in tokens]], dtype=torch.long, device=device)


def clustered_bootstrap_interval(records, field, seed, repetitions=2000):
    """anchor対局をクラスタとして平均差の95%区間を求める。"""
    by_game = defaultdict(list)
    for record in records:
        by_game[str(record["anchor_game_id"])].append(float(record[field]))
    games = sorted(by_game)
    if not games:
        return None
    if len(games) == 1:
        value = sum(by_game[games[0]]) / len(by_game[games[0]])
        return {"lower": value, "upper": value, "repetitions": 0, "clusters": 1}
    generator = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = [generator.choice(games) for _ in games]
        values = [value for game in sampled for value in by_game[game]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return {"lower": lower, "upper": upper, "repetitions": repetitions, "clusters": len(games)}


def evaluate(model, probes, samples, vocabulary, device, amp_dtype, batch_size, progress_every, temperatures, seed):
    accum = {source: defaultdict(new_accumulator) for source in probes}
    turn_accum = {source: defaultdict(new_accumulator) for source in probes}
    paired = {source: defaultdict(dict) for source in probes}
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
                # 盤面81×29 headはこの評価では不要で，特に全層・長い時系列で
                # 大きな一時tensorになるため，14個の持ち駒headだけを計算する。
                hand_logits = tuple(head(features) for head in probe.hand_heads)
                for row, item in enumerate(batch):
                    slot = int(item["slot"]); target = int(item["target_count"])
                    key = (str(item["group"]), int(item["offset"]))
                    count_p, held_p = add_observation(
                        accum[source][key], hand_logits[slot][row], target,
                        temperatures.get(source, 1.0),
                    )
                    turn_key = (
                        str(item["group"]), int(item["offset"]),
                        "tracked_side_to_move" if item["side_to_move_matches_anchor"] else "opponent_to_move",
                    )
                    add_observation(
                        turn_accum[source][turn_key], hand_logits[slot][row], target,
                        temperatures.get(source, 1.0),
                    )
                    if int(item["offset"]) == 0:
                        paired[source][int(item["pair_index"])][str(item["group"])] = {
                            "count_probability": count_p, "held_probability": held_p,
                            "game_id": str(item["game_id"]),
                        }
            done = start + len(batch)
            if progress_every and done // progress_every != start // progress_every:
                print(json.dumps({"event": "drop_relevance_progress", "samples": done, "total": len(ordered)}), flush=True)
    result = {}
    for source, values in accum.items():
        by_group = {group: {} for group in ("drop", "control")}
        by_turn_relation = {group: {} for group in ("drop", "control")}
        for (group, offset), store in values.items():
            by_group[group][str(offset)] = finish(store)
        for (group, offset, relation), store in turn_accum[source].items():
            by_turn_relation[group].setdefault(str(offset), {})[relation] = finish(store)
        differences = []
        for pair_index, groups in paired[source].items():
            if "drop" in groups and "control" in groups:
                differences.append({
                    "pair_index": pair_index,
                    "anchor_game_id": groups["drop"]["game_id"],
                    "count_probability_difference": groups["drop"]["count_probability"] - groups["control"]["count_probability"],
                    "held_probability_difference": groups["drop"]["held_probability"] - groups["control"]["held_probability"],
                })
        result[source] = {
            "trajectory": by_group,
            "trajectory_by_turn_relation": by_turn_relation,
            "offset_zero_paired": {
                "pairs": len(differences),
                "mean_count_probability_difference": None if not differences else sum(x["count_probability_difference"] for x in differences) / len(differences),
                "mean_held_probability_difference": None if not differences else sum(x["held_probability_difference"] for x in differences) / len(differences),
                "count_probability_difference_95ci": clustered_bootstrap_interval(
                    differences, "count_probability_difference", seed + stable_source_seed(source, 1)
                ),
                "held_probability_difference_95ci": clustered_bootstrap_interval(
                    differences, "held_probability_difference", seed + stable_source_seed(source, 2)
                ),
            },
        }
    return result


def stable_source_seed(source: str, salt: int) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(source)) + 1009 * salt


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    vocabulary = load_vocabulary(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = dict(checkpoint.get("new_prompt", {}))
    if metadata.get("move_encoding") != MOVE_ENCODING or metadata.get("terminal_encoding") != TERMINAL_ENCODING:
        raise ValueError("checkpoint is not the current factorized experiment")
    if metadata.get("state_prompt_mode") != "implicit_initial" or metadata.get("start_selection") != "fixed_initial":
        raise ValueError("drop relevance evaluation requires the implicit fixed-initial experiment")
    annotation_mode = "ap" if metadata.get("annotation_mode") == "ap" else "vanilla"
    if annotation_mode == "ap":
        raise ValueError("AP is an oracle condition and is excluded from drop relevance evaluation")
    config = ModelConfig(**checkpoint["config"])
    model_type = str(checkpoint.get("model_type", "vanilla"))
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    del checkpoint

    artifact = torch.load(args.linear_probes, map_location="cpu")
    _, hand_names = label_maps()
    if list(artifact.get("hand_names", [])) != hand_names:
        raise ValueError("linear probe hand labels do not match the evaluator")
    sources = resolve_sources(args.sources, artifact, config.n_layers)
    if not sources:
        raise ValueError("linear probe artifact contains no requested layer sources")
    probes = {}
    for source in sources:
        probe = LinearStateProbe(config.d_model).to(device)
        probe.load_state_dict(artifact["probe_state_dicts"][source]); probe.eval()
        probes[source] = probe

    print(json.dumps({"event": "drop_relevance_scan_start", "path": args.evaluation_jsonl}), flush=True)
    all_positions, census = read_positions(
        args.evaluation_jsonl, metadata["state_prompt_mode"], annotation_mode,
        hand_names, config.max_seq_len,
    )
    # 校正と本分析は対局単位で分離する。action-condition実験と同じ60/20/20分割を使う。
    # evaluation_stepsを持つのはevaluation splitだけなので，校正もここから取る。
    partitions = defaultdict(list)
    for item in all_positions:
        partitions[action_condition_game_partition(str(item["game_id"]), args.partition_seed)].append(item)
    lightweight_positions = partitions["evaluation"]
    if not lightweight_positions:
        raise ValueError("the evaluation partition contains no positions")
    lightweight_pairs, matching = select_anchors_and_controls(
        lightweight_positions, args.max_drops, args.seed
    )
    selected_keys = selected_keys_for_pairs(lightweight_pairs, args.window)
    print(json.dumps({
        "event": "drop_relevance_matching_complete", "positions": len(lightweight_positions),
        "pairs": len(lightweight_pairs), "materialization_keys": len(selected_keys),
    }), flush=True)
    positions, materialization_census = read_positions(
        args.evaluation_jsonl, metadata["state_prompt_mode"], annotation_mode,
        hand_names, config.max_seq_len, selected_keys=selected_keys, materialize=True,
    )
    pairs = rebind_pairs(lightweight_pairs, positions)
    if not pairs:
        raise ValueError("no matched drop/non-drop pairs were found")
    samples = trajectory_samples(positions, pairs, args.window)
    causal_audit = verify_prefix_full_consistency(
        model, samples, sources, vocabulary, device, amp_dtype, config.max_seq_len,
    )
    temperatures = {source: 1.0 for source in sources}
    calibration = {"enabled": False, "examples": 0}
    calibration_source = args.calibration_jsonl or args.evaluation_jsonl
    if calibration_source:
        print(json.dumps({"event": "drop_relevance_calibration_scan_start",
                          "path": calibration_source,
                          "partition": None if args.calibration_jsonl else "calibration"}), flush=True)
        if args.calibration_jsonl:
            calibration_light, calibration_census = read_positions(
                args.calibration_jsonl, metadata["state_prompt_mode"], annotation_mode,
                hand_names, config.max_seq_len,
            )
        else:
            calibration_light, calibration_census = partitions["calibration"], census
        calibration_light = [item for item in calibration_light if int(item["ply"]) > 0]
        calibration_light.sort(key=lambda item: random.Random(
            "{}:{}:{}".format(args.seed, item["game_id"], item["ply"])
        ).random())
        if args.max_calibration_examples > 0:
            calibration_light = calibration_light[: args.max_calibration_examples]
        if not calibration_light:
            raise ValueError("calibration split contains no eligible non-initial positions")
        calibration_keys = {(str(item["game_id"]), int(item["ply"])) for item in calibration_light}
        calibration_samples, _ = read_positions(
            calibration_source, metadata["state_prompt_mode"], annotation_mode,
            hand_names, config.max_seq_len, selected_keys=calibration_keys, materialize=True,
        )
        temperatures = calibration_temperatures(
            model, probes, calibration_samples, vocabulary, device, amp_dtype, args.batch_size,
        )
        calibration = {
            "enabled": True, "examples": len(calibration_samples),
            "census": calibration_census, "temperature_by_source": temperatures,
        }
        print(json.dumps({"event": "drop_relevance_calibration_complete", "examples": len(calibration_samples)}), flush=True)
    print(json.dumps({"event": "drop_relevance_inference_start", "samples": len(samples)}), flush=True)
    metrics = evaluate(
        model, probes, samples, vocabulary, device, amp_dtype,
        args.batch_size, args.progress_every, temperatures, args.seed,
    )
    analysis_games = {str(item["game_id"]) for item in lightweight_positions}
    calibration_games = (
        set() if args.calibration_jsonl
        else {str(item["game_id"]) for item in partitions["calibration"]}
    )
    split_audit = {
        "partition_seed": args.partition_seed,
        "calibration_source": "external" if args.calibration_jsonl else "evaluation_jsonl_calibration_partition",
        "game_counts": {name: len({str(item["game_id"]) for item in values})
                        for name, values in sorted(partitions.items())},
        "analysis_games": len(analysis_games),
        "calibration_games": len(calibration_games),
        "overlap": sorted(analysis_games & calibration_games),
        "passed": not (analysis_games & calibration_games),
    }
    if not split_audit["passed"]:
        raise ValueError("calibration and analysis share {} game(s)".format(len(split_audit["overlap"])))
    result = {
        "format_version": 2,
        "checkpoint": args.checkpoint,
        "linear_probes": args.linear_probes,
        "evaluation_jsonl": args.evaluation_jsonl,
        "model_type": model_type,
        "annotation_mode": str(metadata.get("annotation_mode", "vanilla")),
        "evaluation_input_annotation_mode": annotation_mode,
        "amp": amp_name,
        "settings": vars(args),
        "census": census,
        "split_audit": split_audit,
        "materialization_census": materialization_census,
        "matching": matching,
        "matching_balance": matching_balance(lightweight_pairs),
        "trajectory_samples": len(samples),
        "causal_alignment_audit": causal_audit,
        "calibration": calibration,
        "metrics": metrics,
        "definitions": {
            "h_pre": "current move's first token is not present; the feature is read at <MOVES> or the previous move's destination token",
            "drop": "actual recorded next move is a drop",
            "control": "different-game normal-move position matched on side, held piece/count, check status, ply, event age and legal-drop count",
            "confidence": "temperature-scaled softmax probability from the frozen fitted linear state probe when calibration_jsonl is provided",
        },
        "limitations": [
            "Calibration uses the calibration partition of --evaluation-jsonl unless --calibration-jsonl is given; the two are disjoint at the game level (see split_audit).",
            "High linear decodability does not by itself prove causal use by the language model.",
            "Matching is deterministic nearest-neighbour matching and does not remove all strategic confounding.",
        ],
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_json(output, result)
    print(json.dumps({"event": "drop_relevance_complete", "output": str(output), "pairs": len(pairs), "samples": len(samples)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
