#!/usr/bin/env python3
"""駒打ちに関係する履歴attentionとattention接続遮断を評価する。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch

from data import load_vocabulary
from evaluate_new_prompt_probes import label_maps
from factorized_drop_relevance import (
    matching_balance, read_positions, rebind_pairs,
    relevant_and_control_markers,
    select_anchors_and_controls,
    selected_keys_for_pairs,
)
from factorized_prompt import BASIC_PIECE_TOKENS, DROP_TOKEN, MOVE_ENCODING, TERMINAL_ENCODING
from models import ModelConfig, build_model
from train_model import amp_context, resolve_amp


def parse_args():
    parser = argparse.ArgumentParser(description="駒打ち関連attention観測・遮断評価")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--max-ablation-pairs", type=int, default=250)
    parser.add_argument("--ablation-layers", default="middle,late,all")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="off")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    return torch.device(value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def token_ids(tokens: Sequence[str], vocabulary: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([[vocabulary[token] for token in tokens]], dtype=torch.long, device=device)


def selected_attention_rows(model, input_ids: torch.Tensor, query_position: int):
    """通常forwardを保ったまま，各層の選択query attentionだけを返す。"""
    x = model._embed(input_ids)
    rotary = model._rotary(x.device, x.dtype, 0, x.shape[1])
    rows = []
    for layer in model.layers:
        normalized = layer.attn_norm(x)
        q, k, _ = layer.attn._project(normalized, 0, rotary)
        q = q[:, :, query_position : query_position + 1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(layer.attn.head_dim)
        allowed = torch.arange(x.shape[1], device=x.device) <= query_position
        scores = scores.masked_fill(~allowed[None, None, None, :], -torch.inf)
        rows.append(torch.softmax(scores.float(), dim=-1)[0, :, 0].cpu())
        x = layer(x, rotary=rotary)
    logits = model.lm_head(model.final_norm(x))
    return rows, logits


def forward_with_edge_ablation(
    model,
    input_ids: torch.Tensor,
    query_position: int,
    key_positions: Iterable[int],
    layer_indices: set[int],
    head_indices: set[int] | None = None,
) -> torch.Tensor:
    """指定query→key接続だけを遮断したLLaMA forward。

    遮断しないtoken位置は通常forwardと同じである。直接接続のみを対象とし，
    過去tokenへ既に伝播した間接経路は残す。
    """
    keys = sorted({int(value) for value in key_positions if 0 <= int(value) <= query_position})
    x = model._embed(input_ids)
    rotary = model._rotary(x.device, x.dtype, 0, x.shape[1])
    for layer_index, layer in enumerate(model.layers):
        if layer_index not in layer_indices or not keys:
            x = layer(x, rotary=rotary)
            continue
        normalized = layer.attn_norm(x)
        normal_attention = layer.attn(normalized, rotary=rotary)
        q, k, v = layer.attn._project(normalized, 0, rotary)
        q_row = q[:, :, query_position : query_position + 1]
        scores = torch.matmul(q_row, k.transpose(-2, -1)) / math.sqrt(layer.attn.head_dim)
        allowed = torch.arange(x.shape[1], device=x.device) <= query_position
        scores = scores.masked_fill(~allowed[None, None, None, :], -torch.inf)
        heads = range(layer.attn.n_heads) if head_indices is None else head_indices
        key_mask = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        key_mask[keys] = True
        for head in heads:
            scores[:, int(head), :, key_mask] = -torch.inf
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        query_output = torch.matmul(weights, v)
        query_output = query_output.transpose(1, 2).contiguous().view(1, 1, layer.attn.d_model)
        query_output = layer.attn.resid_dropout(layer.attn.out_proj(query_output))
        attention = normal_attention.clone()
        attention[:, query_position : query_position + 1] = query_output
        x = x + attention
        x = x + layer.ffn(layer.ffn_norm(x))
    return model.lm_head(model.final_norm(x))


def resolve_layer_sets(value: str, n_layers: int) -> dict[str, set[int]]:
    result = {}
    aliases = {
        "early": {max(n_layers // 4, 0)},
        "middle": {max(n_layers // 2, 0)},
        "late": {max((3 * n_layers) // 4, 0)},
        "penultimate": {max(n_layers - 2, 0)},
        "all": set(range(n_layers)),
    }
    for item in value.split(","):
        item = item.strip()
        if item in aliases:
            result[item] = aliases[item]
        elif item.startswith("layer_"):
            index = int(item.split("_", 1)[1])
            if not 0 <= index < n_layers:
                raise ValueError("ablation layer is out of range: {}".format(item))
            result[item] = {index}
        else:
            raise ValueError("unknown ablation layer selection: {}".format(item))
    return result


def restricted_metrics(logits: torch.Tensor, allowed_ids: Sequence[int], target_id: int) -> dict:
    values = logits[0, -1, list(allowed_ids)].float()
    probability = torch.softmax(values, dim=-1)
    target_local = list(allowed_ids).index(int(target_id))
    rank = int((values > values[target_local]).sum()) + 1
    return {
        "target_log_probability": float(torch.log_softmax(values, dim=-1)[target_local]),
        "target_probability": float(probability[target_local]),
        "target_rank": rank,
    }


def add_attention(accum, rows, relevant, control, group, query_name):
    if not relevant or not control:
        return
    for layer, weights in enumerate(rows):
        for head in range(weights.shape[0]):
            key = (group, query_name, layer, head)
            accum[key]["examples"] += 1
            accum[key]["relevant_mass"] += float(weights[head, relevant].sum())
            accum[key]["control_mass"] += float(weights[head, control].sum())


def finish_attention(accum):
    result = defaultdict(lambda: defaultdict(dict))
    for (group, query, layer, head), values in accum.items():
        n = int(values["examples"])
        relevant = values["relevant_mass"] / n
        control = values["control_mass"] / n
        result[group][query]["layer_{}_head_{}".format(layer, head)] = {
            "examples": n,
            "mean_relevant_mass": relevant,
            "mean_distance_matched_control_mass": control,
            "mass_difference": relevant - control,
            "enrichment_ratio": relevant / max(control, 1e-12),
        }
    return {group: dict(queries) for group, queries in result.items()}


def add_ablation(accum, name, baseline, masked):
    key = name
    accum[key]["examples"] += 1
    accum[key]["baseline_log_probability"] += baseline["target_log_probability"]
    accum[key]["masked_log_probability"] += masked["target_log_probability"]
    accum[key]["log_probability_change"] += masked["target_log_probability"] - baseline["target_log_probability"]
    accum[key]["baseline_probability"] += baseline["target_probability"]
    accum[key]["masked_probability"] += masked["target_probability"]


def finish_ablation(accum):
    return {
        name: {
            "examples": int(values["examples"]),
            **{
                key: value / values["examples"]
                for key, value in values.items() if key != "examples"
            },
        }
        for name, values in accum.items() if values["examples"]
    }


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
        raise ValueError("attention evaluation requires the implicit fixed-initial experiment")
    annotation_mode = "ap" if metadata.get("annotation_mode") == "ap" else "vanilla"
    if annotation_mode == "ap":
        raise ValueError("AP is excluded because its oracle piece annotations change the evaluation history")
    config = ModelConfig(**checkpoint["config"])
    model_type = str(checkpoint.get("model_type", "vanilla"))
    if model_type != "llama":
        raise ValueError("selected-query attention intervention currently supports the reference LLaMA model only")
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    del checkpoint
    _, hand_names = label_maps()
    print(json.dumps({"event": "drop_attention_scan_start", "path": args.evaluation_jsonl}), flush=True)
    lightweight_positions, census = read_positions(
        args.evaluation_jsonl, metadata["state_prompt_mode"], annotation_mode,
        hand_names, config.max_seq_len - 1,
    )
    lightweight_pairs, matching = select_anchors_and_controls(
        lightweight_positions, args.max_pairs, args.seed
    )
    selected_keys = selected_keys_for_pairs(lightweight_pairs)
    print(json.dumps({
        "event": "drop_attention_matching_complete", "positions": len(lightweight_positions),
        "pairs": len(lightweight_pairs), "materialization_keys": len(selected_keys),
    }), flush=True)
    positions, materialization_census = read_positions(
        args.evaluation_jsonl, metadata["state_prompt_mode"], annotation_mode,
        hand_names, config.max_seq_len - 1, selected_keys=selected_keys, materialize=True,
    )
    pairs = rebind_pairs(lightweight_pairs, positions)
    if not pairs:
        raise ValueError("no matched drop/non-drop pairs were found")
    layer_sets = resolve_layer_sets(args.ablation_layers, config.n_layers)
    square_ids = [vocabulary["<SQ_{}{}>".format(file, rank)] for file in "123456789" for rank in "abcdefghi"]
    first_ids = square_ids + [vocabulary[DROP_TOKEN]]
    piece_ids = [vocabulary[token] for token in BASIC_PIECE_TOKENS]
    attention_accum = defaultdict(lambda: defaultdict(float))
    ablation_accum = defaultdict(lambda: defaultdict(float))
    no_mask_max_error = 0.0
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for pair_index, pair in enumerate(pairs):
            piece = int(pair["piece"])
            for group, item in (("drop", pair["anchor"]), ("control", pair["control"])):
                relevant, controls = relevant_and_control_markers(item, int(item["side"]), piece, args.seed)
                if not relevant or len(controls) != len(relevant):
                    continue
                pre_ids = token_ids(item["prefix_tokens"], vocabulary, device)
                pre_query = pre_ids.shape[1] - 1
                pre_rows, pre_logits = selected_attention_rows(model, pre_ids, pre_query)
                add_attention(attention_accum, pre_rows, relevant, controls, group, "pre")
                drop_ids = torch.cat((pre_ids, torch.tensor([[vocabulary[DROP_TOKEN]]], device=device)), dim=1)
                drop_query = drop_ids.shape[1] - 1
                drop_rows, drop_logits = selected_attention_rows(model, drop_ids, drop_query)
                add_attention(attention_accum, drop_rows, relevant, controls, group, "after_drop")

                if pair_index == 0 and group == "drop":
                    standard_pre = model(pre_ids).logits
                    empty_pre = forward_with_edge_ablation(model, pre_ids, pre_query, [], set(range(config.n_layers)))
                    standard_drop = model(drop_ids).logits
                    empty_drop = forward_with_edge_ablation(model, drop_ids, drop_query, [], set(range(config.n_layers)))
                    no_mask_max_error = max(
                        float((standard_pre - pre_logits).abs().max()),
                        float((standard_drop - drop_logits).abs().max()),
                        float((standard_pre - empty_pre).abs().max()),
                        float((standard_drop - empty_drop).abs().max()),
                    )
                    # SDPA kernelと明示softmaxの丸め差を許すが，介入経路の
                    # 実装誤りを見逃さない程度に十分小さい閾値とする。
                    tolerance = 2e-3 if amp_dtype is not None else 1e-4
                    if no_mask_max_error > tolerance:
                        raise RuntimeError("attention observability path changes logits: {}".format(no_mask_max_error))

                if pair_index >= args.max_ablation_pairs:
                    continue
                pre_baseline = restricted_metrics(pre_logits, first_ids, vocabulary[DROP_TOKEN])
                drop_baseline = restricted_metrics(drop_logits, piece_ids, piece_ids[piece])
                for layer_name, layer_indices in layer_sets.items():
                    for mask_name, keys in (("relevant", relevant), ("matched_control", controls)):
                        pre_masked_logits = forward_with_edge_ablation(model, pre_ids, pre_query, keys, layer_indices)
                        drop_masked_logits = forward_with_edge_ablation(model, drop_ids, drop_query, keys, layer_indices)
                        add_ablation(
                            ablation_accum,
                            "{}:{}:{}:pre".format(group, layer_name, mask_name),
                            pre_baseline,
                            restricted_metrics(pre_masked_logits, first_ids, vocabulary[DROP_TOKEN]),
                        )
                        add_ablation(
                            ablation_accum,
                            "{}:{}:{}:after_drop".format(group, layer_name, mask_name),
                            drop_baseline,
                            restricted_metrics(drop_masked_logits, piece_ids, piece_ids[piece]),
                        )
            done = pair_index + 1
            if args.progress_every and done // args.progress_every != pair_index // args.progress_every:
                print(json.dumps({"event": "drop_attention_progress", "pairs": done, "total": len(pairs)}), flush=True)

    result = {
        "format_version": 1,
        "checkpoint": args.checkpoint,
        "evaluation_jsonl": args.evaluation_jsonl,
        "model_type": model_type,
        "annotation_mode": str(metadata.get("annotation_mode", "vanilla")),
        "amp": amp_name,
        "settings": vars(args),
        "census": census,
        "materialization_census": materialization_census,
        "matching": matching,
        "matching_balance": matching_balance(lightweight_pairs),
        "no_mask_forward_max_absolute_logit_error": no_mask_max_error,
        "attention": finish_attention(attention_accum),
        "ablation": finish_ablation(ablation_accum),
        "definitions": {
            "relevant_marker": "destination token of an earlier move that changed the tracked side/piece hand count",
            "control_marker": "non-relevant move destination selected to match distance to relevant markers",
            "pre": "query at the previous move destination before the current move's first token",
            "after_drop": "query after teacher-forcing <DROP> and before selecting the dropped piece",
            "edge_ablation": "sets only the selected query-to-key attention logits to -infinity and renormalizes",
        },
        "limitations": [
            "Attention mass alone is descriptive and is not an explanation of model behaviour.",
            "Direct-edge ablation leaves indirect paths through later token representations intact.",
            "The after-DROP analysis is teacher-forced and cannot establish why the model initiated a drop.",
        ],
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "drop_attention_complete", "output": str(output), "pairs": len(pairs)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
