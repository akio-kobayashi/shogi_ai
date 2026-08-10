#!/usr/bin/env python3
"""factorized_v2 checkpointの高速な指手評価。

正解接頭辞を使う構成要素評価と，自律的な文法制約greedy生成を分離して報告する。
全合法手の系列確率を列挙する ``legal probability mass`` は本軽量評価には含めない。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from data import load_vocabulary
from factorized_prompt import DROP_TOKENS, EOM_TOKEN, PROMOTE_TOKEN, factorize_usi, unfactorize_usi
from models import ModelConfig, build_model
from new_prompt import square_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="factorized_v2指手評価")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-distances", default="0,8,32")
    parser.add_argument("--primary-history-distances", default="8,32")
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--candidates-per-game", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_distances(value):
    result = tuple(dict.fromkeys(int(item.strip()) for item in str(value).split(",") if item.strip()))
    if not result or min(result) < 0:
        raise ValueError("history distances must be nonnegative")
    return result


def resolve_device(value):
    return torch.device(value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def position_scope(record, candidate, ply):
    values = record.get("position_scope_by_ply", [])
    if 0 <= ply < len(values):
        return str(values[ply])
    for example in record.get("probe_examples", []):
        if int(example.get("start_ply", -1)) == int(candidate["start_ply"]) and int(example.get("ply", -1)) == ply:
            return str(example.get("position_scope", "unknown_position_scope"))
    return str(candidate.get("position_scope", "unknown_position_scope"))


def iter_query_batches(args, vocabulary, max_seq_len, statistics):
    """JSONLを逐次走査し，評価queryをbatch以上保持しない。"""
    batch = []
    wanted = set(args.history_distances)
    with Path(args.evaluation_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if statistics["games"] >= args.max_games or statistics["queries"] >= args.max_queries:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = list(record.get("start_candidates", []))
            if args.start_selection == "fixed_initial":
                candidates = [value for value in candidates if int(value.get("start_ply", -1)) == 0]
            candidates = candidates[: args.candidates_per_game]
            if not candidates:
                continue
            statistics["games"] += 1
            steps = {int(step["ply"]): step for step in record.get("evaluation_steps", [])}
            for candidate in candidates:
                start = int(candidate["start_ply"])
                state = [] if args.state_prompt_mode == "implicit_initial" else list(candidate["state_prompt_tokens"])
                base = ["<BOS>", *state, "<MOVES>"]
                history = []
                for distance in range(max(wanted) + 1):
                    ply = start + distance
                    if distance in wanted and ply < len(record["move_tokens"]):
                        step = steps.get(ply)
                        prefix = base + history
                        if step is not None and len(prefix) + 4 <= max_seq_len:
                            batch.append({
                                "prefix_ids": [vocabulary[token] for token in prefix],
                                "target": str(record["move_tokens"][ply]),
                                "legal_moves": set(str(value) for value in step["legal_moves"]),
                                "distance": distance,
                                "position_scope": position_scope(record, candidate, ply),
                                "trajectory_scope": str(record.get("trajectory_scope", "unknown_position_scope")),
                            })
                            statistics["queries"] += 1
                            if len(batch) >= args.batch_size:
                                yield batch
                                batch = []
                            if statistics["queries"] >= args.max_queries:
                                break
                    if ply >= len(record["move_tokens"]):
                        break
                    history.extend(factorize_usi(str(record["move_tokens"][ply])))
                if statistics["queries"] >= args.max_queries:
                    break
    if batch:
        yield batch


def padded_forward(model, sequences, pad_id, device):
    lengths_cpu = torch.tensor([len(value) for value in sequences], dtype=torch.long)
    lengths = lengths_cpu.to(device)
    width = int(lengths_cpu.max())
    ids_cpu = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for row, values in enumerate(sequences):
        ids_cpu[row, : len(values)] = torch.as_tensor(values, dtype=torch.long)
    mask_cpu = torch.arange(width)[None, :] < lengths_cpu[:, None]
    ids = ids_cpu.to(device, non_blocking=device.type == "cuda")
    mask = mask_cpu.to(device, non_blocking=device.type == "cuda")
    return model(ids, attention_mask=mask, output_hidden_states=False).logits, lengths


def constrained_top(logits, allowed, k):
    selected = logits[allowed]
    indices = torch.topk(selected, min(k, len(allowed))).indices
    return [allowed[int(index)] for index in indices]


def beam_single_cached(model, prefix_ids, vocabulary, device, beam_size=5):
    square_ids = [vocabulary[token] for token in square_tokens()]
    drop_ids = [vocabulary[token] for token in DROP_TOKENS]
    promote_id, eom_id = vocabulary[PROMOTE_TOKEN], vocabulary[EOM_TOKEN]
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    next_logits, prefix_cache = model.prefill(prefix)
    beams = [([], 0.0, False, prefix_cache, next_logits[0, -1])]
    for _ in range(4):
        candidates = []
        for current, score, finished, cache, vector in beams:
            if finished:
                candidates.append((current, score, True, cache, vector))
                continue
            if not current:
                allowed = square_ids + drop_ids
            elif len(current) == 1:
                allowed = square_ids
            elif current[0] in drop_ids:
                allowed = [eom_id]
            elif len(current) == 2:
                allowed = [promote_id, eom_id]
            else:
                allowed = [eom_id]
            log_probabilities = torch.log_softmax(vector.float(), dim=-1)
            for token in constrained_top(log_probabilities, allowed, beam_size):
                values = current + [token]
                new_score = score + float(log_probabilities[token])
                if token == eom_id:
                    candidates.append((values, new_score, True, cache, vector))
                else:
                    token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
                    logits, next_cache, _, _, _ = model.step(
                        token_tensor, len(prefix_ids) + len(current), cache,
                    )
                    candidates.append((values, new_score, False, next_cache, logits[0, -1]))
        beams = sorted(candidates, key=lambda value: value[1], reverse=True)[:beam_size]
        if all(value[2] for value in beams):
            break
    id_to_token = {index: token for token, index in vocabulary.items()}
    decoded = []
    for values, score, finished, _, _ in beams:
        if finished:
            try:
                decoded.append((unfactorize_usi([id_to_token[value] for value in values]), score))
            except ValueError:
                pass
    return decoded


def beam_batch(model, prefix_ids, vocabulary, device, beam_size=5):
    # query間でcacheを保持しないため，ピークは1 query×beam数に制限される。
    return [
        beam_single_cached(model, values, vocabulary, device, beam_size)
        for values in prefix_ids
    ]


def empty_total():
    return defaultdict(float)


def add(total, query, values):
    total["queries"] += 1
    for key, value in values.items():
        total[key] += float(value)


def summarize(total):
    n = int(total["queries"])
    if not n:
        return None
    result = {"queries": n}
    for key, value in total.items():
        if key != "queries":
            result[key] = value / n
    if total.get("promotion_or_end_applicable", 0):
        result["promotion_or_end_top1"] = (
            total["promotion_or_end_correct"] / total["promotion_or_end_applicable"]
        )
        result["promotion_or_end_examples"] = int(total["promotion_or_end_applicable"])
    if total.get("eom_targets", 0):
        result["eom_unconstrained_top1"] = total["eom_correct"] / total["eom_targets"]
        result["eom_targets"] = int(total["eom_targets"])
    result.pop("promotion_or_end_correct", None)
    result.pop("promotion_or_end_applicable", None)
    result.pop("eom_correct", None)
    result["move_perplexity"] = math.exp(min(result["move_nll"], 20.0))
    return result


def main():
    args = parse_args()
    args.history_distances = parse_distances(args.history_distances)
    args.primary_history_distances = parse_distances(args.primary_history_distances)
    vocabulary = load_vocabulary(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("new_prompt", {}).get("move_encoding") != "factorized_v2":
        raise ValueError("checkpoint is not marked as factorized_v2")
    config = ModelConfig(**checkpoint["config"])
    checkpoint_settings = checkpoint.get("new_prompt", {})
    args.state_prompt_mode = str(checkpoint_settings.get("state_prompt_mode", "explicit"))
    args.start_selection = str(checkpoint_settings.get("start_selection", "random_candidates"))
    if config.vocab_size != len(vocabulary):
        raise ValueError("checkpoint and vocabulary sizes differ")
    device = resolve_device(args.device)
    model_type = str(checkpoint.get("model_type", "vanilla"))
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # best.ptに旧形式のoptimizer stateが含まれていても評価中は保持しない。
    del checkpoint
    model.eval()

    square_ids = [vocabulary[token] for token in square_tokens()]
    source_ids = square_ids + [vocabulary[token] for token in DROP_TOKENS]
    promote_ids = [vocabulary[PROMOTE_TOKEN], vocabulary[EOM_TOKEN]]
    totals = {"all": empty_total(), "primary": empty_total()}
    by_distance = defaultdict(empty_total)
    by_position = defaultdict(empty_total)
    started = time.perf_counter()
    statistics = {"games": 0, "queries": 0}
    with torch.inference_mode():
        for batch in iter_query_batches(args, vocabulary, config.max_seq_len, statistics):
            targets = [factorize_usi(query["target"]) for query in batch]
            target_ids = [[vocabulary[token] for token in values] for values in targets]
            sequences = [query["prefix_ids"] + values for query, values in zip(batch, target_ids)]
            logits, _ = padded_forward(model, sequences, vocabulary["<PAD>"], device)
            beam_moves = beam_batch(model, [query["prefix_ids"] for query in batch], vocabulary, device)
            for row, (query, tokens, ids, generated) in enumerate(zip(batch, targets, target_ids, beam_moves)):
                prefix_length = len(query["prefix_ids"])
                nll = 0.0
                component_top1 = component_top5 = True
                source_top1 = source_top5 = destination_top1 = destination_top5 = 0
                promotion_correct = promotion_applicable = 0
                eom_correct = eom_targets = 0
                for offset, target_id in enumerate(ids):
                    vector = logits[row, prefix_length + offset - 1].float()
                    nll -= float(torch.log_softmax(vector, dim=-1)[target_id])
                    if tokens[offset] == EOM_TOKEN:
                        eom_targets += 1
                        eom_correct += int(int(vector.argmax()) == target_id)
                    if offset == 0:
                        top = constrained_top(vector, source_ids, 5)
                        source_top1, source_top5 = int(top[0] == target_id), int(target_id in top)
                    elif offset == 1:
                        top = constrained_top(vector, square_ids, 5)
                        destination_top1, destination_top5 = int(top[0] == target_id), int(target_id in top)
                    elif tokens[offset] in (PROMOTE_TOKEN, EOM_TOKEN) and tokens[0] not in DROP_TOKENS:
                        top = constrained_top(vector, promote_ids, 2)
                        promotion_applicable = 1
                        promotion_correct = int(top[0] == target_id)
                    else:
                        top = constrained_top(vector, [vocabulary[EOM_TOKEN]], 1)
                    component_top1 = component_top1 and top[0] == target_id
                    component_top5 = component_top5 and target_id in top
                predicted = generated[0][0] if generated else None
                generated_moves = [move for move, _ in generated]
                legal_beam_mass = sum(
                    math.exp(score) for move, score in generated if move in query["legal_moves"]
                )
                values = {
                    "move_nll": nll,
                    "source_top1": source_top1, "source_top5": source_top5,
                    "destination_given_source_top1": destination_top1,
                    "destination_given_source_top5": destination_top5,
                    "promotion_or_end_correct": promotion_correct,
                    "promotion_or_end_applicable": promotion_applicable,
                    "eom_correct": eom_correct,
                    "eom_targets": eom_targets,
                    "teacher_forced_full_top1": int(component_top1),
                    "teacher_forced_full_top5": int(component_top5),
                    "greedy_full_move_top1": int(predicted == query["target"]),
                    "beam_full_move_top5": int(query["target"] in generated_moves[:5]),
                    "greedy_syntactic_rate": int(predicted is not None),
                    "greedy_legal_rate": int(predicted in query["legal_moves"]),
                    "beam_top5_contains_legal_rate": int(any(move in query["legal_moves"] for move in generated_moves[:5])),
                    "beam_legal_probability_lower_bound": legal_beam_mass,
                }
                add(totals["all"], query, values)
                add(by_distance[str(query["distance"])], query, values)
                if query["distance"] in args.primary_history_distances:
                    add(totals["primary"], query, values)
                    add(by_position[query["position_scope"]], query, values)
            done = int(totals["all"]["queries"])
            if args.progress_every and done // args.progress_every != (done - len(batch)) // args.progress_every:
                print(json.dumps({"event": "evaluation_progress", "queries": done, "max_queries": args.max_queries, "elapsed_sec": round(time.perf_counter() - started, 1)}), flush=True)

    if not statistics["queries"]:
        raise ValueError("no evaluation queries")

    output = {
        "format_version": 2,
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "move_encoding": "factorized_v2",
        "settings": vars(args),
        "metrics": {
            "games": statistics["games"],
            "primary": summarize(totals["primary"]),
            "all_reported_distances": summarize(totals["all"]),
            "by_history_distance": {key: summarize(value) for key, value in by_distance.items()},
            "by_position_scope": {key: summarize(value) for key, value in by_position.items()},
        },
        "notes": {
            "teacher_forced": "destination and later components are conditioned on the gold preceding components",
            "greedy_and_beam": "autonomous decoding is constrained only by the factorized USI grammar, not by shogi legality",
            "legal_probability_mass": "beam_legal_probability_lower_bound sums only legal moves retained by beam search; exact mass requires scoring every legal move sequence",
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "evaluation_complete", "output": str(path), "queries": statistics["queries"], "elapsed_sec": round(time.perf_counter() - started, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
