#!/usr/bin/env python3
"""factorized_v3 checkpointの高速な指手評価．

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
from factorized_prompt import BASIC_PIECE_TOKENS, DROP_TOKEN, MOVE_ENCODING, PROMOTE_TOKEN, TERMINAL_ENCODING, factorize_usi, unfactorize_usi
from models import ModelConfig, build_model
from new_prompt import square_tokens
from train_model import amp_context, resolve_amp


def parse_args():
    parser = argparse.ArgumentParser(description="factorized_v3指手評価")
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
    parser.add_argument(
        "--length-bucket-pool-batches", type=int, default=16,
        help="このbatch数分のqueryを長さ順に並べてpaddingを減らす。0で無効",
    )
    parser.add_argument(
        "--beam-micro-batch-size", type=int, default=8,
        help="同じprefix長のqueryを同時にbeam生成する上限。1で逐次相当",
    )
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
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
    """JSONLを逐次走査し，小さなpool内で長さを揃えてbatch化する。"""
    batch = []
    pool_batches = max(1, int(getattr(args, "length_bucket_pool_batches", 0) or 1))
    pool_size = args.batch_size * pool_batches

    def flush_pool(values):
        if pool_batches > 1:
            values.sort(key=lambda query: len(query["prefix_ids"]))
        return [values[index : index + args.batch_size] for index in range(0, len(values), args.batch_size)]

    wanted = set(args.history_distances)
    with Path(args.evaluation_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if statistics["games"] >= args.max_games or statistics["queries"] >= args.max_queries:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = list(record.get("start_candidates", []))
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
                        if step is not None and len(prefix) + 3 <= max_seq_len:
                            batch.append({
                                "prefix_ids": [vocabulary[token] for token in prefix],
                                "target": str(record["move_tokens"][ply]),
                                "legal_moves": set(str(value) for value in step["legal_moves"]),
                                "distance": distance,
                                "position_scope": position_scope(record, candidate, ply),
                                "trajectory_scope": str(record.get("trajectory_scope", "unknown_position_scope")),
                            })
                            statistics["queries"] += 1
                            if len(batch) >= pool_size:
                                yield from flush_pool(batch)
                                batch = []
                            if statistics["queries"] >= args.max_queries:
                                break
                    if ply >= len(record["move_tokens"]):
                        break
                    history.extend(factorize_usi(str(record["move_tokens"][ply])))
                if statistics["queries"] >= args.max_queries:
                    break
    if batch:
        yield from flush_pool(batch)


def padded_forward(model, sequences, pad_id, device):
    lengths_cpu = torch.tensor([len(value) for value in sequences], dtype=torch.long)
    lengths = lengths_cpu.to(device)
    width = int(lengths_cpu.max())
    ids_cpu = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for row, values in enumerate(sequences):
        ids_cpu[row, : len(values)] = torch.as_tensor(values, dtype=torch.long)
    ids = ids_cpu.to(device, non_blocking=device.type == "cuda")
    mask = None
    if not bool((lengths_cpu == width).all()):
        mask_cpu = torch.arange(width)[None, :] < lengths_cpu[:, None]
        mask = mask_cpu.to(device, non_blocking=device.type == "cuda")
    return model(ids, attention_mask=mask, output_hidden_states=False).logits, lengths


def constrained_top(logits, allowed, k):
    selected = logits[allowed]
    indices = torch.topk(selected, min(k, len(allowed))).indices
    return [allowed[int(index)] for index in indices]


def grammar_allowed(current, vocabulary):
    squares = [vocabulary[token] for token in square_tokens()]
    if not current:
        return squares + [vocabulary[DROP_TOKEN]]
    if current[0] == vocabulary[DROP_TOKEN]:
        if len(current) == 1:
            return [vocabulary[token] for token in BASIC_PIECE_TOKENS]
        if len(current) == 2:
            return squares
    else:
        if len(current) == 1:
            return squares + [vocabulary[PROMOTE_TOKEN]]
        if len(current) == 2 and current[1] == vocabulary[PROMOTE_TOKEN]:
            return squares
    return []


def grammar_finished(current, vocabulary):
    if not current:
        return False
    square_set = {vocabulary[token] for token in square_tokens()}
    return current[-1] in square_set and (
        (len(current) == 2 and current[0] in square_set)
        or (len(current) == 3 and (current[0] == vocabulary[DROP_TOKEN] or current[1] == vocabulary[PROMOTE_TOKEN]))
    )


def restricted_candidates(vector, allowed, k):
    """文法上許されたtoken内で正規化した候補とlog確率を返す．"""
    selected = vector[allowed].float()
    log_probabilities = torch.log_softmax(selected, dim=-1)
    local = torch.topk(log_probabilities, min(k, len(allowed))).indices
    return [(allowed[int(index)], float(log_probabilities[int(index)].detach())) for index in local]


def beam_single_cached(model, prefix_ids, vocabulary, device, beam_size=5):
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    next_logits, prefix_cache = model.prefill(prefix)
    beams = [([], 0.0, False, prefix_cache, next_logits[0, -1])]
    for _ in range(3):
        candidates = []
        for current, score, finished, cache, vector in beams:
            if finished:
                candidates.append((current, score, True, cache, vector))
                continue
            allowed = grammar_allowed(current, vocabulary)
            for token, token_score in restricted_candidates(vector, allowed, beam_size):
                values = current + [token]
                new_score = score + token_score
                if grammar_finished(values, vocabulary):
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
    return beam_batch_cached(
        model, prefix_ids, vocabulary, device, beam_size=beam_size,
        micro_batch_size=8,
    )


def _slice_cache(cache, index):
    return tuple((key[index : index + 1], value[index : index + 1]) for key, value in cache)


def _stack_caches(caches):
    return tuple(
        (
            torch.cat([cache[layer][0] for cache in caches], dim=0),
            torch.cat([cache[layer][1] for cache in caches], dim=0),
        )
        for layer in range(len(caches[0]))
    )


def beam_equal_length_cached(model, prefix_ids, vocabulary, device, beam_size=5):
    """同じ長さのprefix群を，query×beam方向にまとめて生成する。"""
    if not prefix_ids:
        return []
    prefix_length = len(prefix_ids[0])
    if any(len(values) != prefix_length for values in prefix_ids):
        raise ValueError("beam_equal_length_cached requires equal prefix lengths")
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    next_logits, prefix_cache = model.prefill(prefix)
    beams = [
        [{"tokens": [], "score": 0.0, "finished": False,
          "cache": _slice_cache(prefix_cache, row), "vector": next_logits[row, -1]}]
        for row in range(len(prefix_ids))
    ]

    for _ in range(3):
        selected_by_query = []
        active = []
        for query_index, query_beams in enumerate(beams):
            candidates = []
            for beam in query_beams:
                if beam["finished"]:
                    candidates.append(beam)
                    continue
                current = beam["tokens"]
                allowed = grammar_allowed(current, vocabulary)
                for token, token_score in restricted_candidates(beam["vector"], allowed, beam_size):
                    values = current + [token]
                    candidates.append({
                        "tokens": values,
                        "score": beam["score"] + token_score,
                        "finished": grammar_finished(values, vocabulary),
                        "cache": beam["cache"],
                        "vector": beam["vector"],
                    })
            selected = sorted(candidates, key=lambda value: value["score"], reverse=True)[:beam_size]
            selected_by_query.append(selected)
            for beam_index, candidate in enumerate(selected):
                if not candidate["finished"]:
                    active.append((query_index, beam_index, candidate))

        if active:
            step_tokens = torch.tensor(
                [[candidate["tokens"][-1]] for _, _, candidate in active],
                dtype=torch.long, device=device,
            )
            past = _stack_caches([candidate["cache"] for _, _, candidate in active])
            generated_length = len(active[0][2]["tokens"])
            logits, next_cache, _, _, _ = model.step(
                step_tokens, prefix_length + generated_length - 1, past,
            )
            for row, (query_index, beam_index, candidate) in enumerate(active):
                candidate["cache"] = _slice_cache(next_cache, row)
                candidate["vector"] = logits[row, -1]
                selected_by_query[query_index][beam_index] = candidate
        beams = selected_by_query
        if all(all(value["finished"] for value in query_beams) for query_beams in beams):
            break

    id_to_token = {index: token for token, index in vocabulary.items()}
    result = []
    for query_beams in beams:
        decoded = []
        for beam in query_beams:
            if beam["finished"]:
                try:
                    decoded.append((
                        unfactorize_usi([id_to_token[value] for value in beam["tokens"]]),
                        beam["score"],
                    ))
                except ValueError:
                    pass
        result.append(decoded)
    return result


def beam_batch_cached(model, prefix_ids, vocabulary, device, beam_size=5, micro_batch_size=8):
    """prefix長ごとに分け，出力順を保ったまま実バッチbeamを行う。"""
    if micro_batch_size <= 0:
        raise ValueError("beam micro-batch size must be positive")
    result = [None] * len(prefix_ids)
    by_length = defaultdict(list)
    for index, values in enumerate(prefix_ids):
        by_length[len(values)].append((index, values))
    for entries in by_length.values():
        for start in range(0, len(entries), micro_batch_size):
            chunk = entries[start : start + micro_batch_size]
            decoded = beam_equal_length_cached(
                model, [values for _, values in chunk], vocabulary, device, beam_size,
            )
            for (index, _), values in zip(chunk, decoded):
                result[index] = values
    return result


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
    if total.get("promotion_decision_applicable", 0):
        result["promotion_decision_top1"] = (
            total["promotion_decision_correct"] / total["promotion_decision_applicable"]
        )
        result["promotion_decision_examples"] = int(total["promotion_decision_applicable"])
    result.pop("promotion_decision_correct", None)
    result.pop("promotion_decision_applicable", None)
    result["move_perplexity"] = math.exp(min(result["move_nll"], 20.0))
    result["grammar_normalized_move_perplexity"] = math.exp(min(result["grammar_normalized_move_nll"], 20.0))
    return result


def main():
    args = parse_args()
    args.history_distances = parse_distances(args.history_distances)
    args.primary_history_distances = parse_distances(args.primary_history_distances)
    vocabulary = load_vocabulary(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("new_prompt", {}).get("move_encoding") != MOVE_ENCODING:
        raise ValueError("checkpoint is not marked as {}".format(MOVE_ENCODING))
    if checkpoint.get("new_prompt", {}).get("terminal_encoding") != TERMINAL_ENCODING:
        raise ValueError("checkpoint was not trained with complete-game EOS supervision")
    config = ModelConfig(**checkpoint["config"])
    checkpoint_settings = checkpoint.get("new_prompt", {})
    args.state_prompt_mode = str(checkpoint_settings.get("state_prompt_mode", "explicit"))
    args.start_selection = str(checkpoint_settings.get("start_selection", "random_candidates"))
    if args.state_prompt_mode != "implicit_initial" or args.start_selection != "fixed_initial":
        raise ValueError("current factorized_v3 evaluation accepts only implicit fixed-initial checkpoints")
    if config.vocab_size != len(vocabulary):
        raise ValueError("checkpoint and vocabulary sizes differ")
    device = resolve_device(args.device)
    amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    model_type = str(checkpoint.get("model_type", "vanilla"))
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # best.ptに旧形式のoptimizer stateが含まれていても評価中は保持しない。
    del checkpoint
    model.eval()

    square_ids = [vocabulary[token] for token in square_tokens()]
    totals = {"all": empty_total(), "primary": empty_total()}
    by_distance = defaultdict(empty_total)
    by_position = defaultdict(empty_total)
    started = time.perf_counter()
    statistics = {"games": 0, "queries": 0}
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for batch in iter_query_batches(args, vocabulary, config.max_seq_len, statistics):
            targets = [factorize_usi(query["target"]) for query in batch]
            target_ids = [[vocabulary[token] for token in values] for values in targets]
            sequences = [query["prefix_ids"] + values for query, values in zip(batch, target_ids)]
            logits, _ = padded_forward(model, sequences, vocabulary["<PAD>"], device)
            beam_moves = beam_batch_cached(
                model, [query["prefix_ids"] for query in batch], vocabulary, device,
                micro_batch_size=args.beam_micro_batch_size,
            )
            for row, (query, tokens, ids, generated) in enumerate(zip(batch, targets, target_ids, beam_moves)):
                prefix_length = len(query["prefix_ids"])
                nll = grammar_nll = 0.0
                component_top1 = component_top5 = True
                source_top1 = source_top5 = destination_top1 = destination_top5 = 0
                promotion_correct = promotion_applicable = 0
                current_ids = []
                for offset, target_id in enumerate(ids):
                    vector = logits[row, prefix_length + offset - 1].float()
                    nll -= float(torch.log_softmax(vector, dim=-1)[target_id])
                    allowed = grammar_allowed(current_ids, vocabulary)
                    allowed_logits = vector[allowed]
                    grammar_nll -= float(torch.log_softmax(allowed_logits, dim=-1)[allowed.index(target_id)])
                    top = constrained_top(vector, allowed, 5)
                    if offset == 0:
                        source_top1, source_top5 = int(top[0] == target_id), int(target_id in top)
                    if tokens[offset].startswith("<SQ_") and offset > 0:
                        destination_top1, destination_top5 = int(top[0] == target_id), int(target_id in top)
                    if offset == 1 and tokens[0].startswith("<SQ_"):
                        promotion_applicable = 1
                        predicted_promote = top[0] == vocabulary[PROMOTE_TOKEN]
                        promotion_correct = int(predicted_promote == (tokens[offset] == PROMOTE_TOKEN))
                    component_top1 = component_top1 and top[0] == target_id
                    component_top5 = component_top5 and target_id in top
                    current_ids.append(target_id)
                predicted = generated[0][0] if generated else None
                generated_moves = [move for move, _ in generated]
                legal_beam_mass = sum(
                    math.exp(score) for move, score in generated if move in query["legal_moves"]
                )
                values = {
                    "move_nll": nll,
                    "grammar_normalized_move_nll": grammar_nll,
                    "source_top1": source_top1, "source_top5": source_top5,
                    "destination_given_source_top1": destination_top1,
                    "destination_given_source_top5": destination_top5,
                    "promotion_decision_correct": promotion_correct,
                    "promotion_decision_applicable": promotion_applicable,
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
        "format_version": 3,
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "move_encoding": MOVE_ENCODING,
        "evaluation_input_rap": False,
        "settings": vars(args),
        "amp": amp_name,
        "metrics": {
            "games": statistics["games"],
            "primary": summarize(totals["primary"]),
            "all_reported_distances": summarize(totals["all"]),
            "by_history_distance": {key: summarize(value) for key, value in by_distance.items()},
            "by_position_scope": {key: summarize(value) for key, value in by_position.items()},
        },
        "notes": {
            "teacher_forced": "destination and later components are conditioned on the gold preceding components",
            "greedy_and_beam": "autonomous decoding is normalized within the factorized move grammar, not shogi legality",
            "legal_probability_mass": "beam_legal_probability_lower_bound sums only legal moves retained by beam search; exact mass requires scoring every legal move sequence",
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "evaluation_complete", "output": str(path), "queries": statistics["queries"], "elapsed_sec": round(time.perf_counter() - started, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
