#!/usr/bin/env python3
"""CoT-like読み筋の自由生成と最終指手を評価する。"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import torch

from create_dataset import all_usi_move_tokens, import_cshogi
from data import load_vocabulary
from evaluate_probes import load_backbone, resolve_device
from generate_reasoning_traces import sample_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="synthetic reasoning traceを自由生成し、形式・合法性・最終指手を評価する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run_prefix(model, token_ids: Sequence[int], moves_marker: int, device):
    past_key_values = None
    recurrent_state = None
    next_logits = None
    with torch.inference_mode():
        for position, token_id in enumerate(token_ids):
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            recurrent_active = torch.tensor(
                [position > moves_marker], dtype=torch.bool, device=device
            )
            (
                logits,
                past_key_values,
                recurrent_state,
                _,
                _,
            ) = model.step(
                token,
                position,
                past_key_values,
                recurrent_state,
                recurrent_active,
            )
            next_logits = logits[0, 0]
    if next_logits is None:
        raise ValueError("empty prefix")
    return next_logits, past_key_values, recurrent_state


def generate_completion(
    model,
    prefix_ids: Sequence[int],
    moves_marker: int,
    eos_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device,
) -> List[int]:
    next_logits, past_key_values, recurrent_state = run_prefix(
        model, prefix_ids, moves_marker, device
    )
    generated: List[int] = []
    position = len(prefix_ids)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            token_id = sample_token(next_logits, temperature, top_p)
            generated.append(token_id)
            if token_id == eos_id:
                break
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            (
                logits,
                past_key_values,
                recurrent_state,
                _,
                _,
            ) = model.step(
                token,
                position,
                past_key_values,
                recurrent_state,
                torch.ones(1, dtype=torch.bool, device=device),
            )
            next_logits = logits[0, 0]
            position += 1
    return generated


def split_trace_lines(tokens: Sequence[str], syntactic_moves) -> List[List[str]]:
    if "<THINK>" not in tokens or "</THINK>" not in tokens:
        return []
    start = tokens.index("<THINK>") + 1
    stop = tokens.index("</THINK>", start)
    lines: List[List[str]] = [[]]
    for token in tokens[start:stop]:
        if token == "<SEP>":
            lines.append([])
        elif token in syntactic_moves:
            lines[-1].append(token)
    return [line for line in lines if line]


def extract_answer(tokens: Sequence[str]):
    if "<ANSWER>" not in tokens:
        return None
    index = tokens.index("<ANSWER>") + 1
    return tokens[index] if index < len(tokens) else None


def root_board(record, cshogi):
    board = cshogi.Board(str(record["start_sfen"]))
    for ply, move_usi in enumerate(record["history_moves"], 1):
        move = board.move_from_usi(str(move_usi))
        if not board.is_legal(move):
            raise ValueError(
                "illegal history move in {} at {}: {}".format(
                    record["game_id"], ply, move_usi
                )
            )
        board.push(move)
    return board


def line_legality(lines, board, cshogi):
    total_moves = 0
    legal_moves = 0
    fully_legal_lines = 0
    for line in lines:
        branch = cshogi.Board(board.sfen())
        full = True
        total_moves += len(line)
        for move_usi in line:
            move = branch.move_from_usi(str(move_usi))
            if not branch.is_legal(move):
                full = False
                break
            legal_moves += 1
            branch.push(move)
        fully_legal_lines += int(full)
    return total_moves, legal_moves, fully_legal_lines


def gold_trace_prefix(record):
    tokens = (
        ["<BOS>"]
        + list(record["initial_state_tokens"])
        + ["<MOVES>"]
        + list(record["history_moves"])
        + ["<THINK>"]
    )
    for line_index, line in enumerate(record["reasoning_lines"]):
        if line_index:
            tokens.append("<SEP>")
        tokens.extend(line)
    tokens.extend(["</THINK>", "<ANSWER>"])
    return tokens


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    vocabulary = load_vocabulary(args.vocab)
    id_to_token = {index: token for token, index in vocabulary.items()}
    syntactic_moves = set(all_usi_move_tokens())
    model, model_type, _ = load_backbone(args.checkpoint, device, False)
    cshogi = import_cshogi()
    moves_marker = 1 + 96
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "reasoning_details.jsonl"

    counters = {
        "examples": 0,
        "format_complete": 0,
        "generated_eos": 0,
        "answer_exact": 0,
        "answer_syntactic": 0,
        "answer_legal": 0,
        "answer_in_trace_first_moves": 0,
        "target_in_trace_first_moves": 0,
        "teacher_forced_answer_top1": 0,
        "teacher_forced_answer_top5": 0,
        "trace_lines": 0,
        "trace_moves": 0,
        "trace_legal_moves": 0,
        "fully_legal_lines": 0,
    }

    with Path(args.trace_jsonl).open("r", encoding="utf-8") as source, \
            details_path.open("w", encoding="utf-8") as details:
        for line in source:
            if not line.strip():
                continue
            if args.max_examples > 0 and counters["examples"] >= args.max_examples:
                break
            record = json.loads(line)
            prompt_tokens = (
                ["<BOS>"]
                + list(record["initial_state_tokens"])
                + ["<MOVES>"]
                + list(record["history_moves"])
            )
            prompt_ids = [vocabulary[token] for token in prompt_tokens]
            generated_ids = generate_completion(
                model,
                prompt_ids,
                moves_marker,
                vocabulary["<EOS>"],
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                device,
            )
            generated_tokens = [id_to_token[index] for index in generated_ids]
            lines = split_trace_lines(generated_tokens, syntactic_moves)
            answer = extract_answer(generated_tokens)
            board = root_board(record, cshogi)
            legal_set = {
                cshogi.move_to_usi(move) for move in board.legal_moves
            }
            total, legal, fully_legal = line_legality(lines, board, cshogi)
            first_moves = {trace_line[0] for trace_line in lines if trace_line}
            target = str(record["target_move"])

            gold_prefix = gold_trace_prefix(record)
            gold_prefix_ids = [vocabulary[token] for token in gold_prefix]
            answer_logits, _, _ = run_prefix(
                model, gold_prefix_ids, moves_marker, device
            )
            target_id = vocabulary[target]
            top_k = min(5, answer_logits.shape[-1])
            answer_top = answer_logits.topk(top_k).indices.tolist()

            format_complete = (
                "<THINK>" in generated_tokens
                and "</THINK>" in generated_tokens
                and "<ANSWER>" in generated_tokens
            )
            counters["examples"] += 1
            counters["format_complete"] += int(format_complete)
            counters["generated_eos"] += int(
                vocabulary["<EOS>"] in generated_ids
            )
            counters["answer_exact"] += int(answer == target)
            counters["answer_syntactic"] += int(answer in syntactic_moves)
            counters["answer_legal"] += int(answer in legal_set)
            counters["answer_in_trace_first_moves"] += int(answer in first_moves)
            counters["target_in_trace_first_moves"] += int(target in first_moves)
            counters["teacher_forced_answer_top1"] += int(answer_top[0] == target_id)
            counters["teacher_forced_answer_top5"] += int(target_id in answer_top)
            counters["trace_lines"] += len(lines)
            counters["trace_moves"] += total
            counters["trace_legal_moves"] += legal
            counters["fully_legal_lines"] += fully_legal
            details.write(
                json.dumps(
                    {
                        "game_id": record["game_id"],
                        "target_move": target,
                        "generated_tokens": generated_tokens,
                        "reasoning_lines": lines,
                        "answer": answer,
                        "answer_exact": answer == target,
                        "answer_legal": answer in legal_set,
                        "format_complete": format_complete,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    examples = max(counters["examples"], 1)
    trace_moves = max(counters["trace_moves"], 1)
    trace_lines = max(counters["trace_lines"], 1)
    metrics = {
        "model_type": model_type,
        "checkpoint": args.checkpoint,
        "examples": counters["examples"],
        "format_complete_rate": counters["format_complete"] / examples,
        "eos_rate": counters["generated_eos"] / examples,
        "answer_exact_rate": counters["answer_exact"] / examples,
        "answer_syntactic_rate": counters["answer_syntactic"] / examples,
        "answer_legal_rate": counters["answer_legal"] / examples,
        "answer_trace_consistency_rate": counters[
            "answer_in_trace_first_moves"
        ]
        / examples,
        "target_in_trace_recall": counters["target_in_trace_first_moves"]
        / examples,
        "teacher_forced_answer_top1": counters[
            "teacher_forced_answer_top1"
        ]
        / examples,
        "teacher_forced_answer_top5": counters[
            "teacher_forced_answer_top5"
        ]
        / examples,
        "trace_move_legal_rate": counters["trace_legal_moves"] / trace_moves,
        "fully_legal_line_rate": counters["fully_legal_lines"] / trace_lines,
        "mean_generated_lines": counters["trace_lines"] / examples,
        "mean_generated_trace_moves": counters["trace_moves"] / examples,
        "uses_legality_mask": False,
    }
    with (output_dir / "reasoning_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
