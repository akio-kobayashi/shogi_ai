#!/usr/bin/env python3
"""次手予測backbone自身からCoT-likeな複数読み筋を生成する。"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Mapping, Sequence

import torch

from create_dataset import all_usi_move_tokens
from data import load_vocabulary
from evaluate_probes import load_backbone, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="外部探索を使わず、backboneからsynthetic reasoning traceを生成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--positions-per-game", type=int, default=4)
    parser.add_argument("--lines", type=int, default=3)
    parser.add_argument("--line-length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="この局数ごとに進捗を表示する。0なら表示しない",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_decision_plies(total_moves: int, count: int) -> List[int]:
    """次手が存在するdecision plyを0..total_moves-1から等間隔に選ぶ。"""
    if total_moves <= 0:
        return []
    if count <= 0 or count >= total_moves:
        return list(range(total_moves))
    if count == 1:
        return [total_moves // 2]
    selected = [
        round(index * (total_moves - 1) / (count - 1))
        for index in range(count)
    ]
    return list(dict.fromkeys(selected))


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    scaled = logits.float() / temperature
    probabilities = torch.softmax(scaled, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities /= sorted_probabilities.sum()
        sampled = torch.multinomial(sorted_probabilities, 1)
        return int(sorted_indices[sampled])
    return int(torch.multinomial(probabilities, 1))


def advance_prefix(
    model,
    prompt_ids: Sequence[int],
    moves_marker: int,
    decision_plies: Sequence[int],
    device: torch.device,
):
    """1局のprefixを一度だけ再生し、指定局面のKV状態を保存する。

    旧実装は、同じprefixを読み筋ごとに再生していた。ここでは局面ごとの
    状態を共有し、各読み筋は保存済み状態から分岐させる。
    """
    requested = set(decision_plies)
    states = {}
    past_key_values = None
    recurrent_state = None
    with torch.inference_mode():
        for position, token_id in enumerate(prompt_ids):
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            recurrent_active = torch.tensor(
                [position > moves_marker], dtype=torch.bool, device=device
            )
            (
                next_logits,
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
            decision_ply = position - moves_marker
            if position >= moves_marker and decision_ply in requested:
                states[decision_ply] = (
                    next_logits[0, 0],
                    past_key_values,
                    recurrent_state,
                )
    missing = requested.difference(states)
    if missing:
        raise ValueError("prefix did not reach decision plies: {}".format(sorted(missing)))
    return states


def generate_line(
    model,
    initial_state,
    id_to_token: Mapping[int, str],
    syntactic_moves,
    line_length: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[str]:
    next_logits, past_key_values, recurrent_state = initial_state
    with torch.inference_mode():
        generated: List[str] = []
        position = past_key_values[0][0].shape[2] if past_key_values else 0
        for _ in range(line_length):
            token_id = sample_token(next_logits, temperature, top_p)
            token_text = id_to_token[token_id]
            # 分布はマスクしない。非指手が出た時点でその読み筋を終了する。
            if token_text not in syntactic_moves:
                break
            generated.append(token_text)
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


def load_records(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    if args.lines <= 0 or args.line_length <= 0:
        raise ValueError("lines and line-length must be positive")
    if args.positions_per_game == 0:
        raise ValueError("positions-per-game must be positive or negative for all")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    vocabulary = load_vocabulary(args.vocab)
    id_to_token = {index: token for token, index in vocabulary.items()}
    model, model_type, config = load_backbone(args.checkpoint, device, False)
    syntactic_moves = set(all_usi_move_tokens())
    moves_marker = 1 + 96
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_empty = 0
    processed_games = 0
    started_at = time.perf_counter()

    def report_progress() -> None:
        if args.progress_every <= 0:
            return
        if processed_games == 1 or processed_games % args.progress_every != 0:
            return
        elapsed = time.perf_counter() - started_at
        print(
            "[progress] games={} traces={} elapsed={:.1f}s".format(
                processed_games, written, elapsed
            ),
            flush=True,
        )

    maximum_trace_tokens = (
        1
        + args.lines * args.line_length
        + max(args.lines - 1, 0)
        + 4
    )

    with output_path.open("w", encoding="utf-8") as output:
        for record in load_records(args.input_jsonl):
            if args.max_games > 0 and processed_games >= args.max_games:
                break
            processed_games += 1
            moves = [str(move) for move in record["move_tokens"]]
            decision_count = (
                len(moves) if args.positions_per_game < 0 else args.positions_per_game
            )
            decision_plies = select_decision_plies(len(moves), decision_count)
            decision_plies = [
                decision_ply
                for decision_ply in decision_plies
                if 98 + decision_ply + maximum_trace_tokens <= config.max_seq_len
            ]
            if not decision_plies:
                report_progress()
                continue
            max_decision_ply = max(decision_plies)
            prompt_tokens = (
                ["<BOS>"]
                + list(record["initial_state_tokens"])
                + ["<MOVES>"]
                + moves[:max_decision_ply]
            )
            prompt_ids = [vocabulary[token] for token in prompt_tokens]
            prefix_states = advance_prefix(
                model,
                prompt_ids,
                moves_marker,
                decision_plies,
                device,
            )
            for decision_ply in decision_plies:
                history = moves[:decision_ply]
                reasoning_lines = []
                for _ in range(args.lines):
                    line = generate_line(
                        model,
                        prefix_states[decision_ply],
                        id_to_token,
                        syntactic_moves,
                        args.line_length,
                        args.temperature,
                        args.top_p,
                        device,
                    )
                    if line and line not in reasoning_lines:
                        reasoning_lines.append(line)
                if not reasoning_lines:
                    skipped_empty += 1
                    continue
                trace_record = {
                    "schema_version": 1,
                    "game_id": "{}:{}".format(record["game_id"], decision_ply),
                    "source_game_id": str(record["game_id"]),
                    "engine_scope": str(record.get("engine_scope", "")),
                    "start_sfen": str(record["initial_sfen"]),
                    "initial_state_tokens": list(record["initial_state_tokens"]),
                    "history_moves": history,
                    "target_move": moves[decision_ply],
                    "reasoning_lines": reasoning_lines,
                    "generation": {
                        "model_type": model_type,
                        "checkpoint": str(args.checkpoint),
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "requested_lines": args.lines,
                        "line_length": args.line_length,
                        "uses_legality_mask": False,
                        "uses_engine_search": False,
                    },
                }
                output.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
                written += 1

            report_progress()

    summary = {
        "input": args.input_jsonl,
        "output": str(output_path),
        "processed_games": processed_games,
        "written_traces": written,
        "skipped_empty": skipped_empty,
        "uses_legality_mask": False,
        "uses_engine_search": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
