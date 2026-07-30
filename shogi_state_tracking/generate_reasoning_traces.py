#!/usr/bin/env python3
"""指手予測backbone自身からCoT-likeな複数読み筋を生成する。"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Mapping, Sequence

import torch

from create_dataset import all_usi_move_tokens
from data import load_vocabulary
from evaluate_probes import load_backbone, resolve_device


LOGGER_NAME = "shogi_state_tracking.generate_reasoning_traces"


def configure_logging(log_file: Path, level: str) -> logging.Logger:
    """標準出力とファイルへ同じ実行ログを出す。"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def install_exception_logging(logger: logging.Logger) -> None:
    """未処理例外もログファイルへ残す。"""
    default_hook = sys.excepthook

    def exception_hook(exc_type, exc_value, traceback):
        logger.error(
            "run_failed",
            exc_info=(exc_type, exc_value, traceback),
        )
        default_hook(exc_type, exc_value, traceback)

    sys.excepthook = exception_hook


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
    parser.add_argument(
        "--line-batch-size",
        type=int,
        default=0,
        help="同一局面から同時生成する読み筋数。0なら--linesと同じ",
    )
    parser.add_argument("--line-length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument(
        "--log-file",
        default="",
        help="実行ログの保存先。空ならoutput-jsonl.log",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="実行summaryの保存先。空ならoutput-jsonl.summary.json",
    )
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
    """指手が存在するdecision plyを0..total_moves-1から等間隔に選ぶ。"""
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


def _sample_token_batch(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """バッチ内の各行から1 tokenをサンプルする。

    読み筋を1本ずつ生成すると、同じprefixからのTransformer計算をPython
    ループで繰り返すことになる。ここではlogitsを[lines, vocab]として扱い、
    samplingとTransformerのstepをまとめて実行する。

    top-pのための並べ替えは語彙全体に対して必要なので、語彙数を変えずに
    nucleus samplingの意味を保つ。sort対象をlogitsにすることで、softmax後の
    sortより中間テンソルを一つ減らす。
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if temperature <= 0:
        return logits.argmax(dim=-1)

    scaled = logits.float() / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = scaled.sort(dim=-1, descending=True)
        normalizer = torch.logsumexp(scaled, dim=-1, keepdim=True)
        sorted_probabilities = (sorted_logits - normalizer).exp()
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_rank = torch.multinomial(sorted_probabilities, 1)
        return sorted_indices.gather(1, sampled_rank).squeeze(1)

    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, 1).squeeze(1)


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> int:
    """互換用の単一行sampling API。

    evaluate_reasoning.pyからも利用されるため、公開関数として残す。
    実際のtrace生成では下のバッチ版を使う。
    """
    if logits.ndim != 1:
        raise ValueError("logits must have shape [vocab]")
    return int(_sample_token_batch(logits.unsqueeze(0), temperature, top_p)[0])


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
    prompt_tensor = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    inactive = torch.zeros(1, dtype=torch.bool, device=device)
    active = torch.ones(1, dtype=torch.bool, device=device)
    with torch.inference_mode():
        for position, token_id in enumerate(prompt_ids):
            # 毎tokenのtensor生成を避け、既存promptのsliceをそのまま渡す。
            token = prompt_tensor[position : position + 1].view(1, 1)
            recurrent_active = active if position > moves_marker else inactive
            decision_ply = position - moves_marker
            needs_logits = position >= moves_marker and decision_ply in requested
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
                return_logits=needs_logits,
            )
            if needs_logits:
                states[decision_ply] = (
                    next_logits[0, 0],
                    past_key_values,
                    recurrent_state,
                )
    missing = requested.difference(states)
    if missing:
        raise ValueError("prefix did not reach decision plies: {}".format(sorted(missing)))
    return states


def advance_prefix_vanilla(
    model,
    prompt_ids: Sequence[int],
    moves_marker: int,
    decision_plies: Sequence[int],
    device: torch.device,
):
    """Vanilla decoderのprefixをparallel forwardで一度だけ評価する。

    Vanillaでは時間再帰がないため、tokenごとのstepを繰り返す必要はない。
    通常のcausal forwardで全prefixのlogitsと各層のK/Vを得て、指定位置までの
    K/V viewを各decision stateとして返す。T²MLRはこの経路を使わず、厳密な
    recurrenceを保つためadvance_prefix()を使う。
    """
    requested = set(decision_plies)
    if not requested:
        return {}
    input_ids = torch.as_tensor(
        prompt_ids, dtype=torch.long, device=device
    ).view(1, -1)
    with torch.inference_mode():
        # model.forward()だと全prefix位置のvocab logitsまで計算する。語彙が
        # 大きいので、ここではbackboneだけをparallel実行し、要求位置のlogits
        # だけを最後にlm_headへ通す。
        x = model._embed(input_ids)
        key_values = []
        for layer in model.layers:
            x, key_value = layer.forward_with_cache(x)
        positions = [moves_marker + decision_ply for decision_ply in requested]
        position_tensor = torch.as_tensor(
            positions, dtype=torch.long, device=device
        )
        selected_hidden = model.final_norm(x[:, position_tensor, :])
        selected_logits = model.lm_head(selected_hidden)[0]
        logits_by_position = {
            position: selected_logits[index]
            for index, position in enumerate(positions)
        }
        states = {}
        for decision_ply in requested:
            position = moves_marker + decision_ply
            if position >= input_ids.shape[1]:
                continue
            prefix_length = position + 1
            states[decision_ply] = (
                logits_by_position[position],
                tuple(
                    (key[:, :, :prefix_length], value[:, :, :prefix_length])
                    for key, value in key_values
                ),
                None,
            )
    missing = requested.difference(states)
    if missing:
        raise ValueError("prefix did not reach decision plies: {}".format(sorted(missing)))
    return states


def _expand_generation_state(initial_state, batch_size: int):
    """1本分のKV状態を、読み筋batch用にexpandする。

    expandはbatch方向のviewだけを作るため、prefix cacheを読み筋数分コピー
    しない。次のmodel.step内で新しいKVが連結されるため、元のprefix stateは
    破壊されない。
    """
    next_logits, past_key_values, recurrent_state = initial_state
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batched_logits = next_logits.unsqueeze(0).expand(batch_size, -1)
    if past_key_values is not None:
        batched_past = tuple(
            (
                past_key.expand(batch_size, -1, -1, -1),
                past_value.expand(batch_size, -1, -1, -1),
            )
            for past_key, past_value in past_key_values
        )
    else:
        batched_past = None
    if recurrent_state is not None:
        batched_recurrent = recurrent_state.expand(
            batch_size, -1, -1
        )
    else:
        batched_recurrent = None
    return batched_logits, batched_past, batched_recurrent


def generate_lines(
    model,
    initial_state,
    id_to_token: Sequence[str],
    move_token_mask: torch.Tensor,
    line_count: int,
    line_length: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    device: torch.device,
) -> List[List[str]]:
    """同じ局面から複数の読み筋をbatch生成する。

    旧実装では、各lineについてmodel.stepをline_length回呼んでいた。
    ここではbatch次元をlineに割り当て、同じstepを一度だけ呼ぶ。
    生成されたtokenが指手語彙でない行はその時点で終了するが、batch全体の
    shapeを保つために残りの行と一緒にplaceholderを1回だけ流す。
    """
    if line_count <= 0 or line_length <= 0:
        return []
    next_logits, past_key_values, recurrent_state = _expand_generation_state(
        initial_state, line_count
    )
    generated: List[List[str]] = [[] for _ in range(line_count)]
    sampled_ids: List[torch.Tensor] = []
    sampled_valid: List[torch.Tensor] = []
    active = torch.ones(line_count, dtype=torch.bool, device=device)
    position = past_key_values[0][0].shape[2] if past_key_values else 0
    placeholder = torch.full(
        (line_count,), int(pad_token_id), dtype=torch.long, device=device
    )

    with torch.inference_mode():
        for _ in range(line_length):
            if not bool(active.any()):
                break
            sampled = _sample_token_batch(next_logits, temperature, top_p)
            selected = torch.where(active, sampled, placeholder)
            valid = active & move_token_mask[selected]
            # GPUからの転送はstepごとに行わず、生成終了後にまとめて行う。
            sampled_ids.append(selected)
            sampled_valid.append(valid)
            active = valid

            # 全行を一度に進める。終了した行の状態は以後参照しない。
            logits, past_key_values, recurrent_state, _, _ = model.step(
                selected[:, None],
                position,
                past_key_values,
                recurrent_state,
                active,
            )
            next_logits = logits[:, 0]
            position += 1

    if sampled_ids:
        ids_cpu = torch.stack(sampled_ids).detach().cpu()
        valid_cpu = torch.stack(sampled_valid).detach().cpu()
        for step_ids, step_valid in zip(ids_cpu, valid_cpu):
            for row, (token_id, is_valid) in enumerate(
                zip(step_ids.tolist(), step_valid.tolist())
            ):
                if is_valid:
                    generated[row].append(id_to_token[token_id])
    return generated


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
    # 外部からの既存呼出しとの互換用。mainのtrace生成はbatch版を使う。
    if isinstance(id_to_token, Mapping):
        max_token_id = max(id_to_token)
        token_list = ["<UNKNOWN>"] * (max_token_id + 1)
        for token_id, token_text in id_to_token.items():
            token_list[token_id] = token_text
    else:
        token_list = list(id_to_token)
    move_token_mask = torch.tensor(
        [token in syntactic_moves for token in token_list],
        dtype=torch.bool,
        device=device,
    )
    return generate_lines(
        model,
        initial_state,
        token_list,
        move_token_mask,
        line_count=1,
        line_length=line_length,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=0,
        device=device,
    )[0]


def load_records(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    if args.lines <= 0 or args.line_length <= 0:
        raise ValueError("lines and line-length must be positive")
    if args.line_batch_size < 0:
        raise ValueError("line-batch-size must be non-negative")
    if args.positions_per_game == 0:
        raise ValueError("positions-per-game must be positive or negative for all")
    output_path = Path(args.output_jsonl)
    log_file = (
        Path(args.log_file)
        if args.log_file
        else Path(str(output_path) + ".log")
    )
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else Path(str(output_path) + ".summary.json")
    )
    logger = configure_logging(log_file, args.log_level)
    install_exception_logging(logger)
    started_at = time.perf_counter()
    logger.info(
        "run_start args=%s",
        json.dumps(vars(args), ensure_ascii=False, sort_keys=True),
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    vocabulary = load_vocabulary(args.vocab)
    max_token_id = max(vocabulary.values())
    id_to_token = ["<UNKNOWN>"] * (max_token_id + 1)
    for token, index in vocabulary.items():
        id_to_token[index] = token
    model, model_type, config = load_backbone(args.checkpoint, device, False)
    syntactic_moves = set(all_usi_move_tokens())
    move_token_mask = torch.tensor(
        [token in syntactic_moves for token in id_to_token],
        dtype=torch.bool,
        device=device,
    )
    pad_token_id = vocabulary.get("<PAD>", 0)
    moves_marker = 1 + 96
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_empty = 0
    processed_games = 0
    total_decisions = 0
    total_generated_lines = 0
    logger.info(
        "model_ready model_type=%s device=%s vocab_size=%d max_seq_len=%d config=%s",
        model_type,
        device,
        len(vocabulary),
        config.max_seq_len,
        json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True),
    )
    logger.info(
        "paths input=%s output=%s log=%s summary=%s checkpoint=%s",
        args.input_jsonl,
        args.output_jsonl,
        log_file,
        summary_path,
        args.checkpoint,
    )

    def report_progress() -> None:
        if args.progress_every <= 0:
            return
        if processed_games == 1 or processed_games % args.progress_every != 0:
            return
        elapsed = time.perf_counter() - started_at
        logger.info(
            "progress games=%d traces=%d decisions=%d generated_lines=%d "
            "skipped_empty=%d elapsed_sec=%.1f games_per_sec=%.3f",
            processed_games,
            written,
            total_decisions,
            total_generated_lines,
            skipped_empty,
            elapsed,
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
            game_started_at = time.perf_counter()
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
            logger.info(
                "game_start index=%d game_id=%s moves=%d decisions=%s",
                processed_games,
                record.get("game_id", ""),
                len(moves),
                decision_plies,
            )
            if not decision_plies:
                logger.warning(
                    "game_skip game_id=%s reason=no_decision_ply_within_max_seq_len",
                    record.get("game_id", ""),
                )
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
            if model_type == "vanilla":
                prefix_states = advance_prefix_vanilla(
                    model,
                    prompt_ids,
                    moves_marker,
                    decision_plies,
                    device,
                )
            else:
                prefix_states = advance_prefix(
                    model,
                    prompt_ids,
                    moves_marker,
                    decision_plies,
                    device,
                )
            for decision_ply in decision_plies:
                decision_started_at = time.perf_counter()
                total_decisions += 1
                history = moves[:decision_ply]
                reasoning_lines = []
                line_batch_size = args.line_batch_size or args.lines
                generated_lines: List[List[str]] = []
                for line_start in range(0, args.lines, line_batch_size):
                    generated_lines.extend(
                        generate_lines(
                            model,
                            prefix_states[decision_ply],
                            id_to_token,
                            move_token_mask,
                            min(line_batch_size, args.lines - line_start),
                            args.line_length,
                            args.temperature,
                            args.top_p,
                            pad_token_id,
                            device,
                        )
                    )
                for line in generated_lines:
                    if line and line not in reasoning_lines:
                        reasoning_lines.append(line)
                total_generated_lines += sum(bool(line) for line in generated_lines)
                if not reasoning_lines:
                    skipped_empty += 1
                    logger.warning(
                        "decision_empty game_id=%s ply=%d elapsed_sec=%.3f",
                        record.get("game_id", ""),
                        decision_ply,
                        time.perf_counter() - decision_started_at,
                    )
                    continue
                trace_record = {
                    "schema_version": 1,
                    "game_id": "{}:{}".format(record["game_id"], decision_ply),
                    "source_game_id": str(record["game_id"]),
                    "player_scope": str(
                        record.get("player_scope", record.get("engine_scope", ""))
                    ),
                    "engine_scope": str(record.get("engine_scope", "")),
                    "position_scope": str(
                        record.get("position_scope", "unknown_position_scope")
                    ),
                    "trajectory_scope": str(
                        record.get("trajectory_scope", "unknown_position_scope")
                    ),
                    "start_sfen": str(record["initial_sfen"]),
                    "initial_state_tokens": list(record["initial_state_tokens"]),
                    "history_moves": history,
                    "target_move": moves[decision_ply],
                    "reasoning_lines": reasoning_lines,
                    "generation": {
                        "model_type": model_type,
                        "checkpoint": str(args.checkpoint),
                        "seed": args.seed,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "requested_lines": args.lines,
                        "line_batch_size": args.line_batch_size or args.lines,
                        "line_length": args.line_length,
                        "uses_legality_mask": False,
                        "uses_engine_search": False,
                    },
                }
                output.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
                written += 1
                logger.info(
                    "decision_complete game_id=%s ply=%d unique_lines=%d "
                    "elapsed_sec=%.3f",
                    record.get("game_id", ""),
                    decision_ply,
                    len(reasoning_lines),
                    time.perf_counter() - decision_started_at,
                )

            output.flush()
            logger.info(
                "game_complete index=%d game_id=%s traces_total=%d "
                "elapsed_sec=%.3f",
                processed_games,
                record.get("game_id", ""),
                written,
                time.perf_counter() - game_started_at,
            )
            report_progress()

    elapsed_sec = time.perf_counter() - started_at
    summary = {
        "input": args.input_jsonl,
        "output": str(output_path),
        "processed_games": processed_games,
        "written_traces": written,
        "skipped_empty": skipped_empty,
        "total_decisions": total_decisions,
        "generated_lines": total_generated_lines,
        "elapsed_sec": elapsed_sec,
        "games_per_sec": processed_games / elapsed_sec if elapsed_sec > 0 else 0.0,
        "traces_per_sec": written / elapsed_sec if elapsed_sec > 0 else 0.0,
        "generated_lines_per_sec": (
            total_generated_lines / elapsed_sec if elapsed_sec > 0 else 0.0
        ),
        "log_file": str(log_file),
        "summary_file": str(summary_path),
        "uses_legality_mask": False,
        "uses_engine_search": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "run_complete summary=%s",
        json.dumps(summary, ensure_ascii=False, sort_keys=True),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
