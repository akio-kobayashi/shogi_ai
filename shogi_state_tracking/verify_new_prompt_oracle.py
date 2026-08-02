#!/usr/bin/env python3
"""cshogiを用い，新prompt artifactの意味的な正しさをデータ作成機で照合する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_new_prompt_dataset import legal_sources_by_piece, state_targets
from create_dataset import import_cshogi
from new_prompt import encode_state_prompt, move_annotation


def compare(name, actual, expected, game_id, ply):
    if actual != expected:
        raise ValueError("{} mismatch game={} ply={}".format(name, game_id, ply))


def verify_record(record, cshogi_module):
    game_id = str(record["game_id"]); moves = [str(value) for value in record["move_tokens"]]
    annotations = record["move_annotations"]; candidates = {int(value["start_ply"]): value for value in record["start_candidates"]}
    steps = {int(value["ply"]): value for value in record.get("evaluation_steps", [])}
    board = cshogi_module.Board(str(record["initial_sfen"]))
    checked = {"moves": 0, "candidates": 0, "steps": 0, "probes": 0}
    for ply, move_usi in enumerate(moves):
        if ply in candidates:
            candidate = candidates[ply]
            compare("candidate start_sfen", str(candidate["start_sfen"]), board.sfen(), game_id, ply)
            compare("candidate state_prompt", candidate["state_prompt_tokens"], encode_state_prompt(board, cshogi_module), game_id, ply)
            compare("candidate probe_targets", candidate["probe_targets"], state_targets(board, cshogi_module), game_id, ply)
            checked["candidates"] += 1
        if ply in steps:
            step = steps[ply]
            compare("evaluation target_move", str(step["target_move"]), move_usi, game_id, ply)
            expected_legal = sorted(cshogi_module.move_to_usi(move) for move in board.legal_moves)
            compare("evaluation legal_moves", step["legal_moves"], expected_legal, game_id, ply)
            compare("evaluation legal_sources", step["legal_sources_by_piece"], legal_sources_by_piece(board, cshogi_module), game_id, ply)
            compare("evaluation probe_targets", step["probe_targets"], state_targets(board, cshogi_module), game_id, ply)
            checked["steps"] += 1
        move = board.move_from_usi(move_usi)
        if not board.is_legal(move):
            raise ValueError("illegal saved move game={} ply={}: {}".format(game_id, ply, move_usi))
        compare("move annotation", annotations[ply], move_annotation(board, move_usi), game_id, ply)
        board.push(move); checked["moves"] += 1

    for example in record.get("probe_examples", []):
        start = int(example["start_ply"]); ply = int(example["ply"])
        replay = cshogi_module.Board(str(record["initial_sfen"]))
        for move_usi in moves[:start]: replay.push(replay.move_from_usi(move_usi))
        compare("probe start prompt", example["state_prompt_tokens"], encode_state_prompt(replay, cshogi_module), game_id, ply)
        compare("probe history", example["history_moves"], moves[start:ply], game_id, ply)
        for move_usi in moves[start:ply]: replay.push(replay.move_from_usi(move_usi))
        compare("probe targets", example["probe_targets"], state_targets(replay, cshogi_module), game_id, ply)
        checked["probes"] += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description="新prompt artifactをcshogi oracleと照合する")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-games-per-split", type=int, default=0, help="0なら全gameを照合する")
    parser.add_argument("--output", help="検査結果JSON。省略時はdataset-dir/oracle_validation_report.json")
    args = parser.parse_args(); cshogi_module = import_cshogi(); directory = Path(args.dataset_dir)
    totals = {"games": 0, "moves": 0, "candidates": 0, "steps": 0, "probes": 0}; splits = {}
    for split in ("train", "validation", "evaluation"):
        counts = {key: 0 for key in totals}
        with (directory / (split + ".jsonl")).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or (args.max_games_per_split and counts["games"] >= args.max_games_per_split): continue
                checked = verify_record(json.loads(line), cshogi_module)
                counts["games"] += 1
                for key, value in checked.items(): counts[key] += value
        splits[split] = counts
        for key, value in counts.items(): totals[key] += value
    report = {"valid": True, "oracle": "cshogi", "max_games_per_split": args.max_games_per_split, "splits": splits, "total": totals}
    output = Path(args.output) if args.output else directory / "oracle_validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
