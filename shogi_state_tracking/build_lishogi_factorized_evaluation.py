#!/usr/bin/env python3
"""Convert normalized non-BOT Lishogi games into an evaluation-only dataset.

The input is ``collect_lishogi_games.py``'s privacy-preserving ``games.jsonl``.
All games remain evaluation-only; this command never creates training labels from
the collected games.  The output uses the same factorized_v3 vocabulary and
evaluation-step schema as the trained model.
"""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence

from build_factorized_prompt_dataset import transform_record
from build_new_prompt_dataset import materialize_record
from create_dataset import import_cshogi
from factorized_prompt import FACTORIZED_SCHEMA_VERSION, TERMINAL_ENCODING, MOVE_ENCODING, write_factorized_vocabulary


def read_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield value


def initial_sfen(record: Mapping[str, object], cshogi_module) -> str:
    value = str(record.get("initial_sfen") or "startpos")
    board = cshogi_module.Board() if value == "startpos" else cshogi_module.Board(value)
    # Keep the move number, although position-scope hashing ignores it.  The
    # factorized prompt validator accepts the canonical full SFEN form.
    return str(board.sfen())


def game_result(record: Mapping[str, object]) -> int:
    winner = str(record.get("winner") or "").lower()
    if winner in {"sente", "black"}:
        return 1
    if winner in {"gote", "white"}:
        return -1
    raise ValueError("evaluation requires a decisive game with winner=sente/gote")


def source_record(record: Mapping[str, object], cshogi_module) -> Dict[str, object]:
    game_id = str(record.get("source_game_id") or "")
    if not game_id:
        raise ValueError("source game has no source_game_id")
    moves = record.get("moves_usi")
    if not isinstance(moves, list) or not moves:
        raise ValueError(f"{game_id} has no moves_usi")
    normalized_moves = [str(move) for move in moves]
    # The collected games are external to the training corpus.  The labels are
    # intentionally conservative: they describe dataset provenance, not a
    # claim that every trajectory is globally unseen in every corpus.
    return {
        "game_id": "lishogi:" + game_id,
        "game_result": game_result(record),
        "split": "evaluation",
        "initial_sfen": initial_sfen(record, cshogi_module),
        "move_tokens": normalized_moves,
        "player_scope": "external_non_bot",
        "engine_scope": "external_non_bot",
        "position_scope": "unseen_position",
        "position_scope_by_ply": ["unseen_position"] * len(normalized_moves),
        "trajectory_scope": "strict_unseen_position",
        "source": "lishogi_public_api_non_bot",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--min-plies", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_games <= 0 or args.min_plies < 0:
        raise ValueError("--max-games must be positive and --min-plies nonnegative")

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation.jsonl"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path}; use --overwrite to replace it")

    cshogi_module = import_cshogi()
    vocabulary = write_factorized_vocabulary(output_dir / "vocab.json")
    token_to_id = vocabulary["token_to_id"]
    seen_ids = set()
    records = 0
    moves = 0
    rejected: Dict[str, int] = {}
    lengths = array("I")
    with output_path.open("w", encoding="utf-8") as output:
        for raw in read_jsonl(input_path):
            if records >= args.max_games:
                break
            game_id = str(raw.get("source_game_id") or "")
            if game_id in seen_ids:
                rejected["duplicate_game"] = rejected.get("duplicate_game", 0) + 1
                continue
            seen_ids.add(game_id)
            raw_moves = raw.get("moves_usi")
            if not isinstance(raw_moves, list) or len(raw_moves) < args.min_plies:
                rejected["too_short"] = rejected.get("too_short", 0) + 1
                continue
            try:
                source = source_record(raw, cshogi_module)
                intermediate = materialize_record(
                    source,
                    cshogi_module,
                    required_start_plies=(0,),
                    candidate_count=1,
                    min_suffix_moves=1,
                    seed=0,
                    include_evaluation_steps=True,
                    probe_offsets=(0, 8, 32),
                    probe_start_plies=(0,),
                )
                factorized = transform_record(
                    intermediate,
                    "evaluation",
                    records + 1,
                    cshogi_module,
                    token_to_id,
                )
            except Exception as exc:
                reason = type(exc).__name__ + ":" + str(exc)
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            json.dump(factorized, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")
            records += 1
            moves += len(factorized["move_tokens"])
            lengths.append(
                3
                + len(factorized["state_prompt_token_ids"])
                + sum(len(value) for value in factorized["factorized_move_ids"])
                + sum(bool(value.get("eligible", False)) for value in factorized["move_annotations"])
            )

    if records == 0:
        raise ValueError(f"no qualifying evaluation games were written from {input_path}")
    length_path = output_path.with_suffix(".lengths.u32")
    with length_path.open("wb") as handle:
        lengths.tofile(handle)
    manifest = {
        "schema_version": FACTORIZED_SCHEMA_VERSION,
        "format": "shogi_external_non_bot_factorized_evaluation",
        "move_encoding": MOVE_ENCODING,
        "terminal_encoding": TERMINAL_ENCODING,
        "terminal_supervision": "complete_game_only",
        "stage_1_2_input_mode": "implicit_standard_initial",
        "probe_annotations": ["legal_drop_available_by_ply", "promotion_choice_available_by_ply"],
        "state_prompt": "stored_for_future_explicit_start_experiments",
        "evaluation_only": True,
        "source_jsonl": str(input_path.resolve()),
        "source_scope": "verified_non_bot_registered_users",
        "records": records,
        "moves": moves,
        "min_plies": args.min_plies,
        "max_games": args.max_games,
        "rejected": rejected,
        "files": {"evaluation": output_path.name, "vocab": "vocab.json", "length_index": length_path.name},
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "evaluation_dataset_complete", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
