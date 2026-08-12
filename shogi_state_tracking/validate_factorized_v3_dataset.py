#!/usr/bin/env python3
"""factorized_v3 artifactの形式・語彙・注釈整合性を学習前に検査する．"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import load_vocabulary
from factorized_prompt import FACTORIZED_SCHEMA_VERSION, MOVE_ENCODING, TERMINAL_ENCODING, annotation_piece_token, factorize_usi, factorized_vocabulary_tokens, validate_state_prompt_tokens
from factorized_prompt_data import is_standard_initial_sfen
from new_prompt import source_square_from_usi, square_token


def validate_record(record, vocabulary, split, line_number):
    where = "{}:{}".format(split, line_number)
    if int(record.get("schema_version", -1)) != FACTORIZED_SCHEMA_VERSION or record.get("move_encoding") != MOVE_ENCODING:
        raise ValueError("{} has an obsolete schema".format(where))
    if record.get("terminal_encoding") != TERMINAL_ENCODING or record.get("terminal_token") != "<EOS>":
        raise ValueError("{} has no current terminal supervision declaration".format(where))
    if int(record.get("game_result", 0)) == 0:
        raise ValueError("{} contains a draw game".format(where))
    if not is_standard_initial_sfen(str(record.get("initial_sfen", ""))):
        raise ValueError("{} is not standard-initial".format(where))
    state = [str(token) for token in record.get("state_prompt_tokens", ())]
    validate_state_prompt_tokens(state)
    if len(state) != 85:
        raise ValueError("{} standard initial prompt must contain 85 tokens".format(where))
    if [vocabulary[token] for token in state] != list(record.get("state_prompt_token_ids", ())):
        raise ValueError("{} state prompt IDs disagree with vocab".format(where))
    candidates = record.get("start_candidates", ())
    if len(candidates) != 1 or int(candidates[0].get("start_ply", -1)) != 0 or list(candidates[0].get("state_prompt_tokens", ())) != state:
        raise ValueError("{} must contain one canonical ply-0 candidate".format(where))
    moves = [str(move) for move in record.get("move_tokens", ())]
    annotations = list(record.get("move_annotations", ()))
    ids = list(record.get("factorized_move_ids", ()))
    legal_drop = record.get("legal_drop_available_by_ply")
    promotion_choice = record.get("promotion_choice_available_by_ply")
    if not moves or len(moves) != len(annotations) or len(moves) != len(ids):
        raise ValueError("{} move arrays differ in length".format(where))
    if not isinstance(legal_drop, list) or len(legal_drop) != len(moves) or any(not isinstance(value, bool) for value in legal_drop):
        raise ValueError("{} legal_drop_available_by_ply must be a boolean per move".format(where))
    if not isinstance(promotion_choice, list) or len(promotion_choice) != len(moves) or any(not isinstance(value, bool) for value in promotion_choice):
        raise ValueError("{} promotion_choice_available_by_ply must be a boolean per move".format(where))
    for index, (move, annotation, stored_ids) in enumerate(zip(moves, annotations, ids)):
        tokens = factorize_usi(move)
        if [vocabulary[token] for token in tokens] != list(stored_ids):
            raise ValueError("{} move {} IDs disagree with vocab".format(where, index))
        if bool(promotion_choice[index]) and "*" in move:
            raise ValueError("{} drop move {} cannot have an optional-promotion label".format(where, index))
        if "*" in move and not legal_drop[index]:
            raise ValueError("{} actual drop move {} contradicts legal_drop_available".format(where, index))
        if "*" in move:
            if bool(annotation.get("eligible", False)):
                raise ValueError("{} drop {} has a RAP annotation".format(where, index))
        else:
            if not bool(annotation.get("eligible", False)):
                raise ValueError("{} normal move {} lacks RAP annotation".format(where, index))
            annotation_piece_token(str(annotation.get("piece", "")))
            if str(annotation.get("source", "")) != square_token(source_square_from_usi(move)):
                raise ValueError("{} move {} has a wrong source annotation".format(where, index))
    if split == "evaluation" and len(record.get("evaluation_steps", ())) != len(moves):
        raise ValueError("{} evaluation_steps must cover every move".format(where))
    return len(moves)


def main():
    parser = argparse.ArgumentParser(description="factorized_v3 datasetを検査する")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--splits", default="train,validation,evaluation")
    parser.add_argument("--max-records", type=int, default=0, help="0は全件")
    args = parser.parse_args()
    root = Path(args.dataset_dir)
    vocabulary = load_vocabulary(root / "vocab.json")
    if list(vocabulary) != factorized_vocabulary_tokens():
        raise ValueError("vocab.json is not the canonical 125-token vocabulary")
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FACTORIZED_SCHEMA_VERSION or manifest.get("move_encoding") != MOVE_ENCODING:
        raise ValueError("dataset manifest is obsolete")
    if manifest.get("terminal_encoding") != TERMINAL_ENCODING or manifest.get("terminal_supervision") != "complete_game_only":
        raise ValueError("dataset manifest does not declare complete-game EOS supervision")
    if manifest.get("stage_1_2_input_mode") != "implicit_standard_initial":
        raise ValueError(
            "dataset manifest does not declare stage_1_2_input_mode=implicit_standard_initial; "
            "rebuild the factorized_v3 dataset"
        )
    if manifest.get("probe_annotations") != [
        "legal_drop_available_by_ply", "promotion_choice_available_by_ply"
    ]:
        raise ValueError(
            "dataset manifest does not declare the current probe annotations; "
            "rebuild the factorized_v3 dataset"
        )
    summary = {}
    for split in (value.strip() for value in args.splits.split(",") if value.strip()):
        records = moves = 0
        with (root / (split + ".jsonl")).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                moves += validate_record(json.loads(line), vocabulary, split, line_number)
                records += 1
                if args.max_records and records >= args.max_records:
                    break
        summary[split] = {"records": records, "moves": moves}
    print(json.dumps({"event": "validation_complete", "schema_version": FACTORIZED_SCHEMA_VERSION, "vocab_size": len(vocabulary), "splits": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
