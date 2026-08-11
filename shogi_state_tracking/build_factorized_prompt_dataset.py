#!/usr/bin/env python3
"""旧new-prompt JSONLからfactorized_v3の固定初期局面datasetを作る．"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path

from create_dataset import import_cshogi
from build_new_prompt_dataset import state_targets
from factorized_prompt import (
    FACTORIZED_SCHEMA_VERSION,
    MOVE_ENCODING,
    annotation_piece_token,
    encode_state_prompt,
    factorize_usi,
    unfactorize_usi,
    validate_state_prompt_tokens,
    write_factorized_vocabulary,
)
from factorized_prompt_data import is_standard_initial_sfen
from new_prompt import validate_move_annotations


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_evaluation_steps(steps):
    result = []
    for source in steps or ():
        step = dict(source)
        step["legal_sources_by_piece"] = {
            annotation_piece_token(piece): list(squares)
            for piece, squares in dict(step.get("legal_sources_by_piece", {})).items()
        }
        result.append(step)
    return result


def transform_record(record, split, line_number, cshogi_module, token_to_id):
    moves = record.get("move_tokens")
    annotations = record.get("move_annotations")
    if not isinstance(moves, list) or not isinstance(annotations, list):
        raise ValueError("{}:{} missing move_tokens/move_annotations".format(split, line_number))
    if not is_standard_initial_sfen(str(record.get("initial_sfen", ""))):
        raise ValueError("{}:{} is not a standard-initial game".format(split, line_number))
    validate_move_annotations([str(value) for value in moves], [dict(value) for value in annotations])
    factorized = []
    for move in moves:
        tokens = factorize_usi(str(move))
        if unfactorize_usi(tokens) != str(move):
            raise ValueError("{}:{} USI round trip failed: {}".format(split, line_number, move))
        factorized.append(tokens)

    board = cshogi_module.Board(str(record["initial_sfen"]))
    state_tokens = encode_state_prompt(board, cshogi_module)
    validate_state_prompt_tokens(state_tokens)
    candidate = {
        "start_ply": 0,
        "start_sfen": str(record["initial_sfen"]),
        "state_prompt_tokens": state_tokens,
        "state_prompt_token_ids": [token_to_id[token] for token in state_tokens],
        "probe_targets": state_targets(board, cshogi_module),
        "position_scope": (record.get("position_scope_by_ply") or [record.get("position_scope", "unknown_position_scope")])[0],
    }

    normalized_annotations = []
    for annotation in annotations:
        value = dict(annotation)
        if bool(value.get("eligible", False)):
            value["piece"] = annotation_piece_token(str(value["piece"]))
        normalized_annotations.append(value)

    # 状態プローブは全て初期局面から当該plyまでの履歴に正規化する．
    probe_by_ply = {}
    # 駒打ち可能性は現在局面の合法手集合から得る派生ラベルである．
    # train/validationにも保存し，評価時に同じ定義の線形probeを学習できるようにする．
    legal_drop_available_by_ply = []
    promotion_choice_available_by_ply = []
    for source in record.get("probe_examples", ()):
        ply = int(source.get("ply", -1))
        if not 0 <= ply <= len(moves) or ply in probe_by_ply:
            continue
        probe_by_ply[ply] = {
            "start_ply": 0,
            "ply": ply,
            "state_prompt_tokens": state_tokens,
            "state_prompt_token_ids": candidate["state_prompt_token_ids"],
            "history_moves": [str(move) for move in moves[:ply]],
            "probe_targets": source.get("probe_targets"),
            "position_scope": source.get("position_scope", "unknown_position_scope"),
            "trajectory_scope": source.get("trajectory_scope", record.get("trajectory_scope", "unknown_position_scope")),
        }
    scopes = list(record.get("position_scope_by_ply", ()))
    replay = cshogi_module.Board(str(record["initial_sfen"]))
    for ply in range(len(moves) + 1):
        if ply in {8, 32} and ply not in probe_by_ply:
            probe_by_ply[ply] = {
                "start_ply": 0,
                "ply": ply,
                "state_prompt_tokens": state_tokens,
                "state_prompt_token_ids": candidate["state_prompt_token_ids"],
                "history_moves": [str(move) for move in moves[:ply]],
                "probe_targets": state_targets(replay, cshogi_module),
                "position_scope": scopes[ply] if ply < len(scopes) else "unknown_position_scope",
                "trajectory_scope": record.get("trajectory_scope", "unknown_position_scope"),
            }
        if ply < len(moves):
            legal_usi = [cshogi_module.move_to_usi(legal_move) for legal_move in replay.legal_moves]
            legal_drop_available_by_ply.append(any("*" in value for value in legal_usi))
            target_move = str(moves[ply])
            promotion_choice_available_by_ply.append(
                "*" not in target_move
                and target_move[:4] in legal_usi
                and target_move[:4] + "+" in legal_usi
            )
            move = replay.move_from_usi(str(moves[ply]))
            if not replay.is_legal(move):
                raise ValueError("{}:{} illegal move at ply {}".format(split, line_number, ply + 1))
            replay.push(move)

    output = dict(record)
    output.update({
        "schema_version": FACTORIZED_SCHEMA_VERSION,
        "move_encoding": MOVE_ENCODING,
        "state_prompt_tokens": state_tokens,
        "state_prompt_token_ids": candidate["state_prompt_token_ids"],
        "move_annotations": normalized_annotations,
        "factorized_move_ids": [[token_to_id[token] for token in tokens] for tokens in factorized],
        "legal_drop_available_by_ply": legal_drop_available_by_ply,
        "promotion_choice_available_by_ply": promotion_choice_available_by_ply,
        "start_candidates": [candidate],
        "probe_examples": [probe_by_ply[ply] for ply in sorted(probe_by_ply)],
    })
    if split == "evaluation":
        output["evaluation_steps"] = _canonical_evaluation_steps(record.get("evaluation_steps"))
        if not output["evaluation_steps"]:
            raise ValueError("evaluation:{} has no evaluation_steps".format(line_number))
    return output


def copy_split(source: Path, destination: Path, split: str, cshogi_module=None, token_to_id=None):
    cshogi_module = cshogi_module or import_cshogi()
    if token_to_id is None:
        token_to_id = write_factorized_vocabulary(destination.parent / "vocab.json")["token_to_id"]
    records = moves = probe_examples = rejected_nonstandard = 0
    lengths = array("I")
    with source.open(encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            source_record = json.loads(line)
            if not is_standard_initial_sfen(str(source_record.get("initial_sfen", ""))):
                rejected_nonstandard += 1
                continue
            record = transform_record(source_record, split, line_number, cshogi_module, token_to_id)
            output_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            records += 1
            moves += len(record["move_tokens"])
            probe_examples += len(record.get("probe_examples", []))
            hint_count = sum(bool(value.get("eligible", False)) for value in record["move_annotations"])
            lengths.append(3 + len(record["state_prompt_token_ids"]) + sum(len(value) for value in record["factorized_move_ids"]) + hint_count)
    length_path = destination.with_suffix(".lengths.u32")
    with length_path.open("wb") as handle:
        lengths.tofile(handle)
    if records == 0:
        raise ValueError("{} contains no standard-initial records".format(source))
    return {"records": records, "rejected_nonstandard": rejected_nonstandard, "moves": moves, "start_candidates": records, "probe_examples": probe_examples, "length_index": length_path.name, "sha256": sha256(destination)}


def main():
    parser = argparse.ArgumentParser(description="factorized_v3 datasetを構築する")
    parser.add_argument("--input-dir", required=True, help="既存new-prompt dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="train,validation,evaluation")
    args = parser.parse_args()
    source_root, output_root = Path(args.input_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    vocab_path = output_root / "vocab.json"
    vocabulary = write_factorized_vocabulary(vocab_path)
    cshogi_module = import_cshogi()
    split_metrics = {}
    for split in (value.strip() for value in args.splits.split(",") if value.strip()):
        source = source_root / (split + ".jsonl")
        if not source.is_file():
            raise FileNotFoundError(source)
        split_metrics[split] = copy_split(source, output_root / (split + ".jsonl"), split, cshogi_module, vocabulary["token_to_id"])
        print(json.dumps({"event": "split_complete", "split": split, **split_metrics[split]}, ensure_ascii=False), flush=True)
    source_manifest = source_root / "dataset_manifest.json"
    manifest = {
        "schema_version": FACTORIZED_SCHEMA_VERSION,
        "format": "shogi_canonical_state_prompt_factorized_moves",
        "move_encoding": MOVE_ENCODING,
        "state_prompt": "stored_for_future_explicit_start_experiments",
        "stage_1_2_input_mode": "implicit_standard_initial",
        "probe_annotations": ["legal_drop_available_by_ply", "promotion_choice_available_by_ply"],
        "source_dataset": str(source_root.resolve()),
        "source_manifest": str(source_manifest.resolve()) if source_manifest.is_file() else None,
        "vocab_sha256": sha256(vocab_path),
        "vocab_size": len(vocabulary["token_to_id"]),
        "splits": split_metrics,
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "dataset_complete", "output_dir": str(output_root.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
