#!/usr/bin/env python3
"""既存new-prompt artifactを検査し，factorized_v2 datasetとして包装する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from factorized_prompt import FACTORIZED_SCHEMA_VERSION, factorize_usi, unfactorize_usi, write_factorized_vocabulary
from new_prompt import validate_move_annotations, validate_state_prompt_tokens


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record, split, line_number):
    moves = record.get("move_tokens")
    annotations = record.get("move_annotations")
    if not isinstance(moves, list) or not isinstance(annotations, list):
        raise ValueError("{}:{} missing move_tokens/move_annotations".format(split, line_number))
    validate_move_annotations([str(value) for value in moves], [dict(value) for value in annotations])
    for move in moves:
        tokens = factorize_usi(str(move))
        if unfactorize_usi(tokens) != str(move):
            raise ValueError("{}:{} USI round trip failed: {}".format(split, line_number, move))
    candidates = record.get("start_candidates", [])
    if not candidates:
        raise ValueError("{}:{} has no start candidates".format(split, line_number))
    for candidate in candidates:
        validate_state_prompt_tokens([str(value) for value in candidate["state_prompt_tokens"]])
        if not 0 <= int(candidate["start_ply"]) < len(moves):
            raise ValueError("{}:{} candidate start_ply is invalid".format(split, line_number))
    if split == "evaluation" and not record.get("evaluation_steps"):
        raise ValueError("evaluation:{} has no evaluation_steps".format(line_number))


def copy_split(source: Path, destination: Path, split: str):
    records = moves = candidates = probe_examples = 0
    with source.open(encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record, split, line_number)
            output_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            records += 1
            moves += len(record["move_tokens"])
            candidates += len(record.get("start_candidates", []))
            probe_examples += len(record.get("probe_examples", []))
    return {"records": records, "moves": moves, "start_candidates": candidates, "probe_examples": probe_examples, "sha256": sha256(destination)}


def main():
    parser = argparse.ArgumentParser(description="factorized_v2 datasetを構築する")
    parser.add_argument("--input-dir", required=True, help="既存new-prompt dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="train,validation,evaluation")
    args = parser.parse_args()
    source_root, output_root = Path(args.input_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    vocab_path = output_root / "vocab.json"
    vocabulary = write_factorized_vocabulary(vocab_path)
    split_metrics = {}
    for split in (value.strip() for value in args.splits.split(",") if value.strip()):
        source = source_root / (split + ".jsonl")
        if not source.is_file():
            raise FileNotFoundError(source)
        split_metrics[split] = copy_split(source, output_root / (split + ".jsonl"), split)
        print(json.dumps({"event": "split_complete", "split": split, **split_metrics[split]}, ensure_ascii=False), flush=True)
    source_manifest = source_root / "dataset_manifest.json"
    manifest = {
        "schema_version": FACTORIZED_SCHEMA_VERSION,
        "format": "shogi_piece_coordinate_prompt_factorized_moves",
        "move_encoding": "factorized_v2",
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

