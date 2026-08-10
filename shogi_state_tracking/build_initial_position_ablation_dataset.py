#!/usr/bin/env python3
"""A–D ablation用に平手初期局面開始の同一棋譜集合を作る。"""

from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path

from build_factorized_prompt_dataset import sha256, validate_record
from factorized_prompt import FACTORIZED_SCHEMA_VERSION, write_factorized_vocabulary
from factorized_prompt import factorize_usi
from factorized_prompt_data import is_standard_initial_sfen


def build_split(source: Path, destination: Path, split: str):
    input_records = output_records = moves = probes = 0
    with source.open(encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            input_records += 1
            record = json.loads(line)
            if not is_standard_initial_sfen(str(record.get("initial_sfen", ""))):
                continue
            candidates = [
                value for value in record.get("start_candidates", [])
                if int(value.get("start_ply", -1)) == 0
            ]
            if not candidates:
                continue
            # A–Dで同じ一つの開始候補を使う。明示条件だけがこのpromptを入力する。
            record["start_candidates"] = [candidates[0]]
            record["probe_examples"] = [
                value for value in record.get("probe_examples", [])
                if int(value.get("start_ply", -1)) == 0
            ]
            validate_record(record, split, line_number)
            output_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_records += 1
            moves += len(record["move_tokens"])
            probes += len(record["probe_examples"])
    if output_records == 0:
        raise ValueError("{} contains no standard-initial records with a ply-0 candidate".format(source))
    return {
        "input_records": input_records,
        "records": output_records,
        "excluded_records": input_records - output_records,
        "moves": moves,
        "probe_examples": probes,
        "sha256": sha256(destination),
    }


def build_runtime_split(source: Path, destination: Path, token_to_id):
    """学習に不要なprobe/evaluation情報を除き，指手を事前ID化する。"""
    records = 0
    lengths = array("I")
    with source.open(encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            compact = {
                "game_id": record["game_id"],
                "initial_sfen": record["initial_sfen"],
                "factorized_move_ids": [
                    [token_to_id[token] for token in factorize_usi(str(move))]
                    for move in record["move_tokens"]
                ],
                "move_annotations": record["move_annotations"],
                "start_candidates": [
                    {
                        "start_ply": int(candidate["start_ply"]),
                        "state_prompt_token_ids": [
                            token_to_id[str(token)] for token in candidate["state_prompt_tokens"]
                        ],
                    }
                    for candidate in record["start_candidates"]
                ],
            }
            lengths.append(
                3
                + len(compact["start_candidates"][0]["state_prompt_token_ids"])
                + sum(len(move) for move in compact["factorized_move_ids"])
            )
            output_handle.write(json.dumps(compact, separators=(",", ":")) + "\n")
            records += 1
    length_path = destination.with_suffix(".lengths.u32")
    with length_path.open("wb") as handle:
        lengths.tofile(handle)
    return {"records": records, "sha256": sha256(destination), "length_index": length_path.name}


def main():
    parser = argparse.ArgumentParser(description="標準初期局面2x2 ablation datasetを作る")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source_root, output_root = Path(args.input_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    vocab_path = output_root / "vocab.json"
    vocabulary = write_factorized_vocabulary(vocab_path)
    token_to_id = vocabulary["token_to_id"]
    splits = {}
    for split in ("train", "validation", "evaluation"):
        source = source_root / (split + ".jsonl")
        if not source.is_file():
            raise FileNotFoundError(source)
        splits[split] = build_split(source, output_root / (split + ".jsonl"), split)
        if split in {"train", "validation"}:
            splits[split]["runtime"] = build_runtime_split(
                output_root / (split + ".jsonl"),
                output_root / (split + ".runtime.jsonl"),
                token_to_id,
            )
        print(json.dumps({"event": "split_complete", "split": split, **splits[split]}, ensure_ascii=False), flush=True)
    manifest = {
        "schema_version": FACTORIZED_SCHEMA_VERSION,
        "format": "shogi_initial_position_prompt_ablation",
        "move_encoding": "factorized_v2",
        "start_selection": "fixed_initial",
        "standard_initial_only": True,
        "source_dataset": str(source_root.resolve()),
        "vocab_sha256": sha256(vocab_path),
        "vocab_size": len(vocabulary["token_to_id"]),
        "splits": splits,
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "dataset_complete", "output_dir": str(output_root.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
