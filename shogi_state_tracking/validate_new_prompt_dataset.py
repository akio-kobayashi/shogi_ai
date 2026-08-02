#!/usr/bin/env python3
"""新prompt artifactのcshogi非依存の整合性検査。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

from new_prompt import (
    NEW_PROMPT_SCHEMA_VERSION,
    new_prompt_vocabulary_tokens,
    validate_move_annotations,
    validate_state_prompt_tokens,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: Mapping[str, object], expected_vocab: set[str]) -> int:
    if int(record.get("schema_version", -1)) != NEW_PROMPT_SCHEMA_VERSION:
        raise ValueError("unexpected schema_version")
    moves = record.get("move_tokens")
    annotations = record.get("move_annotations")
    candidates = record.get("start_candidates")
    if not isinstance(moves, list) or not isinstance(annotations, list) or not isinstance(candidates, list):
        raise ValueError("missing moves, annotations, or candidates")
    validate_move_annotations(moves, annotations)
    if not candidates:
        raise ValueError("record has no start candidate")
    previous_ply = -1
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid start candidate")
        ply = int(candidate.get("start_ply", -1))
        if not previous_ply < ply < len(moves):
            raise ValueError("candidate plies must be sorted and point to a move")
        previous_ply = ply
        tokens = candidate.get("state_prompt_tokens")
        if not isinstance(tokens, list):
            raise ValueError("candidate state prompt is absent")
        validate_state_prompt_tokens(tokens)
        unsupported = set(tokens) - expected_vocab
        if unsupported:
            raise ValueError("state prompt uses unsupported token: {}".format(sorted(unsupported)[0]))
        targets = candidate.get("probe_targets")
        if not isinstance(targets, dict) or len(targets.get("board_labels_cshogi_order", [])) != 81:
            raise ValueError("candidate has malformed probe targets")
    probe_examples = record.get("probe_examples")
    if not isinstance(probe_examples, list):
        raise ValueError("record has no probe_examples")
    for example in probe_examples:
        if not isinstance(example, dict):
            raise ValueError("invalid probe example")
        if int(example.get("start_ply", -1)) > int(example.get("ply", -1)):
            raise ValueError("probe example has negative history")
        if not isinstance(example.get("history_moves"), list):
            raise ValueError("probe example has no history_moves")
        validate_state_prompt_tokens(example.get("state_prompt_tokens", []))
        targets = example.get("probe_targets")
        if not isinstance(targets, dict) or len(targets.get("board_labels_cshogi_order", [])) != 81:
            raise ValueError("probe example has malformed probe targets")
    return len(moves)


def validate_split(path: Path, expected_vocab: set[str]) -> Dict[str, int]:
    games = moves = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    moves += validate_record(json.loads(line), expected_vocab)
                except Exception as exc:
                    raise ValueError("{}:{} {}".format(path, number, exc)) from exc
                games += 1
    if not games:
        raise ValueError("split is empty: {}".format(path))
    return {"games": games, "moves": moves}


def main() -> None:
    parser = argparse.ArgumentParser(description="新prompt dataset artifactを検査する")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", help="検査結果JSON。省略時は標準出力のみ")
    args = parser.parse_args()
    directory = Path(args.dataset_dir)
    manifest_path = directory / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != NEW_PROMPT_SCHEMA_VERSION:
        raise ValueError("manifest schema_version mismatch")
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    expected = new_prompt_vocabulary_tokens()
    if list(vocab.get("token_to_id", {}).keys()) != expected:
        raise ValueError("vocab token order does not match the schema")
    if sha256(directory / "vocab.json") != manifest.get("vocab_sha256"):
        raise ValueError("vocab checksum mismatch")
    report = {}
    for split, details in manifest["splits"].items():
        path = directory / str(details["file"])
        if sha256(path) != details.get("sha256"):
            raise ValueError("{} checksum mismatch".format(split))
        report[split] = validate_split(path, set(expected))
    result = {"valid": True, "splits": report}
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
