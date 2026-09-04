#!/usr/bin/env python3
"""datasetのmanifestとvocabのhashを凍結記録として書き出す。

学習開始後にdatasetが差し替わっていないことを後から照合するために使う。
`verify_study_integrity.py`のdataset-hash検査はrun_manifest側を読むが，
こちらはdataset自身の側に残す独立した記録である。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from provenance import write_metrics_json


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="freeze a factorized-v3 dataset signature")
    parser.add_argument("dataset_dir")
    parser.add_argument("--output", help="既定はDATASET_DIR/dataset_frozen.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset_dir).expanduser().resolve()
    manifest = dataset / "dataset_manifest.json"
    vocab = dataset / "vocab.json"
    missing = [path.name for path in (manifest, vocab) if not path.is_file()]
    if missing:
        print(json.dumps({"event": "freeze_failed", "missing": missing}, ensure_ascii=False))
        return 2

    record = {
        "format_version": 1,
        "dataset_dir": str(dataset),
        "dataset_manifest_sha256": sha256_file(manifest),
        "vocab_sha256": sha256_file(vocab),
        "splits": {
            name: sha256_file(dataset / f"{name}.jsonl")
            for name in ("train", "validation", "evaluation")
        },
    }
    output = Path(args.output) if args.output else dataset / "dataset_frozen.json"
    write_metrics_json(output, record)
    print(json.dumps({"event": "dataset_frozen", "output": str(output),
                      "dataset_manifest_sha256": record["dataset_manifest_sha256"],
                      "vocab_sha256": record["vocab_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
