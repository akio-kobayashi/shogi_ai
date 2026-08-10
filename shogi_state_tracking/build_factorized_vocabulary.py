#!/usr/bin/env python3
"""既存のnew-prompt JSONLに対応するfactorized_v2語彙を生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factorized_prompt import write_factorized_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="factorized_v2 vocabularyを生成する")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = write_factorized_vocabulary(args.output)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "schema_version": payload["schema_version"],
        "move_encoding": payload["move_encoding"],
        "vocab_size": len(payload["token_to_id"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

