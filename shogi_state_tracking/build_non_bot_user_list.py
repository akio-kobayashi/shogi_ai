#!/usr/bin/env python3
"""Build a verified non-BOT user list from an existing discovery cache.

This command performs no network access.  It is intended for reusing a
completed or partially completed discovery run when the collection scope is
changed from PRO/LP to all verified non-BOT users.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from discover_lishogi_titled_users import verified_non_bot_users


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("discovery_dir", type=Path)
    args = parser.parse_args()
    profile_path = args.discovery_dir / "profile_cache.json"
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    value = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{profile_path} must contain a JSON object")
    users = verified_non_bot_users(value)
    json_path = args.discovery_dir / "non_bot_users.json"
    text_path = args.discovery_dir / "non_bot_users.txt"
    json_path.write_text(json.dumps(users, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(
        "".join(str(row["username"]) + "\n" for row in users),
        encoding="utf-8",
    )
    manifest_path = args.discovery_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            counts = manifest.setdefault("counts", {})
            files = manifest.setdefault("files", {})
            if isinstance(counts, dict):
                counts["verified_non_bot_users"] = len(users)
            if isinstance(files, dict):
                files["non_bot_user_list"] = text_path.name
                files["non_bot_users"] = json_path.name
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    print(json.dumps({"verified_non_bot_users": len(users), "text_file": str(text_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
