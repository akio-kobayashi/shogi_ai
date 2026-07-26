#!/usr/bin/env python3
"""Create and verify checksums for an experiment migration bundle.

The virtual environment and uv cache are intentionally not part of a bundle;
they are platform- and accelerator-specific.  This utility verifies the files
that are portable: source code, lock files, generated datasets, and optional
results/checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


SCHEMA_VERSION = 1
MANIFEST_NAME = "MIGRATION_MANIFEST.json"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file_paths(root: Path, excluded: Path) -> Iterable[Path]:
    """Yield regular files below root in deterministic POSIX order."""
    excluded = excluded.resolve()
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.resolve() == excluded:
            continue
        paths.append(path.relative_to(root))
    for path in sorted(paths, key=lambda value: value.as_posix()):
        yield path


def validate_relative_path(value: str) -> Path:
    """Reject absolute and parent-traversal paths from a manifest."""
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError("unsafe manifest path: {}".format(value))
    if value in ("", "."):
        raise ValueError("empty manifest path")
    return Path(*posix.parts)


def write_manifest(
    root: Path,
    output: Path,
    source_commit: str,
    data_mode: str,
    artifacts: str,
) -> None:
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise ValueError("bundle root does not exist: {}".format(root))
    if root not in output.parents:
        raise ValueError("manifest must be inside bundle root")

    files = []
    for relative in relative_file_paths(root, output):
        absolute = root / relative
        files.append(
            {
                "path": relative.as_posix(),
                "size": absolute.stat().st_size,
                "sha256": sha256_file(absolute),
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "bundle_root": root.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "data_mode": data_mode,
        "artifacts": artifacts,
        "source_platform": platform.platform(),
        "source_python": platform.python_version(),
        "file_count": len(files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_manifest(root: Path, manifest: Path) -> int:
    root = root.resolve()
    manifest = manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported migration manifest schema")

    entries = payload.get("files")
    if not isinstance(entries, list):
        raise ValueError("manifest files must be a list")

    failures = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            failures.append("invalid file entry")
            continue
        try:
            relative = validate_relative_path(str(entry["path"]))
        except (KeyError, ValueError) as exc:
            failures.append(str(exc))
            continue
        path = root / relative
        if not path.is_file() or path.is_symlink():
            failures.append("missing file: {}".format(relative.as_posix()))
            continue
        expected_size = int(entry["size"])
        expected_hash = str(entry["sha256"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            failures.append(
                "size mismatch: {} ({} != {})".format(
                    relative.as_posix(), actual_size, expected_size
                )
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            failures.append("checksum mismatch: {}".format(relative.as_posix()))

    if failures:
        for failure in failures:
            print("migration verify: " + failure, file=sys.stderr)
        print(
            "migration verify: FAILED ({} problem(s))".format(len(failures)),
            file=sys.stderr,
        )
        return 1

    print(
        "migration verify: OK ({} files, source commit {})".format(
            len(entries), payload.get("source_commit", "unknown")
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write", help="write a checksum manifest")
    write.add_argument("--root", type=Path, required=True)
    write.add_argument("--output", type=Path, required=True)
    write.add_argument("--source-commit", default="unknown")
    write.add_argument("--data-mode", default="unknown")
    write.add_argument("--artifacts", default="none")

    verify = subparsers.add_parser("verify", help="verify a checksum manifest")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "write":
        write_manifest(
            args.root,
            args.output,
            args.source_commit,
            args.data_mode,
            args.artifacts,
        )
        print("migration manifest: {}".format(args.output))
        return 0

    manifest = args.manifest or (args.root / MANIFEST_NAME)
    return verify_manifest(args.root, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
