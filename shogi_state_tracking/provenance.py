#!/usr/bin/env python3
"""評価成果物へ生成来歴を付けて書き出す。

どのcommitのコードがその数値を出したのかを後から辿れるようにする。
`verify_study_integrity.py`のartifact-commit検査はここが書く`provenance`を読む。
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


PROVENANCE_VERSION = 1
REPOSITORY = Path(__file__).resolve().parent


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY), *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


@lru_cache(maxsize=1)
def git_state() -> tuple[str | None, bool | None]:
    """Return (commit, dirty). Both are None when git is unavailable."""
    commit = _git("rev-parse", "HEAD") or None
    status = _git("status", "--porcelain")
    return commit, None if status is None else bool(status.strip())


def provenance_record(**extra: Any) -> dict[str, Any]:
    commit, dirty = git_state()
    record: dict[str, Any] = {
        "provenance_version": PROVENANCE_VERSION,
        "git_commit": commit,
        "git_dirty": dirty,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else None,
        "python": sys.version.split()[0],
    }
    record.update(extra)
    return record


def with_provenance(payload: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """Copy the payload and attach a provenance block.

    Callers keep their dict unchanged. An existing "provenance" key is preserved
    under "provenance_inner" rather than being silently overwritten.
    """
    document = dict(payload)
    if "provenance" in document:
        document["provenance_inner"] = document.pop("provenance")
    document["provenance"] = provenance_record(**extra)
    return document


def write_metrics_json(path: Any, payload: Mapping[str, Any], **extra: Any) -> Path:
    """Write an evaluation artifact with a provenance block attached."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = with_provenance(payload, **extra)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


@lru_cache(maxsize=1)
def cshogi_provenance() -> dict[str, Any]:
    """Describe the cshogi build that produced evaluation labels.

    The installed distribution is authoritative because it is what actually ran.
    PEP 610 records the resolved commit for a VCS install in direct_url.json.
    The declared pin in pyproject.toml is only a fallback for reporting.
    """
    record: dict[str, Any] = {
        "version": None,
        "commit": None,
        "requested_revision": None,
        "url": None,
        "source": "unavailable",
    }
    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib
        return record | {"declared_pin": _declared_cshogi_pin()}
    try:
        installed = distribution("cshogi")
    except PackageNotFoundError:
        return record | {"declared_pin": _declared_cshogi_pin()}
    record["version"] = installed.version
    record["source"] = "installed"
    try:
        direct = installed.read_text("direct_url.json")
    except (OSError, ValueError):
        direct = None
    if direct:
        try:
            payload = json.loads(direct)
        except json.JSONDecodeError:
            payload = {}
        vcs = payload.get("vcs_info") if isinstance(payload, dict) else None
        record["url"] = payload.get("url") if isinstance(payload, dict) else None
        if isinstance(vcs, dict):
            record["commit"] = vcs.get("commit_id")
            record["requested_revision"] = vcs.get("requested_revision")
            record["source"] = "direct_url"
    record["declared_pin"] = _declared_cshogi_pin()
    return record


def _declared_cshogi_pin() -> str | None:
    """Read the cshogi revision declared in pyproject.toml, if present."""
    pyproject = REPOSITORY / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if "cshogi" in line and "git+" in line:
            _, _, revision = line.partition("@git+")
            revision = revision or line
            _, _, tail = revision.rpartition(".git@")
            candidate = tail.strip().strip('",\'')
            return candidate or None
    return None
