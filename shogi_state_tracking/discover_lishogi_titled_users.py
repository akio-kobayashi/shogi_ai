#!/usr/bin/env python3
"""Discover public Lishogi PRO/LP accounts through public game metadata.

Lishogi has no public endpoint that enumerates all titled users.  This program
therefore performs a bounded, resumable breadth-first crawl of the public user
game graph.  It downloads metadata first, verifies discovered accounts through
the public profile API in batches, and writes a user list consumable by
``collect_lishogi_games.py``.  Full move records are never saved in this phase.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Set

from collect_lishogi_games import (
    DEFAULT_BASE_URL,
    LishogiClient,
    game_players,
    parse_time,
    player_id,
    player_title,
    read_users,
    utc_now,
    write_json,
)


FORMAT_VERSION = 1
DEFAULT_TITLES = ("PRO", "LP")


def canonical_user(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text if text else None


def enqueue_user(
    queue: List[Dict[str, str]],
    known_ids: Set[str],
    username: object,
    source: str,
    maximum: int,
) -> bool:
    value = canonical_user(username)
    if value is None:
        return False
    key = value.lower()
    if key in known_ids or len(queue) >= maximum:
        return False
    known_ids.add(key)
    queue.append({"username": value, "source": source})
    return True


def profile_record(profile: Mapping[str, object]) -> Optional[Dict[str, object]]:
    username = canonical_user(profile.get("username") or profile.get("name") or profile.get("id"))
    if username is None:
        return None
    title = str(profile.get("title") or "").upper() or None
    return {
        "username": username,
        "id": canonical_user(profile.get("id")) or username.lower(),
        "title": title,
        "disabled": bool(profile.get("disabled")),
        "checked_at": utc_now(),
    }


def profile_is_stale(record: Mapping[str, object], max_age_hours: float) -> bool:
    checked_at = record.get("checked_at")
    if not isinstance(checked_at, str):
        return True
    try:
        checked = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    age_seconds = (datetime.now(timezone.utc) - checked).total_seconds()
    return age_seconds >= max_age_hours * 3600.0


def verified_non_bot_users(
    profiles: Mapping[str, object],
) -> List[Dict[str, object]]:
    """Return unique users whose public profile was actually verified.

    The discovery cache contains entries indexed both by user id and by
    username.  It can also contain negative ``not_returned`` placeholders
    when the batch profile endpoint did not return an account.  Those
    placeholders must not be treated as human users merely because they have
    no BOT title.
    """
    unique: MutableMapping[str, Dict[str, object]] = {}
    for value in profiles.values():
        if not isinstance(value, Mapping):
            continue
        username = canonical_user(value.get("username") or value.get("name") or value.get("id"))
        if username is None:
            continue
        if value.get("not_returned") or value.get("disabled"):
            continue
        if "checked_at" not in value:
            continue
        title = str(value.get("title") or "").upper() or None
        if title == "BOT":
            continue
        unique[username.lower()] = {
            "username": username,
            "title": title,
            "profile_checked_at": value.get("checked_at"),
            "verified_non_bot": True,
        }
    return sorted(unique.values(), key=lambda row: str(row["username"]).lower())


def load_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {
            "format_version": FORMAT_VERSION,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "queue": [],
            "next_index": 0,
            "title_hints": {},
            "failed_users": [],
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported discovery state: {path}")
    if not isinstance(value.get("queue"), list):
        raise ValueError(f"{path} has no queue")
    return value


def state_settings(args: argparse.Namespace) -> Dict[str, object]:
    """Settings that must not silently change during a resumed crawl."""
    return {
        "base_url": args.base_url.rstrip("/"),
        "titles": sorted(set(args.title or DEFAULT_TITLES)),
        "leaderboard_size": args.leaderboard_size,
        "since_ms": parse_time(args.since),
        "until_ms": parse_time(args.until),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="公開対局メタデータからLishogiのPRO／LP利用者を探索する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/lishogi-title-discovery"))
    parser.add_argument("--seed-users-file", type=Path)
    parser.add_argument("--seed-user", action="append", default=[])
    parser.add_argument("--title", action="append", choices=DEFAULT_TITLES, default=[])
    parser.add_argument("--leaderboard-size", type=int, default=200, choices=range(0, 201), metavar="0..200")
    parser.add_argument("--max-discovered-users", type=int, default=10000)
    parser.add_argument("--max-users-this-run", type=int, default=500)
    parser.add_argument("--max-profile-users-this-run", type=int, default=3000)
    parser.add_argument("--max-games-per-user", type=int, default=50)
    parser.add_argument("--since", help="Unix ms又はISO-8601")
    parser.add_argument("--until", help="Unix ms又はISO-8601")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token-env", default="LISHOGI_TOKEN")
    parser.add_argument("--request-delay", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--ca-file")
    parser.add_argument(
        "--refresh-profiles",
        action="store_true",
        help="既存profile cacheも再照合する（現在のタイトルを優先する場合に指定）",
    )
    parser.add_argument("--profile-cache-ttl-hours", type=float, default=24.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.profile_cache_ttl_hours < 0:
        raise ValueError("--profile-cache-ttl-hours must be non-negative")
    for name in (
        "max_discovered_users",
        "max_users_this_run",
        "max_profile_users_this_run",
        "max_games_per_user",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "discovery_state.json"
    profile_path = args.output_dir / "profile_cache.json"
    titled_json_path = args.output_dir / "titled_users.json"
    titled_text_path = args.output_dir / "titled_users.txt"
    non_bot_json_path = args.output_dir / "non_bot_users.json"
    non_bot_text_path = args.output_dir / "non_bot_users.txt"
    manifest_path = args.output_dir / "manifest.json"
    if args.overwrite:
        for path in (
            state_path,
            profile_path,
            titled_json_path,
            titled_text_path,
            non_bot_json_path,
            non_bot_text_path,
            manifest_path,
        ):
            path.unlink(missing_ok=True)

    state = load_state(state_path)
    current_settings = state_settings(args)
    saved_settings = state.get("settings")
    if saved_settings is not None and saved_settings != current_settings:
        raise ValueError(
            "discovery settings differ from the existing state; use --overwrite "
            "or resume with the same base URL, title filter, leaderboard size, and time window"
        )
    state["settings"] = current_settings
    queue = state["queue"]
    assert isinstance(queue, list)
    known_ids = {
        str(item.get("username") or "").lower()
        for item in queue
        if isinstance(item, Mapping) and item.get("username")
    }
    title_hints = state.setdefault("title_hints", {})
    failed_users = state.setdefault("failed_users", [])
    if not isinstance(title_hints, dict) or not isinstance(failed_users, list):
        raise ValueError(f"invalid discovery state: {state_path}")

    client = LishogiClient(
        base_url=args.base_url,
        token=os.environ.get(args.token_env),
        request_delay=args.request_delay,
        timeout=args.timeout,
        retries=args.retries,
        ca_file=args.ca_file,
    )

    seeds = list(args.seed_user)
    if args.seed_users_file is not None:
        seeds.extend(read_users(args.seed_users_file))
    for username in seeds:
        enqueue_user(queue, known_ids, username, "explicit_seed", args.max_discovered_users)

    if args.leaderboard_size:
        try:
            for user in client.leaderboard(args.leaderboard_size):
                enqueue_user(
                    queue,
                    known_ids,
                    user.get("username") or user.get("id"),
                    "realTime_leaderboard",
                    args.max_discovered_users,
                )
                title = str(user.get("title") or "").upper()
                identifier = canonical_user(user.get("id") or user.get("username"))
                if identifier and title:
                    title_hints[identifier.lower()] = title
        except (RuntimeError, ValueError) as exc:
            if not queue:
                raise
            print(f"warning: leaderboard discovery failed: {exc}", file=sys.stderr)

    # Persist newly supplied seeds before the first network request.  If the
    # process is interrupted during discovery, the next invocation can resume
    # from exactly this queue.
    state["updated_at"] = utc_now()
    write_json(state_path, state)

    if not queue:
        raise ValueError("no discovery seeds; provide --seed-user(s) or enable the leaderboard")

    next_index = int(state.get("next_index", 0))
    scanned_this_run = 0
    metadata_games = 0
    while next_index < len(queue) and scanned_this_run < args.max_users_this_run:
        entry = queue[next_index]
        if not isinstance(entry, Mapping) or not entry.get("username"):
            raise ValueError(f"invalid queue entry at index {next_index}")
        username = str(entry["username"])
        print(
            json.dumps(
                {
                    "event": "scan_user_games",
                    "queue_index": next_index,
                    "queued_users": len(queue),
                    "username": username,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        scan_failed = False
        try:
            for game in client.games(
                username,
                maximum=args.max_games_per_user,
                since=parse_time(args.since),
                until=parse_time(args.until),
                include_moves=False,
            ):
                metadata_games += 1
                players = game_players(game)
                if players is None:
                    continue
                game_id = str(game.get("id") or "unknown")
                for player in players:
                    identifier = player_id(player)
                    if identifier is None:
                        continue
                    enqueue_user(
                        queue,
                        known_ids,
                        identifier,
                        "opponent_in_game:" + game_id,
                        args.max_discovered_users,
                    )
                    hint = player_title(player)
                    if hint:
                        title_hints[identifier.lower()] = hint
        except (RuntimeError, ValueError) as exc:
            failed_users.append({"username": username, "error": str(exc), "failed_at": utc_now()})
            print(f"warning: game metadata scan failed for {username}: {exc}", file=sys.stderr)
            # Keep next_index unchanged.  A later invocation retries this user
            # instead of silently losing a graph component after a transient
            # network or API failure.
            scan_failed = True
        if scan_failed:
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            break
        next_index += 1
        scanned_this_run += 1
        state["next_index"] = next_index
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    if profile_path.exists():
        profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    else:
        profiles = {}
    if not isinstance(profiles, dict):
        raise ValueError(f"{profile_path} is not an object")

    unchecked = [
        str(entry["username"])
        for entry in queue
        if isinstance(entry, Mapping)
        and entry.get("username")
        and (
            args.refresh_profiles
            or str(entry["username"]).lower() not in profiles
            or profile_is_stale(
                profiles[str(entry["username"]).lower()], args.profile_cache_ttl_hours
            )
        )
    ][: args.max_profile_users_this_run]
    profile_requests = 0
    for offset in range(0, len(unchecked), 300):
        batch = unchecked[offset : offset + 300]
        returned = client.profiles(batch)
        returned_ids = set()
        for profile in returned:
            record = profile_record(profile)
            if record is None:
                continue
            key = str(record["id"]).lower()
            returned_ids.add(key)
            profiles[key] = record
            profiles[str(record["username"]).lower()] = record
        for username in batch:
            key = username.lower()
            if key not in profiles and key not in returned_ids:
                profiles[key] = {
                    "username": username,
                    "id": key,
                    "title": None,
                    "disabled": None,
                    "checked_at": utc_now(),
                    "not_returned": True,
                }
        profile_requests += 1
        write_json(profile_path, profiles)

    wanted_titles = set(args.title or DEFAULT_TITLES)
    unique_titled: MutableMapping[str, Dict[str, object]] = {}
    for record in profiles.values():
        if not isinstance(record, Mapping):
            continue
        title = str(record.get("title") or "").upper()
        username = canonical_user(record.get("username"))
        if title not in wanted_titles or username is None or bool(record.get("disabled")):
            continue
        unique_titled[username.lower()] = {
            "username": username,
            "title": title,
            "profile_checked_at": record.get("checked_at"),
            "discovery_complete": False,
        }
    titled = sorted(unique_titled.values(), key=lambda row: (str(row["title"]), str(row["username"]).lower()))
    write_json(titled_json_path, titled)
    temporary = titled_text_path.with_suffix(".txt.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in titled:
            handle.write(str(row["username"]) + "\n")
    temporary.replace(titled_text_path)

    non_bot = verified_non_bot_users(profiles)
    write_json(non_bot_json_path, non_bot)
    temporary = non_bot_text_path.with_suffix(".txt.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in non_bot:
            handle.write(str(row["username"]) + "\n")
    temporary.replace(non_bot_text_path)

    title_counts: Dict[str, int] = {}
    for row in titled:
        title = str(row["title"])
        title_counts[title] = title_counts.get(title, 0) + 1
    manifest = {
        "format_version": FORMAT_VERSION,
        "source": "Lishogi public API",
        "completed_at": utc_now(),
        "method": "bounded breadth-first crawl of public user-game metadata followed by profile verification",
        "completeness": "not exhaustive; users without a traversed public game can be missed",
        "settings": {
            "titles": sorted(wanted_titles),
            "leaderboard_size": args.leaderboard_size,
            "max_discovered_users": args.max_discovered_users,
            "max_users_this_run": args.max_users_this_run,
            "max_profile_users_this_run": args.max_profile_users_this_run,
            "max_games_per_user": args.max_games_per_user,
            "refresh_profiles": args.refresh_profiles,
            "profile_cache_ttl_hours": args.profile_cache_ttl_hours,
            "since_ms": parse_time(args.since),
            "until_ms": parse_time(args.until),
        },
        "counts": {
            "queued_users": len(queue),
            "scanned_users_total": next_index,
            "scanned_users_this_run": scanned_this_run,
            "metadata_games_this_run": metadata_games,
            "profiles_cached": len(profiles),
            "profile_requests_this_run": profile_requests,
            "titled_users": len(titled),
            "titles": dict(sorted(title_counts.items())),
            "verified_non_bot_users": len(non_bot),
            "failed_user_scans_total": len(failed_users),
        },
        "files": {
            "collector_user_list": titled_text_path.name,
            "titled_users": titled_json_path.name,
            "non_bot_user_list": non_bot_text_path.name,
            "non_bot_users": non_bot_json_path.name,
            "profile_cache": profile_path.name,
            "resume_state": state_path.name,
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"event": "complete", **manifest["counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
