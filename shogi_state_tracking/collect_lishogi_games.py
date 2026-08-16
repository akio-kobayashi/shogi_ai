#!/usr/bin/env python3
"""Collect a reproducible, pseudonymised Lishogi human-game evaluation set.

User discovery is handled separately by ``discover_lishogi_titled_users.py``.  This
collector consumes the resulting explicit user list and revalidates public profile
metadata before downloading full game records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


FORMAT_VERSION = 1
DEFAULT_BASE_URL = "https://lishogi.org"
TERMINAL_WINNERS = {"sente", "gote", "white", "black"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_time(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def stable_hash(value: str, salt: str) -> str:
    return hashlib.sha256((salt + "\0" + value.lower()).encode("utf-8")).hexdigest()


def read_users(path: Path) -> List[str]:
    users: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.split("#", 1)[0].strip()
            if not value or value.lower() in seen:
                continue
            seen.add(value.lower())
            users.append(value)
    if not users:
        raise ValueError(f"{path} contains no user names")
    return users


def player_user(player: object) -> Optional[Mapping[str, object]]:
    if not isinstance(player, Mapping):
        return None
    user = player.get("user")
    return user if isinstance(user, Mapping) else None


def player_id(player: object) -> Optional[str]:
    user = player_user(player)
    if user is None:
        return None
    value = user.get("id") or user.get("name")
    return str(value) if value else None


def player_title(player: object) -> Optional[str]:
    user = player_user(player)
    if user is None or user.get("title") is None:
        return None
    return str(user["title"]).upper()


def game_players(game: Mapping[str, object]) -> Optional[Tuple[Mapping[str, object], Mapping[str, object]]]:
    players = game.get("players")
    if not isinstance(players, Mapping):
        return None
    sente = players.get("sente") or players.get("white")
    gote = players.get("gote") or players.get("black")
    if not isinstance(sente, Mapping) or not isinstance(gote, Mapping):
        return None
    return sente, gote


def variant_name(game: Mapping[str, object]) -> Optional[str]:
    variant = game.get("variant")
    if isinstance(variant, Mapping):
        value = variant.get("key") or variant.get("name")
        return str(value) if value else None
    return str(variant) if variant is not None else None


def move_list(game: Mapping[str, object]) -> List[str]:
    moves = game.get("moves")
    if isinstance(moves, str):
        return [move for move in moves.split() if move]
    if isinstance(moves, Sequence) and not isinstance(moves, (str, bytes)):
        return [str(move) for move in moves]
    return []


def rejection_reason(
    game: Mapping[str, object],
    *,
    min_plies: int,
    min_rating: Optional[int],
    max_rating: Optional[int],
    decisive_only: bool,
) -> Optional[str]:
    if not bool(game.get("rated")):
        return "not_rated"
    if (variant_name(game) or "").lower() != "standard":
        return "not_standard"
    if str(game.get("perf") or "").lower() != "realtime":
        return "not_realtime"
    initial_sfen = game.get("initialSfen") or game.get("initial_sfen")
    if initial_sfen not in (None, "", "startpos"):
        return "not_standard_initial"
    players = game_players(game)
    if players is None:
        return "missing_players"
    for player in players:
        if player_id(player) is None:
            return "anonymous_player"
        if player_title(player) == "BOT" or player.get("aiLevel") is not None:
            return "bot_or_ai"
        rating = player.get("rating")
        if not isinstance(rating, (int, float)):
            return "missing_rating"
        if min_rating is not None and rating < min_rating:
            return "rating_below_minimum"
        if max_rating is not None and rating > max_rating:
            return "rating_above_maximum"
    moves = move_list(game)
    if len(moves) < min_plies:
        return "too_short"
    if decisive_only and str(game.get("winner") or "").lower() not in TERMINAL_WINNERS:
        return "not_decisive"
    return None


class LishogiClient:
    def __init__(
        self,
        base_url: str,
        token: Optional[str],
        request_delay: float,
        timeout: float,
        retries: int,
        ca_file: Optional[str],
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.request_delay = request_delay
        self.timeout = timeout
        self.retries = retries
        self.ssl_context = ssl.create_default_context(cafile=ca_file) if ca_file else None
        self._last_request = 0.0

    def _request(
        self,
        path: str,
        accept: str,
        *,
        method: str = "GET",
        data: Optional[bytes] = None,
        content_type: Optional[str] = None,
    ) -> bytes:
        url = self.base_url + path
        headers = {
            "Accept": accept,
            "User-Agent": "shogi-state-tracking-research/1.0 (Lishogi public API)",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if content_type:
            headers["Content-Type"] = content_type
        for attempt in range(self.retries + 1):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=self.ssl_context
                ) as response:
                    body = response.read()
                self._last_request = time.monotonic()
                return body
            except urllib.error.HTTPError as exc:
                self._last_request = time.monotonic()
                if exc.code == 429 or 500 <= exc.code < 600:
                    if attempt < self.retries:
                        retry_after = exc.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after else 2.0 ** attempt
                        time.sleep(max(delay, self.request_delay))
                        continue
                raise RuntimeError(f"Lishogi API request failed: HTTP {exc.code} {url}") from exc
            except urllib.error.URLError as exc:
                self._last_request = time.monotonic()
                if attempt < self.retries:
                    time.sleep(2.0 ** attempt)
                    continue
                raise RuntimeError(f"Lishogi API request failed: {url}: {exc}") from exc
        raise AssertionError("unreachable")

    def games(
        self,
        username: str,
        *,
        maximum: int,
        since: Optional[int],
        until: Optional[int],
        include_moves: bool = True,
    ) -> Iterator[Mapping[str, object]]:
        query: Dict[str, object] = {
            "max": maximum,
            "rated": "true",
            "perfType": "realTime",
            "moves": str(include_moves).lower(),
            "clocks": str(include_moves).lower(),
            "evals": "false",
            "ongoing": "false",
            "finished": "true",
            "sort": "dateDesc",
        }
        if since is not None:
            query["since"] = since
        if until is not None:
            query["until"] = until
        path = "/api/games/user/{user}?{query}".format(
            user=urllib.parse.quote(username, safe=""),
            query=urllib.parse.urlencode(query),
        )
        body = self._request(path, "application/x-ndjson")
        for number, line in enumerate(body.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{username}: response line {number} is not an object")
            yield value

    def leaderboard(self, maximum: int = 200) -> List[Mapping[str, object]]:
        path = "/player/top/{}/realTime".format(maximum)
        value = json.loads(
            self._request(path, "application/vnd.lishogi.v3+json").decode("utf-8")
        )
        users = value.get("users") if isinstance(value, Mapping) else None
        if not isinstance(users, list):
            raise ValueError("leaderboard response has no users list")
        return [user for user in users if isinstance(user, Mapping)]

    def profiles(self, usernames: Sequence[str]) -> List[Mapping[str, object]]:
        """Return public profiles in batches supported by POST /api/users."""
        if not usernames:
            return []
        if len(usernames) > 300:
            raise ValueError("Lishogi accepts at most 300 user IDs per profile request")
        data = ",".join(usernames).encode("utf-8")
        value = json.loads(
            self._request(
                "/api/users",
                "application/json",
                method="POST",
                data=data,
                content_type="text/plain",
            ).decode("utf-8")
        )
        if not isinstance(value, list):
            raise ValueError("profile batch response is not a list")
        return [profile for profile in value if isinstance(profile, Mapping)]

    def profile(self, username: str) -> Mapping[str, object]:
        path = "/api/user/" + urllib.parse.quote(username, safe="")
        value = json.loads(self._request(path, "application/json").decode("utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"profile for {username} is not an object")
        return value


def verified_profile(
    client: LishogiClient,
    username: str,
    cache: MutableMapping[str, Mapping[str, object]],
    salt: str,
) -> Optional[Mapping[str, object]]:
    key = stable_hash(username, salt)
    cached = cache.get(key)
    # Cache entries written by format_version 1 before title-aware collection did
    # not contain ``title``.  Refresh those entries instead of treating the
    # missing field as a verified untitled account.
    if cached is not None and "title" in cached:
        return cached
    try:
        profile = client.profile(username)
        title = str(profile.get("title") or "").upper()
        verified = title != "BOT" and not bool(profile.get("disabled"))
        cache[key] = {
            "verified_non_bot": verified,
            "title": title or None,
            "title_is_bot": title == "BOT",
            "disabled": bool(profile.get("disabled")),
            "checked_at": utc_now(),
        }
    except RuntimeError as exc:
        print(f"warning: profile verification failed for {key[:12]}: {exc}", file=sys.stderr)
        # A transient network failure must not become a permanent negative cache entry.
        return None
    return cache[key]


def verified_non_bot(
    client: LishogiClient,
    username: str,
    cache: MutableMapping[str, Mapping[str, object]],
    salt: str,
) -> bool:
    profile = verified_profile(client, username, cache, salt)
    return bool(profile and profile.get("verified_non_bot"))


def normalized_game(
    game: Mapping[str, object],
    salt: str,
    retrieved_at: str,
    profile_cache: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[str, object]:
    players = game_players(game)
    if players is None:
        raise ValueError("game has no players")

    def normalize_player(player: Mapping[str, object]) -> Dict[str, object]:
        identifier = player_id(player)
        if identifier is None:
            raise ValueError("anonymous player")
        result: Dict[str, object] = {
            "id_hash": stable_hash(identifier, salt),
            "rating": player.get("rating"),
        }
        if player.get("ratingDiff") is not None:
            result["rating_diff"] = player.get("ratingDiff")
        title = player_title(player)
        if profile_cache is not None:
            cached = profile_cache.get(stable_hash(identifier, salt))
            if cached and cached.get("title"):
                title = str(cached["title"]).upper()
        if title:
            result["title"] = title
        return result

    game_id = str(game.get("id") or "")
    return {
        "format_version": FORMAT_VERSION,
        "source": "lishogi_public_api",
        "source_game_id": game_id,
        "source_url": "https://lishogi.org/" + game_id,
        "retrieved_at": retrieved_at,
        "created_at_ms": game.get("createdAt"),
        "last_move_at_ms": game.get("lastMoveAt"),
        "rated": bool(game.get("rated")),
        "variant": variant_name(game),
        "perf": game.get("perf"),
        "status": game.get("status"),
        "winner": game.get("winner"),
        "initial_sfen": game.get("initialSfen") or "startpos",
        "moves_usi": move_list(game),
        "players": {
            "sente": normalize_player(players[0]),
            "gote": normalize_player(players[1]),
        },
        "clock": game.get("clock"),
    }


def load_existing(path: Path) -> Tuple[List[Dict[str, object]], set]:
    rows: List[Dict[str, object]] = []
    identifiers = set()
    if not path.exists():
        return rows, identifiers
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            game_id = row.get("source_game_id")
            if not game_id:
                raise ValueError(f"{path}:{number} has no source_game_id")
            rows.append(row)
            identifiers.add(str(game_id))
    return rows, identifiers


def latest_user_game_since(
    rows: Sequence[Mapping[str, object]], username: str, salt: str, fallback: Optional[int]
) -> Optional[int]:
    """Return a per-user high-water mark for incremental API retrieval."""
    target = stable_hash(username, salt)
    latest: Optional[int] = None
    for row in rows:
        players = row.get("players")
        if not isinstance(players, Mapping):
            continue
        appears = any(
            isinstance(player, Mapping) and player.get("id_hash") == target
            for player in players.values()
        )
        if not appears:
            continue
        created = row.get("created_at_ms")
        if isinstance(created, (int, float)):
            created_int = int(created)
            latest = created_int if latest is None else max(latest, created_int)
    if latest is None:
        return fallback
    # Lishogi's ``since`` boundary is inclusive.  Keep the boundary itself so
    # games imported in the same millisecond cannot be missed; the collector's
    # source-game-id set removes that one (usually tiny) overlap.
    candidate = latest
    return candidate if fallback is None else max(candidate, fallback)


def selection_for_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "rated": True,
        "variant": "standard",
        "perf": "realTime",
        "registered_players_only": True,
        "profile_verified_non_bot": True,
        "required_seed_titles": sorted(set(args.required_user_title)),
        "standard_initial_position": True,
        "decisive_only": not args.include_draws,
        "min_plies": args.min_plies,
        "min_rating": args.min_rating,
        "max_rating": args.max_rating,
        "since_ms": parse_time(args.since),
        "until_ms": parse_time(args.until),
        "max_games_per_user": args.max_games_per_user,
        "target_games": None if args.target_new_games is not None else args.target_games,
        "target_new_games": args.target_new_games,
    }


def incremental_selection_matches(path: Path, current: Mapping[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    previous = value.get("selection") if isinstance(value, Mapping) else None
    return isinstance(previous, Mapping) and all(previous.get(key) == val for key, val in current.items())


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lishogi公開APIから非BOT同士の評価用棋譜を収集する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--users-file", type=Path, required=True, help="1行1ユーザ名の収集起点")
    parser.add_argument(
        "--required-user-title",
        action="append",
        choices=("PRO", "LP"),
        default=[],
        help="収集起点に要求する現在の公開タイトル（複数回指定可）",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/lishogi-human"))
    parser.add_argument("--target-games", type=int, default=1000)
    parser.add_argument(
        "--target-new-games",
        type=int,
        help="既存games.jsonlに対して今回新たに追記する局数の上限",
    )
    parser.add_argument("--max-games-per-user", type=int, default=200)
    parser.add_argument("--min-plies", type=int, default=80)
    parser.add_argument("--min-rating", type=int)
    parser.add_argument("--max-rating", type=int)
    parser.add_argument("--since", help="Unix ms又はISO-8601")
    parser.add_argument("--until", help="Unix ms又はISO-8601")
    parser.add_argument("--include-draws", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token-env", default="LISHOGI_TOKEN")
    parser.add_argument("--hash-salt", default="lishogi-human-evaluation-v1")
    parser.add_argument("--request-delay", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--ca-file", help="TLS検証に使うCA bundle（通常は指定不要）")
    parser.add_argument("--keep-raw", action="store_true", help="公開ユーザ名を含むAPI応答も保存する")
    parser.add_argument(
        "--full-rescan",
        action="store_true",
        help="既存games.jsonlがあっても各利用者の全指定範囲を再取得する",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.target_games <= 0 or args.max_games_per_user <= 0 or args.min_plies < 0:
        raise ValueError("game counts and min-plies must be positive")
    if args.target_new_games is not None and args.target_new_games <= 0:
        raise ValueError("--target-new-games must be positive")
    users = read_users(args.users_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    games_path = args.output_dir / "games.jsonl"
    raw_path = args.output_dir / "raw_games.ndjson"
    cache_path = args.output_dir / "profile_cache.json"
    if args.overwrite:
        for path in (games_path, raw_path, cache_path, args.output_dir / "manifest.json"):
            path.unlink(missing_ok=True)
    rows, seen_game_ids = load_existing(games_path)
    current_selection = selection_for_args(args)
    incremental = bool(rows) and not args.overwrite and not args.full_rescan and incremental_selection_matches(
        args.output_dir / "manifest.json", current_selection
    )
    if rows and not incremental and not args.full_rescan and not args.overwrite:
        print(
            "warning: existing selection differs or has no manifest; using full rescan. "
            "Use identical options for incremental retrieval.",
            file=sys.stderr,
        )
    if cache_path.exists():
        profile_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        profile_cache = {}
    if not isinstance(profile_cache, dict):
        raise ValueError(f"{cache_path} is not an object")

    client = LishogiClient(
        base_url=args.base_url,
        token=os.environ.get(args.token_env),
        request_delay=args.request_delay,
        timeout=args.timeout,
        retries=args.retries,
        ca_file=args.ca_file,
    )
    rejected: Counter[str] = Counter()
    retrieved = 0
    duplicate = 0
    selected_this_run = 0
    started_at = utc_now()
    mode = "a" if games_path.exists() and not args.overwrite else "w"
    raw_mode = "a" if raw_path.exists() and not args.overwrite else "w"
    raw_handle = raw_path.open(raw_mode, encoding="utf-8") if args.keep_raw else None

    def limit_reached() -> bool:
        if args.target_new_games is not None:
            return selected_this_run >= args.target_new_games
        return len(rows) >= args.target_games

    try:
        with games_path.open(mode, encoding="utf-8") as output:
            for user_index, username in enumerate(users, 1):
                if limit_reached():
                    break
                seed_profile = verified_profile(client, username, profile_cache, args.hash_salt)
                if seed_profile is None or not seed_profile.get("verified_non_bot"):
                    rejected["seed_profile_not_verified_non_bot"] += 1
                    continue
                required_titles = {str(title).upper() for title in args.required_user_title}
                if required_titles and str(seed_profile.get("title") or "").upper() not in required_titles:
                    rejected["seed_title_not_allowed"] += 1
                    continue
                print(
                    json.dumps(
                        {
                            "event": "fetch_user",
                            "index": user_index,
                            "users": len(users),
                            "user_hash": stable_hash(username, args.hash_salt)[:12],
                            "selected": len(rows),
                            "incremental": incremental,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                user_since = parse_time(args.since)
                if incremental:
                    user_since = latest_user_game_since(rows, username, args.hash_salt, user_since)
                for game in client.games(
                    username,
                    maximum=args.max_games_per_user,
                    since=user_since,
                    until=parse_time(args.until),
                ):
                    retrieved += 1
                    game_id = str(game.get("id") or "")
                    if not game_id:
                        rejected["missing_game_id"] += 1
                        continue
                    if game_id in seen_game_ids:
                        duplicate += 1
                        continue
                    reason = rejection_reason(
                        game,
                        min_plies=args.min_plies,
                        min_rating=args.min_rating,
                        max_rating=args.max_rating,
                        decisive_only=not args.include_draws,
                    )
                    if reason is not None:
                        rejected[reason] += 1
                        continue
                    players = game_players(game)
                    assert players is not None
                    identifiers = [player_id(player) for player in players]
                    if any(identifier is None for identifier in identifiers):
                        rejected["anonymous_player"] += 1
                        continue
                    if not all(
                        verified_non_bot(client, str(identifier), profile_cache, args.hash_salt)
                        for identifier in identifiers
                    ):
                        rejected["profile_not_verified_non_bot"] += 1
                        continue
                    retrieved_at = utc_now()
                    row = normalized_game(game, args.hash_salt, retrieved_at, profile_cache)
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    output.flush()
                    if raw_handle is not None:
                        raw_handle.write(json.dumps(game, ensure_ascii=False, separators=(",", ":")) + "\n")
                        raw_handle.flush()
                    rows.append(row)
                    seen_game_ids.add(game_id)
                    selected_this_run += 1
                    if limit_reached():
                        break
                write_json(cache_path, profile_cache)
    finally:
        if raw_handle is not None:
            raw_handle.close()

    manifest = {
        "format_version": FORMAT_VERSION,
        "source": "Lishogi public API",
        "base_url": args.base_url,
        "api_endpoint": "/api/games/user/{username}",
        "started_at": started_at,
        "completed_at": utc_now(),
        "selection": current_selection,
        "sampling": {
            "target_games": args.target_games,
            "target_new_games": args.target_new_games,
            "max_games_per_user": args.max_games_per_user,
            "seed_user_hashes": [stable_hash(user, args.hash_salt) for user in users],
            "sort": "dateDesc",
            "incremental_retrieval": incremental,
        },
        "counts": {
            "api_games_retrieved_this_run": retrieved,
            "duplicates_this_run": duplicate,
            "selected_games_total": len(rows),
            "selected_games_this_run": selected_this_run,
            "rejected_this_run": dict(sorted(rejected.items())),
        },
        "privacy": {
            "player_names_stored_in_games_jsonl": False,
            "player_id_hash": "sha256(salt + NUL + lowercase(user_id))",
            "raw_api_responses_saved": bool(args.keep_raw),
        },
        "files": {
            "games": games_path.name,
            "profile_cache": cache_path.name,
            "raw_games": raw_path.name if args.keep_raw else None,
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"event": "complete", **manifest["counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
