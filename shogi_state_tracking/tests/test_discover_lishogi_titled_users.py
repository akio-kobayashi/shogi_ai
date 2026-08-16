import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from discover_lishogi_titled_users import (
    enqueue_user,
    main,
    profile_is_stale,
    profile_record,
    verified_non_bot_users,
)


class FakeClient:
    def __init__(self, **kwargs):
        pass

    def leaderboard(self, maximum):
        return []

    def games(self, username, **kwargs):
        if username.lower() != "alice":
            return iter(())
        return iter(
            [
                {
                    "id": "game0001",
                    "players": {
                        "sente": {"user": {"id": "alice", "name": "Alice"}},
                        "gote": {"user": {"id": "bob", "name": "Bob", "title": "LP"}},
                    },
                }
            ]
        )

    def profiles(self, usernames):
        records = {
            "alice": {"id": "alice", "username": "Alice", "title": "PRO", "disabled": False},
            "bob": {"id": "bob", "username": "Bob", "title": "LP", "disabled": False},
        }
        return [records[name.lower()] for name in usernames]


class DiscoverLishogiTitledUsersTest(unittest.TestCase):
    def test_enqueue_is_case_insensitive_and_bounded(self):
        queue = []
        known = set()
        self.assertTrue(enqueue_user(queue, known, "Alice", "seed", 1))
        self.assertFalse(enqueue_user(queue, known, "alice", "duplicate", 1))
        self.assertFalse(enqueue_user(queue, known, "Bob", "bounded", 1))

    def test_profile_record_normalizes_title(self):
        record = profile_record({"id": "alice", "username": "Alice", "title": "pro"})
        self.assertEqual(record["title"], "PRO")

    def test_profile_cache_ttl(self):
        self.assertFalse(profile_is_stale({"checked_at": "2099-01-01T00:00:00+00:00"}, 24))
        self.assertTrue(profile_is_stale({"checked_at": "2020-01-01T00:00:00+00:00"}, 24))

    def test_verified_non_bot_users_excludes_bot_and_unverified_placeholders(self):
        users = verified_non_bot_users(
            {
                "alice": {
                    "username": "Alice",
                    "title": None,
                    "disabled": False,
                    "checked_at": "2026-01-01T00:00:00+00:00",
                },
                "alice-id": {
                    "username": "Alice",
                    "title": None,
                    "disabled": False,
                    "checked_at": "2026-01-01T00:00:00+00:00",
                },
                "bot": {
                    "username": "BotUser",
                    "title": "BOT",
                    "disabled": False,
                    "checked_at": "2026-01-01T00:00:00+00:00",
                },
                "missing": {
                    "username": "Missing",
                    "title": None,
                    "disabled": None,
                    "checked_at": "2026-01-01T00:00:00+00:00",
                    "not_returned": True,
                },
            }
        )
        self.assertEqual([row["username"] for row in users], ["Alice"])

    @patch("discover_lishogi_titled_users.LishogiClient", FakeClient)
    def test_discovers_and_verifies_titled_opponent(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            main(
                [
                    "--output-dir",
                    str(output),
                    "--seed-user",
                    "Alice",
                    "--leaderboard-size",
                    "0",
                    "--max-discovered-users",
                    "2",
                    "--max-users-this-run",
                    "2",
                    "--max-profile-users-this-run",
                    "2",
                    "--max-games-per-user",
                    "1",
                ]
            )
            users = (output / "titled_users.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(users, ["Bob", "Alice"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["titled_users"], 2)
            self.assertEqual(manifest["counts"]["metadata_games_this_run"], 1)


if __name__ == "__main__":
    unittest.main()
