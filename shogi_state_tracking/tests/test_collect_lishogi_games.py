import unittest

from collect_lishogi_games import latest_user_game_since, normalized_game, rejection_reason, stable_hash


def sample_game():
    return {
        "id": "abcdefgh",
        "rated": True,
        "variant": "standard",
        "perf": "realTime",
        "createdAt": 1,
        "lastMoveAt": 2,
        "status": "resign",
        "winner": "sente",
        "players": {
            "sente": {"user": {"id": "alice", "name": "Alice"}, "rating": 1800},
            "gote": {"user": {"id": "bob", "name": "Bob"}, "rating": 1700},
        },
        "moves": " ".join(["7g7f", "3c3d"] * 40),
        "clock": {"initial": 600, "increment": 0, "byoyomi": 10},
    }


class CollectLishogiGamesTest(unittest.TestCase):
    def test_accepts_standard_rated_realtime_non_bot_game(self):
        self.assertIsNone(
            rejection_reason(
                sample_game(), min_plies=80, min_rating=None, max_rating=None, decisive_only=True
            )
        )

    def test_rejects_bot_anonymous_short_and_custom_games(self):
        game = sample_game()
        game["players"]["gote"]["user"]["title"] = "BOT"
        self.assertEqual(
            rejection_reason(game, min_plies=80, min_rating=None, max_rating=None, decisive_only=True),
            "bot_or_ai",
        )
        game = sample_game()
        game["players"]["gote"].pop("user")
        self.assertEqual(
            rejection_reason(game, min_plies=80, min_rating=None, max_rating=None, decisive_only=True),
            "anonymous_player",
        )
        game = sample_game()
        game["moves"] = "7g7f 3c3d"
        self.assertEqual(
            rejection_reason(game, min_plies=80, min_rating=None, max_rating=None, decisive_only=True),
            "too_short",
        )
        game = sample_game()
        game["initialSfen"] = "custom sfen"
        self.assertEqual(
            rejection_reason(game, min_plies=80, min_rating=None, max_rating=None, decisive_only=True),
            "not_standard_initial",
        )

    def test_normalized_output_removes_public_user_names(self):
        row = normalized_game(sample_game(), "test-salt", "2026-01-01T00:00:00+00:00")
        encoded = str(row)
        self.assertNotIn("Alice", encoded)
        self.assertNotIn("alice", encoded)
        self.assertEqual(row["players"]["sente"]["id_hash"], stable_hash("alice", "test-salt"))
        self.assertEqual(len(row["moves_usi"]), 80)

    def test_normalized_output_can_include_verified_public_title(self):
        salt = "test-salt"
        cache = {
            stable_hash("alice", salt): {"verified_non_bot": True, "title": "PRO"},
            stable_hash("bob", salt): {"verified_non_bot": True, "title": None},
        }
        row = normalized_game(sample_game(), salt, "2026-01-01T00:00:00+00:00", cache)
        self.assertEqual(row["players"]["sente"]["title"], "PRO")
        self.assertNotIn("title", row["players"]["gote"])

    def test_incremental_user_cursor_keeps_inclusive_created_at_boundary(self):
        salt = "test-salt"
        rows = [
            {
                "created_at_ms": 100,
                "players": {
                    "sente": {"id_hash": stable_hash("alice", salt)},
                    "gote": {"id_hash": stable_hash("bob", salt)},
                },
            },
            {
                "created_at_ms": 250,
                "players": {
                    "sente": {"id_hash": stable_hash("alice", salt)},
                    "gote": {"id_hash": stable_hash("carol", salt)},
                },
            },
        ]
        self.assertEqual(latest_user_game_since(rows, "alice", salt, None), 250)
        self.assertEqual(latest_user_game_since(rows, "alice", salt, 300), 300)
        self.assertIsNone(latest_user_game_since(rows, "unknown", salt, None))


if __name__ == "__main__":
    unittest.main()
