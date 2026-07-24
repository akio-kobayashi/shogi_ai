import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import preprocess

try:
    preprocess.import_cshogi()
    HAS_CSHOGI = True
except RuntimeError:
    HAS_CSHOGI = False


class RandomStartTest(unittest.TestCase):
    def test_candidates_cover_game_with_minimum_suffix(self):
        candidates = preprocess.candidate_start_plies(
            total_moves=140,
            candidate_count=40,
            min_suffix_moves=40,
        )
        self.assertEqual(len(candidates), 40)
        self.assertEqual(candidates[0], 0)
        self.assertEqual(candidates[-1], 100)
        self.assertTrue(all(140 - start >= 40 for start in candidates))

    def test_choice_is_reproducible_and_changes_across_epochs(self):
        candidates = list(range(40))
        first = preprocess.choose_start_ply("game", candidates, 7, 3)
        second = preprocess.choose_start_ply("game", candidates, 7, 3)
        self.assertEqual(first, second)
        choices = {
            preprocess.choose_start_ply("game", candidates, 7, epoch)
            for epoch in range(20)
        }
        self.assertGreater(len(choices), 1)

    @unittest.skipUnless(HAS_CSHOGI, "cshogi is not installed")
    def test_materialize_segment_replays_to_new_start(self):
        cshogi = preprocess.import_cshogi()
        record = {
            "game_id": "g",
            "engine_scope": "open",
            "initial_sfen": cshogi.Board().sfen(),
            "move_tokens": ["7g7f", "3c3d"],
        }
        segment = preprocess.materialize_segment(record, start_ply=1)
        self.assertEqual(segment["move_tokens"], ["3c3d"])
        self.assertEqual(segment["start_ply"], 1)
        self.assertEqual(len(segment["initial_state_tokens"]), 96)
        self.assertIn(" w ", segment["start_sfen"])


if __name__ == "__main__":
    unittest.main()
