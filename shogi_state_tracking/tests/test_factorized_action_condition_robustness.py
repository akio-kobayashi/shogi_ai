import unittest


class ActionConditionRobustnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("PyTorch is unavailable") from exc

    def test_game_partition_is_deterministic_and_game_disjoint(self):
        from evaluate_factorized_action_condition_robustness import split_positions

        positions = [
            {"game_id": "game-{}".format(game), "ply": ply}
            for game in range(100) for ply in range(3)
        ]
        first = split_positions(positions, 7)
        second = split_positions(list(reversed(positions)), 7)
        first_games = {name: {item["game_id"] for item in values} for name, values in first.items()}
        second_games = {name: {item["game_id"] for item in values} for name, values in second.items()}
        self.assertEqual(first_games, second_games)
        self.assertFalse(first_games["probe_train"] & first_games["calibration"])
        self.assertFalse(first_games["probe_train"] & first_games["evaluation"])
        self.assertFalse(first_games["calibration"] & first_games["evaluation"])

    def test_pooled_indices_balance_three_branch_positions(self):
        from evaluate_factorized_action_condition_robustness import balanced_family_indices

        branches = ["pre"] * 3 + ["drop"] * 3 + ["normal"] * 9
        indices = balanced_family_indices(branches, "pooled", 11).tolist()
        selected = [branches[index] for index in indices]
        self.assertEqual(selected.count("pre"), 3)
        self.assertEqual(selected.count("drop"), 3)
        self.assertEqual(selected.count("normal"), 3)

    def test_normal_branch_selection_prefers_different_piece(self):
        from evaluate_factorized_action_condition_robustness import select_normal_branches

        item = {
            "game_id": "g", "ply": 12,
            "normal_branches": [
                {"piece": "<P>", "source": "<SQ_7g>"},
                {"piece": "<B>", "source": "<SQ_8h>"},
                {"piece": "<S>", "source": "<SQ_7i>"},
            ],
        }
        selected = select_normal_branches(item, 2, 3, "<P>")
        self.assertEqual({value["piece"] for value in selected}, {"<B>", "<S>"})


if __name__ == "__main__":
    unittest.main()
