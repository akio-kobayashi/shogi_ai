import unittest


class FactorizedDropRelevancePureTest(unittest.TestCase):
    def test_actual_normal_move_selects_its_own_source(self):
        from factorized_drop_relevance import choose_normal_branch

        item = {
            "game_id": "g", "ply": 12, "move": "7g7f",
            "normal_branches": [
                {"piece": "<R>", "source": "<SQ_2h>"},
                {"piece": "<P>", "source": "<SQ_7g>"},
            ],
        }
        self.assertEqual(choose_normal_branch(item, 7), {"piece": "<P>", "source": "<SQ_7g>"})
        self.assertEqual(
            choose_normal_branch(item, 7, "<P>"),
            {"piece": "<R>", "source": "<SQ_2h>"},
        )

    def test_irrelevant_slot_is_nonzero_and_from_same_side(self):
        from factorized_drop_relevance import choose_irrelevant_hand_slot

        hands = [2, 0, 1, 0, 0, 0, 0] + [4, 0, 0, 0, 0, 0, 0]
        item = {"game_id": "g", "ply": 20, "hands": hands}
        self.assertEqual(choose_irrelevant_hand_slot(item, 0, 3), 2)
        self.assertIsNone(choose_irrelevant_hand_slot(item, 7, 3))


if __name__ == "__main__":
    unittest.main()
