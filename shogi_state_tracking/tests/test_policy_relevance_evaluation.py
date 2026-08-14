import sys
import unittest
from pathlib import Path

import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))


class PolicyRelevanceEvaluationTest(unittest.TestCase):
    def test_move_squares(self):
        from evaluate_factorized_policy_relevance import move_squares, square_index

        self.assertEqual(move_squares("7g7f"), (square_index("7g"), square_index("7f")))
        self.assertEqual(move_squares("P*7f"), (None, square_index("7f")))

    def test_roles_separate_actual_candidates_and_background(self):
        from evaluate_factorized_policy_relevance import role_squares, square_index

        roles = role_squares("7g7f", ["7g7f", "2g2f", "8h3c"])
        self.assertEqual(roles["actual_source"], (square_index("7g"),))
        self.assertEqual(roles["actual_destination"], (square_index("7f"),))
        self.assertIn(square_index("2g"), roles["candidate_source"])
        self.assertIn(square_index("3c"), roles["candidate_destination"])
        self.assertNotIn(square_index("7g"), roles["background"])

    def test_coordinate_piece_recency_matching(self):
        from evaluate_factorized_policy_relevance import _role_metrics

        targets = torch.zeros((2, 81), dtype=torch.long)
        prediction = targets.clone()
        prediction[1, 0] = 1
        queries = [
            {
                "roles": {name: tuple() for name in (
                    "actual_source", "actual_destination", "actual_move", "endpoint_attacker",
                    "actual_local_context", "candidate_source",
                    "candidate_destination", "candidate_related", "background",
                )},
                "recency": ["never"] * 81,
            }
            for _ in range(2)
        ]
        queries[0]["roles"]["actual_source"] = (0,)
        queries[0]["roles"]["background"] = tuple(range(1, 81))
        queries[1]["roles"]["background"] = tuple(range(81))
        metrics = _role_metrics(prediction == targets, targets, queries)
        matched = metrics["actual_source"]["coordinate_piece_recency_matched"]
        self.assertEqual(matched["matched_observations"], 1)
        self.assertEqual(matched["relevant_accuracy"], 1.0)
        self.assertEqual(matched["background_accuracy"], 0.0)
        self.assertEqual(matched["difference"], 1.0)


if __name__ == "__main__":
    unittest.main()
