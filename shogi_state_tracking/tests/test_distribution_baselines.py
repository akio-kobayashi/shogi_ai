import unittest
from collections import Counter

from evaluate_factorized_distribution_baselines import (
    entropy_bits,
    evaluation_concentration,
    frequency_bin,
    summarize_queries,
)


class DistributionBaselineTest(unittest.TestCase):
    def test_training_position_majority_and_coverage(self):
        queries = [
            {"position_hash": "h1", "target": "7g7f"},
            {"position_hash": "h1", "target": "2g2f"},
            {"position_hash": "h2", "target": "7g7f"},
        ]
        counts = {"h1": Counter({"7g7f": 3, "2g2f": 1})}
        result = summarize_queries(queries, counts, "7g7f")
        self.assertEqual(result["train_position_majority"]["covered_queries"], 2)
        self.assertAlmostEqual(result["train_position_majority"]["coverage"], 2.0 / 3.0)
        self.assertAlmostEqual(
            result["train_position_majority"]["accuracy_covered_queries"], 0.5
        )
        self.assertAlmostEqual(result["global_train_move_majority_accuracy"], 2.0 / 3.0)
        self.assertEqual(result["train_position_distribution"]["mean_occurrences_per_query"], 4.0)
        self.assertEqual(result["train_position_distribution"]["mean_distinct_next_moves_per_query"], 2.0)
        self.assertAlmostEqual(result["train_position_distribution"]["mean_majority_share_per_query"], 0.75)

    def test_descriptive_concentration_is_marked_in_sample(self):
        queries = [
            {"position_hash": "h1", "target": "7g7f"},
            {"position_hash": "h1", "target": "2g2f"},
            {"position_hash": "h2", "target": "7g7f"},
        ]
        result = evaluation_concentration(queries)
        self.assertEqual(result["unique_positions"], 2)
        self.assertEqual(result["singleton_positions"], 1)
        self.assertAlmostEqual(result["in_sample_position_majority_accuracy_descriptive_only"], 2.0 / 3.0)
        self.assertIn("not a predictive baseline", result["warning"])

    def test_entropy_and_frequency_bins(self):
        self.assertEqual(entropy_bits(Counter({"a": 1})), 0.0)
        self.assertEqual(entropy_bits(Counter({"a": 1, "b": 1})), 1.0)
        self.assertEqual([frequency_bin(value) for value in (0, 1, 2, 4, 5, 9, 10, 100)], [
            "0", "1", "2_4", "2_4", "5_9", "5_9", "10_plus", "10_plus",
        ])


if __name__ == "__main__":
    unittest.main()
