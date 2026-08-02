import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NewPromptMetricTest(unittest.TestCase):
    def test_oracle_action_scores_are_perfect(self):
        from evaluate_new_prompt_moves import _add_metrics, _empty_totals, summarize

        totals = _empty_totals()
        # 正解指手だけに確率1を置いたoracle distributionを模す。
        _add_metrics(totals, target_id=17, top=[17, 18, 19], legal_ids=[17, 23], log_prob=0.0, legal_prob=1.0)
        metrics = summarize(totals)
        self.assertEqual(metrics["cross_entropy"], 0.0)
        self.assertEqual(metrics["perplexity"], 1.0)
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["top5_accuracy"], 1.0)
        self.assertEqual(metrics["legality"]["top1_legal_rate"], 1.0)
        self.assertEqual(metrics["legality"]["top5_contains_legal_rate"], 1.0)
        self.assertEqual(metrics["legality"]["mean_legal_probability_mass"], 1.0)


if __name__ == "__main__":
    unittest.main()
