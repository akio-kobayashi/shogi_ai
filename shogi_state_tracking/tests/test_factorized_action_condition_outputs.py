import json
import sys
import tempfile
import unittest
from pathlib import Path


class FactorizedActionConditionOutputTest(unittest.TestCase):
    @staticmethod
    def source_metrics(value):
        branch = {
            "actual_drop": {
                name: {"relevant": {"mean_true_count_probability": value + offset}}
                for name, offset in (("pre", 0.0), ("drop", 0.1), ("normal", 0.02))
            },
            "actual_normal": {
                name: {"relevant": {"mean_true_count_probability": value + offset}}
                for name, offset in (("pre", 0.0), ("drop", 0.04), ("normal", 0.01))
            },
        }
        contrasts = {}
        for group in ("all", "actual_drop", "actual_normal"):
            contrasts[group] = {
                "relevant_count_drop_minus_normal": {"mean": 0.08},
                "irrelevant_count_drop_minus_normal": {"mean": 0.01},
                "selective_count_difference_in_differences": {"mean": 0.07},
            }
        return {"branch_metrics": branch, "within_prefix_contrasts": contrasts}

    def test_visualizer_writes_all_figures(self):
        import visualize_factorized_action_condition as visualizer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.json"
            metrics.write_text(json.dumps({
                "metrics": {
                    "layer_0": self.source_metrics(0.2),
                    "layer_1": self.source_metrics(0.3),
                }
            }), encoding="utf-8")
            output = root / "figures"
            original = sys.argv
            sys.argv = ["visualize", "--metrics", str(metrics), "--output-dir", str(output)]
            try:
                visualizer.main()
            finally:
                sys.argv = original
            self.assertTrue((output / "within_prefix_action_contrast.svg").is_file())
            self.assertTrue((output / "selective_action_condition.svg").is_file())
            self.assertTrue((output / "relevant_hand_by_branch.svg").is_file())

    def test_matrix_keeps_primary_and_ap_separate(self):
        import collect_factorized_action_condition_matrix as collector

        conditions = (
            ("vanilla-p0.0", "primary"),
            ("rap-p0.15-proportional-rap-v1", "primary"),
            ("rap-p0.25-proportional-rap-v1", "primary"),
            ("ap-p1.0-proportional-annotation-v1", "oracle-native"),
            ("ap-p1.0-proportional-annotation-v1", "sensitivity-no-annotation"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for condition, category in conditions:
                path = (
                    root / "llama-reference" / "implicit-initial" / condition / "seed-7"
                    / "evaluation" / "action-condition" / category / "action_condition_metrics.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "checkpoint": "model.pt", "protocol": {"primary_comparable": category == "primary"},
                    "metrics": {"layer_0": self.source_metrics(0.2)},
                }), encoding="utf-8")
            output = root / "summary.json"
            original = sys.argv
            sys.argv = [
                "collect", "--results-dir", str(root), "--seeds", "7", "--output", str(output),
            ]
            try:
                collector.main()
            finally:
                sys.argv = original
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["primary"]), {"vanilla", "rap_0.15", "rap_0.25"})
            self.assertEqual(set(payload["oracle"]), {"ap_native", "ap_no_annotation"})
            self.assertFalse(payload["design"]["pool_ap_with_primary"])


if __name__ == "__main__":
    unittest.main()
