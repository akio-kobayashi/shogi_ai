import argparse
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import package_analysis_results as package_results


class PackageAnalysisResultsTest(unittest.TestCase):
    def test_collects_hand_evaluation_from_canonical_run_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "factorized_v3_eos_results"
            hand = (
                results / "llama-small" / "implicit-initial" / "vanilla-p0.0"
                / "seed-20260802" / "evaluation" / "hand-evaluation"
            )
            hand.mkdir(parents=True)
            (hand / "hand_dynamics_metrics.json").write_text(
                json.dumps({"metrics": {"drop_top1_fully_legal_rate": 0.5}}),
                encoding="utf-8",
            )
            output = root / "analysis.tar.gz"
            args = argparse.Namespace(
                results_dir=str(results), output=str(output), dataset_dir=None,
                include_probe_artifacts=False, include_tensorboard=False,
                no_logs=False, force=False,
            )
            original_parse_args = package_results.parse_args
            package_results.parse_args = lambda: args
            try:
                package_results.main()
            finally:
                package_results.parse_args = original_parse_args

            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
                expected = (
                    "analysis_bundle/results/llama-small/implicit-initial/vanilla-p0.0/"
                    "seed-20260802/evaluation/hand-evaluation/hand_dynamics_metrics.json"
                )
                self.assertIn(expected, names)
                manifest = json.load(
                    archive.extractfile("analysis_bundle/COLLECTION_MANIFEST.json")
                )
            self.assertIn("hand_dynamics_metrics.json", manifest["present_result_types"])
            self.assertEqual(
                manifest["result_locations"]["hand_dynamics_metrics.json"],
                [
                    "llama-small/implicit-initial/vanilla-p0.0/seed-20260802/"
                    "evaluation/hand-evaluation/hand_dynamics_metrics.json"
                ],
            )


if __name__ == "__main__":
    unittest.main()
