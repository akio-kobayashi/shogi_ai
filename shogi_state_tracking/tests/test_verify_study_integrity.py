import json
import tempfile
import unittest
from pathlib import Path

import verify_study_integrity as verify


class StudyIntegrityContractTest(unittest.TestCase):
    def test_ap_uses_only_first_seed_and_excludes_drop_relevance(self):
        runs = list(verify.expected_runs(verify.CONDITIONS, verify.DEFAULT_SEEDS))
        self.assertEqual(len(runs), 10)
        self.assertEqual([seed for condition, seed in runs if condition.startswith("ap-")], ["20260802"])
        self.assertNotIn("drop-relevance", verify.artifact_contract(verify.CONDITIONS[-1]))

    def test_primary_contract_has_all_twelve_stages(self):
        self.assertEqual(len(verify.artifact_contract(verify.CONDITIONS[0])), 12)

    def test_locates_nested_study_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = (root / "results/llama-reference/implicit-initial/vanilla-p0.0/seed-20260802")
            expected.mkdir(parents=True)
            self.assertEqual(
                verify.locate_run(verify.results_root(root), "vanilla-p0.0", "20260802"), expected
            )

    def test_provenance_accepts_only_nonempty_commit(self):
        self.assertIsNone(verify.provenance_commit({"provenance": {"git_commit": ""}}))
        self.assertEqual(
            verify.provenance_commit({"provenance": {"git_commit": "abc123"}}), "abc123"
        )


if __name__ == "__main__":
    unittest.main()
