import json
import tempfile
import unittest
from pathlib import Path

import verify_study_integrity as verify


def write_run(root: Path, condition: str, seed: str, manifest_value: str) -> Path:
    run = root / "results/llama-reference/implicit-initial" / condition / f"seed-{seed}"
    (run / "evaluation/probes").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"dataset": {"schema_version": 4, "manifest": manifest_value,
                                "vocab_sha256": "deadbeef"}}),
        encoding="utf-8",
    )
    return run


def run_gate(study: Path, report: Path, **overrides: str) -> dict:
    arguments = {
        "study_root": str(study),
        "report": str(report),
        "allow_missing": "",
        "conditions": ",".join(verify.CONDITIONS),
        "seeds": ",".join(verify.DEFAULT_SEEDS),
        "oracle_seed": None,
        "dataset_dir": None,
        "target_epochs": 50,
        "causal_threshold": 1e-4,
        "check_torch_provenance": False,
    }
    arguments.update(overrides)
    namespace = type("Args", (), arguments)()
    original = verify.parse_args
    verify.parse_args = lambda: namespace
    try:
        verify.main()
    finally:
        verify.parse_args = original
    return json.loads(report.read_text(encoding="utf-8"))


def findings_for(report: dict, check: str) -> list[dict]:
    return [item for item in report["findings"] if item["check"] == check]


class StudyIntegrityContractTest(unittest.TestCase):
    def test_ap_uses_only_first_seed_and_excludes_drop_relevance(self):
        runs = list(verify.expected_runs(verify.CONDITIONS, verify.DEFAULT_SEEDS))
        self.assertEqual(len(runs), 10)
        self.assertEqual([seed for condition, seed in runs if condition.startswith("ap-")], ["20260802"])
        self.assertNotIn("drop-relevance", verify.artifact_contract(verify.CONDITIONS[-1]))

    def test_oracle_seed_is_selectable_and_empty_seeds_rejected(self):
        runs = list(verify.expected_runs(verify.CONDITIONS, ("20260803", "20260804"), "20260802"))
        self.assertEqual([seed for condition, seed in runs if condition.startswith("ap-")], ["20260802"])
        with self.assertRaises(ValueError):
            list(verify.expected_runs(verify.CONDITIONS, ()))

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


class ManifestResolutionTest(unittest.TestCase):
    def test_placeholder_is_expanded_against_dataset_dir(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest = base / "dataset_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            resolved = verify.resolve_manifest(
                f"{verify.DATASET_DIR_PLACEHOLDER}/dataset_manifest.json", (base,)
            )
            self.assertEqual(resolved, manifest.resolve())

    def test_relative_path_resolves_against_a_later_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "data").mkdir()
            manifest = base / "data/dataset_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            resolved = verify.resolve_manifest(
                "data/dataset_manifest.json", (Path(temporary) / "absent", base)
            )
            self.assertEqual(resolved, manifest.resolve())

    def test_unresolvable_path_returns_none(self):
        self.assertIsNone(verify.resolve_manifest("data/dataset_manifest.json", ()))
        self.assertIsNone(verify.resolve_manifest(None, (Path("/tmp"),)))


class DatasetConsistencyTest(unittest.TestCase):
    def test_partially_resolved_signatures_do_not_pass(self):
        """A run whose manifest cannot be resolved must not be silently excluded."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "dataset_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            for index, condition in enumerate(verify.CONDITIONS):
                value = str(manifest) if index == 0 else "/nonexistent/dataset_manifest.json"
                write_run(root / "study", condition, "20260802", value)
            report = run_gate(
                root / "study", root / "report.json",
                seeds="20260802", conditions=",".join(verify.CONDITIONS),
            )
            consistency = findings_for(report, "dataset-consistency")
            self.assertEqual(len(consistency), 1)
            self.assertEqual(consistency[0]["status"], "fail")
            self.assertIn("unresolved=3", consistency[0]["detail"])

    def test_fully_resolved_matching_signatures_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "dataset_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            for condition in verify.CONDITIONS:
                write_run(root / "study", condition, "20260802", str(manifest))
            report = run_gate(root / "study", root / "report.json", seeds="20260802")
            consistency = findings_for(report, "dataset-consistency")
            self.assertEqual(consistency[0]["status"], "pass")


class MalformedJsonTest(unittest.TestCase):
    def test_corrupt_probe_metrics_is_reported_not_raised(self):
        """A malformed artifact must produce a finding and still write the report."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "dataset_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            run = write_run(root / "study", verify.CONDITIONS[0], "20260802", str(manifest))
            (run / "evaluation/probes/probe_metrics.json").write_text(
                "{ not valid json", encoding="utf-8"
            )
            report_path = root / "report.json"
            report = run_gate(
                root / "study", report_path,
                seeds="20260802", conditions=verify.CONDITIONS[0],
            )
            self.assertTrue(report_path.is_file())
            broken = findings_for(report, "artifact-json")
            self.assertTrue(broken, "expected an artifact-json finding")
            self.assertTrue(all(item["status"] == "fail" for item in broken))
            # The gate must keep going rather than aborting on the first bad file.
            self.assertTrue(findings_for(report, "stage-log"))


class AllowSelectorTest(unittest.TestCase):
    def test_selector_granularity(self):
        allowed = {"artifacts:chess-protocol"}
        self.assertTrue(verify.selector_matches("artifacts", "run-a", "chess-protocol", allowed))
        self.assertFalse(verify.selector_matches("artifacts", "run-a", "moves", allowed))
        self.assertTrue(
            verify.selector_matches("artifacts", "run-a", "moves", {"artifacts:run-a:moves"})
        )
        self.assertTrue(verify.selector_matches("artifacts", None, None, {"artifacts"}))




class TorchProvenanceOptInTest(unittest.TestCase):
    """The .pt provenance check must be opt-in so the gate needs no torch by default."""

    def _gate(self, check_torch: bool):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "dataset_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            run = write_run(root / "study", verify.CONDITIONS[0], "20260802", str(manifest))
            (run / "evaluation/probes/linear_probes.pt").write_bytes(b"stub")
            report = run_gate(
                root / "study", root / "report.json",
                seeds="20260802", conditions=verify.CONDITIONS[0],
                check_torch_provenance=check_torch,
            )
            return [item for item in findings_for(report, "artifact-commit")
                    if (item["path"] or "").endswith(".pt")]

    def test_disabled_by_default(self):
        self.assertEqual(self._gate(False), [])

    def test_enabled_reports_a_finding_without_crashing(self):
        findings = self._gate(True)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
