import json
import re
import tempfile
import unittest
from pathlib import Path

import provenance


class WithProvenanceTest(unittest.TestCase):
    def test_payload_is_copied_and_block_is_attached(self):
        payload = {"format_version": 3, "metrics": {"a": 1}}
        document = provenance.with_provenance(payload)
        self.assertNotIn("provenance", payload, "caller dict must not be mutated")
        self.assertEqual(document["format_version"], 3)
        self.assertEqual(document["metrics"], {"a": 1})
        block = document["provenance"]
        self.assertEqual(block["provenance_version"], provenance.PROVENANCE_VERSION)
        for field in ("git_commit", "git_dirty", "generated_at", "script", "python"):
            self.assertIn(field, block)

    def test_existing_provenance_is_preserved_not_overwritten(self):
        document = provenance.with_provenance({"provenance": {"legacy": True}})
        self.assertEqual(document["provenance_inner"], {"legacy": True})
        self.assertIn("git_commit", document["provenance"])

    def test_extra_fields_are_merged(self):
        document = provenance.with_provenance({}, stage="moves")
        self.assertEqual(document["provenance"]["stage"], "moves")

    def test_generated_at_is_utc_iso8601(self):
        stamp = provenance.with_provenance({})["provenance"]["generated_at"]
        self.assertRegex(stamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class WriteMetricsJsonTest(unittest.TestCase):
    def test_creates_parent_directories_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested/deeper/metrics.json"
            provenance.write_metrics_json(target, {"value": 1})
            document = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(document["value"], 1)
            self.assertIn("provenance", document)

    def test_trailing_newline_matches_previous_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "metrics.json"
            provenance.write_metrics_json(target, {"value": 1})
            self.assertTrue(target.read_text(encoding="utf-8").endswith("}\n"))


class CshogiProvenanceTest(unittest.TestCase):
    def test_declared_pin_matches_pyproject(self):
        """The fallback pin must be the revision the project actually declares."""
        pinned = provenance._declared_cshogi_pin()
        self.assertIsNotNone(pinned, "pyproject.toml should declare a cshogi revision")
        self.assertRegex(pinned, r"^[0-9a-f]{40}$")
        pyproject = (provenance.REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(pinned, pyproject)

    def test_record_always_reports_a_source(self):
        record = provenance.cshogi_provenance()
        self.assertIn(record["source"], {"installed", "direct_url", "unavailable"})
        self.assertIn("declared_pin", record)


class ContractCoverageTest(unittest.TestCase):
    """Every evaluator that writes a contract artifact must route through the helper."""

    EVALUATORS = (
        "evaluate_factorized_moves.py",
        "evaluate_factorized_distribution_baselines.py",
        "evaluate_new_prompt_probes.py",
        "evaluate_factorized_token_probe.py",
        "evaluate_factorized_chess_protocol.py",
        "evaluate_factorized_action_probes.py",
        "evaluate_factorized_hand_dynamics.py",
        "evaluate_factorized_policy_relevance.py",
        "evaluate_factorized_action_condition.py",
        "evaluate_factorized_action_condition_robustness.py",
        "evaluate_factorized_drop_attention.py",
        "evaluate_factorized_drop_relevance.py",
    )
    RAW_WRITE = re.compile(r"\.write_text\(json\.dumps\([A-Za-z_][A-Za-z0-9_]*, ensure_ascii=False")

    def test_no_evaluator_writes_metrics_json_directly(self):
        offenders = []
        for name in self.EVALUATORS:
            source = (provenance.REPOSITORY / name).read_text(encoding="utf-8")
            if self.RAW_WRITE.search(source):
                offenders.append(name)
            if "write_metrics_json" not in source:
                offenders.append(f"{name} (helper not imported)")
        self.assertEqual(offenders, [], f"evaluators bypassing provenance: {offenders}")

    def test_torch_probe_artifacts_carry_provenance(self):
        for name in ("evaluate_probes.py", "evaluate_new_prompt_probes.py",
                     "evaluate_factorized_action_probes.py",
                     "evaluate_factorized_action_condition_robustness.py"):
            source = (provenance.REPOSITORY / name).read_text(encoding="utf-8")
            self.assertIn("with_provenance", source, f"{name} saves .pt without provenance")


if __name__ == "__main__":
    unittest.main()
