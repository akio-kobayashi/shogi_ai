"""評価器の版番号による作り直し判定。

ファイルの有無だけで判定すると評価器を変えた後も古い成果物が残り，git の commit で
判定すると値の変わらない変更でも全成果物を作り直すことになる。版番号はその中間である。
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

import artifact_versions as versions
import summarize_factorized_study as summarize
import verify_study_integrity as verify

PROJECT = Path(__file__).resolve().parents[1]


def write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


class WriterRegistryTest(unittest.TestCase):
    def test_every_writer_defines_and_records_a_version(self):
        for name, script in versions.WRITERS.items():
            source = (PROJECT / script).read_text(encoding="utf-8")
            with self.subTest(artifact=name):
                self.assertRegex(source, versions.VERSION_PATTERN)
                self.assertIn("evaluator_version=EVALUATOR_VERSION", source)

    def test_every_contract_json_has_a_registered_writer(self):
        """契約に成果物を足したとき，書き手の登録漏れで判定が素通りしないように。"""
        names = {Path(path).name
                 for condition in verify.CONDITIONS
                 for paths in verify.artifact_contract(condition).values()
                 for path in paths if Path(path).suffix == ".json"}
        self.assertEqual(names - set(versions.WRITERS), set())


class StatusTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.version = versions.current_version(versions.WRITERS["move_metrics.json"])

    def tearDown(self):
        self.directory.cleanup()

    def test_artifact_without_a_version_counts_as_version_one(self):
        path = write(self.root / "move_metrics.json", {"provenance": {"git_commit": "abc"}})
        self.assertEqual(versions.recorded_version(json.loads(path.read_text())), versions.UNVERSIONED)
        self.assertEqual(versions.status(path)[0], "current" if self.version <= 1 else "stale")

    def test_older_version_is_stale(self):
        path = write(self.root / "move_metrics.json",
                     {"provenance": {"evaluator_version": self.version - 1}})
        self.assertEqual(versions.status(path)[0], "stale")

    def test_current_version_is_current(self):
        path = write(self.root / "move_metrics.json",
                     {"provenance": {"evaluator_version": self.version}})
        self.assertEqual(versions.status(path)[0], "current")

    def test_unreadable_and_unregistered_are_distinguished(self):
        self.assertEqual(versions.status(write(self.root / "move_metrics.json", "{ broken"))[0], "unreadable")
        self.assertEqual(versions.status(write(self.root / "something_else.json", {}))[0], "unregistered")

    def test_exit_codes(self):
        stale = write(self.root / "a/move_metrics.json", {"provenance": {"evaluator_version": self.version - 1}})
        current = write(self.root / "b/move_metrics.json", {"provenance": {"evaluator_version": self.version}})
        self.assertEqual(versions.main(["x", "check", str(current)]), 0)
        self.assertEqual(versions.main(["x", "check", str(stale)]), 1)
        self.assertEqual(versions.main(["x", "check", str(write(self.root / "c/move_metrics.json", "{"))]), 3)


class SummarizerStaleTest(unittest.TestCase):
    def test_stale_artifact_is_named(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "seed-20260802"
            write(run / "evaluation/move_metrics.json",
                  {"provenance": {"git_commit": "abc", "evaluator_version": 0}})
            self.assertEqual(summarize.stale_artifacts(run, "vanilla-p0.0"), ["move_metrics.json"])

    def test_current_artifact_is_not_named(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "seed-20260802"
            write(run / "evaluation/move_metrics.json", {"provenance": {"git_commit": "abc"}})
            self.assertEqual(summarize.stale_artifacts(run, "vanilla-p0.0"), [])


class DriverUsesVersionsTest(unittest.TestCase):
    def test_driver_and_runner_check_versions_not_only_existence(self):
        for script in ("scripts/run_factorized_full_evaluation.sh",
                       "scripts/run_full_history_move_evaluation.sh"):
            with self.subTest(script=script):
                self.assertIn("artifact_versions.py", (PROJECT / script).read_text(encoding="utf-8"))

    def test_runner_no_longer_uses_git_to_judge_staleness(self):
        source = (PROJECT / "scripts/run_full_history_move_evaluation.sh").read_text(encoding="utf-8")
        self.assertNotIn("merge-base", source)
