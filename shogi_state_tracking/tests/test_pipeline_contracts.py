"""パイプライン4層の成果物一覧が整合していることを検査する。

生成（run_factorized_full_evaluation.sh）→ 監査（verify_study_integrity）→
収集（collect_factorized_analysis）→ 集約（summarize_factorized_study）の
どこかで一覧がずれると，成果物が「作られたのに収集されない」「期待されたのに
作られない」「収集されたのに集約されない」という取りこぼしが静かに起きる。

各層の一覧を突き合わせ，宣言のない差分を失敗として検出する。
"""

import re
import unittest
from pathlib import Path

import collect_factorized_analysis as collect
import summarize_factorized_study as summarize
import verify_study_integrity as verify

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/run_factorized_full_evaluation.sh"
# action-conditionは4条件を横断するstudy単位の評価であり，checkpoint単位の
# 駆動には含めない。distribution-baselinesはmoves段階が生成する。
DRIVER_EXEMPT_STAGES = {"action-condition", "distribution-baselines"}


def contract_files(condition: str) -> dict[str, str]:
    """成果物の相対パス -> 段階名。"""
    mapping: dict[str, str] = {}
    for stage, paths in verify.artifact_contract(condition).items():
        for path in paths:
            mapping[path.as_posix()] = stage
    return mapping


def all_contract_files() -> dict[str, str]:
    merged: dict[str, str] = {}
    for condition in verify.CONDITIONS:
        merged.update(contract_files(condition))
    return merged


def driver_artifacts() -> tuple[set[str], set[str]]:
    """駆動スクリプトが完了判定に使う成果物と段階名を読み取る。"""
    source = DRIVER.read_text(encoding="utf-8")
    block = source.split("stage_artifact()")[1].split("\n}")[0]
    files = set(re.findall(r'"([A-Za-z0-9_./-]+\.(?:json|pt))"', block))
    stages = set(re.findall(r"^\s{4}([a-z-]+)\)", block, re.M))
    return files, stages


def summarizer_files() -> set[str]:
    files = {relative for _, relative, _ in summarize.ARTIFACTS}
    files |= {relative for _, relative, _ in summarize.ORACLE_ONLY}
    files |= set(summarize.ORACLE_REPLACEMENTS.values())
    return files


class DriverCoversContractTest(unittest.TestCase):
    def test_driver_produces_every_checkpoint_level_contract_artifact(self):
        contract = contract_files(verify.CONDITIONS[0])
        expected = {path for path, stage in contract.items()
                    if stage not in DRIVER_EXEMPT_STAGES}
        produced, _ = driver_artifacts()
        self.assertEqual(
            expected - produced, set(),
            "the gate requires artifacts the per-checkpoint driver never checks for",
        )

    def test_driver_checks_nothing_outside_the_contract(self):
        produced, _ = driver_artifacts()
        self.assertEqual(
            produced - set(all_contract_files()), set(),
            "the driver waits on artifacts no contract requires",
        )

    def test_driver_stage_names_are_contract_stage_names_or_declared(self):
        contract_stages = set(all_contract_files().values())
        _, stages = driver_artifacts()
        # 駆動側の段階名は契約の段階名と対応する。tokenとchessとterminalは
        # run_factorized_evaluation.sh側の呼び出し名なので別名を許す。
        aliases = {"token": "token-probe", "chess": "chess-protocol",
                   "terminal": "terminal-probe", "moves": "moves"}
        unknown = {aliases.get(stage, stage) for stage in stages} - contract_stages
        self.assertEqual(unknown, set(), f"driver stages not in any contract: {unknown}")


class CollectorCoversContractTest(unittest.TestCase):
    def test_collector_tracks_every_contract_filename(self):
        """収集されない成果物はbundleに入らず，集約から永久に落ちる。"""
        names = {path.rsplit("/", 1)[-1] for path in all_contract_files()
                 if path.endswith(".json")}
        untracked = names - set(collect.TRACKED_RESULT_NAMES)
        self.assertEqual(untracked, set(),
                         f"contract artifacts the collector never gathers: {untracked}")

    def test_collector_tracks_the_probe_artifacts_the_contract_names(self):
        names = {path.rsplit("/", 1)[-1] for path in all_contract_files()
                 if path.endswith(".pt")}
        self.assertEqual(names - set(collect.PROBE_ARTIFACTS), set())


class SummarizerCoversContractTest(unittest.TestCase):
    def test_every_contract_artifact_is_read_pending_or_declared(self):
        """集約されない成果物は，pendingか意図的除外として宣言されていること。"""
        contract = all_contract_files()
        read = summarizer_files()
        undeclared = []
        for path, stage in sorted(contract.items()):
            if path.endswith(".pt") or path in read:
                continue
            if stage in summarize.PENDING_ARTIFACTS:
                continue
            if path.rsplit("/", 1)[-1] in summarize.NOT_SUMMARIZED:
                continue
            undeclared.append(f"{path} (stage={stage})")
        self.assertEqual(undeclared, [],
                         "contract artifacts neither summarized nor declared: " + "; ".join(undeclared))

    def test_summarizer_reads_nothing_outside_the_contract(self):
        self.assertEqual(summarizer_files() - set(all_contract_files()), set())

    def test_pending_artifacts_are_real_contract_stages(self):
        stages = set(all_contract_files().values())
        self.assertEqual(set(summarize.PENDING_ARTIFACTS) - stages, set(),
                         "PENDING_ARTIFACTS names a stage no contract defines")

    def test_not_summarized_entries_carry_a_reason(self):
        for name, reason in summarize.NOT_SUMMARIZED.items():
            self.assertTrue(reason.strip(), f"{name} needs a stated reason")
            self.assertIn(name, {path.rsplit("/", 1)[-1] for path in all_contract_files()})


class ConditionContractTest(unittest.TestCase):
    def test_oracle_condition_keeps_both_annotation_protocols(self):
        oracle = contract_files(verify.CONDITIONS[-1])
        self.assertTrue(any("oracle-native" in path for path in oracle))
        self.assertTrue(any("sensitivity-no-annotation" in path for path in oracle))

    def test_oracle_condition_has_no_attention_ablation(self):
        oracle = contract_files(verify.CONDITIONS[-1])
        self.assertFalse(any("attention_ablation" in path for path in oracle))

    def test_condition_lists_agree_across_layers(self):
        self.assertEqual(set(verify.CONDITIONS), set(summarize.CONDITIONS))
        self.assertEqual(set(verify.PRIMARY_CONDITIONS), set(summarize.PRIMARY_CONDITIONS))
        self.assertEqual(set(verify.CONDITIONS), set(collect.CONDITIONS))


if __name__ == "__main__":
    unittest.main()
