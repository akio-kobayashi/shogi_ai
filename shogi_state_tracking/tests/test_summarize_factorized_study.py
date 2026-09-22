import json
import tempfile
import unittest
from pathlib import Path

import summarize_factorized_study as summarize


class DigTest(unittest.TestCase):
    def test_missing_keys_return_none_without_raising(self):
        self.assertIsNone(summarize.dig({"a": {"b": 1}}, "a", "missing"))
        self.assertIsNone(summarize.dig({"a": 1}, "a", "b"))
        self.assertIsNone(summarize.dig(None, "a"))
        self.assertEqual(summarize.dig({"a": {"b": 1}}, "a", "b"), 1)


class CanonicalPerplexityTest(unittest.TestCase):
    """The reported perplexity must be the canonical one.

    For RAP conditions the raw value differs substantially because canonical
    masks the annotation-only piece-token logits before normalization.
    """

    def test_move_perplexity_maps_to_canonical(self):
        self.assertEqual(summarize.MOVE_FIELDS["move_perplexity"], "canonical_move_perplexity")
        self.assertEqual(summarize.MOVE_FIELDS["move_perplexity_raw"], "move_perplexity")

    def test_both_variants_are_extracted_and_distinct(self):
        payload = {
            "metrics": {"primary": {
                "move_perplexity": 4.2421,
                "canonical_move_perplexity": 3.6636,
                "grammar_normalized_move_perplexity": 3.6632,
            }}
        }
        values = summarize.extract_moves(payload)
        self.assertAlmostEqual(values["move_perplexity"], 3.6636)
        self.assertAlmostEqual(values["move_perplexity_raw"], 4.2421)


class ProbeLayerSelectionTest(unittest.TestCase):
    def test_layer_with_lowest_validation_loss_is_selected(self):
        results = {
            "layer_0": {"best_validation_loss": 1.0},
            "layer_9": {"best_validation_loss": 0.2},
            "layer_12": {"best_validation_loss": 0.4},
        }
        self.assertEqual(summarize.select_probe_layer(results), "layer_9")

    def test_missing_losses_yield_no_selection(self):
        self.assertIsNone(summarize.select_probe_layer({"layer_0": {}}))
        self.assertIsNone(summarize.select_probe_layer({}))

    def test_input_and_final_positions_follow_layer_order(self):
        payload = {"probe_results": {
            "layer_0": {"best_validation_loss": 1.0, "evaluation": {"board_macro_f1": 0.1}},
            "layer_9": {"best_validation_loss": 0.2, "evaluation": {"board_macro_f1": 0.8}},
            "layer_12": {"best_validation_loss": 0.4, "evaluation": {"board_macro_f1": 0.7}},
        }}
        values = summarize.extract_probes(payload)
        self.assertEqual(values["probe_selected_layer"], 9)
        self.assertAlmostEqual(values["input_board_macro_f1"], 0.1)
        self.assertAlmostEqual(values["selected_board_macro_f1"], 0.8)
        self.assertAlmostEqual(values["final_board_macro_f1"], 0.7)


class AblationDeltaTest(unittest.TestCase):
    def test_delta_is_masked_minus_baseline(self):
        payload = {"ablation": {
            "drop:all:relevant:after_drop": {
                "baseline_probability": 0.5722599943645764,
                "masked_probability": 0.30505056998114743,
                "examples": 250,
            }
        }}
        values = summarize.extract_attention_ablation(payload)
        self.assertAlmostEqual(values["ablation_all_relevant_delta"] * 100, -26.72, places=2)

    def test_missing_block_yields_none_delta(self):
        values = summarize.extract_attention_ablation({})
        self.assertIsNone(values["ablation_all_relevant_delta"])


class AggregateTest(unittest.TestCase):
    def test_std_is_none_for_a_single_run(self):
        summary = summarize.aggregate([{"m": 1.0}], ["m"])
        self.assertEqual(summary["m"]["n"], 1)
        self.assertIsNone(summary["m"]["std"])

    def test_sample_std_across_seeds(self):
        summary = summarize.aggregate([{"m": 1.0}, {"m": 2.0}, {"m": 3.0}], ["m"])
        self.assertAlmostEqual(summary["m"]["mean"], 2.0)
        self.assertAlmostEqual(summary["m"]["std"], 1.0)
        self.assertEqual(summary["m"]["n"], 3)

    def test_non_numeric_and_boolean_values_are_ignored(self):
        summary = summarize.aggregate([{"m": True}, {"m": "x"}, {"m": None}], ["m"])
        self.assertEqual(summary["m"]["n"], 0)
        self.assertIsNone(summary["m"]["mean"])


class OracleContractTest(unittest.TestCase):
    def test_ap_reads_oracle_native_and_skips_attention_ablation(self):
        self.assertIn("action-condition", summarize.ORACLE_REPLACEMENTS)
        self.assertIn("oracle-native", summarize.ORACLE_REPLACEMENTS["action-condition"])
        self.assertIn("attention-ablation", summarize.ORACLE_EXCLUDED)

    def test_excluded_artifact_is_not_reported_missing_for_ap(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "seed-20260802"
            (run / "evaluation").mkdir(parents=True)
            _, missing, _ = summarize.collect_run(run, "ap-p1.0-proportional-annotation-v1")
            self.assertNotIn("attention-ablation", missing)
            _, primary_missing, _ = summarize.collect_run(run, "vanilla-p0.0")
            self.assertIn("attention-ablation", primary_missing)


class SingleSeedFlagTest(unittest.TestCase):
    def test_summary_records_single_seed_primary_conditions(self):
        """A one-seed primary condition must be flagged so it stays exploratory."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "analysis_bundle"
            for condition in summarize.PRIMARY_CONDITIONS[:1]:
                run = bundle / "results/llama-reference/implicit-initial" / condition / "seed-20260802"
                (run / "evaluation").mkdir(parents=True)
                (run / "evaluation/move_metrics.json").write_text(
                    json.dumps({"metrics": {"primary": {"canonical_move_perplexity": 3.7}}}),
                    encoding="utf-8",
                )
            namespace = type("A", (), {
                "bundle": str(bundle), "output": str(root / "summary"),
                "conditions": summarize.PRIMARY_CONDITIONS[0], "seeds": "",
            })()
            original = summarize.parse_args
            summarize.parse_args = lambda: namespace
            try:
                self.assertEqual(summarize.main(), 0)
            finally:
                summarize.parse_args = original
            document = json.loads((root / "summary/study_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(document["single_seed_conditions"], [summarize.PRIMARY_CONDITIONS[0]])
            self.assertTrue(document["interpretation_limits"])
            self.assertIn("provenance", document)


if __name__ == "__main__":
    unittest.main()


class DistributionBaselineTest(unittest.TestCase):
    """暗記ベースラインの抽出。実物のフィールド名に合わせてある。"""

    PAYLOAD = {
        "evaluation": "factorized_move_distribution_baselines_v1",
        "global_train_move": {"majority_move": "7g7f", "share": 0.031},
        "metrics": {
            "primary": {
                "queries": 10000,
                "global_train_move_majority_accuracy": 0.031,
                "train_position_majority": {
                    "covered_queries": 6595,
                    "coverage": 0.6595,
                    "accuracy_all_queries_uncovered_wrong": 0.4102,
                    "accuracy_covered_queries": 0.6220,
                },
                "train_position_distribution": {
                    "mean_occurrences_per_query": 12.4,
                    "mean_distinct_next_moves_per_query": 3.87,
                    "mean_majority_share_per_query": 0.641,
                    "mean_entropy_bits_per_query": 1.204,
                },
            },
            "by_history_distance": {
                "8": {"queries": 5000,
                      "train_position_majority": {"coverage": 0.93,
                                                  "accuracy_all_queries_uncovered_wrong": 0.61}},
                "32": {"queries": 5000,
                       "train_position_majority": {"coverage": 0.39,
                                                   "accuracy_all_queries_uncovered_wrong": 0.21}},
            },
            "evaluation_position_concentration": {
                "queries": 10000,
                "unique_positions": 9814,
                "repeated_position_query_rate": 0.019,
                "macro_entropy_bits": 0.04,
                "in_sample_position_majority_accuracy_descriptive_only": 0.99,
                "warning": "in-sample descriptive statistic; it is not a predictive baseline",
            },
        },
    }

    def test_primary_fields_are_read(self):
        values = summarize.extract_distribution_baselines(self.PAYLOAD, {})
        self.assertEqual(values["baseline_train_position_top1"], 0.4102)
        self.assertEqual(values["baseline_train_position_top1_covered"], 0.6220)
        self.assertEqual(values["baseline_train_position_coverage"], 0.6595)
        self.assertEqual(values["baseline_global_move_top1"], 0.031)
        self.assertEqual(values["baseline_mean_entropy_bits"], 1.204)
        self.assertEqual(values["baseline_global_move_share"], 0.031)

    def test_history_distance_is_split(self):
        values = summarize.extract_distribution_baselines(self.PAYLOAD, {})
        self.assertEqual(values["baseline_train_position_coverage_h8"], 0.93)
        self.assertEqual(values["baseline_train_position_coverage_h32"], 0.39)

    def test_in_sample_statistic_stays_out(self):
        """成果物側が「予測ベースラインではない」と警告する値は取り込まない。"""
        values = summarize.extract_distribution_baselines(self.PAYLOAD, {})
        self.assertNotIn(
            "baseline_evaluation_in_sample_position_majority_accuracy_descriptive_only", values)
        self.assertEqual(values["baseline_evaluation_unique_positions"], 9814)

    def test_every_field_carries_the_dataset_level_prefix(self):
        """条件比較の行へ紛れ込ませないための印。"""
        values = summarize.extract_distribution_baselines(self.PAYLOAD, {})
        self.assertTrue(values)
        for name in values:
            self.assertTrue(name.startswith(summarize.DATASET_LEVEL_PREFIX), name)

    def test_missing_blocks_do_not_raise(self):
        values = summarize.extract_distribution_baselines({}, {})
        self.assertIsNone(values["baseline_train_position_top1"])

    def test_nothing_is_pending_any_more(self):
        self.assertEqual(summarize.PENDING_ARTIFACTS, ())


class ProvenanceReportingTest(unittest.TestCase):
    """来歴のない成果物を名指しできること。

    ``verify_study_integrity``のartifact-commitと同じ判定だが，verifyは日常的に
    実行されていない。集約側で毎回検査しないと，古いコードで作られた成果物が
    値だけ出して混ざる。
    """

    def write(self, run: Path, payload: dict) -> None:
        (run / "evaluation").mkdir(parents=True, exist_ok=True)
        (run / "evaluation/move_metrics.json").write_text(
            json.dumps(payload), encoding="utf-8")

    def test_missing_provenance_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "seed-20260802"
            self.write(run, {"metrics": {"primary": {"queries": 1}}})
            _, _, unprovenanced = summarize.collect_run(run, "vanilla-p0.0")
            self.assertIn("move_metrics.json", unprovenanced)

    def test_present_provenance_is_not_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "seed-20260802"
            self.write(run, {"provenance": {"git_commit": "abc123"},
                             "metrics": {"primary": {"queries": 1}}})
            _, _, unprovenanced = summarize.collect_run(run, "vanilla-p0.0")
            self.assertEqual(unprovenanced, [])

    def test_empty_commit_counts_as_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "seed-20260802"
            self.write(run, {"provenance": {"git_commit": ""},
                             "metrics": {"primary": {"queries": 1}}})
            _, _, unprovenanced = summarize.collect_run(run, "vanilla-p0.0")
            self.assertIn("move_metrics.json", unprovenanced)

    def test_matches_the_verifier(self):
        """判定をverify側と一致させる。片方だけ緩むと検査が意味を失う。"""
        import verify_study_integrity as verify
        for payload in ({"provenance": {"git_commit": "abc"}},
                        {"provenance": {"git_commit": ""}},
                        {"git_commit": "abc"},
                        {}):
            self.assertEqual(summarize.provenance_commit(payload),
                             verify.provenance_commit(payload), payload)


class PartialMetricsTest(unittest.TestCase):
    """シード間の指標の揃い具合。評価器の版の混在を検出するための検査。"""

    def row(self, seed, condition="vanilla-p0.0", **values):
        return {"condition": condition, "seed": seed, **values}

    def test_metric_missing_in_one_seed_is_reported(self):
        rows = [self.row("1", fh_x=1.0), self.row("2", fh_x=2.0), self.row("3")]
        partial, runs = summarize.find_partial_metrics(rows, ["fh_x"], ["vanilla-p0.0"])
        self.assertEqual(partial, ["fh_x"])
        self.assertEqual(runs, ["vanilla-p0.0/seed-3"])

    def test_zero_sample_bin_is_not_reported(self):
        """標本0件で値が出ない欠落は，版の混在ではない。"""
        rows = [self.row("1", chess_a_lgm_card4_7=1.0, chess_a_queries_card4_7=1),
                self.row("2", chess_a_queries_card4_7=0)]
        partial, runs = summarize.find_partial_metrics(
            rows, ["chess_a_lgm_card4_7", "chess_a_queries_card4_7"], ["vanilla-p0.0"])
        self.assertEqual(partial, [])
        self.assertEqual(runs, [])

    def test_missing_bin_with_nonzero_samples_is_reported(self):
        rows = [self.row("1", chess_a_lgm_card4_7=1.0, chess_a_queries_card4_7=1),
                self.row("2", chess_a_queries_card4_7=5)]
        partial, _ = summarize.find_partial_metrics(
            rows, ["chess_a_lgm_card4_7", "chess_a_queries_card4_7"], ["vanilla-p0.0"])
        self.assertEqual(partial, ["chess_a_lgm_card4_7"])

    def test_differences_across_conditions_are_not_compared(self):
        """APのoracle除外のような条件間の差は設計上のもの。"""
        rows = [self.row("1", condition="vanilla-p0.0", droprel=1.0),
                self.row("2", condition="vanilla-p0.0", droprel=1.0),
                self.row("1", condition="ap-p1.0-proportional-annotation-v1"),
                self.row("2", condition="ap-p1.0-proportional-annotation-v1")]
        partial, _ = summarize.find_partial_metrics(
            rows, ["droprel"], ["vanilla-p0.0", "ap-p1.0-proportional-annotation-v1"])
        self.assertEqual(partial, [])
