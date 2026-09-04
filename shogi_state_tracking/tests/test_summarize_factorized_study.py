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
            _, missing = summarize.collect_run(run, "ap-p1.0-proportional-annotation-v1")
            self.assertNotIn("attention-ablation", missing)
            _, primary_missing = summarize.collect_run(run, "vanilla-p0.0")
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
