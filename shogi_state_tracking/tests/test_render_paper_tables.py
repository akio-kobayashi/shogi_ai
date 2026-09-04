import json
import tempfile
import unittest
from pathlib import Path

import render_paper_tables as render


def summary(metrics: dict, runs: int = 1, condition: str = "vanilla-p0.0") -> dict:
    return {
        "conditions": [condition],
        "single_seed_conditions": [condition] if runs == 1 else [],
        "by_condition": {condition: {"runs": runs, "metrics": {
            name: {"mean": value[0], "std": value[1], "n": runs, "values": []}
            for name, value in metrics.items()
        }}},
        "series": {},
    }


class ValueFormattingTest(unittest.TestCase):
    def test_percentage_without_seed_variance(self):
        document = summary({"m": (0.8288, None)})
        self.assertEqual(render.render_value(document, "vanilla-p0.0", render.Column("x", "m")), "82.88\\%")

    def test_percentage_with_seed_variance(self):
        document = summary({"m": (0.8288, 0.0012)}, runs=3)
        self.assertEqual(
            render.render_value(document, "vanilla-p0.0", render.Column("x", "m")),
            "82.88\\% $\\pm$ 0.12",
        )

    def test_num3_and_int_formats(self):
        document = summary({"pp": (3.70565, None), "n": (10000.0, None)})
        self.assertEqual(render.render_value(document, "vanilla-p0.0", render.Column("x", "pp", "num3")), "3.706")
        self.assertEqual(render.render_value(document, "vanilla-p0.0", render.Column("x", "n", "int")), "10,000")

    def test_missing_metric_renders_a_dash(self):
        document = summary({})
        self.assertEqual(render.render_value(document, "vanilla-p0.0", render.Column("x", "absent")), "---")

    def test_confidence_interval_is_composed_from_three_metrics(self):
        document = summary({
            "d": (0.15382, None),
            "d_ci_lower": (0.14805, None),
            "d_ci_upper": (0.15979, None),
        })
        self.assertEqual(
            render.render_value(document, "vanilla-p0.0", render.Column("x", "d", "ci")),
            "0.154 [0.148, 0.160]",
        )

    def test_interval_carries_seed_variance_when_available(self):
        document = summary({
            "d": (0.15382, 0.004),
            "d_ci_lower": (0.14805, None),
            "d_ci_upper": (0.15979, None),
        }, runs=3)
        self.assertIn("$\\pm$ 0.004", render.render_value(document, "vanilla-p0.0", render.Column("x", "d", "ci")))


class SingleSeedMarkerTest(unittest.TestCase):
    def test_single_seed_condition_is_marked(self):
        document = summary({}, runs=1)
        self.assertIn("†", render.condition_label(document, "vanilla-p0.0"))

    def test_multi_seed_condition_is_not_marked(self):
        document = summary({}, runs=3)
        self.assertNotIn("†", render.condition_label(document, "vanilla-p0.0"))


class TableContractTest(unittest.TestCase):
    def _table(self, key: str) -> render.Table:
        return next(table for table in render.TABLES if table.key == key)

    def test_move_tables_exclude_the_oracle_condition(self):
        """AP's canonical_move_perplexity is a piece-conditioned diagnostic, not comparable."""
        for key in ("move_prediction", "move_prediction_unseen", "move_prediction_lishogi"):
            conditions = self._table(key).conditions
            self.assertIsNotNone(conditions, f"{key} must pin its conditions")
            self.assertNotIn("ap-p1.0-proportional-annotation-v1", conditions)

    def test_ap_sensitivity_reports_both_perplexity_definitions(self):
        table = self._table("ap_sensitivity")
        metrics = [column.metric for column in table.columns]
        self.assertIn("move_perplexity_ap_canonical", metrics)
        self.assertIn("move_perplexity", metrics)
        self.assertIn("比較できない", table.note)

    def test_attention_ablation_excludes_the_oracle_condition(self):
        self.assertNotIn("ap-p1.0-proportional-annotation-v1",
                         self._table("attention_ablation").conditions)

    def test_every_table_key_is_unique(self):
        keys = [table.key for table in render.TABLES]
        self.assertEqual(len(keys), len(set(keys)))


class OutputTest(unittest.TestCase):
    def test_tex_and_markdown_are_written_with_a_seed_note(self):
        document = summary({"move_queries": (10000.0, None), "move_perplexity": (3.705, None),
                            "move_top1": (0.5657, None), "move_top5": (0.9114, None),
                            "move_top1_legal": (0.9992, None)})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "study_summary.json").write_text(json.dumps(document), encoding="utf-8")
            namespace = type("A", (), {"summary": str(root / "study_summary.json"),
                                       "output": str(root / "paper"), "tables": "move_prediction"})()
            original = render.parse_args
            render.parse_args = lambda: namespace
            try:
                self.assertEqual(render.main(), 0)
            finally:
                render.parse_args = original
            tex = (root / "paper/tables/move_prediction.tex").read_text(encoding="utf-8")
            markdown = (root / "paper/tables/move_prediction.md").read_text(encoding="utf-8")
            self.assertIn("\\toprule", tex)
            self.assertIn("\\bottomrule", tex)
            self.assertIn("自動生成", tex)
            self.assertIn("単一シード", tex)
            self.assertIn("| 56.57% |", markdown)
            self.assertNotIn("\\%", markdown, "markdown must not keep TeX escapes")
            self.assertTrue((root / "paper/tables/all_tables.tex").is_file())


class FigureTest(unittest.TestCase):
    def test_series_are_averaged_over_seeds(self):
        document = {
            "conditions": ["vanilla-p0.0"], "by_condition": {}, "single_seed_conditions": [],
            "series": {
                "vanilla-p0.0/seed-1": {"probe_by_layer": {"9": {"board_macro_f1": 0.80}}},
                "vanilla-p0.0/seed-2": {"probe_by_layer": {"9": {"board_macro_f1": 0.90}}},
            },
        }
        series = render.figure_series(document, "board_macro_f1")
        self.assertEqual(len(series["RAPなし"]), 1)
        layer, value = series["RAPなし"][0]
        self.assertEqual(layer, 9)
        self.assertAlmostEqual(value, 0.85)

    def test_svg_is_written_and_well_formed(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "figure.svg"
            render.line_figure(target, "title", "y", {"a": [(0, 0.5), (1, 0.6)]}, (0.0, 1.0))
            text = target.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("<svg"))
            self.assertTrue(text.rstrip().endswith("</svg>"))
            self.assertIn("<polyline", text)

    def test_empty_series_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "figure.svg"
            render.line_figure(target, "title", "y", {}, (0.0, 1.0))
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
