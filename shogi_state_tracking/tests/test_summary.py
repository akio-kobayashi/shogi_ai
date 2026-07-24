import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]


class SummaryTest(unittest.TestCase):
    def test_collects_complete_2x2_outputs(self):
        with tempfile.TemporaryDirectory() as temp_text:
            experiment = Path(temp_text) / "experiment"
            seed_root = experiment / "seed_7"
            for model_type in ("vanilla", "t2mlr"):
                files = (
                    seed_root
                    / model_type
                    / "answer-only"
                    / "training_history.json",
                    seed_root
                    / model_type
                    / "cot"
                    / "probes-answer-only"
                    / "probe_metrics.json",
                    seed_root
                    / model_type
                    / "cot"
                    / "training"
                    / "training_history.json",
                    seed_root
                    / model_type
                    / "cot"
                    / "probes-cot"
                    / "probe_metrics.json",
                    seed_root
                    / model_type
                    / "cot"
                    / "evaluation"
                    / "reasoning_metrics.json",
                )
                for index, path in enumerate(files):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps({"metric": index, "nested": {"rate": 0.5}}),
                        encoding="utf-8",
                    )

            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_DIR / "summarize_2x2.py"),
                    "--experiment-dir",
                    str(experiment),
                    "--seed",
                    "7",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            csv_path = seed_root / "summary" / "comparison_metrics.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertEqual(
                {row["model_type"] for row in rows}, {"vanilla", "t2mlr"}
            )
            self.assertEqual(
                {row["training_condition"] for row in rows},
                {"answer-only", "cot"},
            )


if __name__ == "__main__":
    unittest.main()
