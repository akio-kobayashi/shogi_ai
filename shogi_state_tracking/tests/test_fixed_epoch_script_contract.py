import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_DEFAULT = 'RAP_RATES="${RAP_RATES:-0.0,0.15,0.25,1.0}"'
AP_SWITCH = '[[ "${rate}" == 1 || "${rate}" == 1.0 || "${rate}" == 1.00 ]] && mode=ap'


class FixedEpochScriptContractTest(unittest.TestCase):
    def test_training_and_evaluation_keep_ap_in_the_default_condition_set(self):
        for relative in (
            "scripts/run_factorized_fixed_epoch_training.sh",
            "scripts/run_factorized_fixed_epoch_evaluation.sh",
        ):
            text = (PROJECT / relative).read_text(encoding="utf-8")
            self.assertIn(EXPECTED_DEFAULT, text, relative)
            self.assertIn(AP_SWITCH, text, relative)

    def test_collector_default_condition_set_contains_ap(self):
        from collect_factorized_analysis import CONDITIONS

        self.assertIn("ap-p1.0-proportional-annotation-v1", CONDITIONS)


if __name__ == "__main__":
    unittest.main()
