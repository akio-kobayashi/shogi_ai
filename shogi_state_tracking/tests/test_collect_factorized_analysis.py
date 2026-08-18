import unittest

from collect_factorized_analysis import required_matrix


class RequiredMatrixTest(unittest.TestCase):
    def test_probe_only_collection_does_not_require_move_metrics(self):
        observed = {
            ("vanilla-p0.0", "standard"): {"probe_metrics.json"},
        }

        self.assertEqual(
            required_matrix(
                observed,
                ("vanilla-p0.0",),
                ("standard",),
                ("probe_metrics.json",),
            ),
            [],
        )

    def test_default_pair_can_still_require_move_and_probe_metrics(self):
        observed = {
            ("vanilla-p0.0", "standard"): {"probe_metrics.json"},
        }

        self.assertEqual(
            required_matrix(
                observed,
                ("vanilla-p0.0",),
                ("standard",),
                ("move_metrics.json", "probe_metrics.json"),
            ),
            [
                {
                    "condition": "vanilla-p0.0",
                    "dataset": "standard",
                    "missing": ["move_metrics.json"],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
