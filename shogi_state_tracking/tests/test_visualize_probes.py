import tempfile
import unittest
from pathlib import Path

import torch


MODULE_DIR = Path(__file__).resolve().parents[1]


class ProbeVisualizationTest(unittest.TestCase):
    def payload(self):
        target = torch.zeros((2, 81), dtype=torch.long)
        target[0, 0] = 1
        target[1, 80] = 15
        prediction = target.clone()
        prediction[1, 80] = 0
        return {
            "format_version": 1,
            "evaluation": {
                "layer_1": {
                    "board_target": target,
                    "board_prediction": prediction,
                    "board_target_probability": torch.full((2, 81), 0.9),
                    "board_prediction_probability": torch.full((2, 81), 0.9),
                    "hand_target": torch.zeros((2, 14), dtype=torch.long),
                    "hand_prediction": torch.zeros((2, 14), dtype=torch.long),
                    "turn_target": torch.zeros(2, dtype=torch.long),
                    "turn_prediction": torch.zeros(2, dtype=torch.long),
                    "distances": torch.tensor([0, 1]),
                    "scopes": ["open", "closed"],
                    "game_ids": ["a", "b"],
                }
            },
        }

    def test_aggregate_and_position_svg(self):
        import visualize_probes

        payload = self.payload()
        with tempfile.TemporaryDirectory() as temp_dir_text:
            temp_dir = Path(temp_dir_text)
            aggregate = visualize_probes.aggregate_svg(
                payload["evaluation"]["layer_1"], "layer_1", "accuracy"
            )
            position = visualize_probes.position_svg(
                payload["evaluation"]["layer_1"], "layer_1", 1
            )
            (temp_dir / "aggregate.svg").write_text(aggregate, encoding="utf-8")
            (temp_dir / "position.svg").write_text(position, encoding="utf-8")
            self.assertIn("<svg", aggregate)
            self.assertIn("0.50", aggregate)
            self.assertIn("→", position)

    def test_difference_requires_matching_targets(self):
        import visualize_probes

        payload_a = self.payload()
        payload_b = self.payload()
        svg = visualize_probes.difference_svg(payload_a, payload_b, "layer_1")
        self.assertIn("+0.00", svg)


if __name__ == "__main__":
    unittest.main()
