import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

try:
    import torch
except ImportError:
    torch = None

try:
    import cshogi
except ImportError:
    cshogi = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ProbeMetricTest(unittest.TestCase):
    def test_perfect_predictions_have_perfect_metrics(self):
        from probes import ProbeTargets, state_metrics

        targets = ProbeTargets(
            board=torch.tensor([[0] * 80 + [1], [0] * 79 + [15, 2]]),
            hands=torch.tensor([[0] * 14, [1] + [0] * 13]),
            turn=torch.tensor([0, 1]),
        )
        metrics = state_metrics(
            targets,
            targets.board.clone(),
            targets.hands.clone(),
            targets.turn.clone(),
        )
        self.assertEqual(metrics["board_exact_match"], 1.0)
        self.assertEqual(metrics["board_occupancy_accuracy"], 1.0)
        self.assertEqual(metrics["board_piece_accuracy_on_occupied"], 1.0)
        self.assertEqual(metrics["board_occupied_accuracy"], 1.0)
        self.assertEqual(metrics["hand_exact_match"], 1.0)
        self.assertEqual(metrics["full_state_exact_match"], 1.0)

    def test_majority_baseline_is_position_specific(self):
        from probes import ProbeTargets, majority_predictions

        targets = ProbeTargets(
            board=torch.tensor([[1] + [0] * 80, [1] + [2] + [0] * 79]),
            hands=torch.zeros((2, 14), dtype=torch.long),
            turn=torch.tensor([0, 0]),
        )
        board, hands, turn = majority_predictions(targets, 3)
        self.assertEqual(board.shape, (3, 81))
        self.assertTrue(bool((board[:, 0] == 1).all()))
        self.assertEqual(hands.shape, (3, 14))
        self.assertTrue(bool((turn == 0).all()))


@unittest.skipIf(torch is None or cshogi is None, "torch/cshogi is not installed")
class ProbeReplayTest(unittest.TestCase):
    def test_replay_aligns_state_zero_and_moves(self):
        from probes import replay_probe_targets

        targets = replay_probe_targets(
            cshogi.Board().sfen(), ["7g7f", "3c3d"]
        )
        self.assertEqual(targets.board.shape, (3, 81))
        self.assertEqual(targets.hands.shape, (3, 14))
        self.assertEqual(targets.turn.tolist(), [0, 1, 0])

    def test_evaluation_cli_writes_metrics_and_probes(self):
        import create_dataset
        from models import ModelConfig, VanillaTransformer

        board = cshogi.Board()
        moves = ["7g7f", "3c3d"]
        vocabulary_tokens = create_dataset.base_vocabulary() + moves
        token_to_id = {
            token: index for index, token in enumerate(vocabulary_tokens)
        }
        config = ModelConfig(
            vocab_size=len(token_to_id),
            max_seq_len=128,
            d_model=8,
            n_layers=1,
            n_heads=1,
            d_ff=16,
            dropout=0.0,
        )
        model = VanillaTransformer(config)
        record = {
            "game_id": "synthetic",
            "engine_scope": "closed",
            "initial_sfen": board.sfen(),
            "initial_state_tokens": create_dataset.encode_initial_state(
                board, cshogi
            ),
            "move_tokens": moves,
        }

        with tempfile.TemporaryDirectory() as temp_dir_text:
            temp_dir = Path(temp_dir_text)
            vocab_path = temp_dir / "vocab.json"
            vocab_path.write_text(
                json.dumps({"token_to_id": token_to_id}), encoding="utf-8"
            )
            dataset_path = temp_dir / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            checkpoint_path = temp_dir / "model.pt"
            torch.save(
                {
                    "model_type": "vanilla",
                    "config": config.to_dict(),
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            output_dir = temp_dir / "results"
            command = [
                sys.executable,
                str(MODULE_DIR / "evaluate_probes.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--vocab",
                str(vocab_path),
                "--train-jsonl",
                str(dataset_path),
                "--validation-jsonl",
                str(dataset_path),
                "--evaluation-jsonl",
                str(dataset_path),
                "--output-dir",
                str(output_dir),
                "--candidate-count",
                "1",
                "--min-suffix-moves",
                "1",
                "--positions-per-game",
                "0",
                "--probe-epochs",
                "1",
                "--patience",
                "1",
                "--batch-size",
                "4",
                "--sources",
                "final,token_embedding",
                "--device",
                "cpu",
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics_path = output_dir / "probe_metrics.json"
            self.assertTrue(metrics_path.exists())
            self.assertTrue((output_dir / "linear_probes.pt").exists())
            self.assertTrue((output_dir / "probe_predictions.pt").exists())
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("layer_1", report["probe_results"])
            self.assertIn("token_embedding", report["probe_results"])
            legality = report["language_model"]["evaluation"]["legality"]
            self.assertEqual(legality["move_positions"], 2)
            self.assertGreater(
                legality["mean_legal_move_vocabulary_coverage"], 0.0
            )
            self.assertIn(
                "state_tracking_1_plus",
                report["probe_results"]["layer_1"]["evaluation"]["strata"],
            )

            shell_output = temp_dir / "shell-results"
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON_BIN": sys.executable,
                    "CHECKPOINT": str(checkpoint_path),
                    "VOCAB_PATH": str(vocab_path),
                    "TRAIN_JSONL": str(dataset_path),
                    "VALIDATION_JSONL": str(dataset_path),
                    "EVALUATION_JSONL": str(dataset_path),
                    "OUTPUT_DIR": str(shell_output),
                    "POSITIONS_PER_GAME": "0",
                    "PROBE_EPOCHS": "1",
                    "PATIENCE": "1",
                    "BATCH_SIZE": "4",
                }
            )
            shell_completed = subprocess.run(
                [
                    str(MODULE_DIR / "scripts" / "run_probe_evaluation.sh"),
                    "standard",
                    "--candidate-count",
                    "1",
                    "--min-suffix-moves",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(
                shell_completed.returncode, 0, shell_completed.stderr
            )
            self.assertTrue((shell_output / "probe_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
