import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

try:
    import torch
    import cshogi
except ImportError:
    torch = None
    cshogi = None


@unittest.skipIf(torch is None or cshogi is None, "torch/cshogi is not installed")
class CoTPipelineTest(unittest.TestCase):
    def setUp(self):
        import create_dataset
        from models import ModelConfig, VanillaTransformer

        self.create_dataset = create_dataset
        self.moves = ["7g7f", "3c3d", "2g2f"]
        vocabulary_tokens = create_dataset.base_vocabulary() + self.moves
        self.vocabulary = {
            token: index for index, token in enumerate(vocabulary_tokens)
        }
        self.config = ModelConfig(
            vocab_size=len(self.vocabulary),
            max_seq_len=128,
            d_model=8,
            n_layers=1,
            n_heads=1,
            d_ff=16,
            dropout=0.0,
            tie_embeddings=False,
        )
        self.model = VanillaTransformer(self.config)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.token_embedding.weight.fill_(1.0)
            for name, parameter in self.model.named_parameters():
                if parameter.ndim == 1 and name.endswith("weight"):
                    parameter.fill_(1.0)
            self.model.lm_head.weight[
                self.vocabulary["7g7f"]
            ].fill_(1.0)

    def write_inputs(self, temp_dir: Path):
        board = cshogi.Board()
        vocab_path = temp_dir / "vocab.json"
        vocab_path.write_text(
            json.dumps({"token_to_id": self.vocabulary}), encoding="utf-8"
        )
        game = {
            "game_id": "cot-game",
            "engine_scope": "closed",
            "initial_sfen": board.sfen(),
            "initial_state_tokens": self.create_dataset.encode_initial_state(
                board, cshogi
            ),
            "move_tokens": self.moves[:2],
        }
        game_path = temp_dir / "games.jsonl"
        game_path.write_text(json.dumps(game) + "\n", encoding="utf-8")
        checkpoint = temp_dir / "base.pt"
        torch.save(
            {
                "model_type": "vanilla",
                "config": self.config.to_dict(),
                "model_state_dict": self.model.state_dict(),
            },
            checkpoint,
        )
        return vocab_path, game_path, checkpoint

    def test_reasoning_dataset_masks_prompt_and_weights_answer(self):
        from cot_data import ReasoningTraceDataset

        board = cshogi.Board()
        trace = {
            "schema_version": 1,
            "game_id": "trace",
            "engine_scope": "closed",
            "start_sfen": board.sfen(),
            "initial_state_tokens": self.create_dataset.encode_initial_state(
                board, cshogi
            ),
            "history_moves": [],
            "target_move": "7g7f",
            "reasoning_lines": [["7g7f", "3c3d"], ["2g2f"]],
        }
        with tempfile.TemporaryDirectory() as temp_text:
            path = Path(temp_text) / "trace.jsonl"
            path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
            dataset = ReasoningTraceDataset(
                str(path), self.vocabulary, answer_weight=3.0
            )
            example = dataset[0]
        active = (example["labels"] != -100).nonzero().flatten()
        self.assertEqual(int(active[0]), 97)
        target_position = int(
            (example["input_ids"] == self.vocabulary["<ANSWER>"])
            .nonzero()
            .flatten()[0]
        )
        self.assertEqual(float(example["loss_weights"][target_position]), 3.0)

    def test_generation_sft_and_reasoning_evaluation_cli(self):
        with tempfile.TemporaryDirectory() as temp_text:
            temp_dir = Path(temp_text)
            vocab_path, game_path, checkpoint = self.write_inputs(temp_dir)
            trace_path = temp_dir / "traces.jsonl"
            generate = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_DIR / "generate_reasoning_traces.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--vocab",
                    str(vocab_path),
                    "--input-jsonl",
                    str(game_path),
                    "--output-jsonl",
                    str(trace_path),
                    "--positions-per-game",
                    "1",
                    "--lines",
                    "2",
                    "--line-length",
                    "2",
                    "--temperature",
                    "0",
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(generate.returncode, 0, generate.stderr)
            self.assertTrue(trace_path.read_text(encoding="utf-8").strip())

            training_dir = temp_dir / "training"
            train = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_DIR / "train_model.py"),
                    "--stage",
                    "cot",
                    "--model-type",
                    "vanilla",
                    "--vocab",
                    str(vocab_path),
                    "--train-jsonl",
                    str(trace_path),
                    "--validation-jsonl",
                    str(trace_path),
                    "--output-dir",
                    str(training_dir),
                    "--init-checkpoint",
                    str(checkpoint),
                    "--epochs",
                    "1",
                    "--max-steps",
                    "1",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(train.returncode, 0, train.stderr)
            self.assertTrue((training_dir / "best.pt").exists())

            evaluation_dir = temp_dir / "evaluation"
            evaluate = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_DIR / "evaluate_reasoning.py"),
                    "--checkpoint",
                    str(training_dir / "best.pt"),
                    "--vocab",
                    str(vocab_path),
                    "--trace-jsonl",
                    str(trace_path),
                    "--output-dir",
                    str(evaluation_dir),
                    "--max-new-tokens",
                    "8",
                    "--max-examples",
                    "1",
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(evaluate.returncode, 0, evaluate.stderr)
            metrics = json.loads(
                (evaluation_dir / "reasoning_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metrics["examples"], 1)
            self.assertIn("trace_move_legal_rate", metrics)


if __name__ == "__main__":
    unittest.main()
