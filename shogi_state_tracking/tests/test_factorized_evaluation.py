import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FactorizedEvaluationTest(unittest.TestCase):
    @staticmethod
    def record():
        state = [
            "<STATE>", "<BOARD>", "<W_K>", "<SQ_5a>", "<B_K>", "<SQ_5i>",
            "<B_P>", "<SQ_7g>", "</BOARD>", "<HANDS>", "</HANDS>", "<TURN_BLACK>",
        ]
        return {
            "game_id": "g",
            "initial_sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "move_tokens": ["7g7f"],
            "move_annotations": [{"eligible": True, "piece": "<B_P>", "source": "<SQ_7g>"}],
            "start_candidates": [{"start_ply": 0, "state_prompt_tokens": state, "position_scope": "unseen_position"}],
            "evaluation_steps": [{"ply": 0, "target_move": "7g7f", "legal_moves": ["7g7f"], "legal_sources_by_piece": {"<B_P>": ["<SQ_7g>"]}}],
            "position_scope_by_ply": ["unseen_position"], "trajectory_scope": "strict_unseen_position",
        }

    def test_explicit_and_implicit_initial_use_the_same_ply_zero_game(self):
        from factorized_prompt import factorized_vocabulary_tokens
        from factorized_prompt_data import FactorizedPromptSequenceDataset

        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(self.record()) + "\n", encoding="utf-8")
            explicit = FactorizedPromptSequenceDataset(
                str(path), vocab, state_prompt_mode="explicit", start_selection="fixed_initial",
                randomize_each_epoch=False,
            )[0]
            implicit = FactorizedPromptSequenceDataset(
                str(path), vocab, state_prompt_mode="implicit_initial", start_selection="fixed_initial",
                randomize_each_epoch=False,
            )[0]
        self.assertEqual(explicit["start_ply"], 0)
        self.assertEqual(implicit["start_ply"], 0)
        self.assertGreater(len(explicit["input_ids"]), len(implicit["input_ids"]))
        self.assertEqual(int(explicit["move_boundary_mask"].sum()), int(implicit["move_boundary_mask"].sum()))

    def test_standard_initial_sfen_requires_all_nine_pawns(self):
        from factorized_prompt_data import is_standard_initial_sfen

        self.assertTrue(is_standard_initial_sfen(self.record()["initial_sfen"]))
        self.assertFalse(is_standard_initial_sfen(
            "lnsgkgsnl/1r5b1/p1ppppppp/9/9/9/P1PPPPPPP/1B5R1/LNSGKGSNL b - 1"
        ))

    def test_ablation_builder_keeps_only_standard_initial_candidate(self):
        from build_initial_position_ablation_dataset import build_split

        standard = self.record()
        standard["start_candidates"].append(dict(standard["start_candidates"][0], start_ply=1))
        nonstandard = self.record()
        nonstandard["game_id"] = "handicap"
        nonstandard["initial_sfen"] = "9/9/9/9/9/9/9/9/9 b - 1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "train.jsonl", root / "out.jsonl"
            source.write_text(json.dumps(standard) + "\n" + json.dumps(nonstandard) + "\n", encoding="utf-8")
            metrics = build_split(source, destination, "train")
            output = json.loads(destination.read_text(encoding="utf-8").strip())
        self.assertEqual(metrics["input_records"], 2)
        self.assertEqual(metrics["records"], 1)
        self.assertEqual([value["start_ply"] for value in output["start_candidates"]], [0])

    def test_dataset_builder_validates_and_packages_jsonl(self):
        from build_factorized_prompt_dataset import copy_split

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.jsonl", root / "evaluation.jsonl"
            source.write_text(json.dumps(self.record()) + "\n", encoding="utf-8")
            metrics = copy_split(source, destination, "evaluation")
            self.assertEqual(metrics["records"], 1)
            self.assertEqual(metrics["moves"], 1)
            self.assertTrue(destination.is_file())

    def test_tiny_llama_and_vanilla_training_loss(self):
        from factorized_prompt import factorized_vocabulary_tokens
        from factorized_prompt_data import FactorizedPromptSequenceDataset
        from models import ModelConfig, build_model
        from new_prompt_data import collate_new_prompt_sequences
        from train_new_prompt import loss_for_batch

        record = self.record()
        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            example = FactorizedPromptSequenceDataset(
                str(path), vocab, annotation_mode="rap", annotation_probability=1.0,
                randomize_each_epoch=False,
            )[0]
            batch = collate_new_prompt_sequences([example], vocab["<PAD>"], 64)
        for model_type in ("vanilla", "llama"):
            config = ModelConfig(vocab_size=len(vocab), max_seq_len=64, d_model=16, n_layers=1, n_heads=4, d_ff=32, dropout=0.0)
            model = build_model(model_type, config)
            loss, metrics = loss_for_batch(model, batch, torch.device("cpu"))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertAlmostEqual(float(batch["move_unit_weight"].sum()), 3.0)
            self.assertEqual(int(batch["move_boundary_mask"].sum()), 1)
            self.assertEqual(int(metrics["hint"]["targets"]), 1)

    def test_tiny_llama_and_vanilla_evaluation(self):
        from evaluate_factorized_moves import main
        from factorized_prompt import factorized_vocabulary_tokens, write_factorized_vocabulary
        from models import ModelConfig, build_model

        record = self.record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab_path = root / "vocab.json"
            write_factorized_vocabulary(vocab_path)
            vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
            evaluation = root / "evaluation.jsonl"
            evaluation.write_text(json.dumps(record) + "\n", encoding="utf-8")
            for model_type in ("vanilla", "llama"):
                config = ModelConfig(vocab_size=len(vocab), max_seq_len=64, d_model=16, n_layers=1, n_heads=4, d_ff=32, dropout=0.0)
                model = build_model(model_type, config)
                checkpoint = root / (model_type + ".pt")
                torch.save({
                    "model_type": model_type, "config": config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "new_prompt": {"move_encoding": "factorized_v2"},
                }, checkpoint)
                output = root / (model_type + ".json")
                argv = [
                    "evaluate_factorized_moves.py", "--checkpoint", str(checkpoint),
                    "--evaluation-jsonl", str(evaluation), "--vocab", str(vocab_path),
                    "--output", str(output), "--history-distances", "0",
                    "--primary-history-distances", "0", "--max-queries", "1",
                    "--batch-size", "1", "--device", "cpu", "--progress-every", "0",
                ]
                with patch.object(sys, "argv", argv):
                    main()
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(payload["model_type"], model_type)
                self.assertEqual(payload["metrics"]["primary"]["queries"], 1)
                self.assertEqual(payload["metrics"]["primary"]["greedy_syntactic_rate"], 1.0)

    def test_move_query_reader_is_bounded_by_batch_size(self):
        from evaluate_factorized_moves import iter_query_batches
        from factorized_prompt import factorized_vocabulary_tokens

        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            path.write_text(
                "".join(json.dumps(dict(self.record(), game_id="g{}".format(index))) + "\n" for index in range(7)),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                evaluation_jsonl=str(path), history_distances=(0,), max_games=7,
                max_queries=5, candidates_per_game=1, start_selection="fixed_initial",
                state_prompt_mode="explicit", batch_size=2,
            )
            statistics = {"games": 0, "queries": 0}
            batches = list(iter_query_batches(args, vocab, 64, statistics))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(statistics["queries"], 5)
        self.assertLessEqual(max(map(len, batches)), args.batch_size)

    def test_batched_beam_matches_single_query_beam(self):
        from evaluate_factorized_moves import beam_batch_cached, beam_single_cached
        from factorized_prompt import factorized_vocabulary_tokens
        from models import ModelConfig, build_model

        vocabulary = {
            token: index for index, token in enumerate(factorized_vocabulary_tokens())
        }
        prefixes = [
            [vocabulary["<BOS>"], vocabulary["<MOVES>"]],
            [vocabulary["<BOS>"], vocabulary["<MOVES>"]],
        ]
        for model_type in ("vanilla", "llama"):
            torch.manual_seed(7)
            config = ModelConfig(
                vocab_size=len(vocabulary), max_seq_len=16, d_model=16,
                n_layers=1, n_heads=4, d_ff=32, dropout=0.0,
            )
            model = build_model(model_type, config).eval()
            expected = [
                beam_single_cached(model, prefix, vocabulary, torch.device("cpu"), 5)
                for prefix in prefixes
            ]
            actual = beam_batch_cached(
                model, prefixes, vocabulary, torch.device("cpu"),
                beam_size=5, micro_batch_size=2,
            )
            self.assertEqual(
                [[move for move, _ in values] for values in actual],
                [[move for move, _ in values] for values in expected],
            )
            for actual_query, expected_query in zip(actual, expected):
                self.assertEqual(len(actual_query), len(expected_query))
                for (_, actual_score), (_, expected_score) in zip(actual_query, expected_query):
                    self.assertAlmostEqual(actual_score, expected_score, places=5)

    def test_best_checkpoint_can_omit_optimizer_state(self):
        from factorized_prompt import factorized_vocabulary_tokens
        from models import ModelConfig, build_model
        from train_new_prompt import save_checkpoint

        config = ModelConfig(
            vocab_size=len(factorized_vocabulary_tokens()), max_seq_len=16,
            d_model=8, n_layers=1, n_heads=2, d_ff=16, dropout=0.0,
        )
        model = build_model("vanilla", config)
        optimizer = torch.optim.AdamW(model.parameters())
        args = SimpleNamespace(
            model_type="vanilla", model_size="small", move_encoding="factorized_v2",
            state_prompt_mode="explicit", start_selection="fixed_initial",
            annotation_mode="vanilla", annotation_probability=0.0,
            hint_loss_weight=1.0, max_hints=0, max_moves=1, seed=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, optimizer, args, 1, 1, 1.0, include_optimizer=False)
            payload = torch.load(path, map_location="cpu")
        self.assertNotIn("optimizer_state_dict", payload)


if __name__ == "__main__":
    unittest.main()
