import json
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
class DataTest(unittest.TestCase):
    def test_only_move_targets_contribute_to_loss(self):
        from data import IGNORE_INDEX, ShogiSequenceDataset, collate_sequences

        state = ["SQ_EMPTY"] * 81 + ["HAND_0"] * 14 + ["TURN_BLACK"]
        all_tokens = (
            ["<PAD>", "<BOS>", "<MOVES>", "<EOS>", "SQ_EMPTY", "HAND_0", "TURN_BLACK"]
            + ["7g7f", "3c3d"]
        )
        vocab = {token: index for index, token in enumerate(all_tokens)}
        record = {
            "game_id": "g1",
            "engine_scope": "open",
            "initial_state_tokens": state,
            "move_tokens": ["7g7f", "3c3d"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = ShogiSequenceDataset(str(path), vocab)
            example = dataset[0]
            batch = collate_sequences([example], vocab["<PAD>"], max_seq_len=128)

        target_positions = (batch["labels"] != IGNORE_INDEX).nonzero()
        self.assertEqual(target_positions.shape[0], 3)  # 2 moves + EOS
        moves_marker = 1 + 96
        self.assertFalse(bool(batch["recurrent_mask"][0, moves_marker]))
        self.assertTrue(bool(batch["recurrent_mask"][0, moves_marker + 1]))

    @unittest.skipIf(cshogi is None, "cshogi is not installed")
    def test_random_start_batch_runs_through_both_models(self):
        import create_dataset
        from data import RandomStartSequenceDataset, collate_sequences
        from models import (
            ModelConfig,
            T2MLRConfig,
            T2MLRTransformer,
            VanillaTransformer,
        )

        board = cshogi.Board()
        record = {
            "game_id": "g2",
            "engine_scope": "closed",
            "initial_sfen": board.sfen(),
            "initial_state_tokens": create_dataset.encode_initial_state(board, cshogi),
            "move_tokens": ["7g7f", "3c3d"],
        }
        vocabulary_tokens = create_dataset.base_vocabulary() + ["7g7f", "3c3d"]
        vocab = {token: index for index, token in enumerate(vocabulary_tokens)}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = RandomStartSequenceDataset(
                str(path),
                vocab,
                candidate_count=2,
                min_suffix_moves=1,
            )
            dataset.set_epoch(3)
            batch = collate_sequences(
                [dataset[0]], vocab["<PAD>"], max_seq_len=128
            )

        common = dict(
            vocab_size=len(vocab),
            max_seq_len=128,
            d_model=16,
            n_layers=2,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
        )
        vanilla = VanillaTransformer(ModelConfig(**common))
        t2mlr = T2MLRTransformer(
            T2MLRConfig(**common, l_start=0, l_end=0, jacobi_depth=1)
        )
        vanilla_output = vanilla(
            batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        t2mlr_output = t2mlr(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            recurrent_mask=batch["recurrent_mask"],
        )
        self.assertEqual(vanilla_output.logits.shape, t2mlr_output.logits.shape)
        self.assertGreaterEqual(int(batch["start_plies"][0]), 0)


if __name__ == "__main__":
    unittest.main()
