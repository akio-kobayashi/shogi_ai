import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


class NewPromptSchemaTest(unittest.TestCase):
    def test_prompt_and_move_annotation_validation(self):
        from new_prompt import validate_move_annotations, validate_state_prompt_tokens

        prompt = [
            "<STATE>", "<BOARD>", "<W_K>", "<SQ_5a>", "<B_K>", "<SQ_5i>",
            "<B_P>", "<SQ_7g>", "</BOARD>", "<HANDS>", "</HANDS>", "<TURN_BLACK>",
        ]
        validate_state_prompt_tokens(prompt)
        validate_move_annotations(
            ["7g7f", "P*5e"],
            [
                {"eligible": True, "piece": "<B_P>", "source": "<SQ_7g>"},
                {"eligible": False},
            ],
        )

    def test_duplicate_square_is_rejected(self):
        from new_prompt import validate_state_prompt_tokens

        prompt = [
            "<STATE>", "<BOARD>", "<W_K>", "<SQ_5a>", "<B_K>", "<SQ_5i>",
            "<B_P>", "<SQ_5i>", "</BOARD>", "<HANDS>", "</HANDS>", "<TURN_BLACK>",
        ]
        with self.assertRaises(ValueError):
            validate_state_prompt_tokens(prompt)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NewPromptDatasetTest(unittest.TestCase):
    def test_partial_action_labels_hints_and_moves(self):
        from new_prompt import new_prompt_vocabulary_tokens
        from new_prompt_data import NewPromptSequenceDataset, collate_new_prompt_sequences

        record = {
            "game_id": "g1",
            "state_prompt_tokens": [
                "<STATE>", "<BOARD>", "<W_K>", "<SQ_5a>", "<B_K>", "<SQ_5i>",
                "<B_P>", "<SQ_7g>", "</BOARD>", "<HANDS>", "</HANDS>", "<TURN_BLACK>",
            ],
            "start_ply": 0,
            "move_tokens": ["7g7f", "P*5e"],
            "move_annotations": [
                {"eligible": True, "piece": "<B_P>", "source": "<SQ_7g>"},
                {"eligible": False},
            ],
        }
        vocab = {token: index for index, token in enumerate(new_prompt_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = NewPromptSequenceDataset(
                str(path), vocab, annotation_mode="partial_action", annotation_probability=1.0,
                randomize_each_epoch=False,
            )
            example = dataset[0]
            batch = collate_new_prompt_sequences([example], vocab["<PAD>"], max_seq_len=64)
        self.assertEqual(int(batch["move_target_mask"].sum()), 2)
        self.assertEqual(int(batch["hint_target_mask"].sum()), 2)
        self.assertEqual(int(batch["labels"].ne(-100).sum()), 4)

    def test_start_candidate_is_selected_without_cshogi(self):
        from new_prompt import new_prompt_vocabulary_tokens
        from new_prompt_data import NewPromptSequenceDataset

        state = [
            "<STATE>", "<BOARD>", "<W_K>", "<SQ_5a>", "<B_K>", "<SQ_5i>",
            "<B_P>", "<SQ_7g>", "</BOARD>", "<HANDS>", "</HANDS>", "<TURN_BLACK>",
        ]
        record = {
            "game_id": "g2", "move_tokens": ["7g7f", "3c3d"],
            "move_annotations": [
                {"eligible": True, "piece": "<B_P>", "source": "<SQ_7g>"},
                {"eligible": True, "piece": "<W_P>", "source": "<SQ_3c>"},
            ],
            "start_candidates": [{"start_ply": 1, "state_prompt_tokens": state, "start_sfen": "x"}],
        }
        vocab = {token: index for index, token in enumerate(new_prompt_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            example = NewPromptSequenceDataset(str(path), vocab, max_moves=1)[0]
        self.assertEqual(int(example["start_ply"]), 1)
        self.assertEqual(example["move_tokens"], ["7g7f", "3c3d"])

    def test_streaming_dataset_keeps_offsets_and_training_omits_metadata(self):
        """学習時に全JSON recordや棋譜metadataを常駐・転送しない。"""
        from new_prompt import new_prompt_vocabulary_tokens
        from new_prompt_data import NewPromptSequenceDataset, collate_new_prompt_sequences

        state = [
            "<STATE>", "<BOARD>", "<W_K>", "<SQ_5a>", "<B_K>", "<SQ_5i>",
            "<B_P>", "<SQ_7g>", "</BOARD>", "<HANDS>", "</HANDS>", "<TURN_BLACK>",
        ]
        record = {
            "game_id": "g3", "state_prompt_tokens": state, "start_ply": 0,
            "move_tokens": ["7g7f"],
            "move_annotations": [{"eligible": True, "piece": "<B_P>", "source": "<SQ_7g>"}],
        }
        vocab = {token: index for index, token in enumerate(new_prompt_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for _ in range(2)), encoding="utf-8")
            dataset = NewPromptSequenceDataset(str(path), vocab, return_metadata=False)
            self.assertEqual(len(dataset), 2)
            self.assertFalse(hasattr(dataset, "records"))
            example = dataset[1]
            self.assertNotIn("move_tokens", example)
            batch = collate_new_prompt_sequences([example], vocab["<PAD>"], max_seq_len=64)
        self.assertNotIn("move_tokens", batch)
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 16))

    def test_training_loss_reports_move_and_hint_targets_separately(self):
        """注釈tokenがcombined lossに混ざるだけで消えないことを検査する。"""
        from types import SimpleNamespace
        import torch
        from train_new_prompt import loss_for_batch

        class ConstantModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, recurrent_mask=None):
                del attention_mask, recurrent_mask
                return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 7), device=input_ids.device))

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.bool),
            "recurrent_mask": torch.zeros((1, 4), dtype=torch.bool),
            "labels": torch.tensor([[-100, 2, 3, -100]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
            "move_target_mask": torch.tensor([[False, True, False, False]]),
            "hint_target_mask": torch.tensor([[False, False, True, False]]),
        }
        loss, metrics = loss_for_batch(ConstantModel(), batch, torch.device("cpu"))
        expected = torch.log(torch.tensor(7.0))
        self.assertTrue(torch.allclose(loss.detach(), expected))
        self.assertEqual(int(metrics["move"]["targets"]), 1)
        self.assertEqual(int(metrics["hint"]["targets"]), 1)
        self.assertTrue(torch.allclose(metrics["move"]["nll_sum"], expected))
        self.assertTrue(torch.allclose(metrics["hint"]["nll_sum"], expected))


if __name__ == "__main__":
    unittest.main()
