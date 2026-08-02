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


if __name__ == "__main__":
    unittest.main()
