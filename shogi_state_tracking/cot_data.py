"""自己生成した将棋読み筋をCoT-like SFT系列へ変換する。"""

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from data import IGNORE_INDEX


TRACE_SCHEMA_VERSION = 1


class ReasoningTraceDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        token_to_id: Mapping[str, int],
        answer_weight: float = 1.0,
    ):
        if answer_weight <= 0:
            raise ValueError("answer_weight must be positive")
        self.token_to_id = dict(token_to_id)
        self.answer_weight = float(answer_weight)
        self.records = []
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record.get("schema_version", 0)) != TRACE_SCHEMA_VERSION:
                    raise ValueError(
                        "{}:{} unsupported trace schema".format(
                            jsonl_path, line_number
                        )
                    )
                if len(record.get("initial_state_tokens", [])) != 96:
                    raise ValueError(
                        "{}:{} initial_state_tokens must have length 96".format(
                            jsonl_path, line_number
                        )
                    )
                if not record.get("reasoning_lines"):
                    raise ValueError(
                        "{}:{} reasoning_lines must not be empty".format(
                            jsonl_path, line_number
                        )
                    )
                self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def _id(self, token: str) -> int:
        try:
            return self.token_to_id[token]
        except KeyError as exc:
            raise KeyError("token is absent from vocab: {}".format(token)) from exc

    def __getitem__(self, index: int):
        record = self.records[index]
        prompt = (
            ["<BOS>"]
            + list(record["initial_state_tokens"])
            + ["<MOVES>"]
            + list(record["history_moves"])
        )
        trace = ["<THINK>"]
        for line_index, reasoning_line in enumerate(record["reasoning_lines"]):
            if line_index:
                trace.append("<SEP>")
            trace.extend(str(move) for move in reasoning_line)
        trace.extend(["</THINK>", "<ANSWER>", str(record["target_move"]), "<EOS>"])
        tokens = prompt + trace
        input_ids = torch.tensor([self._id(token) for token in tokens], dtype=torch.long)

        labels = torch.full((len(tokens),), IGNORE_INDEX, dtype=torch.long)
        # prompt末尾から<THINK>を予測し、以後trace、answer、EOSを学習する。
        labels[len(prompt) - 1 : -1] = input_ids[len(prompt) :]
        loss_weights = torch.zeros(len(tokens), dtype=torch.float32)
        loss_weights[len(prompt) - 1 : -1] = 1.0
        answer_token_position = len(tokens) - 2
        # target_moveを予測する位置は<ANSWER>の位置。
        loss_weights[answer_token_position - 1] = self.answer_weight

        recurrent_mask = torch.zeros(len(tokens), dtype=torch.bool)
        moves_marker = 1 + 96
        recurrent_mask[moves_marker + 1 :] = True
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "recurrent_mask": recurrent_mask,
            "game_id": str(record["game_id"]),
            "player_scope": str(
                record.get("player_scope", record.get("engine_scope", ""))
            ),
            "engine_scope": str(record.get("engine_scope", "")),
            "position_scope": str(
                record.get("position_scope", "unknown_position_scope")
            ),
            "trajectory_scope": str(
                record.get("trajectory_scope", "unknown_position_scope")
            ),
        }


def collate_reasoning_traces(
    examples: Sequence[Mapping[str, object]],
    pad_token_id: int,
    max_seq_len: int,
) -> Dict[str, object]:
    if not examples:
        raise ValueError("empty batch")
    batch_length = max(int(example["input_ids"].shape[0]) for example in examples)
    if batch_length > max_seq_len:
        raise ValueError(
            "batch sequence length {} exceeds max_seq_len {}".format(
                batch_length, max_seq_len
            )
        )
    batch_size = len(examples)
    input_ids = torch.full(
        (batch_size, batch_length), pad_token_id, dtype=torch.long
    )
    labels = torch.full(
        (batch_size, batch_length), IGNORE_INDEX, dtype=torch.long
    )
    loss_weights = torch.zeros((batch_size, batch_length), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, batch_length), dtype=torch.bool)
    recurrent_mask = torch.zeros((batch_size, batch_length), dtype=torch.bool)
    for row, example in enumerate(examples):
        length = int(example["input_ids"].shape[0])
        input_ids[row, :length] = example["input_ids"]
        labels[row, :length] = example["labels"]
        loss_weights[row, :length] = example["loss_weights"]
        attention_mask[row, :length] = True
        recurrent_mask[row, :length] = example["recurrent_mask"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_weights": loss_weights,
        "attention_mask": attention_mask,
        "recurrent_mask": recurrent_mask,
        "game_ids": [str(example["game_id"]) for example in examples],
        "player_scopes": [str(example["player_scope"]) for example in examples],
        "engine_scopes": [str(example["engine_scope"]) for example in examples],
        "position_scopes": [
            str(example["position_scope"]) for example in examples
        ],
        "trajectory_scopes": [
            str(example["trajectory_scope"]) for example in examples
        ],
    }
