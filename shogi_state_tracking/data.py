"""JSONLを本実験のcausal decoder入力へ変換する。

対局者名、レート、game_resultはデータ監査専用であり、モデル入力へ入れない。
"""

import json
import multiprocessing
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from preprocess import (
    candidate_start_plies,
    choose_start_ply,
    materialize_segment,
)


IGNORE_INDEX = -100


def load_vocabulary(path: str) -> Dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(token): int(index) for token, index in payload["token_to_id"].items()}


class ShogiSequenceDataset(Dataset):
    def __init__(self, jsonl_path: str, token_to_id: Mapping[str, int]):
        self.token_to_id = dict(token_to_id)
        self.records = []
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                state_tokens = record.get("initial_state_tokens", [])
                if len(state_tokens) != 96:
                    raise ValueError(
                        "{}:{} initial_state_tokens must have length 96".format(
                            jsonl_path, line_number
                        )
                    )
                self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def _token_id(self, token: str) -> int:
        try:
            return self.token_to_id[token]
        except KeyError as exc:
            raise KeyError("token is absent from vocab: {}".format(token)) from exc

    def __getitem__(self, index: int):
        record = self.records[index]
        return self._encode_record(record)

    def _encode_record(self, record: Mapping[str, object]):
        tokens = (
            ["<BOS>"]
            + list(record["initial_state_tokens"])
            + ["<MOVES>"]
            + list(record["move_tokens"])
            + ["<EOS>"]
        )
        moves_position = 1 + 96
        input_ids = torch.tensor(
            [self._token_id(token) for token in tokens], dtype=torch.long
        )
        recurrent_mask = torch.zeros(len(tokens), dtype=torch.bool)
        # <MOVES>自体はprompt。最初のmove tokenから再帰を有効にする。
        recurrent_mask[moves_position + 1 :] = True

        labels = torch.full((len(tokens),), IGNORE_INDEX, dtype=torch.long)
        # <MOVES>位置の出力で第1手を予測し、最後は<EOS>を予測する。
        labels[moves_position:-1] = input_ids[moves_position + 1 :]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "recurrent_mask": recurrent_mask,
            "game_id": record["game_id"],
            "engine_scope": record["engine_scope"],
            "start_ply": int(record.get("start_ply", 0)),
            "start_sfen": str(record.get("start_sfen", record.get("initial_sfen", ""))),
        }


class RandomStartSequenceDataset(ShogiSequenceDataset):
    """各epochで対局ごとの開始局面を選び直す学習用Dataset。

    `set_epoch()`の値はmultiprocessing.Valueでworker間共有する。DistributedSamplerの
    `set_epoch()`と同じepochを設定すれば、再現可能でworker数にも依存しない。
    """

    def __init__(
        self,
        jsonl_path: str,
        token_to_id: Mapping[str, int],
        candidate_count: int = 40,
        min_suffix_moves: int = 40,
        seed: int = 20260724,
        samples_per_game: int = 1,
        randomize_each_epoch: bool = True,
    ):
        super().__init__(jsonl_path, token_to_id)
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if samples_per_game <= 0:
            raise ValueError("samples_per_game must be positive")
        self.candidate_count = candidate_count
        self.min_suffix_moves = min_suffix_moves
        self.seed = seed
        self.samples_per_game = samples_per_game
        self.randomize_each_epoch = randomize_each_epoch
        self._epoch = multiprocessing.Value("q", 0)

        for record in self.records:
            candidates = candidate_start_plies(
                len(record["move_tokens"]),
                candidate_count=self.candidate_count,
                min_suffix_moves=self.min_suffix_moves,
            )
            if not candidates:
                raise ValueError(
                    "game {} has no valid random start".format(record["game_id"])
                )
            record["_start_candidates"] = candidates

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_game

    def __getitem__(self, index: int):
        record_index, replica = divmod(index, self.samples_per_game)
        record = self.records[record_index]
        epoch = self._epoch.value if self.randomize_each_epoch else 0
        start_ply = choose_start_ply(
            str(record["game_id"]),
            record["_start_candidates"],
            seed=self.seed,
            epoch=epoch,
            replica=replica,
        )
        segment = materialize_segment(record, start_ply)
        return self._encode_record(segment)


def collate_sequences(
    examples: Sequence[Mapping[str, object]],
    pad_token_id: int,
    max_seq_len: int,
):
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
    attention_mask = torch.zeros((batch_size, batch_length), dtype=torch.bool)
    recurrent_mask = torch.zeros((batch_size, batch_length), dtype=torch.bool)
    for row, example in enumerate(examples):
        length = int(example["input_ids"].shape[0])
        input_ids[row, :length] = example["input_ids"]
        labels[row, :length] = example["labels"]
        attention_mask[row, :length] = True
        recurrent_mask[row, :length] = example["recurrent_mask"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "recurrent_mask": recurrent_mask,
        "game_ids": [str(example["game_id"]) for example in examples],
        "engine_scopes": [str(example["engine_scope"]) for example in examples],
        "start_plies": torch.tensor(
            [int(example["start_ply"]) for example in examples], dtype=torch.long
        ),
        # probeの正解局面再生用。モデル入力にはしない。
        "start_sfens": [str(example["start_sfen"]) for example in examples],
    }


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence shapes do not match")
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    """token別重み付きcross entropy。prompt/paddingはweight 0とする。"""
    if logits.shape[:2] != labels.shape or labels.shape != loss_weights.shape:
        raise ValueError("logits, labels, and loss_weights shapes do not match")
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(labels)
    active = labels != IGNORE_INDEX
    weights = torch.where(active, loss_weights.to(per_token.dtype), 0.0)
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError("weighted loss has no active targets")
    return (per_token * weights).sum() / denominator
