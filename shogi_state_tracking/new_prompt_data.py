"""新しい駒・座標prompt用のPyTorch Dataset。

入力artifactはデータセット計算機でmaterialize済みであり，このモジュールはcshogiを
要求しない。部分的行動教師の挿入だけを学習時に決定論的に行う。
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data import IGNORE_INDEX
from new_prompt import (
    move_token,
    validate_move_annotations,
    validate_state_prompt_tokens,
)


ANNOTATION_MODES = ("vanilla", "partial_action", "random_control")


def _annotation_color(annotation: Mapping[str, object]) -> str:
    piece = str(annotation.get("piece", ""))
    if piece.startswith("<B_"):
        return "B"
    if piece.startswith("<W_"):
        return "W"
    raise ValueError("annotation has no colored piece token")


def _seeded_rng(*parts: object) -> random.Random:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big"))


class NewPromptSequenceDataset(Dataset):
    """materialize済み新schema segmentをcausal decoder入力へ変換する。"""

    def __init__(
        self,
        jsonl_path: str,
        token_to_id: Mapping[str, int],
        annotation_mode: str = "vanilla",
        annotation_probability: float = 0.0,
        hint_loss_weight: float = 1.0,
        max_hints: int | None = None,
        max_moves: int | None = None,
        max_seq_len: int | None = None,
        seed: int = 20260802,
        randomize_each_epoch: bool = True,
    ):
        if annotation_mode not in ANNOTATION_MODES:
            raise ValueError("unknown annotation_mode: {}".format(annotation_mode))
        if not 0.0 <= annotation_probability <= 1.0:
            raise ValueError("annotation_probability must be in [0, 1]")
        if hint_loss_weight < 0.0:
            raise ValueError("hint_loss_weight must be nonnegative")
        if max_hints is not None and max_hints < 0:
            raise ValueError("max_hints must be nonnegative")
        if max_moves is not None and max_moves <= 0:
            raise ValueError("max_moves must be positive")
        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if annotation_mode == "vanilla" and annotation_probability != 0.0:
            raise ValueError("vanilla mode requires annotation_probability=0")

        self.token_to_id = dict(token_to_id)
        self._bos_id = self._token_id("<BOS>")
        self._moves_id = self._token_id("<MOVES>")
        self._eos_id = self._token_id("<EOS>")
        self.annotation_mode = annotation_mode
        self.annotation_probability = float(annotation_probability)
        self.hint_loss_weight = float(hint_loss_weight)
        self.max_hints = max_hints
        self.max_moves = max_moves
        self.max_seq_len = max_seq_len
        self.seed = int(seed)
        self.randomize_each_epoch = bool(randomize_each_epoch)
        self.records: List[Dict[str, object]] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    self._validate_record(record)
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError("{}:{} {}".format(jsonl_path, line_number, exc)) from exc
                self.records.append(self._prepare_record(record))
        if not self.records:
            raise ValueError("new prompt dataset is empty: {}".format(jsonl_path))
        self._epoch = multiprocessing.Value("q", 0)
        self._control_pool = self._make_control_pool()

    def _prepare_record(self, record: Mapping[str, object]) -> Dict[str, object]:
        """文字列artifactを一度だけ整数IDへ変換する。

        学習中の ``__getitem__`` では文字列変換・語彙検索を行わない。
        開始候補の局面も同じタイミングで前処理しておく。
        """
        prepared = dict(record)
        prepared["_state_prompt_ids"] = tuple(
            self._token_id(str(token)) for token in record.get("state_prompt_tokens", [])
        ) if record.get("state_prompt_tokens") is not None else None
        prepared["_move_ids"] = tuple(
            self._token_id(move_token(str(move))) for move in record["move_tokens"]
        )
        annotation_ids = []
        eligible_indices = []
        annotation_colors = []
        for move_index, annotation in enumerate(record["move_annotations"]):
            if bool(annotation.get("eligible", False)):
                piece = str(annotation["piece"])
                source = str(annotation["source"])
                annotation_ids.append((self._token_id(piece), self._token_id(source)))
                eligible_indices.append(move_index)
                annotation_colors.append(_annotation_color(annotation))
            else:
                annotation_ids.append(None)
                annotation_colors.append(None)
        prepared["_annotation_ids"] = tuple(annotation_ids)
        prepared["_eligible_indices"] = tuple(eligible_indices)
        prepared["_annotation_colors"] = tuple(annotation_colors)
        prepared["_prepared_candidates"] = tuple(
            dict(candidate, _state_prompt_ids=tuple(
                self._token_id(str(token)) for token in candidate["state_prompt_tokens"]
            ))
            for candidate in record.get("start_candidates", [])
        )
        return prepared

    def _validate_record(self, record: Mapping[str, object]) -> None:
        if not str(record.get("game_id", "")):
            raise ValueError("record has no game_id")
        state = record.get("state_prompt_tokens")
        candidates = record.get("start_candidates")
        if state is None and candidates is None:
            raise ValueError("record has neither state_prompt_tokens nor start_candidates")
        if state is not None:
            if not isinstance(state, list):
                raise ValueError("state_prompt_tokens must be a list")
            validate_state_prompt_tokens([str(token) for token in state])
        moves = record.get("move_tokens")
        annotations = record.get("move_annotations")
        if not isinstance(moves, list) or not isinstance(annotations, list):
            raise ValueError("record needs move_tokens and move_annotations lists")
        validate_move_annotations(
            [str(move) for move in moves],
            [dict(annotation) for annotation in annotations],
        )
        if state is not None:
            for token in [str(token) for token in state]:
                self._token_id(token)
        for move in [str(move) for move in moves]:
            self._token_id(move_token(move))
        for annotation in annotations:
            if bool(annotation.get("eligible", False)):
                self._token_id(str(annotation["piece"]))
                self._token_id(str(annotation["source"]))
        if candidates is not None:
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("start_candidates must be a nonempty list")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError("start candidate must be an object")
                start_ply = int(candidate.get("start_ply", -1))
                if not 0 <= start_ply < len(moves):
                    raise ValueError("start candidate ply is outside move_tokens")
                state = candidate.get("state_prompt_tokens")
                if not isinstance(state, list):
                    raise ValueError("start candidate has no state_prompt_tokens")
                validate_state_prompt_tokens([str(token) for token in state])
                for token in state:
                    self._token_id(str(token))

    def _make_control_pool(self) -> Dict[str, List[tuple[int, int]]]:
        pools: Dict[str, List[tuple[int, int]]] = {"B": [], "W": []}
        for record in self.records:
            for annotation, pair, color in zip(
                record["move_annotations"], record["_annotation_ids"], record["_annotation_colors"]
            ):
                if not bool(annotation.get("eligible", False)) or pair is None:
                    continue
                pools[color].append(pair)
        if not pools["B"] or not pools["W"]:
            raise ValueError("random control requires eligible annotations for both sides")
        return pools

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _token_id(self, token: str) -> int:
        try:
            return self.token_to_id[token]
        except KeyError as exc:
            raise KeyError("token is absent from vocab: {}".format(token)) from exc

    def _selected_hint_indices(
        self, record: Mapping[str, object], start_ply: int, end_ply: int, index: int
    ) -> set[int]:
        if self.annotation_mode == "vanilla" or self.annotation_probability == 0.0:
            return set()
        epoch = self._epoch.value if self.randomize_each_epoch else 0
        rng = _seeded_rng(self.seed, epoch, index, record["game_id"], "hint")
        eligible = [move_index for move_index in record["_eligible_indices"] if start_ply <= move_index < end_ply]
        selected = [
            move_index
            for move_index in eligible
            if rng.random() < self.annotation_probability
        ]
        if self.max_hints is not None and len(selected) > self.max_hints:
            selected = rng.sample(selected, self.max_hints)
        return set(selected)

    def _hint_pair(
        self,
        record: Mapping[str, object],
        move_index: int,
        index: int,
    ) -> tuple[int, int]:
        original = record["_annotation_ids"][move_index]
        if original is None:
            raise ValueError("move {} has no eligible annotation".format(move_index))
        if self.annotation_mode != "random_control":
            return original
        epoch = self._epoch.value if self.randomize_each_epoch else 0
        rng = _seeded_rng(self.seed, epoch, index, record["game_id"], move_index, "control")
        pool = self._control_pool[record["_annotation_colors"][move_index]]
        # 現在局面との対応だけを壊す対照なので，同一の駒種・開始位置対を
        # そのまま戻さない。色は保つため，先後や系列長は交絡させない。
        if len(pool) <= 1:
            return pool[0]
        selected_index = rng.randrange(len(pool))
        replacement = pool[selected_index]
        if replacement == original:
            replacement = pool[(selected_index + 1) % len(pool)]
        return replacement

    def _encode_record(self, record: Mapping[str, object], index: int) -> Dict[str, object]:
        record, candidate = self._materialize_candidate(record, index)
        state_ids = candidate["_state_prompt_ids"] if candidate is not None else record["_state_prompt_ids"]
        start_ply = int(candidate.get("start_ply", 0) if candidate is not None else record.get("start_ply", 0))
        end_ply = self._end_ply(record, len(state_ids), start_ply)
        selected = self._selected_hint_indices(record, start_ply, end_ply, index)
        token_ids = [self._bos_id, *state_ids, self._moves_id]
        categories = ["prompt"] * len(token_ids)
        for move_index in range(start_ply, end_ply):
            if move_index in selected:
                token_ids.extend(self._hint_pair(record, move_index, index))
                categories.extend(("hint", "hint"))
            token_ids.append(record["_move_ids"][move_index])
            categories.append("move")
        token_ids.append(self._eos_id)
        categories.append("eos")

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = torch.full((len(token_ids),), IGNORE_INDEX, dtype=torch.long)
        loss_weights = torch.zeros(len(token_ids), dtype=torch.float32)
        move_target_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        hint_target_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        # position iのlogitsでtokens[i+1]を予測する。
        for position in range(len(token_ids) - 1):
            target_category = categories[position + 1]
            if target_category == "move":
                labels[position] = input_ids[position + 1]
                loss_weights[position] = 1.0
                move_target_mask[position] = True
            elif target_category == "hint":
                labels[position] = input_ids[position + 1]
                loss_weights[position] = self.hint_loss_weight
                hint_target_mask[position] = True

        moves_position = 1 + len(state_ids)
        recurrent_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        recurrent_mask[moves_position + 1 :] = True
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "move_target_mask": move_target_mask,
            "hint_target_mask": hint_target_mask,
            "recurrent_mask": recurrent_mask,
            "game_id": str(record["game_id"]),
            "player_scope": str(record.get("player_scope", record.get("engine_scope", ""))),
            "engine_scope": str(record.get("engine_scope", record.get("player_scope", ""))),
            "position_scope": str(record.get("position_scope", "unknown_position_scope")),
            "trajectory_scope": str(record.get("trajectory_scope", "unknown_position_scope")),
            "start_ply": start_ply,
            "start_sfen": str(candidate.get("start_sfen", "") if candidate is not None else record.get("start_sfen", "")),
            "move_tokens": [str(move) for move in record["move_tokens"]],
            "move_annotations": [dict(value) for value in record["move_annotations"]],
        }

    def _end_ply(self, record: Mapping[str, object], state_length: int, start_ply: int) -> int:
        """全条件で同数の実指手を保ちつつ，最悪時のhint数で長さを予約する。"""
        available = len(record["_move_ids"]) - start_ply
        requested = self.max_moves if self.max_moves is not None else available
        if self.max_seq_len is not None:
            # BOS + state + MOVES + EOS に，最大K個の2 token hintを確保する。
            overhead = 3 + state_length
            reserved_hints = 2 * (self.max_hints or 0)
            requested = min(requested, self.max_seq_len - overhead - reserved_hints)
        if requested <= 0:
            raise ValueError("state prompt and hint budget leave no room for a move")
        return start_ply + min(available, requested)

    def _materialize_candidate(self, record: Mapping[str, object], index: int):
        """候補開始局面から一つを決め，学習側でcshogiを使わずsuffixを選ぶ。"""
        candidates = record.get("_prepared_candidates")
        if not candidates:
            return record, None
        epoch = self._epoch.value if self.randomize_each_epoch else 0
        rng = _seeded_rng(self.seed, epoch, index, record["game_id"], "start")
        return record, candidates[rng.randrange(len(candidates))]

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self._encode_record(self.records[index], index)


def collate_new_prompt_sequences(
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
    input_ids = pad_sequence(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = pad_sequence(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    recurrent_mask = pad_sequence(
        [example["recurrent_mask"] for example in examples],
        batch_first=True,
        padding_value=False,
    )
    loss_weights = pad_sequence(
        [example["loss_weights"] for example in examples],
        batch_first=True,
        padding_value=0.0,
    )
    move_target_mask = pad_sequence(
        [example["move_target_mask"] for example in examples],
        batch_first=True,
        padding_value=False,
    )
    hint_target_mask = pad_sequence(
        [example["hint_target_mask"] for example in examples],
        batch_first=True,
        padding_value=False,
    )
    lengths = torch.tensor(
        [int(example["input_ids"].shape[0]) for example in examples], dtype=torch.long
    )
    attention_mask = torch.arange(batch_length)[None, :] < lengths[:, None]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "recurrent_mask": recurrent_mask,
        "loss_weights": loss_weights,
        "move_target_mask": move_target_mask,
        "hint_target_mask": hint_target_mask,
        "game_ids": [str(example["game_id"]) for example in examples],
        "player_scopes": [str(example["player_scope"]) for example in examples],
        "engine_scopes": [str(example["engine_scope"]) for example in examples],
        "position_scopes": [str(example["position_scope"]) for example in examples],
        "trajectory_scopes": [str(example["trajectory_scope"]) for example in examples],
        "start_plies": torch.tensor([int(example["start_ply"]) for example in examples], dtype=torch.long),
        "start_sfens": [str(example["start_sfen"]) for example in examples],
        "move_tokens": [list(example["move_tokens"]) for example in examples],
        "move_annotations": [list(example["move_annotations"]) for example in examples],
    }
