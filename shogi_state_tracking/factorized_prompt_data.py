"""factorized_v3用のストリーミングDataset．"""

from __future__ import annotations

from array import array
from typing import Mapping

import torch

from data import IGNORE_INDEX
from factorized_prompt import annotation_piece_token, factorize_usi, validate_state_prompt_tokens
from new_prompt import source_square_from_usi, square_token
from new_prompt_data import NewPromptSequenceDataset


STANDARD_INITIAL_SFEN_POSITION = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -"
)


def is_standard_initial_sfen(value: str) -> bool:
    """手数fieldを除いたSFENが平手初期局面かを判定する。"""
    return " ".join(str(value).split()[:3]) == STANDARD_INITIAL_SFEN_POSITION


class FactorizedPromptSequenceDataset(NewPromptSequenceDataset):
    """旧JSONL artifactを再利用し，指手だけを実行時に座標分解する。

    RAPは非駒打ち指手の直前へ移動駒種を1 tokenだけ挿入する。開始升は本体の
    第1 tokenに含まれるため，ヒントとして重複させない。
    """

    def __init__(
        self,
        *args,
        state_prompt_mode: str = "implicit_initial",
        start_selection: str = "fixed_initial",
        eos_loss_weight: float = 1.0,
        **kwargs,
    ):
        if state_prompt_mode not in {"implicit_initial", "explicit"} or start_selection != "fixed_initial":
            raise ValueError("factorized_v3 requires fixed initial start")
        self.state_prompt_mode = state_prompt_mode
        self.start_selection = start_selection
        if eos_loss_weight < 0.0:
            raise ValueError("eos_loss_weight must be nonnegative")
        self.eos_loss_weight = float(eos_loss_weight)
        super().__init__(*args, **kwargs)
        self.length_estimates = None
        length_path = self.jsonl_path.with_suffix(".lengths.u32")
        if length_path.is_file():
            values = array("I")
            with length_path.open("rb") as handle:
                values.fromfile(handle, length_path.stat().st_size // values.itemsize)
            if len(values) != len(self):
                raise ValueError("length index and JSONL record counts differ")
            self.length_estimates = values

    def _materialize_candidate(self, record: Mapping[str, object], index: int):
        if not is_standard_initial_sfen(str(record.get("initial_sfen", ""))):
            raise ValueError("fixed_initial requires a standard initial SFEN")
        candidates = [
            candidate for candidate in record.get("_prepared_candidates", ())
            if int(candidate.get("start_ply", -1)) == 0
        ]
        if len(candidates) != 1:
            raise ValueError("fixed_initial requires exactly one start_ply=0 candidate")
        return record, candidates[0]

    def _make_control_pool(self):
        raise ValueError("random_control was retired from factorized_v3")

    def _hint_pair(self, record: Mapping[str, object], move_index: int, index: int):
        original = record["_annotation_ids"][move_index]
        if original is None:
            raise ValueError("move {} has no eligible annotation".format(move_index))
        return original

    def _prepare_record(self, record: Mapping[str, object]):
        prepared = dict(record)
        prepared["_state_prompt_ids"] = tuple(record["state_prompt_token_ids"]) if "state_prompt_token_ids" in record else tuple(
            self._token_id(str(token)) for token in record.get("state_prompt_tokens", [])
        ) if record.get("state_prompt_tokens") is not None else None
        prepared["_move_ids"] = tuple(tuple(int(token) for token in move) for move in record["factorized_move_ids"]) if "factorized_move_ids" in record else tuple(
            tuple(self._token_id(token) for token in factorize_usi(str(move)))
            for move in record["move_tokens"]
        )
        annotation_ids = []
        eligible_indices = []
        annotation_colors = []
        for move_index, annotation in enumerate(record["move_annotations"]):
            if bool(annotation.get("eligible", False)):
                piece_id = self._token_id(annotation_piece_token(str(annotation["piece"])))
                source_id = self._token_id(str(annotation["source"]))
                annotation_ids.append((piece_id, source_id))
                eligible_indices.append(move_index)
                annotation_colors.append(None)
            else:
                annotation_ids.append(None)
                annotation_colors.append(None)
        prepared["_annotation_ids"] = tuple(annotation_ids)
        prepared["_eligible_indices"] = tuple(eligible_indices)
        prepared["_annotation_colors"] = tuple(annotation_colors)
        prepared["_prepared_candidates"] = tuple(
            dict(candidate, _state_prompt_ids=(
                tuple(int(token) for token in candidate["state_prompt_token_ids"])
                if "state_prompt_token_ids" in candidate else
                tuple(self._token_id(str(token)) for token in candidate["state_prompt_tokens"])
            ))
            for candidate in record.get("start_candidates", [])
        )
        return prepared

    def _validate_record(self, record: Mapping[str, object]) -> None:
        if not str(record.get("game_id", "")):
            raise ValueError("record has no game_id")
        if record.get("terminal_token") != "<EOS>" or int(record.get("game_result", 0)) == 0:
            raise ValueError("factorized_v3 requires a decisive complete-game terminal label")
        state = record.get("state_prompt_tokens")
        candidates = record.get("start_candidates")
        if state is None and candidates is None:
            raise ValueError("record has neither state_prompt_tokens nor start_candidates")
        if state is not None:
            validate_state_prompt_tokens([str(token) for token in state])
            for token in state:
                self._token_id(str(token))
        moves = record.get("move_tokens")
        factorized_ids = record.get("factorized_move_ids")
        annotations = record.get("move_annotations")
        if not isinstance(annotations, list) or (not isinstance(moves, list) and not isinstance(factorized_ids, list)):
            raise ValueError("record needs move_tokens and move_annotations lists")
        if moves is not None:
            if len(moves) != len(annotations):
                raise ValueError("move_tokens and move_annotations lengths differ")
            for move, annotation in zip(moves, annotations):
                move = str(move)
                eligible = bool(annotation.get("eligible", False))
                if "*" in move:
                    if eligible:
                        raise ValueError("drop move must not have a RAP annotation")
                elif not eligible or str(annotation.get("source", "")) != square_token(source_square_from_usi(move)):
                    raise ValueError("normal move needs a matching RAP annotation")
            for move in moves:
                for token in factorize_usi(str(move)):
                    self._token_id(token)
        elif len(factorized_ids) != len(annotations):
            raise ValueError("factorized_move_ids and move_annotations lengths differ")
        for annotation in annotations:
            if bool(annotation.get("eligible", False)):
                self._token_id(annotation_piece_token(str(annotation["piece"])))
        if candidates is not None:
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("start_candidates must be a nonempty list")
            for candidate in candidates:
                start_ply = int(candidate.get("start_ply", -1))
                move_count = len(moves) if moves is not None else len(factorized_ids)
                if not 0 <= start_ply < move_count:
                    raise ValueError("start candidate ply is outside move_tokens")
                if "state_prompt_token_ids" not in candidate:
                    validate_state_prompt_tokens([str(token) for token in candidate["state_prompt_tokens"]])
                    for token in candidate["state_prompt_tokens"]:
                        self._token_id(str(token))

    def _end_ply(self, record: Mapping[str, object], state_length: int, start_ply: int) -> int:
        available = len(record["_move_ids"]) - start_ply
        requested = min(available, self.max_moves if self.max_moves is not None else available)
        if self.max_seq_len is None:
            return start_ply + requested
        budget = self.max_seq_len - (3 + state_length) - (self.max_hints or 0)
        used = 0
        count = 0
        for ids in record["_move_ids"][start_ply : start_ply + requested]:
            if used + len(ids) > budget:
                break
            used += len(ids)
            count += 1
        if count <= 0:
            raise ValueError("state prompt and hint budget leave no room for a move")
        return start_ply + count

    def _encode_record(self, record: Mapping[str, object], index: int):
        record, candidate = self._materialize_candidate(record, index)
        state_ids = candidate["_state_prompt_ids"] if candidate is not None else record["_state_prompt_ids"]
        if self.state_prompt_mode == "implicit_initial":
            state_ids = ()
        start_ply = int(candidate.get("start_ply", 0) if candidate is not None else record.get("start_ply", 0))
        end_ply = self._end_ply(record, len(state_ids), start_ply)
        complete_game = start_ply == 0 and end_ply == len(record["_move_ids"])
        selected = self._selected_hint_indices(record, start_ply, end_ply, index)
        token_ids = [self._bos_id, *state_ids, self._moves_id]
        categories = ["prompt"] * len(token_ids)
        move_weights = [0.0] * len(token_ids)
        move_end = [False] * len(token_ids)
        for move_index in range(start_ply, end_ply):
            if move_index in selected:
                token_ids.append(self._hint_pair(record, move_index, index)[0])
                categories.append("hint")
                move_weights.append(0.0)
                move_end.append(False)
            ids = record["_move_ids"][move_index]
            token_ids.extend(ids)
            categories.extend(["move"] * len(ids))
            # 指手内の全subtoken NLLを合計し，後段で移動先座標数（指手数）で割る．
            move_weights.extend([1.0] * len(ids))
            move_end.extend([False] * (len(ids) - 1) + [True])
        if complete_game:
            token_ids.append(self._eos_id)
            categories.append("eos")
            move_weights.append(0.0)
            move_end.append(False)

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = torch.full((len(token_ids),), IGNORE_INDEX, dtype=torch.long)
        loss_weights = torch.zeros(len(token_ids), dtype=torch.float32)
        move_target_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        hint_target_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        eos_target_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        move_unit_weight = torch.zeros(len(token_ids), dtype=torch.float32)
        move_boundary_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        for position in range(len(token_ids) - 1):
            target = categories[position + 1]
            if target == "move":
                labels[position] = input_ids[position + 1]
                loss_weights[position] = move_weights[position + 1]
                move_unit_weight[position] = move_weights[position + 1]
                move_target_mask[position] = True
                move_boundary_mask[position] = move_end[position + 1]
            elif target == "hint":
                labels[position] = input_ids[position + 1]
                loss_weights[position] = self.hint_loss_weight
                hint_target_mask[position] = True
            elif target == "eos":
                labels[position] = input_ids[position + 1]
                loss_weights[position] = self.eos_loss_weight
                eos_target_mask[position] = True
        recurrent_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        recurrent_mask[2 + len(state_ids) :] = True
        example = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "move_unit_weight": move_unit_weight,
            "move_boundary_mask": move_boundary_mask,
            "move_target_mask": move_target_mask,
            "hint_target_mask": hint_target_mask,
            "eos_target_mask": eos_target_mask,
            "recurrent_mask": recurrent_mask,
            "complete_game": complete_game,
        }
        if self.return_metadata:
            example.update({
                "game_id": str(record["game_id"]),
                "player_scope": str(record.get("player_scope", record.get("engine_scope", ""))),
                "engine_scope": str(record.get("engine_scope", record.get("player_scope", ""))),
                "position_scope": str(record.get("position_scope", "unknown_position_scope")),
                "trajectory_scope": str(record.get("trajectory_scope", "unknown_position_scope")),
                "start_ply": start_ply,
                "start_sfen": str(candidate.get("start_sfen", "") if candidate is not None else record.get("start_sfen", "")),
                "move_tokens": [str(move) for move in record["move_tokens"]],
                "move_annotations": [dict(value) for value in record["move_annotations"]],
            })
        return example
