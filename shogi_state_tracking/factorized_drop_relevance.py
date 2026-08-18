"""駒打ちを中心とした行動条件付き持ち駒解析の共通データ処理。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from factorized_prompt import PIECE_TOKENS, annotation_piece_token, factorize_history_move


PIECE_LETTERS = "PLNSGBR"


def base_tokens(record: Mapping[str, object], state_prompt_mode: str) -> list[str]:
    candidates = [
        value for value in record.get("start_candidates", [])
        if int(value.get("start_ply", -1)) == 0
    ]
    if len(candidates) != 1:
        raise ValueError("evaluation record must have exactly one ply-0 start candidate")
    state = (
        [] if state_prompt_mode == "implicit_initial"
        else [str(value) for value in candidates[0]["state_prompt_tokens"]]
    )
    return ["<BOS>", *state, "<MOVES>"]


def hand_vector(step: Mapping[str, object], hand_names: Sequence[str]) -> list[int]:
    hands = dict(step["probe_targets"]["hands"])
    return [int(hands.get(name, 0)) for name in hand_names]


def side_index(step: Mapping[str, object]) -> int:
    return 0 if step["probe_targets"]["turn"] == "<TURN_BLACK>" else 1


def piece_index_from_drop(move: str) -> int:
    if len(move) != 4 or move[1] != "*" or move[0] not in PIECE_LETTERS:
        raise ValueError("not a supported USI drop: {}".format(move))
    return PIECE_LETTERS.index(move[0])


def legal_drop_destinations(step: Mapping[str, object], piece_index: int) -> tuple[str, ...]:
    cached = step.get("legal_drop_destinations_by_piece")
    if cached is not None:
        return tuple(cached.get(str(piece_index), ()))
    prefix = PIECE_LETTERS[piece_index] + "*"
    return tuple(
        str(move)[2:4] for move in step.get("legal_moves", [])
        if str(move).startswith(prefix)
    )


def legal_drop_count(step: Mapping[str, object], piece_index: int) -> int:
    cached = step.get("legal_drop_counts")
    if cached is not None:
        return int(cached[piece_index])
    return len(legal_drop_destinations(step, piece_index))


def bucket(value: int, boundaries: Sequence[int]) -> int:
    return sum(int(value) >= int(boundary) for boundary in boundaries)


def stable_number(*values: object) -> int:
    digest = hashlib.sha256(":".join(str(value) for value in values).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def read_positions(
    path: str,
    state_prompt_mode: str,
    annotation_mode: str,
    hand_names: Sequence[str],
    max_seq_len: int,
    selected_keys: set[tuple[str, int]] | None = None,
    materialize: bool = False,
) -> tuple[list[dict], dict]:
    """全plyを読み，prefixと持ち駒増減イベント位置を付けて返す。

    materialize=Falseの第1走査では対応付け用の軽量metadataだけを返す。
    materialize=Trueの第2走査ではselected_keysだけにprefixとattention位置を付ける。
    これにより全plyのprefix複製による二次的なメモリ増加を避ける。
    """
    positions: list[dict] = []
    counters = defaultdict(int)
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            moves = [str(value) for value in record.get("move_tokens", [])]
            annotations = [dict(value) for value in record.get("move_annotations", [])]
            steps = list(record.get("evaluation_steps", []))
            if len(moves) != len(annotations) or len(moves) != len(steps):
                raise ValueError("{}:{} moves, annotations and steps do not align".format(path, line_number))
            prefix = base_tokens(record, state_prompt_mode)
            event_markers: dict[tuple[int, int], list[int]] = defaultdict(list)
            last_event_ply: dict[tuple[int, int], int] = {}
            all_move_markers: list[int] = []
            previous_hands = None
            for ply, (move, annotation, step) in enumerate(zip(moves, annotations, steps)):
                if int(step.get("ply", ply)) != ply:
                    raise ValueError("{}:{} evaluation step ply is not aligned".format(path, line_number))
                if str(step.get("target_move", move)) != move:
                    raise ValueError("{}:{} target_move is not aligned with move_tokens".format(path, line_number))
                hands = hand_vector(step, hand_names)
                if previous_hands is not None:
                    for slot, (before, after) in enumerate(zip(previous_hands, hands)):
                        if before != after:
                            # 直前の指手の終了token位置に，その状態更新を対応させる。
                            event_markers[(slot // 7, slot % 7)].append(len(prefix) - 1)
                            last_event_ply[(slot // 7, slot % 7)] = ply - 1
                previous_hands = hands
                game_id = str(record.get("game_id", "{}:{}".format(path, line_number)))
                selected = selected_keys is None or (game_id, ply) in selected_keys
                if len(prefix) <= max_seq_len and (not materialize or selected):
                    side = side_index(step)
                    item = {
                        "game_id": game_id,
                        "ply": ply,
                        "prefix_length": len(prefix),
                        "query_position": len(prefix) - 1,
                        "move": move,
                        "is_drop": "*" in move,
                        "side": side,
                        "hands": hands,
                        "in_check": int(bool(step["probe_targets"].get("in_check", False))),
                        "legal_drop_counts": tuple(
                            len(legal_drop_destinations(step, piece)) for piece in range(7)
                        ),
                        "event_age_by_slot": tuple(
                                None if (color, piece) not in last_event_ply
                                else ply - last_event_ply[(color, piece)]
                            for color in range(2) for piece in range(7)
                        ),
                    }
                    if materialize:
                        normal_branches = []
                        for raw_piece, raw_sources in dict(step.get("legal_sources_by_piece", {})).items():
                            piece_token = annotation_piece_token(str(raw_piece))
                            if piece_token not in PIECE_TOKENS:
                                continue
                            for source in raw_sources:
                                source_token = str(source)
                                if source_token.startswith("<SQ_") and source_token.endswith(">"):
                                    normal_branches.append({"piece": piece_token, "source": source_token})
                        normal_branches.sort(key=lambda value: (value["source"], value["piece"]))
                        item.update({
                            "prefix_tokens": list(prefix),
                            "normal_branches": normal_branches,
                            "event_markers": {
                                "{}:{}".format(key[0], key[1]): list(values)
                                for key, values in event_markers.items()
                            },
                            "all_move_markers": list(all_move_markers),
                        })
                    positions.append(item)
                    counters["included_positions"] += 1
                    counters["included_drops"] += int("*" in move)
                elif len(prefix) > max_seq_len:
                    counters["context_excluded_positions"] += 1
                    counters["context_excluded_drops"] += int("*" in move)
                encoded = factorize_history_move(move, annotation, annotation_mode)
                prefix.extend(encoded)
                all_move_markers.append(len(prefix) - 1)
            counters["games"] += 1
    return positions, dict(counters)


def choose_normal_branch(
    item: Mapping[str, object], seed: int, avoid_piece_token: str | None = None,
) -> dict | None:
    """同一prefix比較に用いる合法な通常移動の先頭情報を決定する。"""
    branches = [dict(value) for value in item.get("normal_branches", [])]
    if not branches:
        return None
    preferred = [value for value in branches if value["piece"] != avoid_piece_token]
    if preferred:
        branches = preferred
    move = str(item.get("move", ""))
    if "*" not in move and len(move) >= 4:
        source = "<SQ_{}>".format(move[:2])
        for branch in branches:
            if branch["source"] == source:
                return branch
    return min(
        branches,
        key=lambda value: stable_number(
            seed, item.get("game_id"), item.get("ply"), value["piece"], value["source"]
        ),
    )


def choose_irrelevant_hand_slot(item: Mapping[str, object], relevant_slot: int, seed: int) -> int | None:
    """同じ側が保有する別駒から，枚数の近い非零slotを対照に選ぶ。"""
    side = int(relevant_slot) // 7
    relevant_count = int(item["hands"][relevant_slot])
    candidates = [
        side * 7 + piece for piece in range(7)
        if side * 7 + piece != relevant_slot and int(item["hands"][side * 7 + piece]) > 0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda slot: (
        abs(int(item["hands"][slot]) - relevant_count),
        stable_number(seed, item.get("game_id"), item.get("ply"), slot),
    ))


def _control_score(anchor: Mapping[str, object], candidate: Mapping[str, object], piece: int) -> tuple[int, ...]:
    slot = int(anchor["side"]) * 7 + piece
    anchor_age = anchor.get("event_age_by_slot", (None,) * 14)[slot]
    candidate_age = candidate.get("event_age_by_slot", (None,) * 14)[slot]
    anchor_age = 10**6 if anchor_age is None else int(anchor_age)
    candidate_age = 10**6 if candidate_age is None else int(candidate_age)
    anchor_legal = legal_drop_count(anchor, piece)
    candidate_legal = legal_drop_count(candidate, piece)
    return (
        abs(bucket(int(anchor["ply"]), (16, 32, 64, 96)) - bucket(int(candidate["ply"]), (16, 32, 64, 96))),
        abs(bucket(anchor_age, (2, 5, 17, 33)) - bucket(candidate_age, (2, 5, 17, 33))),
        abs(bucket(anchor_legal, (2, 4, 8, 16)) - bucket(candidate_legal, (2, 4, 8, 16))),
        abs(int(anchor["ply"]) - int(candidate["ply"])),
        stable_number(anchor["game_id"], anchor["ply"], candidate["game_id"], candidate["ply"]),
    )


def select_anchors_and_controls(positions: Sequence[dict], max_drops: int, seed: int) -> tuple[list[dict], dict]:
    """実駒打ちと，保有枚数・合法性を揃えた通常移動を1対1対応させる。"""
    drops = [item for item in positions if item["is_drop"] and int(item["ply"]) > 0]
    drops.sort(key=lambda item: stable_number(seed, item["game_id"], item["ply"]))
    if max_drops > 0:
        drops = drops[:max_drops]
    by_signature: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for item in positions:
        if item["is_drop"] or int(item["ply"]) == 0:
            continue
        side = int(item["side"])
        for piece in range(7):
            count = int(item["hands"][side * 7 + piece])
            if count > 0 and legal_drop_count(item, piece) > 0:
                by_signature[(side, piece, count, int(item["in_check"]))].append(item)
    pairs = []
    unmatched = 0
    used_controls: set[tuple[str, int]] = set()
    for anchor in drops:
        piece = piece_index_from_drop(str(anchor["move"]))
        side = int(anchor["side"])
        count = int(anchor["hands"][side * 7 + piece])
        candidates = [
            item for item in by_signature.get((side, piece, count, int(anchor["in_check"])), [])
            if item["game_id"] != anchor["game_id"]
            and (str(item["game_id"]), int(item["ply"])) not in used_controls
        ]
        if not candidates:
            unmatched += 1
            continue
        control = min(candidates, key=lambda item: _control_score(anchor, item, piece))
        used_controls.add((str(control["game_id"]), int(control["ply"])))
        pairs.append({"anchor": anchor, "control": control, "piece": piece})
    return pairs, {
        "candidate_drops": len(drops),
        "matched_pairs": len(pairs),
        "unmatched_drops": unmatched,
    }


def trajectory_samples(positions: Sequence[dict], pairs: Sequence[dict], window: int) -> list[dict]:
    by_game = defaultdict(dict)
    for item in positions:
        by_game[item["game_id"]][int(item["ply"])] = item
    samples = []
    for pair_index, pair in enumerate(pairs):
        piece = int(pair["piece"])
        for group in ("drop", "control"):
            center = pair["anchor"] if group == "drop" else pair["control"]
            game = by_game[center["game_id"]]
            for offset in range(-window, window + 1):
                item = game.get(int(center["ply"]) + offset)
                # ply=0は全例で同一状態となるため，主集計には含めない。
                if item is None or int(item["ply"]) == 0:
                    continue
                side = int(item["side"])
                # 相対時刻で手番が反転するため，anchorで打つ側の持ち駒を追う。
                tracked_side = int(center["side"])
                slot = tracked_side * 7 + piece
                samples.append({
                    **item,
                    "pair_index": pair_index,
                    "group": group,
                    "offset": offset,
                    "piece": piece,
                    "tracked_side": tracked_side,
                    "slot": slot,
                    "target_count": int(item["hands"][slot]),
                    "side_to_move_matches_anchor": side == tracked_side,
                })
    return samples


def selected_keys_for_pairs(pairs: Sequence[dict], window: int = 0) -> set[tuple[str, int]]:
    keys = set()
    for pair in pairs:
        for center in (pair["anchor"], pair["control"]):
            for offset in range(-window, window + 1):
                ply = int(center["ply"]) + offset
                if ply >= 0:
                    keys.add((str(center["game_id"]), ply))
    return keys


def matching_balance(pairs: Sequence[dict]) -> dict:
    if not pairs:
        return {"pairs": 0}
    ply_differences = []
    age_differences = []
    legal_differences = []
    piece_counts = defaultdict(int)
    for pair in pairs:
        anchor, control, piece = pair["anchor"], pair["control"], int(pair["piece"])
        slot = int(anchor["side"]) * 7 + piece
        anchor_age = anchor["event_age_by_slot"][slot]
        control_age = control["event_age_by_slot"][slot]
        ply_differences.append(abs(int(anchor["ply"]) - int(control["ply"])))
        if anchor_age is not None and control_age is not None:
            age_differences.append(abs(int(anchor_age) - int(control_age)))
        legal_differences.append(abs(legal_drop_count(anchor, piece) - legal_drop_count(control, piece)))
        piece_counts[PIECE_LETTERS[piece]] += 1
    def mean(values): return None if not values else sum(values) / len(values)
    return {
        "pairs": len(pairs),
        "piece_distribution": dict(sorted(piece_counts.items())),
        "mean_absolute_ply_difference": mean(ply_differences),
        "mean_absolute_last_event_age_difference": mean(age_differences),
        "age_comparable_pairs": len(age_differences),
        "mean_absolute_legal_destination_count_difference": mean(legal_differences),
        "exact_side_piece_count_and_check_match": True,
        "controls_used_without_replacement": True,
    }


def rebind_pairs(pairs: Sequence[dict], materialized: Sequence[dict]) -> list[dict]:
    lookup = {(str(item["game_id"]), int(item["ply"])): item for item in materialized}
    result = []
    for pair in pairs:
        anchor_key = (str(pair["anchor"]["game_id"]), int(pair["anchor"]["ply"]))
        control_key = (str(pair["control"]["game_id"]), int(pair["control"]["ply"]))
        if anchor_key in lookup and control_key in lookup:
            result.append({"anchor": lookup[anchor_key], "control": lookup[control_key], "piece": pair["piece"]})
    return result


def relevant_and_control_markers(item: Mapping[str, object], side: int, piece: int, seed: int) -> tuple[list[int], list[int]]:
    relevant = list(item["event_markers"].get("{}:{}".format(side, piece), []))
    excluded = set(relevant)
    candidates = [value for value in item["all_move_markers"] if value not in excluded]
    candidates.sort(key=lambda value: (
        min((abs(value - other) for other in relevant), default=10**6),
        stable_number(seed, item["game_id"], item["ply"], value),
    ))
    return relevant, candidates[: len(relevant)]
