#!/usr/bin/env python3
"""次手に関係する盤面情報の復号可能性と局所的介入効果を評価する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch

from create_dataset import import_cshogi
from data import load_vocabulary
from evaluate_new_prompt_probes import label_maps, target_from_mapping
from factorized_prompt import MOVE_ENCODING, TERMINAL_ENCODING, factorize_history_move
from models import ModelConfig, build_model
from models.layers import prepare_sdpa_mask
from probes import BOARD_CLASS_COUNT, LinearStateProbe
from train_model import amp_context, resolve_amp


ROLE_NAMES = (
    "actual_source",
    "actual_destination",
    "actual_move",
    "endpoint_attacker",
    "actual_local_context",
    "candidate_source",
    "candidate_destination",
    "candidate_related",
    "background",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="factorized_v3の次手関連マス復号・介入評価"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--linear-probes", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources", default="available")
    parser.add_argument("--history-distances", default="8,32")
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steering-sources", default="middle,late,penultimate")
    parser.add_argument("--steering-strengths", default="0.5,1.0,2.0")
    parser.add_argument("--max-steering-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    return parser.parse_args()


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not result or min(result) < 0:
        raise ValueError("history distances must be nonnegative integers")
    return result


def _csv_floats(value: str) -> tuple[float, ...]:
    result = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",") if item.strip()))
    if not result or min(result) <= 0:
        raise ValueError("steering strengths must be positive")
    return result


def _resolve_device(value: str) -> torch.device:
    return torch.device(value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def square_index(square: str) -> int:
    if len(square) != 2 or square[0] not in "123456789" or square[1] not in "abcdefghi":
        raise ValueError("invalid USI square: {}".format(square))
    return (int(square[0]) - 1) * 9 + "abcdefghi".index(square[1])


def move_squares(move: str) -> tuple[int | None, int]:
    move = str(move)
    if "*" in move:
        if len(move) != 4:
            raise ValueError("invalid USI drop: {}".format(move))
        return None, square_index(move[2:4])
    if len(move) not in (4, 5):
        raise ValueError("invalid USI move: {}".format(move))
    return square_index(move[:2]), square_index(move[2:4])


def recency_bucket(age: int | None) -> str:
    if age is None:
        return "never"
    if age == 1:
        return "1"
    if age <= 4:
        return "2-4"
    if age <= 16:
        return "5-16"
    return "17+"


def role_squares(
    actual_move: str,
    legal_moves: Sequence[str],
    endpoint_attackers: Sequence[int] = (),
) -> dict[str, tuple[int, ...]]:
    actual_source, actual_destination = move_squares(actual_move)
    candidate_sources: set[int] = set()
    candidate_destinations: set[int] = set()
    for move in legal_moves:
        source, destination = move_squares(str(move))
        if source is not None:
            candidate_sources.add(source)
        candidate_destinations.add(destination)
    actual_sources = set() if actual_source is None else {actual_source}
    actual_destinations = {actual_destination}
    actual = actual_sources | actual_destinations
    attackers = set(int(value) for value in endpoint_attackers) - actual
    local_context = actual | attackers
    candidate = candidate_sources | candidate_destinations
    background = set(range(81)) - candidate - attackers
    return {
        "actual_source": tuple(sorted(actual_sources)),
        "actual_destination": tuple(sorted(actual_destinations)),
        "actual_move": tuple(sorted(actual)),
        "endpoint_attacker": tuple(sorted(attackers)),
        "actual_local_context": tuple(sorted(local_context)),
        "candidate_source": tuple(sorted(candidate_sources)),
        "candidate_destination": tuple(sorted(candidate_destinations)),
        "candidate_related": tuple(sorted(candidate)),
        "background": tuple(sorted(background)),
    }


def _base_tokens(record: Mapping[str, object], state_prompt_mode: str) -> list[str]:
    candidates = [value for value in record.get("start_candidates", []) if int(value.get("start_ply", -1)) == 0]
    if len(candidates) != 1:
        raise ValueError("evaluation record must have exactly one ply-0 start candidate")
    state = [] if state_prompt_mode == "implicit_initial" else [str(value) for value in candidates[0]["state_prompt_tokens"]]
    return ["<BOS>", *state, "<MOVES>"]


def read_queries(path: str, state_prompt_mode: str, annotation_mode: str, distances: Sequence[int], limit: int, seed: int):
    rng = random.Random(seed)
    cshogi = import_cshogi()
    selected: list[dict] = []
    eligible = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            moves = [str(value) for value in record.get("move_tokens", [])]
            annotations = [dict(value) for value in record.get("move_annotations", [])]
            steps = list(record.get("evaluation_steps", []))
            if len(steps) != len(moves) or len(annotations) != len(moves):
                raise ValueError("{}:{} evaluation data do not align".format(path, line_number))
            prefix = _base_tokens(record, state_prompt_mode)
            board = cshogi.Board(str(record["initial_sfen"]))
            last_touched: list[int | None] = [None] * 81
            for ply, move in enumerate(moves):
                if ply in distances:
                    step = steps[ply]
                    legal_moves = [str(value) for value in step.get("legal_moves", [])]
                    if move not in legal_moves:
                        raise ValueError("{}:{} target move is absent from candidates at ply {}".format(path, line_number, ply))
                    actual_source, actual_destination = move_squares(move)
                    endpoints = [actual_destination]
                    if actual_source is not None:
                        endpoints.append(actual_source)
                    endpoint_attackers = {
                        int(attacker)
                        for endpoint in endpoints
                        for color in (cshogi.BLACK, cshogi.WHITE)
                        for attacker in board.attackers_to(color, endpoint)
                    }
                    ages = [None if value is None else ply - value + 1 for value in last_touched]
                    encoded_next = factorize_history_move(move, annotations[ply], annotation_mode)
                    if annotation_mode == "ap" and actual_source is not None:
                        # APではh_preの次が駒種注釈になる。条件間で測定対象を揃えるため、
                        # 正解駒種だけをteacher-forceし、その次の移動元座標を評価する。
                        steering_tokens = [*prefix, encoded_next[0]]
                        steering_target = encoded_next[1]
                    else:
                        steering_tokens = list(prefix)
                        steering_target = encoded_next[0]
                    item = {
                        "game_id": str(record.get("game_id", "")),
                        "ply": ply,
                        "prefix_tokens": list(prefix),
                        "target_move": move,
                        "steering_tokens": steering_tokens,
                        "steering_state_position": len(prefix) - 1,
                        "steering_target_token": steering_target,
                        "probe_targets": step["probe_targets"],
                        "roles": role_squares(move, legal_moves, endpoint_attackers),
                        "recency": [recency_bucket(value) for value in ages],
                    }
                    eligible += 1
                    if len(selected) < limit:
                        selected.append(item)
                    else:
                        index = rng.randrange(eligible)
                        if index < limit:
                            selected[index] = item
                source, destination = move_squares(move)
                if source is not None:
                    last_touched[source] = ply + 1
                last_touched[destination] = ply + 1
                prefix.extend(factorize_history_move(move, annotations[ply], annotation_mode))
                encoded = board.move_from_usi(move)
                if not board.is_legal(encoded):
                    raise ValueError("{}:{} invalid move at ply {}".format(path, line_number, ply))
                board.push(encoded)
    if not selected:
        raise ValueError("no policy-relevance queries were selected")
    return selected, eligible


def _resolve_sources(value: str, available: Sequence[str], n_layers: int) -> list[str]:
    if value == "available":
        return list(available)
    result = []
    for item in value.split(","):
        item = item.strip()
        aliases = {
            "middle": n_layers // 2,
            "late": (3 * n_layers) // 4,
            "penultimate": max(n_layers - 1, 0),
            "final": n_layers,
        }
        if item in aliases:
            item = "layer_{}".format(aliases[item])
        if item not in available:
            raise ValueError("probe source is unavailable: {}".format(item))
        if item not in result:
            result.append(item)
    return result


def _pad(batch, vocabulary, device, key="prefix_tokens"):
    lengths = torch.tensor([len(item[key]) for item in batch], dtype=torch.long)
    width = int(lengths.max())
    ids = torch.full((len(batch), width), vocabulary["<PAD>"], dtype=torch.long)
    for row, item in enumerate(batch):
        ids[row, : len(item[key])] = torch.tensor(
            [vocabulary[token] for token in item[key]], dtype=torch.long
        )
    mask = torch.arange(width)[None, :] < lengths[:, None]
    return ids.to(device), mask.to(device), lengths.to(device)


def _role_metrics(correct: torch.Tensor, targets: torch.Tensor, queries: Sequence[dict]) -> dict:
    result = {}
    background_by_stratum: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row, query in enumerate(queries):
        for square in query["roles"]["background"]:
            key = (square, int(targets[row, square]), query["recency"][square])
            background_by_stratum[key].append(float(correct[row, square]))

    for role in ROLE_NAMES:
        observations = []
        strata: dict[tuple[int, int, str], list[float]] = defaultdict(list)
        within_position = []
        for row, query in enumerate(queries):
            background = query["roles"]["background"]
            for square in query["roles"][role]:
                value = float(correct[row, square])
                observations.append(value)
                key = (square, int(targets[row, square]), query["recency"][square])
                strata[key].append(value)
                controls = [
                    other for other in background
                    if int(targets[row, other]) == int(targets[row, square])
                    and query["recency"][other] == query["recency"][square]
                ]
                if controls:
                    within_position.append((value, sum(float(correct[row, other]) for other in controls) / len(controls)))
        entry = {
            "observations": len(observations),
            "accuracy": None if not observations else sum(observations) / len(observations),
        }
        pairs = []
        for key, relevant_values in strata.items():
            controls = background_by_stratum.get(key, [])
            if controls:
                pairs.append((len(relevant_values), sum(relevant_values) / len(relevant_values), sum(controls) / len(controls)))
        weight = sum(item[0] for item in pairs)
        entry["coordinate_piece_recency_matched"] = {
            "matched_observations": weight,
            "matched_strata": len(pairs),
            "relevant_accuracy": None if not weight else sum(n * a for n, a, _ in pairs) / weight,
            "background_accuracy": None if not weight else sum(n * b for n, _, b in pairs) / weight,
            "difference": None if not weight else sum(n * (a - b) for n, a, b in pairs) / weight,
        }
        entry["within_position_piece_recency_matched"] = {
            "matched_observations": len(within_position),
            "relevant_accuracy": None if not within_position else sum(a for a, _ in within_position) / len(within_position),
            "background_accuracy": None if not within_position else sum(b for _, b in within_position) / len(within_position),
            "difference": None if not within_position else sum(a - b for a, b in within_position) / len(within_position),
        }
        result[role] = entry
    return result


def _load_probes(artifact, sources, d_model, device):
    result = {}
    for source in sources:
        probe = LinearStateProbe(d_model).to(device)
        probe.load_state_dict(artifact["probe_state_dicts"][source])
        probe.eval()
        result[source] = probe
    return result


def evaluate_decoding(model, probes, queries, vocabulary, board_map, hand_names, device, amp_dtype, batch_size):
    predictions = {source: [] for source in probes}
    targets_all = []
    model.eval()
    ordered = sorted(queries, key=lambda item: len(item["prefix_tokens"]))
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            ids, mask, lengths = _pad(batch, vocabulary, device)
            output = model(ids, attention_mask=mask, output_hidden_states=True)
            rows = torch.arange(len(batch), device=device)
            positions = lengths - 1
            targets = torch.cat([
                target_from_mapping(item["probe_targets"], board_map, hand_names).board
                for item in batch
            ])
            targets_all.append(targets)
            for source, probe in probes.items():
                layer = int(source.split("_", 1)[1])
                features = output.hidden_states[layer][rows, positions]
                predictions[source].append(probe(features).board.argmax(dim=-1).cpu())
    # batch境界をまたぐstratumも同じ比較へ入れるため、全例をまとめて集計する。
    targets = torch.cat(targets_all)
    result = {}
    for source, parts in predictions.items():
        correct = torch.cat(parts) == targets
        by_distance = {}
        for distance in sorted({int(item["ply"]) for item in ordered}):
            indices = [index for index, item in enumerate(ordered) if int(item["ply"]) == distance]
            by_distance[str(distance)] = _role_metrics(
                correct[indices], targets[indices], [ordered[index] for index in indices]
            )
        result[source] = {
            "all": _role_metrics(correct, targets, ordered),
            "by_history_distance": by_distance,
        }
    return result, ordered


def _resume_logits(model, hidden, layer_index: int, attention_mask, model_type: str):
    x = hidden
    mask = prepare_sdpa_mask(attention_mask)
    if model_type == "llama":
        rotary = model._rotary(x.device, x.dtype, 0, x.shape[1])
        for layer in model.layers[layer_index:]:
            x = layer(x, mask, rotary)
    elif model_type == "vanilla":
        for layer in model.layers[layer_index:]:
            x = layer(x, mask)
    else:
        raise ValueError("steering currently supports llama and vanilla models")
    return model.lm_head(model.final_norm(x))


def _stable_control(query, role_square: int, targets: torch.Tensor) -> int | None:
    candidates = [
        square for square in query["roles"]["background"]
        if int(targets[square]) == int(targets[role_square])
        and query["recency"][square] == query["recency"][role_square]
    ]
    if not candidates:
        return None
    digest = hashlib.sha256("{}:{}:{}".format(query["game_id"], query["ply"], role_square).encode()).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def _stable_role_square(query, role: str) -> int | None:
    candidates = list(query["roles"][role])
    if not candidates:
        return None
    digest = hashlib.sha256("{}:{}:{}".format(query["game_id"], query["ply"], role).encode()).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def _steering_delta(probe, features, targets, squares, strength):
    weights = probe.board_head.weight.view(81, BOARD_CLASS_COUNT, -1)
    biases = probe.board_head.bias.view(81, BOARD_CLASS_COUNT)
    result = torch.zeros_like(features)
    for row, square in enumerate(squares):
        if square is None:
            continue
        true_class = int(targets[row, square])
        logits = torch.mv(weights[square], features[row]) + biases[square]
        logits[true_class] = -torch.inf
        alternative = int(logits.argmax())
        direction = weights[square, alternative] - weights[square, true_class]
        result[row] = float(strength) * direction / direction.norm().clamp_min(1e-12)
    return result


def evaluate_steering(model, model_type, probes, queries, vocabulary, board_map, hand_names, strengths, device, amp_dtype, batch_size):
    ordered = sorted(queries, key=lambda item: len(item["steering_tokens"]))
    accum = {
        source: {
            role: {str(value): defaultdict(float) for value in strengths}
            for role in ("actual_source", "actual_destination", "endpoint_attacker")
        }
        for source in probes
    }
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            ids, mask, lengths = _pad(batch, vocabulary, device, "steering_tokens")
            output = model(ids, attention_mask=mask, output_hidden_states=True)
            rows = torch.arange(len(batch), device=device)
            prediction_positions = lengths - 1
            state_positions = torch.tensor(
                [int(item["steering_state_position"]) for item in batch], device=device
            )
            target_ids = torch.tensor([vocabulary[item["steering_target_token"]] for item in batch], device=device)
            baseline = torch.log_softmax(output.logits[rows, prediction_positions].float(), dim=-1)[rows, target_ids]
            targets = torch.cat([
                target_from_mapping(item["probe_targets"], board_map, hand_names).board
                for item in batch
            ]).to(device)
            for source, probe in probes.items():
                layer_index = int(source.split("_", 1)[1])
                base_hidden = output.hidden_states[layer_index]
                features = base_hidden[rows, state_positions]
                for role in ("actual_source", "actual_destination", "endpoint_attacker"):
                    relevant = [_stable_role_square(item, role) for item in batch]
                    controls = [
                        None if square is None else _stable_control(item, square, targets[row].cpu())
                        for row, (item, square) in enumerate(zip(batch, relevant))
                    ]
                    active = torch.tensor([a is not None and b is not None for a, b in zip(relevant, controls)], device=device)
                    if not bool(active.any()):
                        continue
                    for strength in strengths:
                        relevant_delta = _steering_delta(probe, features, targets, relevant, strength)
                        control_delta = _steering_delta(probe, features, targets, controls, strength)
                        relevant_hidden = base_hidden.clone()
                        control_hidden = base_hidden.clone()
                        relevant_hidden[rows, state_positions] += relevant_delta
                        control_hidden[rows, state_positions] += control_delta
                        relevant_logits = _resume_logits(model, relevant_hidden, layer_index, mask, model_type)
                        control_logits = _resume_logits(model, control_hidden, layer_index, mask, model_type)
                        relevant_lp = torch.log_softmax(relevant_logits[rows, prediction_positions].float(), dim=-1)[rows, target_ids]
                        control_lp = torch.log_softmax(control_logits[rows, prediction_positions].float(), dim=-1)[rows, target_ids]
                        store = accum[source][role][str(strength)]
                        store["examples"] += int(active.sum())
                        store["baseline_log_probability"] += float(baseline[active].sum())
                        store["relevant_intervention_log_probability"] += float(relevant_lp[active].sum())
                        store["matched_intervention_log_probability"] += float(control_lp[active].sum())
                        store["differential_damage"] += float((control_lp[active] - relevant_lp[active]).sum())
    result = {}
    for source, roles in accum.items():
        result[source] = {}
        for role, by_strength in roles.items():
            result[source][role] = {}
            for strength, sums in by_strength.items():
                count = int(sums.get("examples", 0))
                result[source][role][strength] = {
                    "examples": count,
                    **{
                        key: None if not count else value / count
                        for key, value in sums.items() if key != "examples"
                    },
                }
    return result


def main():
    args = parse_args()
    distances = _csv_ints(args.history_distances)
    strengths = _csv_floats(args.steering_strengths)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    vocabulary = load_vocabulary(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig(**checkpoint["config"])
    model_type = str(checkpoint.get("model_type", "vanilla"))
    metadata = dict(checkpoint.get("new_prompt", {}))
    annotation_mode = "ap" if metadata.get("annotation_mode") == "ap" else "vanilla"
    if metadata.get("move_encoding") != MOVE_ENCODING or metadata.get("terminal_encoding") != TERMINAL_ENCODING:
        raise ValueError("evaluation requires a current factorized_v3 checkpoint")
    if metadata.get("state_prompt_mode") != "implicit_initial" or metadata.get("start_selection") != "fixed_initial":
        raise ValueError("evaluation requires the implicit fixed-initial experiment")
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    del checkpoint

    artifact = torch.load(args.linear_probes, map_location="cpu")
    available = list(artifact["sources"])
    sources = _resolve_sources(args.sources, available, config.n_layers)
    steering_sources = _resolve_sources(args.steering_sources, available, config.n_layers)
    board_map, hand_names = label_maps()
    if dict(artifact.get("board_label_map", {})) != board_map:
        raise ValueError("linear probe board labels do not match the current evaluator")
    if list(artifact.get("hand_names", [])) != hand_names:
        raise ValueError("linear probe hand labels do not match the current evaluator")
    probes = _load_probes(artifact, sources, config.d_model, device)
    steering_probes = {source: probes.get(source) or _load_probes(artifact, [source], config.d_model, device)[source] for source in steering_sources}
    queries, eligible = read_queries(
        args.evaluation_jsonl, metadata["state_prompt_mode"], annotation_mode,
        distances, args.max_examples, args.seed,
    )
    if any(len(item["prefix_tokens"]) > config.max_seq_len for item in queries):
        raise ValueError("a selected prefix exceeds the checkpoint context length")
    decoding, ordered = evaluate_decoding(
        model, probes, queries, vocabulary, board_map, hand_names,
        device, amp_dtype, args.batch_size,
    )
    steering_count = min(len(queries), args.max_steering_examples)
    steering_queries = random.Random(args.seed + 1).sample(queries, steering_count)
    steering = evaluate_steering(
        model, model_type, steering_probes, steering_queries, vocabulary,
        board_map, hand_names, strengths, device, amp_dtype, args.batch_size,
    ) if args.max_steering_examples > 0 else {}
    result = {
        "format_version": 1,
        "checkpoint": args.checkpoint,
        "linear_probes": args.linear_probes,
        "model_type": model_type,
        "move_encoding": metadata.get("move_encoding"),
        "evaluation_input_annotation_mode": annotation_mode,
        "settings": vars(args),
        "amp": amp_name,
        "eligible_queries": eligible,
        "evaluated_queries": len(queries),
        "definitions": {
            "actual_source": "正解次手の移動元。駒打ちでは該当マスなし。",
            "actual_destination": "正解次手の移動先。",
            "endpoint_attacker": "正解次手の移動元または移動先へ、指手直前の局面で利きを持つ駒が存在するマス。移動元・移動先自体は除く。",
            "actual_local_context": "actual_source、actual_destination、endpoint_attackerの和集合。",
            "candidate_source": "その局面で候補となる全指手に現れる盤上の移動元。",
            "candidate_destination": "その局面で候補となる全指手に現れる移動先。",
            "candidate_related": "candidate_sourceとcandidate_destinationの和集合。",
            "background": "candidate_relatedにもendpoint_attackerにも含まれないマス。",
            "coordinate_piece_recency_matched": "座標、正解駒クラス、最後に指手で更新されてからの距離区分が同じbackgroundとの比較。",
            "within_position_piece_recency_matched": "同一局面内で正解駒クラスと更新距離区分が同じbackgroundとの比較。",
            "differential_damage": "関連マス介入後より比較マス介入後の正解次トークン対数確率がどれだけ高いか。正値は関連マス介入の影響が大きい。",
            "steering_prediction_target": "通常移動では移動元座標、駒打ちでは<DROP>。APの通常移動だけは正解駒種注釈を与えてから移動元座標を評価する。",
        },
        "decoding": decoding,
        "steering": steering,
        "limitations": [
            "復号精度は線形復号可能性を測り、それだけでは次手予測への因果的利用を示さない。",
            "steeringは線形プローブが定める方向への局所的介入であり、盤面変数だけを完全に変更する保証はない。",
            "endpoint_attackerは正解次手の両端への直接の利きだけを表し、盤面全体の利きや複数手先の戦術関係を表すものではない。",
        ],
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
