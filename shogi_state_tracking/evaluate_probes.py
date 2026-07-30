#!/usr/bin/env python3
"""凍結済みdecoderの層別線形プローブを学習・評価する。"""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from create_dataset import all_usi_move_tokens, import_cshogi
from data import (
    FIXED_SEQUENCE_OVERHEAD,
    IGNORE_INDEX,
    RandomStartSequenceDataset,
    load_vocabulary,
)
from models import ModelConfig, T2MLRConfig, build_model
from probes import (
    LinearStateProbe,
    ProbeTargets,
    distance_bin,
    linear_probe_loss,
    majority_predictions,
    predictions_from_logits,
    replay_probe_targets,
    state_metrics,
    stratified_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="指し手境界表現から盤面・持ち駒・手番を線形復号する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sources",
        default="final,recurrent,token_embedding",
        help=(
            "comma区切り。final, layers, layer_0..layer_N, recurrent, "
            "token_embeddingを指定可能"
        ),
    )
    parser.add_argument("--candidate-count", type=int, default=40)
    parser.add_argument("--min-suffix-moves", type=int, default=40)
    parser.add_argument("--samples-per-game", type=int, default=1)
    parser.add_argument(
        "--positions-per-game",
        type=int,
        default=16,
        help="各対局から等間隔に使うstate_1以降の数。0なら全位置",
    )
    parser.add_argument(
        "--include-initial-state",
        dest="include_initial_state",
        action="store_true",
        default=True,
        help="prompt読取りのsanity checkとしてstate_0も評価する",
    )
    parser.add_argument(
        "--exclude-initial-state",
        dest="include_initial_state",
        action="store_false",
    )
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="probe学習・評価のbatch size。GPUメモリに応じて調整する",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="特徴抽出・probe学習の進捗表示間隔。0なら詳細表示をしない",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, mpsなど",
    )
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="checkpointの設定だけを使い、重みをロードしない対照条件",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_backbone(path: str, device: torch.device, untrained: bool):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be a mapping")
    model_type = str(payload.get("model_type", "")).lower()
    config_payload = payload.get("config")
    if not model_type or not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint requires model_type and config")
    config_dict = dict(config_payload)
    config = (
        T2MLRConfig(**config_dict)
        if model_type in {"t2mlr", "t^2mlr", "t²mlr"}
        else ModelConfig(**config_dict)
    )
    model = build_model(model_type, config)
    if not untrained:
        state = payload.get("model_state_dict", payload.get("state_dict"))
        if state is None:
            raise ValueError("checkpoint requires model_state_dict or state_dict")
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_type, config


def fixed_dataset(
    path: str,
    vocabulary: Mapping[str, int],
    candidate_count: int,
    min_suffix_moves: int,
    samples_per_game: int,
    seed: int,
    max_seq_len: int | None = None,
):
    # RandomStartSequenceDatasetはepoch=0固定で、同じゲームから常に同じ開始点を返す。
    max_suffix_moves = None
    if max_seq_len is not None:
        max_suffix_moves = max_seq_len - FIXED_SEQUENCE_OVERHEAD
        if max_suffix_moves <= 0:
            raise ValueError(
                "checkpoint max_seq_len is too short for the fixed state prefix"
            )
    return RandomStartSequenceDataset(
        path,
        vocabulary,
        candidate_count=candidate_count,
        min_suffix_moves=min_suffix_moves,
        samples_per_game=samples_per_game,
        seed=seed,
        randomize_each_epoch=False,
        max_suffix_moves=max_suffix_moves,
    )


def select_distances(
    move_count: int, positions_per_game: int, include_initial: bool
) -> List[int]:
    distances = list(range(1, move_count + 1))
    if positions_per_game > 0 and len(distances) > positions_per_game:
        if positions_per_game == 1:
            distances = [distances[len(distances) // 2]]
        else:
            distances = [
                distances[
                    round(index * (len(distances) - 1) / (positions_per_game - 1))
                ]
                for index in range(positions_per_game)
            ]
            distances = list(dict.fromkeys(distances))
    if include_initial:
        return [0] + distances
    return distances


def expand_sources(
    source_text: str, n_layers: int, has_recurrent: bool
) -> List[str]:
    result: List[str] = []
    for source in (part.strip() for part in source_text.split(",")):
        if not source:
            continue
        if source == "final":
            source = "layer_{}".format(n_layers)
        if source == "layers":
            result.extend("layer_{}".format(index) for index in range(n_layers + 1))
            continue
        if source == "recurrent" and not has_recurrent:
            continue
        if source == "token_embedding" or source == "recurrent":
            result.append(source)
            continue
        if source.startswith("layer_"):
            index = int(source.split("_", 1)[1])
            if not 0 <= index <= n_layers:
                raise ValueError("layer source outside model: {}".format(source))
            result.append(source)
            continue
        raise ValueError("unknown probe source: {}".format(source))
    result = list(dict.fromkeys(result))
    if not result:
        raise ValueError("no applicable probe sources")
    return result


def concatenate_targets(chunks: Sequence[ProbeTargets]) -> ProbeTargets:
    return ProbeTargets(
        board=torch.cat([chunk.board for chunk in chunks], dim=0),
        hands=torch.cat([chunk.hands for chunk in chunks], dim=0),
        turn=torch.cat([chunk.turn for chunk in chunks], dim=0),
    )


def extract_split(
    model,
    model_type: str,
    dataset,
    id_to_token: Mapping[int, str],
    sources: Sequence[str],
    positions_per_game: int,
    include_initial: bool,
    device: torch.device,
    progress_label: str = "split",
    progress_every: int = 10,
):
    feature_chunks: Dict[str, List[torch.Tensor]] = {
        source: [] for source in sources
    }
    target_chunks: List[ProbeTargets] = []
    distances_all: List[int] = []
    scopes_all: List[str] = []
    position_scopes_all: List[str] = []
    trajectory_scopes_all: List[str] = []
    games_all: List[str] = []
    # 指手評価は<EOS>を除く指手位置だけで集計する。<EOS>は対局終了の
    # 制御トークンであり，合法手の選択能力とは異なるためである。
    move_loss_sum = 0.0
    move_targets = 0
    move_top1 = 0
    move_top5 = 0
    eos_loss_sum = 0.0
    eos_targets = 0
    legal_positions = 0
    legal_top1 = 0
    legal_top5 = 0
    syntactic_top1 = 0
    legal_probability_mass = 0.0
    legal_vocabulary_coverage = 0.0
    token_to_id = {token: index for index, token in id_to_token.items()}
    syntactic_moves = set(all_usi_move_tokens())
    cshogi = import_cshogi()
    started_at = time.perf_counter()
    print(
        "feature_extract_start split={} examples={} sources={}".format(
            progress_label, len(dataset), ",".join(sources)
        ),
        flush=True,
    )

    with torch.inference_mode():
        for example_index in range(len(dataset)):
            example = dataset[example_index]
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            recurrent_mask = example["recurrent_mask"].unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            exact = model_type in {"t2mlr", "t^2mlr", "t²mlr"}
            output = model(
                input_ids,
                attention_mask=None if exact else attention_mask,
                recurrent_mask=recurrent_mask,
                exact_recurrence=exact,
            )

            labels = example["labels"].to(device)
            supervised = labels != IGNORE_INDEX
            eos_id = token_to_id.get("<EOS>")
            move_supervised = supervised
            if eos_id is not None:
                move_supervised = supervised & (labels != eos_id)
            move_logits = output.logits[0, move_supervised]
            move_labels = labels[move_supervised]
            if move_labels.numel():
                move_loss_sum += float(
                    torch.nn.functional.cross_entropy(
                        move_logits,
                        move_labels,
                        reduction="sum",
                    )
                )
                move_targets += int(move_labels.numel())
                move_top1 += int(
                    (move_logits.argmax(dim=-1) == move_labels).sum()
                )
                top_k = min(5, move_logits.shape[-1])
                move_top5 += int(
                    (
                        move_logits.topk(top_k, dim=-1).indices
                        == move_labels[:, None]
                    )
                    .any(dim=1)
                    .sum()
                )
            if eos_id is not None:
                eos_supervised = supervised & (labels == eos_id)
                eos_logits = output.logits[0, eos_supervised]
                eos_labels = labels[eos_supervised]
                if eos_labels.numel():
                    eos_loss_sum += float(
                        torch.nn.functional.cross_entropy(
                            eos_logits,
                            eos_labels,
                            reduction="sum",
                        )
                    )
                    eos_targets += int(eos_labels.numel())

            moves_marker = 1 + 96
            move_ids = example["input_ids"][moves_marker + 1 : -1].tolist()
            move_tokens = [id_to_token[int(token_id)] for token_id in move_ids]
            replay_board = cshogi.Board(str(example["start_sfen"]))
            for move_index, target_move in enumerate(move_tokens):
                prediction_position = moves_marker + move_index
                move_logits = output.logits[0, prediction_position]
                legal_moves = [
                    cshogi.move_to_usi(move) for move in replay_board.legal_moves
                ]
                legal_ids = [
                    token_to_id[move]
                    for move in legal_moves
                    if move in token_to_id
                ]
                legal_id_set = set(legal_ids)
                top_k = min(5, move_logits.shape[-1])
                top_ids = move_logits.topk(top_k).indices.tolist()
                top_token = id_to_token[int(top_ids[0])]

                legal_positions += 1
                legal_top1 += int(int(top_ids[0]) in legal_id_set)
                legal_top5 += int(
                    any(int(value) in legal_id_set for value in top_ids)
                )
                syntactic_top1 += int(top_token in syntactic_moves)
                legal_vocabulary_coverage += len(legal_ids) / max(
                    len(legal_moves), 1
                )
                if legal_ids:
                    probabilities = torch.softmax(move_logits, dim=-1)
                    legal_index = torch.tensor(
                        legal_ids, dtype=torch.long, device=device
                    )
                    legal_probability_mass += float(
                        probabilities.index_select(0, legal_index).sum()
                    )

                target = replay_board.move_from_usi(str(target_move))
                if not replay_board.is_legal(target):
                    raise ValueError(
                        "ground-truth move is illegal in game {} at local ply {}: {}".format(
                            example["game_id"], move_index + 1, target_move
                        )
                    )
                replay_board.push(target)

            all_targets = replay_probe_targets(example["start_sfen"], move_tokens)
            distances = select_distances(
                len(move_tokens), positions_per_game, include_initial
            )
            scope_by_ply = list(example.get("position_scope_by_ply", []))
            if scope_by_ply and max(distances, default=0) >= len(scope_by_ply):
                raise ValueError(
                    "position_scope_by_ply is shorter than selected probe distances"
                )
            if scope_by_ply:
                selected_position_scopes = [scope_by_ply[index] for index in distances]
            else:
                selected_position_scopes = [
                    str(example.get("position_scope", "unknown_position_scope"))
                ] * len(distances)
            selected = torch.tensor(distances, dtype=torch.long)
            target_chunks.append(
                ProbeTargets(
                    board=all_targets.board[selected],
                    hands=all_targets.hands[selected],
                    turn=all_targets.turn[selected],
                )
            )
            sequence_positions = moves_marker + selected
            device_positions = sequence_positions.to(device)

            for source in sources:
                if source == "token_embedding":
                    features = model.token_embedding(input_ids)[0, device_positions]
                elif source == "recurrent":
                    if output.recurrent_states is None:
                        raise ValueError("model did not return recurrent states")
                    features = output.recurrent_states[0, device_positions]
                else:
                    layer_index = int(source.split("_", 1)[1])
                    features = output.hidden_states[layer_index][
                        0, device_positions
                    ]
                feature_chunks[source].append(features.detach().cpu())

            distances_all.extend(distances)
            scopes_all.extend([str(example["engine_scope"])] * len(distances))
            position_scopes_all.extend(selected_position_scopes)
            trajectory_scopes_all.extend(
                [str(example.get("trajectory_scope", "unknown_position_scope"))]
                * len(distances)
            )
            games_all.extend([str(example["game_id"])] * len(distances))
            processed = example_index + 1
            if progress_every > 0 and (
                processed == 1 or processed % progress_every == 0
            ):
                elapsed = time.perf_counter() - started_at
                print(
                    "feature_extract_progress split={} examples={}/{} elapsed_sec={:.1f} examples_per_sec={:.3f}".format(
                        progress_label,
                        processed,
                        len(dataset),
                        elapsed,
                        processed / max(elapsed, 1e-9),
                    ),
                    flush=True,
                )

    if not target_chunks:
        raise ValueError("probe dataset produced no examples")
    print(
        "feature_extract_complete split={} examples={} elapsed_sec={:.1f}".format(
            progress_label,
            len(dataset),
            time.perf_counter() - started_at,
        ),
        flush=True,
    )
    return {
        "features": {
            source: torch.cat(chunks, dim=0)
            for source, chunks in feature_chunks.items()
        },
        "targets": concatenate_targets(target_chunks),
        "distances": torch.tensor(distances_all, dtype=torch.long),
        "scopes": scopes_all,
        "position_scopes": position_scopes_all,
        "trajectory_scopes": trajectory_scopes_all,
        "game_ids": games_all,
        "lm": {
            "targets": move_targets,
            "cross_entropy": move_loss_sum / max(move_targets, 1),
            "perplexity": math.exp(min(move_loss_sum / max(move_targets, 1), 20.0)),
            "top1_accuracy": move_top1 / max(move_targets, 1),
            "top5_accuracy": move_top5 / max(move_targets, 1),
            "eos_targets": eos_targets,
            "eos_cross_entropy": eos_loss_sum / max(eos_targets, 1),
            "legality": {
                "move_positions": legal_positions,
                "top1_legal_rate": legal_top1 / max(legal_positions, 1),
                "top5_contains_legal_rate": legal_top5
                / max(legal_positions, 1),
                "top1_syntactic_move_rate": syntactic_top1
                / max(legal_positions, 1),
                "mean_legal_probability_mass": legal_probability_mass
                / max(legal_positions, 1),
                "mean_legal_move_vocabulary_coverage": legal_vocabulary_coverage
                / max(legal_positions, 1),
            },
        },
    }


def target_batch(targets: ProbeTargets, indices: torch.Tensor, device: torch.device):
    return ProbeTargets(
        board=targets.board[indices].to(device),
        hands=targets.hands[indices].to(device),
        turn=targets.turn[indices].to(device),
    )


def validation_loss(
    probe: LinearStateProbe,
    features: torch.Tensor,
    targets: ProbeTargets,
    batch_size: int,
    device: torch.device,
) -> float:
    probe.eval()
    total = 0.0
    samples = 0
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            indices = torch.arange(start, stop)
            logits = probe(features[indices].to(device))
            loss, _ = linear_probe_loss(
                logits, target_batch(targets, indices, device)
            )
            count = stop - start
            total += float(loss) * count
            samples += count
    return total / max(samples, 1)


def train_probe(
    train_features: torch.Tensor,
    train_targets: ProbeTargets,
    validation_features: torch.Tensor,
    validation_targets: ProbeTargets,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[LinearStateProbe, Mapping[str, object]]:
    torch.manual_seed(args.seed)
    probe = LinearStateProbe(train_features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    started_at = time.perf_counter()

    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        permutation = torch.randperm(train_features.shape[0], generator=generator)
        for start in range(0, permutation.numel(), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = probe(train_features[indices].to(device))
            loss, _ = linear_probe_loss(
                logits, target_batch(train_targets, indices, device)
            )
            loss.backward()
            optimizer.step()

        current_loss = validation_loss(
            probe,
            validation_features,
            validation_targets,
            args.batch_size,
            device,
        )
        history.append({"epoch": epoch, "validation_loss": current_loss})
        if args.progress_every > 0 and (
            epoch == 1
            or epoch % args.progress_every == 0
            or epoch == args.probe_epochs
        ):
            print(
                "probe_progress epoch={}/{} validation_loss={:.6f} elapsed_sec={:.1f}".format(
                    epoch,
                    args.probe_epochs,
                    current_loss,
                    time.perf_counter() - started_at,
                ),
                flush=True,
            )
        if current_loss < best_loss - 1e-7:
            best_loss = current_loss
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise AssertionError("probe training did not produce a checkpoint")
    probe.load_state_dict(best_state)
    probe.eval()
    return probe, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
    }


def predict_probe(
    probe: LinearStateProbe,
    features: torch.Tensor,
    targets: ProbeTargets,
    batch_size: int,
    device: torch.device,
):
    boards = []
    hands = []
    turns = []
    board_target_probabilities = []
    board_prediction_probabilities = []
    probe.eval()
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            logits = probe(features[start:stop].to(device))
            board, hand, turn = predictions_from_logits(logits)
            boards.append(board.cpu())
            hands.append(hand.cpu())
            turns.append(turn.cpu())
            board_probabilities = logits.board.softmax(dim=-1)
            target = targets.board[start:stop].to(device)
            board_target_probabilities.append(
                board_probabilities.gather(
                    -1, target.unsqueeze(-1)
                ).squeeze(-1).cpu()
            )
            board_prediction_probabilities.append(
                board_probabilities.max(dim=-1).values.cpu()
            )
    return (
        torch.cat(boards),
        torch.cat(hands),
        torch.cat(turns),
        torch.cat(board_target_probabilities),
        torch.cat(board_prediction_probabilities),
    )


def evaluate_source(
    source: str,
    probe: LinearStateProbe,
    split_data,
    batch_size: int,
    device: torch.device,
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    (
        board,
        hands,
        turn,
        board_target_probability,
        board_prediction_probability,
    ) = predict_probe(
        probe,
        split_data["features"][source],
        split_data["targets"],
        batch_size,
        device,
    )
    metrics = state_metrics(split_data["targets"], board, hands, turn)
    metrics["strata"] = stratified_metrics(
        split_data["targets"],
        board,
        hands,
        turn,
        split_data["distances"],
        split_data["scopes"],
    )
    metrics["position_strata"] = stratified_metrics(
        split_data["targets"],
        board,
        hands,
        turn,
        split_data["distances"],
        split_data["position_scopes"],
    )
    return metrics, {
        "board_target": split_data["targets"].board,
        "board_prediction": board,
        "board_target_probability": board_target_probability,
        "board_prediction_probability": board_prediction_probability,
        "hand_target": split_data["targets"].hands,
        "hand_prediction": hands,
        "turn_target": split_data["targets"].turn,
        "turn_prediction": turn,
        "distances": split_data["distances"],
        "scopes": list(split_data["scopes"]),
        "position_scopes": list(split_data["position_scopes"]),
        "trajectory_scopes": list(split_data["trajectory_scopes"]),
        "game_ids": list(split_data["game_ids"]),
    }


def main() -> None:
    args = parse_args()
    run_started_at = time.perf_counter()
    print(
        "run_start checkpoint={} device={} sources={} probe_epochs={}".format(
            args.checkpoint, args.device, args.sources, args.probe_epochs
        ),
        flush=True,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    vocabulary = load_vocabulary(args.vocab)
    id_to_token = {index: token for token, index in vocabulary.items()}
    model, model_type, config = load_backbone(
        args.checkpoint, device, args.untrained
    )
    sources = expand_sources(
        args.sources,
        config.n_layers,
        model_type in {"t2mlr", "t^2mlr", "t²mlr"},
    )

    datasets = {
        "train": fixed_dataset(
            args.train_jsonl,
            vocabulary,
            args.candidate_count,
            args.min_suffix_moves,
            args.samples_per_game,
            args.seed,
            config.max_seq_len,
        ),
        "validation": fixed_dataset(
            args.validation_jsonl,
            vocabulary,
            args.candidate_count,
            args.min_suffix_moves,
            args.samples_per_game,
            args.seed + 1,
            config.max_seq_len,
        ),
        "evaluation": fixed_dataset(
            args.evaluation_jsonl,
            vocabulary,
            args.candidate_count,
            args.min_suffix_moves,
            args.samples_per_game,
            args.seed + 2,
            config.max_seq_len,
        ),
    }
    print(
        "data_ready train_examples={} validation_examples={} evaluation_examples={} device={}".format(
            len(datasets["train"]),
            len(datasets["validation"]),
            len(datasets["evaluation"]),
            device,
        ),
        flush=True,
    )
    extracted = {}
    for name, dataset in datasets.items():
        extracted[name] = extract_split(
            model,
            model_type,
            dataset,
            id_to_token,
            sources,
            args.positions_per_game,
            args.include_initial_state,
            device,
            progress_label=name,
            progress_every=args.progress_every,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "model_type": model_type,
        "untrained": bool(args.untrained),
        "sources": sources,
        "device": str(device),
        "settings": {
            "positions_per_game": args.positions_per_game,
            "include_initial_state": args.include_initial_state,
            "samples_per_game": args.samples_per_game,
            "seed": args.seed,
        },
        "language_model": {
            name: data["lm"] for name, data in extracted.items()
        },
        "probe_results": {},
    }
    saved_probes = {}

    majority_board, majority_hands, majority_turn = majority_predictions(
        extracted["train"]["targets"],
        extracted["evaluation"]["targets"].board.shape[0],
    )
    majority_metrics = state_metrics(
        extracted["evaluation"]["targets"],
        majority_board,
        majority_hands,
        majority_turn,
    )
    majority_metrics["strata"] = stratified_metrics(
        extracted["evaluation"]["targets"],
        majority_board,
        majority_hands,
        majority_turn,
        extracted["evaluation"]["distances"],
        extracted["evaluation"]["scopes"],
    )
    majority_metrics["position_strata"] = stratified_metrics(
        extracted["evaluation"]["targets"],
        majority_board,
        majority_hands,
        majority_turn,
        extracted["evaluation"]["distances"],
        extracted["evaluation"]["position_scopes"],
    )
    report["majority_baseline"] = majority_metrics

    prediction_payload = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "model_type": model_type,
        "sources": sources,
        "evaluation": {},
    }

    for source in sources:
        print(
            "probe_start source={} train_features={} validation_features={} evaluation_features={}".format(
                source,
                extracted["train"]["features"][source].shape[0],
                extracted["validation"]["features"][source].shape[0],
                extracted["evaluation"]["features"][source].shape[0],
            ),
            flush=True,
        )
        probe, training = train_probe(
            extracted["train"]["features"][source],
            extracted["train"]["targets"],
            extracted["validation"]["features"][source],
            extracted["validation"]["targets"],
            args,
            device,
        )
        validation_metrics, _ = evaluate_source(
            source, probe, extracted["validation"], args.batch_size, device
        )
        evaluation_metrics, evaluation_predictions = evaluate_source(
            source, probe, extracted["evaluation"], args.batch_size, device
        )
        report["probe_results"][source] = {
            "training": training,
            "validation": validation_metrics,
            "evaluation": evaluation_metrics,
        }
        prediction_payload["evaluation"][source] = evaluation_predictions
        saved_probes[source] = {
            key: value.detach().cpu() for key, value in probe.state_dict().items()
        }
        print(
            "probe_complete source={} best_epoch={} evaluation_full_state_exact_match={:.6f}".format(
                source,
                training["best_epoch"],
                evaluation_metrics["full_state_exact_match"],
            ),
            flush=True,
        )

    with (output_dir / "probe_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    torch.save(
        {
            "checkpoint": str(args.checkpoint),
            "model_type": model_type,
            "sources": sources,
            "probe_state_dicts": saved_probes,
        },
        output_dir / "linear_probes.pt",
    )
    torch.save(prediction_payload, output_dir / "probe_predictions.pt")
    print(
        "run_complete output_dir={} elapsed_sec={:.1f}".format(
            output_dir, time.perf_counter() - run_started_at
        ),
        flush=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
