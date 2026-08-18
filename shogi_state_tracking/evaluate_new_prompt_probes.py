#!/usr/bin/env python3
"""新prompt artifactから，開始局面＋履歴表現の層別線形プローブを学習する。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

from create_dataset import HAND_ORDER, PIECE_NAMES
from data import load_vocabulary
from models import ModelConfig, build_model
from new_prompt import piece_token, move_token
from factorized_prompt import MOVE_ENCODING, TERMINAL_ENCODING, factorize_history_move, factorize_usi
from probes import LinearStateProbe, ProbeTargets, binary_classification_metrics, linear_probe_loss, majority_predictions, predictions_from_logits, replay_probe_targets, state_metrics
from train_model import amp_context, resolve_amp


def parse_args():
    parser = argparse.ArgumentParser(description="新prompt用の層別線形状態プローブ")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", default="layers", help="layers，final，layer_0,...")
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-validation-samples", type=int, default=3000)
    parser.add_argument("--max-evaluation-samples", type=int, default=5000)
    parser.add_argument(
        "--history-distances",
        default="8,32",
        help="プローブの学習・主評価へ含める開始局面からの指手距離。0はprompt再読出しになるため既定では除外する。",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--length-bucket-pool-batches", type=int, default=16,
        help="このbatch数分の系列を長さ順に並べ，特徴抽出時のpaddingを減らす",
    )
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--alignment-check-samples", type=int, default=8,
                        help="splitごとのprefix/full causal consistency検査数")
    return parser.parse_args()


def resolve_device(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_maps():
    board = {"<EMPTY>": 0}
    for piece_type in range(1, 15):
        board[piece_token("B", piece_type)] = piece_type
        board[piece_token("W", piece_type)] = 14 + piece_type
    hands = ["<{}_{}>".format(color, piece) for color in ("B", "W") for piece in HAND_ORDER]
    return board, hands


def target_from_mapping(value: Mapping[str, object], board_map, hand_names) -> ProbeTargets:
    labels = value["board_labels_cshogi_order"]
    board = torch.tensor([[board_map[str(label)] for label in labels]], dtype=torch.long)
    hands = torch.tensor([[int(value["hands"].get(name, 0)) for name in hand_names]], dtype=torch.long)
    turn = torch.tensor([0 if value["turn"] == "<TURN_BLACK>" else 1], dtype=torch.long)
    check = torch.tensor([int(bool(value["in_check"]))], dtype=torch.long)
    return ProbeTargets(board=board, hands=hands, turn=turn, in_check=check)


def concat_targets(parts):
    return ProbeTargets(board=torch.cat([part.board for part in parts]), hands=torch.cat([part.hands for part in parts]), turn=torch.cat([part.turn for part in parts]), in_check=torch.cat([part.in_check for part in parts]))


def expand_sources(text, n_layers):
    result = []
    for item in (part.strip() for part in text.split(",")):
        if item == "layers": result.extend("layer_{}".format(index) for index in range(n_layers + 1))
        elif item == "final": result.append("layer_{}".format(n_layers))
        elif item.startswith("layer_") and 0 <= int(item.split("_", 1)[1]) <= n_layers: result.append(item)
        else: raise ValueError("unknown source: {}".format(item))
    return list(dict.fromkeys(result))


def parse_history_distances(value):
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        distance = int(item)
        if distance < 0:
            raise ValueError("history distance must be nonnegative")
        if distance not in result:
            result.append(distance)
    if not result:
        raise ValueError("at least one history distance is required")
    return tuple(result)


def read_examples(path, limit, history_distances, start_selection="random_candidates"):
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            record = json.loads(line)
            for raw_example in record.get("probe_examples", []):
                example = dict(raw_example)
                if start_selection == "fixed_initial" and int(example.get("start_ply", -1)) != 0:
                    continue
                example.setdefault("trajectory_scope", record.get("trajectory_scope", "unknown_position_scope"))
                # alignment検査とprefix/full-sequence比較のため，元対局の開始局面と
                # 切り捨てていない全指し手系列を保持する。
                example["initial_sfen"] = str(record.get("initial_sfen", ""))
                example["full_move_tokens"] = [str(move) for move in record.get("move_tokens", [])]
                example["full_move_annotations"] = [dict(value) for value in record.get("move_annotations", [])]
                example["history_move_annotations"] = example["full_move_annotations"][: int(example["ply"])]
                example["game_id"] = str(record.get("game_id", ""))
                distance = int(example["ply"]) - int(example["start_ply"])
                # ply=0は全例が同一の平手初期局面であり，主状態集計から除外する。
                if distance <= 0 or distance not in history_distances:
                    continue
                examples.append(example)
                if len(examples) >= limit: return examples
    if not examples: raise ValueError("no probe_examples in {}".format(path))
    return examples


def verify_probe_alignment(examples, vocabulary, max_seq_len, state_prompt_mode, evaluation_annotation_mode="vanilla"):
    """h_preと教師局面の時間対応を，系列構文とcshogi再生で検証する．"""
    checked = 0
    for example in examples:
        start_ply = int(example.get("start_ply", -1))
        ply = int(example.get("ply", -1))
        history_moves = [str(move) for move in example.get("history_moves", [])]
        if start_ply != 0:
            raise ValueError("state probe alignment requires start_ply=0")
        if ply <= 0:
            raise ValueError("ply=0 is excluded from the main state-probe aggregate")
        if len(history_moves) != ply:
            raise ValueError("history length {} does not match ply {} for {}".format(len(history_moves), ply, example.get("game_id", "")))
        factorized_history = []
        annotations = list(example.get("history_move_annotations", []))
        if evaluation_annotation_mode == "ap" and len(annotations) != len(history_moves):
            raise ValueError("AP probe history annotations do not align with moves")
        for move_index, move in enumerate(history_moves):
            annotation = annotations[move_index] if move_index < len(annotations) else None
            tokens = factorize_history_move(move, annotation, evaluation_annotation_mode) if MOVE_ENCODING == "factorized_v3_no_eom" else [move_token(move)]
            if not tokens or not tokens[-1].startswith("<SQ_"):
                raise ValueError("history move does not end at a destination square: {}".format(move))
            factorized_history.extend(tokens)
        state = [] if state_prompt_mode == "implicit_initial" else list(example.get("state_prompt_tokens", []))
        prefix = ["<BOS>"] + state + ["<MOVES>"] + factorized_history
        if len(prefix) > max_seq_len:
            raise ValueError("probe prefix exceeds max_seq_len; series must not be truncated: {}".format(example.get("game_id", "")))
        board_map, hand_names = label_maps()
        actual = target_from_mapping(example["probe_targets"], board_map, hand_names)
        replayed = replay_probe_targets(str(example["initial_sfen"]), history_moves)
        expected = ProbeTargets(
            board=replayed.board[-1:].clone(), hands=replayed.hands[-1:].clone(),
            turn=replayed.turn[-1:].clone(), in_check=replayed.in_check[-1:].clone(),
        )
        for name in ("board", "hands", "turn", "in_check"):
            if not torch.equal(getattr(actual, name), getattr(expected, name)):
                raise ValueError("probe label is not the state immediately before ply {} for {}".format(ply, example.get("game_id", "")))
        checked += 1
    return {"examples": checked, "truncated": 0, "ply_zero_excluded": True}


def check_causal_prefix_consistency(model, examples, vocabulary, sources, device, max_seq_len, state_prompt_mode, max_samples, evaluation_annotation_mode="vanilla"):
    """prefix単独と未来を含むfull sequenceのprefix位置を比較する．"""
    checked = 0
    skipped = 0
    max_abs_diff = {source: 0.0 for source in sources}
    # 長い対局をskipしても，後続の比較可能な例を探し続ける．先頭max_samples件
    # だけを調べると，そこがすべてmax_seq_len超過の場合に誤って検査不能となる．
    for example in examples:
        if checked >= max_samples:
            break
        history = [str(move) for move in example.get("history_moves", [])]
        full_moves = [str(move) for move in example.get("full_move_tokens", [])]
        full_annotations = list(example.get("full_move_annotations", []))
        if full_moves[: len(history)] != history:
            raise ValueError(
                "probe history is not a prefix of the full game for {}".format(
                    example.get("game_id", "")
                )
            )
        state = [] if state_prompt_mode == "implicit_initial" else list(example.get("state_prompt_tokens", []))
        prefix_tokens = ["<BOS>", *state, "<MOVES>"]
        for move_index, move in enumerate(history):
            prefix_tokens.extend(factorize_history_move(
                move, full_annotations[move_index] if move_index < len(full_annotations) else None,
                evaluation_annotation_mode,
            ))
        # 対局全体ではなく，max_seq_len内に収まる未来をprefixへ追加する．
        # causal maskの検査には1手以上の未来があればよく，長い対局全体が
        # contextへ収まることを要求する必要はない．
        full_tokens = list(prefix_tokens)
        for move_index, move in enumerate(full_moves[len(history) :], len(history)):
            future = factorize_history_move(
                move, full_annotations[move_index] if move_index < len(full_annotations) else None,
                evaluation_annotation_mode,
            )
            if len(full_tokens) + len(future) > max_seq_len:
                break
            full_tokens.extend(future)
        if len(full_tokens) <= len(prefix_tokens):
            skipped += 1
            continue
        prefix_ids = torch.tensor([[vocabulary[token] for token in prefix_tokens]], dtype=torch.long, device=device)
        full_ids = torch.tensor([[vocabulary[token] for token in full_tokens]], dtype=torch.long, device=device)
        with torch.inference_mode():
            prefix_output = model(prefix_ids, output_hidden_states=True)
            full_output = model(full_ids, output_hidden_states=True)
        position = len(prefix_tokens) - 1
        for source in sources:
            layer = int(source.split("_", 1)[1])
            difference = float((prefix_output.hidden_states[layer][0, -1] - full_output.hidden_states[layer][0, position]).abs().max().cpu())
            max_abs_diff[source] = max(max_abs_diff[source], difference)
            if difference > 2e-4:
                raise AssertionError("causal prefix/full mismatch at {}: {}".format(source, difference))
        checked += 1
    return {"checked": checked, "skipped": skipped, "max_abs_diff": max_abs_diff, "passed": checked > 0}


def assert_disjoint_game_ids(paths):
    """probeの学習・検証・評価間で対局が重複していないことを確認する．"""
    game_ids = {}
    for split, path in paths.items():
        values = set()
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                game_id = str(json.loads(line).get("game_id", ""))
                if not game_id:
                    raise ValueError("{}:{} has no game_id".format(path, line_number))
                values.add(game_id)
        game_ids[split] = values
    split_names = list(paths)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = game_ids[left] & game_ids[right]
            if overlap:
                raise ValueError(
                    "game_id overlap between {} and {}: {} examples (e.g. {})".format(
                        left, right, len(overlap), sorted(overlap)[0]
                    )
                )
    return {split: len(values) for split, values in game_ids.items()}


def _mapped_features(directory, split, sources, rows, d_model):
    result = {}
    for source in sources:
        path = Path(directory) / "{}_{}.f32".format(split, source)
        with path.open("wb") as handle:
            handle.truncate(rows * d_model * 4)
        result[source] = torch.from_file(
            str(path), shared=True, size=rows * d_model, dtype=torch.float32,
        ).reshape(rows, d_model)
    return result


def extract(model, examples, vocabulary, sources, board_map, hand_names, device, max_seq_len, move_encoding, batch_size, state_prompt_mode, feature_directory, split, amp_dtype=None, length_bucket_pool_batches=16, evaluation_annotation_mode="vanilla"):
    features = _mapped_features(
        feature_directory, split, sources, len(examples), model.config.d_model,
    )
    target_chunks = []
    metadata = {"position_scope": [], "trajectory_scope": [], "history_distance": [], "start_ply": []}
    model.eval()
    prepared_pool = []
    written = 0

    def consume(batch):
        nonlocal written
        if not batch:
            return
        lengths = torch.tensor([len(ids) for _, ids in batch], device=device)
        width = int(lengths.max())
        input_ids = torch.full((len(batch), width), vocabulary["<PAD>"], dtype=torch.long, device=device)
        attention_mask = None
        if not bool((lengths == width).all()):
            attention_mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
        for row, (_, ids) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids, device=device)
        output = model(input_ids, attention_mask=attention_mask)
        rows = torch.arange(len(batch), device=device)
        positions = lengths - 1
        end = written + len(batch)
        for source in sources:
            layer = int(source.split("_", 1)[1])
            selected = output.hidden_states[layer][rows, positions].detach().cpu()
            features[source][written:end].copy_(selected)
        for example, _ in batch:
            target_chunks.append(target_from_mapping(example["probe_targets"], board_map, hand_names))
            metadata["position_scope"].append(str(example.get("position_scope", "unknown_position_scope")))
            metadata["trajectory_scope"].append(str(example.get("trajectory_scope", "unknown_position_scope")))
            metadata["history_distance"].append(int(example["ply"]) - int(example["start_ply"]))
            metadata["start_ply"].append(int(example["start_ply"]))
        written = end

    def flush_pool():
        nonlocal prepared_pool
        prepared_pool.sort(key=lambda value: len(value[1]))
        for start in range(0, len(prepared_pool), batch_size):
            consume(prepared_pool[start : start + batch_size])
        prepared_pool = []

    pool_size = batch_size * max(1, int(length_bucket_pool_batches))
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for example in examples:
            history = []
            annotations = list(example.get("history_move_annotations", []))
            for move_index, move in enumerate(example["history_moves"]):
                history.extend(factorize_history_move(
                    move, annotations[move_index] if move_index < len(annotations) else None,
                    evaluation_annotation_mode,
                ) if move_encoding == MOVE_ENCODING else [move_token(move)])
            state = [] if state_prompt_mode == "implicit_initial" else list(example["state_prompt_tokens"])
            tokens = ["<BOS>"] + state + ["<MOVES>"] + history
            if len(tokens) > max_seq_len:
                raise ValueError(
                    "probe prefix exceeds max_seq_len; refusing to truncate {} at ply {}".format(
                        example.get("game_id", ""), example.get("ply", "?")
                    )
                )
            prepared_pool.append((example, [vocabulary[token] for token in tokens]))
            if len(prepared_pool) >= pool_size:
                flush_pool()
        flush_pool()
    if not target_chunks: raise ValueError("all probe examples exceeded max_seq_len")
    return {source: values[:written] for source, values in features.items()}, concat_targets(target_chunks), metadata


def fit_probe(train_x, train_y, validation_x, validation_y, d_model, args, device, seed):
    # checkpoint backboneの構築時に消費した乱数と切り離し，モデル種別間で
    # probe初期値とmini-batch順を揃える。
    torch.manual_seed(seed)
    probe = LinearStateProbe(d_model).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y.board, train_y.hands, train_y.turn, train_y.in_check),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    best_state, best_loss, wait = None, float("inf"), 0
    for _ in range(args.probe_epochs):
        probe.train()
        for x, board, hands, turn, check in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = linear_probe_loss(probe(x.to(device)), ProbeTargets(board.to(device), hands.to(device), turn.to(device), check.to(device)))
            loss.backward(); optimizer.step()
        probe.eval()
        with torch.inference_mode():
            loss, _ = linear_probe_loss(probe(validation_x.to(device)), ProbeTargets(validation_y.board.to(device), validation_y.hands.to(device), validation_y.turn.to(device), validation_y.in_check.to(device)))
        current = float(loss)
        if current < best_loss - 1e-4:
            best_loss, wait = current, 0
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience: break
    probe.load_state_dict(best_state)
    return probe, best_loss


def metric(probe, x, targets, device):
    probe.eval()
    with torch.inference_mode():
        logits = probe(x.to(device)); board, hands, turn = predictions_from_logits(logits)
    result = state_metrics(targets, board.cpu(), hands.cpu(), turn.cpu())
    if targets.in_check is not None:
        check_prediction = logits.in_check.argmax(dim=-1).cpu()
        check_metrics = binary_classification_metrics(targets.in_check, check_prediction)
        result.update({"in_check_{}".format(name): value for name, value in check_metrics.items()})
        result["in_check_positive_rate"] = float(targets.in_check.float().mean())
    return result


def metrics_by_group(probe, x, targets, values, device):
    result = {}
    for group in sorted(set(values), key=str):
        indices = torch.tensor([index for index, value in enumerate(values) if value == group], dtype=torch.long)
        subset = ProbeTargets(targets.board[indices], targets.hands[indices], targets.turn[indices], None if targets.in_check is None else targets.in_check[indices])
        result[str(group)] = metric(probe, x[indices], subset, device)
    return result


def metrics_by_history_distance_and_group(probe, x, targets, distances, groups, device):
    result = {}
    for distance in sorted(set(distances)):
        indices = [index for index, value in enumerate(distances) if value == distance]
        subset = ProbeTargets(
            targets.board[indices], targets.hands[indices], targets.turn[indices],
            None if targets.in_check is None else targets.in_check[indices],
        )
        result[str(distance)] = metrics_by_group(
            probe, x[indices], subset, [groups[index] for index in indices], device,
        )
    return result


def main():
    args = parse_args(); args.history_distances = parse_history_distances(args.history_distances)
    random.seed(args.seed); torch.manual_seed(args.seed)
    vocabulary = load_vocabulary(args.vocab); device = resolve_device(args.device)
    amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig(**checkpoint["config"])
    model_type = str(checkpoint.get("model_type", "vanilla"))
    move_encoding = str(checkpoint.get("new_prompt", {}).get("move_encoding", "atomic_v1"))
    state_prompt_mode = str(checkpoint.get("new_prompt", {}).get("state_prompt_mode", "explicit"))
    start_selection = str(checkpoint.get("new_prompt", {}).get("start_selection", "random_candidates"))
    terminal_encoding = str(checkpoint.get("new_prompt", {}).get("terminal_encoding", ""))
    evaluation_annotation_mode = "ap" if checkpoint.get("new_prompt", {}).get("annotation_mode") == "ap" else "vanilla"
    if move_encoding != MOVE_ENCODING or terminal_encoding != TERMINAL_ENCODING or state_prompt_mode != "implicit_initial" or start_selection != "fixed_initial":
        raise ValueError("linear probe requires the current implicit fixed-initial factorized_v3 checkpoint")
    model = build_model(model_type, config).to(device); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    del checkpoint
    board_map, hand_names = label_maps(); sources = expand_sources(args.sources, config.n_layers)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    feature_cache = tempfile.TemporaryDirectory(prefix=".probe-features-", dir=str(output))
    split_paths = {
        "train": args.train_jsonl,
        "validation": args.validation_jsonl,
        "evaluation": args.evaluation_jsonl,
    }
    split_game_counts = assert_disjoint_game_ids(split_paths)
    raw = {
        split: read_examples(path, getattr(args, "max_{}_samples".format(split)), args.history_distances, start_selection)
        for split, path in split_paths.items()
    }
    print(json.dumps({"event": "probe_alignment_check_start", "examples": {name: len(values) for name, values in raw.items()}}, ensure_ascii=False), flush=True)
    alignment = {
        split: verify_probe_alignment(examples, vocabulary, config.max_seq_len, state_prompt_mode, evaluation_annotation_mode)
        for split, examples in raw.items()
    }
    causal_alignment = {
        split: check_causal_prefix_consistency(
            model, examples, vocabulary, sources, device, config.max_seq_len,
            state_prompt_mode, args.alignment_check_samples, evaluation_annotation_mode,
        )
        for split, examples in raw.items()
    }
    missing_causal_checks = [split for split, report in causal_alignment.items() if not report["passed"]]
    if missing_causal_checks:
        raise ValueError("causal prefix/full-sequence check produced no comparable example: {}".format(", ".join(missing_causal_checks)))
    print(json.dumps({"event": "probe_alignment_check_complete", "alignment": alignment, "causal_prefix_full_alignment": causal_alignment}, ensure_ascii=False), flush=True)
    feature_bytes = sum(len(examples) for examples in raw.values()) * len(sources) * config.d_model * 4
    disk_free = shutil.disk_usage(output).free
    if feature_bytes + 2**30 > disk_free:
        feature_cache.cleanup()
        raise OSError(
            "insufficient disk space for mmap probe features: need {} bytes plus 1 GiB, free {}".format(
                feature_bytes, disk_free,
            )
        )
    print(json.dumps({
        "event": "probe_feature_store",
        "backend": "mmap",
        "bytes": feature_bytes,
        "disk_free_bytes": disk_free,
        "directory": feature_cache.name,
    }), flush=True)
    extracted = {
        name: extract(
            model, examples, vocabulary, sources, board_map, hand_names, device,
            config.max_seq_len, move_encoding, args.batch_size, state_prompt_mode,
            feature_cache.name, name, amp_dtype, args.length_bucket_pool_batches,
            evaluation_annotation_mode,
        )
        for name, examples in raw.items()
    }
    majority = majority_predictions(extracted["train"][1], extracted["evaluation"][1].board.shape[0])
    result = {
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "move_encoding": move_encoding,
        "state_prompt_mode": state_prompt_mode,
        "start_selection": start_selection,
        "evaluation_input_annotation_mode": evaluation_annotation_mode,
        "evaluation_input_rap": False,
        "oracle_piece_conditioned": evaluation_annotation_mode == "ap",
        "split_game_counts": split_game_counts,
        "game_splits_disjoint": True,
        "settings": vars(args),
        "history_distances": list(args.history_distances),
        "sources": sources,
        "amp": amp_name,
        "state_metric_definition": {
            "version": "state_probe_metrics_v2_slot_macro",
            "board_macro_f1": "macro-F1 over the 29-class label universe, excluding classes with zero target support in the reported split",
            "hand_count_macro_f1": "unweighted mean of 14 owner-by-piece slot macro-F1 values; each slot excludes count classes with zero target support in the reported split",
            "hand_count_pooled_macro_f1": "legacy diagnostic computed after flattening all 14 hand slots; not a primary metric",
        },
        "alignment": alignment,
        "causal_prefix_full_alignment": causal_alignment,
        "majority_baseline": state_metrics(extracted["evaluation"][1], *majority),
        "probe_results": {},
    }
    states = {}
    for source_index, source in enumerate(sources):
        probe, best_loss = fit_probe(
            extracted["train"][0][source], extracted["train"][1],
            extracted["validation"][0][source], extracted["validation"][1],
            config.d_model, args, device, args.seed + source_index,
        )
        evaluation_x, evaluation_y, evaluation_metadata = extracted["evaluation"]
        result["probe_results"][source] = {
            "best_validation_loss": best_loss,
            "validation": metric(probe, extracted["validation"][0][source], extracted["validation"][1], device),
            "evaluation": metric(probe, evaluation_x[source], evaluation_y, device),
            "evaluation_by_position_scope": metrics_by_group(
                probe, evaluation_x[source], evaluation_y, evaluation_metadata["position_scope"], device,
            ),
            "evaluation_by_trajectory_scope": metrics_by_group(
                probe, evaluation_x[source], evaluation_y, evaluation_metadata["trajectory_scope"], device,
            ),
            "evaluation_by_history_distance": metrics_by_group(
                probe, evaluation_x[source], evaluation_y, evaluation_metadata["history_distance"], device,
            ),
            "evaluation_by_history_distance_and_position_scope": metrics_by_history_distance_and_group(
                probe, evaluation_x[source], evaluation_y,
                evaluation_metadata["history_distance"], evaluation_metadata["position_scope"], device,
            ),
            "evaluation_by_history_distance_and_trajectory_scope": metrics_by_history_distance_and_group(
                probe, evaluation_x[source], evaluation_y,
                evaluation_metadata["history_distance"], evaluation_metadata["trajectory_scope"], device,
            ),
        }
        states[source] = probe.cpu().state_dict()
    (output / "probe_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"checkpoint": args.checkpoint, "sources": sources, "probe_state_dicts": states, "board_label_map": board_map, "hand_names": hand_names}, output / "linear_probes.pt")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    feature_cache.cleanup()


if __name__ == "__main__": main()
