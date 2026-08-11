#!/usr/bin/env python3
"""factorized_v3の層別指し手probe。

状態probeとは分け，実戦で選ばれた指し手の構成要素が各層から線形に
復号できるかを測る。合法集合の評価は元モデルのlogitを用いる
``evaluate_factorized_moves.py``に任せる。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import load_vocabulary
from factorized_prompt import BASIC_PIECE_TOKENS, DROP_TOKEN, MOVE_ENCODING, PROMOTE_TOKEN, factorize_usi
from models import ModelConfig, build_model
from train_model import amp_context, resolve_amp


TASK_SPECS = {
    "actual_move_kind": ("pre", 2),
    "actual_source": ("pre", 81),
    "drop_available": ("pre", 2),
    "actual_destination_nonpromote": ("src", 81),
    "actual_destination_promote": ("promote", 81),
    "actual_promote": ("src", 2),
    "actual_promote_optional": ("src", 2),
    "actual_drop_piece": ("drop", 7),
    "actual_drop_destination": ("piece", 81),
}


def parse_args():
    parser = argparse.ArgumentParser(description="factorized_v3層別指し手probe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", default="layers", help="layers,final,layer_0,...")
    parser.add_argument("--history-distances", default="8,32")
    parser.add_argument("--max-train-examples", type=int, default=12000)
    parser.add_argument("--max-validation-examples", type=int, default=3000)
    parser.add_argument("--max-evaluation-examples", type=int, default=5000)
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="入力系列の上限。超えるqueryは作らない")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--length-bucket-pool-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    return torch.device(value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def parse_distances(value: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in str(value).split(",") if item.strip()))
    if not values or min(values) < 0:
        raise ValueError("history distances must be nonnegative")
    return values


def expand_sources(value: str, n_layers: int) -> list[str]:
    result: list[str] = []
    for item in (part.strip() for part in value.split(",")):
        if item == "layers":
            result.extend("layer_{}".format(index) for index in range(n_layers + 1))
        elif item == "final":
            result.append("layer_{}".format(n_layers))
        elif item.startswith("layer_"):
            index = int(item.split("_", 1)[1])
            if not 0 <= index <= n_layers:
                raise ValueError("layer source is outside model range: {}".format(item))
            result.append(item)
        else:
            raise ValueError("unknown source: {}".format(item))
    return list(dict.fromkeys(result))


def _target_index(token: str, vocabulary: Mapping[str, int], tokens: Sequence[str]) -> int:
    try:
        return list(tokens).index(token)
    except ValueError as exc:
        raise ValueError("token is not in target vocabulary: {}".format(token)) from exc


def _square_index(token: str) -> int:
    """<SQ_7f>を，語彙の固定順（file * 9 + rank）へ変換する．"""
    value = str(token)
    if not value.startswith("<SQ_") or not value.endswith(">"):
        raise ValueError("invalid square token: {}".format(token))
    square = value[4:-1]
    if len(square) != 2 or square[0] not in "123456789" or square[1] not in "abcdefghi":
        raise ValueError("invalid square token: {}".format(token))
    return (int(square[0]) - 1) * 9 + "abcdefghi".index(square[1])


def _candidate_state(record: Mapping[str, object], state_prompt_mode: str) -> list[str]:
    candidates = [
        value for value in record.get("start_candidates", [])
        if int(value.get("start_ply", -1)) == 0
    ]
    if len(candidates) != 1:
        raise ValueError("factorized_v3 action probe requires exactly one start_ply=0 candidate")
    if state_prompt_mode == "implicit_initial":
        return []
    return [str(token) for token in candidates[0]["state_prompt_tokens"]]


def _append_query(
    queries: Dict[str, list[dict]],
    task: str,
    tokens: list[str],
    target: int,
    distance: int,
    game_id: str,
    recurrent_start: int,
    limit: int,
    max_seq_len: int,
) -> None:
    if len(queries[task]) >= limit or len(tokens) > max_seq_len:
        return
    queries[task].append({
        "tokens": tokens,
        "target": int(target),
        "distance": int(distance),
        "game_id": game_id,
        "recurrent_start": recurrent_start,
    })


def _drop_available_by_ply(record: Mapping[str, object], move_count: int) -> list[bool] | None:
    """データセットに保存した合法駒打ちラベルを取得する．

    factorized_v3の新artifactでは全splitに列を保存する。旧artifactを読む場合は
    evaluation_stepsから復元できるときだけfallbackし，学習側で曖昧なラベルを
    作らない。
    """
    raw = record.get("legal_drop_available_by_ply")
    if isinstance(raw, list) and len(raw) == move_count:
        return [bool(value) for value in raw]
    steps = record.get("evaluation_steps")
    if isinstance(steps, list) and len(steps) == move_count:
        result: list[bool] = []
        for step in steps:
            legal_moves = step.get("legal_moves", []) if isinstance(step, Mapping) else []
            result.append(any("*" in str(move) for move in legal_moves))
        return result
    return None


def _promotion_choice_available_by_ply(record: Mapping[str, object], move_count: int) -> list[bool] | None:
    raw = record.get("promotion_choice_available_by_ply")
    if isinstance(raw, list) and len(raw) == move_count:
        return [bool(value) for value in raw]
    steps = record.get("evaluation_steps")
    if isinstance(steps, list) and len(steps) == move_count:
        result: list[bool] = []
        for step in steps:
            if not isinstance(step, Mapping):
                result.append(False)
                continue
            legal_moves = {str(move) for move in step.get("legal_moves", [])}
            target_move = str(step.get("target_move", ""))
            base_move = target_move[:4]
            result.append(
                "*" not in target_move
                and len(base_move) == 4
                and base_move in legal_moves
                and base_move + "+" in legal_moves
            )
        return result
    return None


def assert_disjoint_game_ids(paths: Mapping[str, str | Path]) -> dict[str, int]:
    """probeの学習・検証・評価間で対局が重複していないことを確認する．"""
    game_ids: dict[str, set[str]] = {}
    for split, path in paths.items():
        values: set[str] = set()
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


def read_queries(
    path: str | Path,
    vocabulary: Mapping[str, int],
    state_prompt_mode: str,
    distances: Sequence[int],
    max_examples: int,
    max_seq_len: int,
) -> Dict[str, list[dict]]:
    queries = {task: [] for task in TASK_SPECS}
    wanted = set(distances)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or all(len(values) >= max_examples for values in queries.values()):
                continue
            record = json.loads(line)
            base_state = _candidate_state(record, state_prompt_mode)
            base = ["<BOS>", *base_state, "<MOVES>"]
            recurrent_start = len(base)
            history: list[str] = []
            moves = [str(value) for value in record.get("move_tokens", [])]
            game_id = str(record.get("game_id", ""))
            drop_available_by_ply = _drop_available_by_ply(record, len(moves))
            promotion_choice_available_by_ply = _promotion_choice_available_by_ply(record, len(moves))
            for ply, move in enumerate(moves):
                parts = factorize_usi(move)
                if ply in wanted:
                    prefix = base + history
                    if parts[0] == DROP_TOKEN:
                        piece_index = _target_index(parts[1], vocabulary, BASIC_PIECE_TOKENS)
                        _append_query(queries, "actual_move_kind", prefix, 1, ply, game_id, recurrent_start, max_examples, max_seq_len)
                        if drop_available_by_ply is not None:
                            _append_query(queries, "drop_available", prefix, int(drop_available_by_ply[ply]), ply, game_id, recurrent_start, max_examples, max_seq_len)
                        _append_query(queries, "actual_drop_piece", prefix + [DROP_TOKEN], piece_index, ply, game_id, recurrent_start, max_examples, max_seq_len)
                        _append_query(queries, "actual_drop_destination", prefix + [DROP_TOKEN, parts[1]], _square_index(parts[2]), ply, game_id, recurrent_start, max_examples, max_seq_len)
                    else:
                        source = _square_index(parts[0])
                        destination = _square_index(parts[-1])
                        _append_query(queries, "actual_move_kind", prefix, 0, ply, game_id, recurrent_start, max_examples, max_seq_len)
                        if drop_available_by_ply is not None:
                            _append_query(queries, "drop_available", prefix, int(drop_available_by_ply[ply]), ply, game_id, recurrent_start, max_examples, max_seq_len)
                        _append_query(queries, "actual_source", prefix, source, ply, game_id, recurrent_start, max_examples, max_seq_len)
                        _append_query(queries, "actual_promote", prefix + [parts[0]], int(len(parts) == 3 and parts[1] == PROMOTE_TOKEN), ply, game_id, recurrent_start, max_examples, max_seq_len)
                        if promotion_choice_available_by_ply is not None and promotion_choice_available_by_ply[ply]:
                            _append_query(queries, "actual_promote_optional", prefix + [parts[0]], int(len(parts) == 3 and parts[1] == PROMOTE_TOKEN), ply, game_id, recurrent_start, max_examples, max_seq_len)
                        if len(parts) == 3:
                            _append_query(queries, "actual_destination_promote", prefix + [parts[0], PROMOTE_TOKEN], destination, ply, game_id, recurrent_start, max_examples, max_seq_len)
                        else:
                            _append_query(queries, "actual_destination_nonpromote", prefix + [parts[0]], destination, ply, game_id, recurrent_start, max_examples, max_seq_len)
                    # 既存の学習・評価スクリプトと同様，評価対象はRAPなし系列である。
                history.extend(parts)
    return queries


def _pad_batch(batch: Sequence[dict], vocabulary: Mapping[str, int], device: torch.device):
    lengths_cpu = torch.tensor([len(item["tokens"]) for item in batch], dtype=torch.long)
    width = int(lengths_cpu.max())
    pad_id = int(vocabulary["<PAD>"])
    ids_cpu = torch.full((len(batch), width), pad_id, dtype=torch.long)
    recurrent_cpu = torch.zeros((len(batch), width), dtype=torch.bool)
    for row, item in enumerate(batch):
        ids = torch.tensor([vocabulary[token] for token in item["tokens"]], dtype=torch.long)
        ids_cpu[row, : len(ids)] = ids
        recurrent_cpu[row, item["recurrent_start"] : len(ids)] = True
    attention_cpu = torch.arange(width)[None, :] < lengths_cpu[:, None]
    return (
        ids_cpu.to(device, non_blocking=device.type == "cuda"),
        attention_cpu.to(device, non_blocking=device.type == "cuda"),
        recurrent_cpu.to(device, non_blocking=device.type == "cuda"),
        lengths_cpu.to(device, non_blocking=device.type == "cuda"),
    )


def extract_features(model, queries, vocabulary, sources, device, batch_size, amp_dtype, pool_batches, progress, label):
    feature_chunks = {source: [] for source in sources}
    if not queries:
        return (
            {source: torch.empty((0, model.config.d_model)) for source in sources},
            torch.empty((0,), dtype=torch.long),
        )
    ordered = sorted(queries, key=lambda item: len(item["tokens"])) if pool_batches > 1 else list(queries)
    started = time.perf_counter()
    with torch.inference_mode(), amp_context(device, amp_dtype):
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            ids, attention, recurrent, lengths = _pad_batch(batch, vocabulary, device)
            output = model(ids, attention_mask=attention, recurrent_mask=recurrent, output_hidden_states=True)
            rows = torch.arange(len(batch), device=device)
            positions = lengths - 1
            for source in sources:
                layer = int(source.split("_", 1)[1])
                feature_chunks[source].append(output.hidden_states[layer][rows, positions].float().cpu())
            if progress and (start + len(batch)) % progress < len(batch):
                print(json.dumps({"event": "action_probe_feature_progress", "task": label, "queries": start + len(batch), "total": len(ordered), "elapsed_sec": round(time.perf_counter() - started, 1)}), flush=True)
    features = {source: torch.cat(chunks, dim=0) for source, chunks in feature_chunks.items()}
    labels = torch.tensor([item["target"] for item in ordered], dtype=torch.long)
    return features, labels


def _macro_metrics(prediction: torch.Tensor, target: torch.Tensor, classes: int) -> dict:
    confusion = torch.zeros((classes, classes), dtype=torch.long)
    for expected, actual in zip(target.tolist(), prediction.tolist()):
        confusion[int(expected), int(actual)] += 1
    recalls, precisions, f1s = [], [], []
    for index in range(classes):
        tp = float(confusion[index, index])
        support = float(confusion[index].sum())
        predicted = float(confusion[:, index].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            recalls.append(recall); precisions.append(precision); f1s.append(f1)
    return {
        "accuracy": float((prediction == target).float().mean()),
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "support": [int(value) for value in confusion.sum(dim=1)],
    }


def classification_metrics(logits: torch.Tensor, target: torch.Tensor, classes: int) -> dict:
    top_k = min(5, classes)
    top = logits.topk(top_k, dim=-1).indices
    result = _macro_metrics(logits.argmax(dim=-1), target, classes)
    result["top5_accuracy"] = float((top == target[:, None]).any(dim=1).float().mean())
    return result


def fit_head(train_x, train_y, valid_x, valid_y, classes, args, device, seed):
    torch.manual_seed(seed)
    head = nn.Linear(train_x.shape[-1], classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, generator=generator)
    best_state, best_loss, wait = None, float("inf"), 0
    for _ in range(args.probe_epochs):
        head.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(features.to(device)), labels.to(device))
            loss.backward(); optimizer.step()
        head.eval()
        with torch.inference_mode():
            valid_loss = float(nn.functional.cross_entropy(head(valid_x.to(device)), valid_y.to(device)))
        if valid_loss < best_loss - 1e-5:
            best_loss, wait = valid_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("action probe did not produce a checkpoint")
    head.load_state_dict(best_state)
    return head, best_loss


def evaluate_head(head, features, labels, classes, device):
    head.eval()
    with torch.inference_mode():
        logits = head(features.to(device)).cpu()
    return classification_metrics(logits, labels, classes)


def majority_baseline(train_y, eval_y, classes):
    majority = int(torch.bincount(train_y, minlength=classes).argmax())
    return {"majority_class": majority, **_macro_metrics(torch.full_like(eval_y, majority), eval_y, classes)}


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    vocabulary = load_vocabulary(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    settings = checkpoint.get("new_prompt", {})
    if settings.get("move_encoding") != MOVE_ENCODING:
        raise ValueError("checkpoint is not factorized_v3_no_eom")
    state_prompt_mode = str(settings.get("state_prompt_mode", "implicit_initial"))
    if str(settings.get("start_selection", "fixed_initial")) != "fixed_initial":
        raise ValueError("action probe requires fixed_initial checkpoint")
    config = ModelConfig(**checkpoint["config"])
    device = resolve_device(args.device)
    amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    model = build_model(str(checkpoint.get("model_type", "vanilla")), config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sources = expand_sources(args.sources, config.n_layers)
    distances = parse_distances(args.history_distances)
    limits = {"train": args.max_train_examples, "validation": args.max_validation_examples, "evaluation": args.max_evaluation_examples}
    paths = {"train": args.train_jsonl, "validation": args.validation_jsonl, "evaluation": args.evaluation_jsonl}
    split_game_counts = assert_disjoint_game_ids(paths)
    all_queries = {
        split: read_queries(path, vocabulary, state_prompt_mode, distances, limits[split], args.max_seq_len)
        for split, path in paths.items()
    }
    print(json.dumps({"event": "action_probe_query_counts", "counts": {split: {task: len(values) for task, values in tasks.items()} for split, tasks in all_queries.items()}}, ensure_ascii=False), flush=True)
    result = {"format_version": 1, "checkpoint": args.checkpoint, "settings": vars(args), "state_prompt_mode": state_prompt_mode, "evaluation_input_rap": False, "split_game_counts": split_game_counts, "game_splits_disjoint": True, "tasks": {}}
    saved = {"checkpoint": args.checkpoint, "sources": sources, "tasks": {}}
    for task, (_, classes) in TASK_SPECS.items():
        result["tasks"][task] = {}
        saved["tasks"][task] = {}
        # タスク単位で特徴量を抽出・学習・解放する。全タスク×全層の特徴を同時に
        # 保持すると，largeモデルでCPUメモリを不必要に占有するためである。
        task_features = {}
        task_labels = {}
        for split in ("train", "validation", "evaluation"):
            queries = all_queries[split][task]
            task_features[split], task_labels[split] = extract_features(
                model, queries, vocabulary, sources, device, args.batch_size,
                amp_dtype, args.length_bucket_pool_batches, args.progress_every,
                "{}:{}".format(split, task),
            )
        train_y = task_labels["train"]
        valid_y = task_labels["validation"]
        eval_y = task_labels["evaluation"]
        if min(len(train_y), len(valid_y), len(eval_y)) == 0:
            result["tasks"][task] = {"status": "unavailable", "counts": {split: len(task_labels[split]) for split in task_labels}}
            del task_features, task_labels
            continue
        for source in sources:
            head, validation_loss = fit_head(task_features["train"][source], train_y, task_features["validation"][source], valid_y, classes, args, device, args.seed)
            result["tasks"][task][source] = {
                "train_examples": len(train_y), "validation_examples": len(valid_y), "evaluation_examples": len(eval_y),
                "validation_loss": validation_loss,
                "evaluation": evaluate_head(head, task_features["evaluation"][source], eval_y, classes, device),
                "majority_baseline": majority_baseline(train_y, eval_y, classes),
            }
            saved["tasks"][task][source] = head.state_dict()
        del task_features, task_labels
    del model, checkpoint, all_queries
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "action_probe_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(saved, output / "action_probes.pt")
    print(json.dumps({"event": "action_probe_complete", "output": str(output), "tasks": list(result["tasks"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
