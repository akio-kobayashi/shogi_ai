#!/usr/bin/env python3
"""均衡化した王手状態集合に対する層別線形probe。"""

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
from torch import nn

from data import ShogiSequenceDataset, load_vocabulary
from evaluate_probes import expand_sources, load_backbone, resolve_device


class CheckProbeDataset(ShogiSequenceDataset):
    """各レコードの末尾局面に付いたin_checkラベルを返す。"""

    def __init__(self, path: str, vocabulary: Mapping[str, int]):
        super().__init__(path, vocabulary)
        self.check_labels = []
        for index, record in enumerate(self.records):
            if "in_check" not in record:
                raise ValueError("{} record {} has no in_check label".format(path, index))
            self.check_labels.append(int(bool(record["in_check"])))

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        item["in_check"] = self.check_labels[index]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="王手状態を隠れ表現から線形復号する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", default="final,recurrent,token_embedding")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def extract_features(model, model_type: str, dataset, sources: Sequence[str], device):
    """同一棋譜からの複数状態は一度のforwardでまとめて復号する。

    check用JSONLはstate単位だが，同じsource_game_idの各行は同一開始局面からの
    指手prefixである。各prefixを個別に再生すると，特にT²MLRの逐次評価が必要以上に
    遅くなるため，最長prefixを一度だけforwardして途中位置を取り出す。
    """
    features = {source: [] for source in sources}
    labels = []
    grouped = {}
    for record in dataset.records:
        key = (str(record.get("source_game_id", record["game_id"])), str(record.get("initial_sfen", "")))
        grouped.setdefault(key, []).append(record)

    with torch.inference_mode():
        for group_index, records in enumerate(grouped.values(), 1):
            records.sort(key=lambda record: len(record["move_tokens"]))
            longest = records[-1]
            longest_moves = list(longest["move_tokens"])
            for record in records[:-1]:
                prefix = list(record["move_tokens"])
                if longest_moves[: len(prefix)] != prefix:
                    raise ValueError(
                        "check probe records with the same source_game_id are not prefixes"
                    )
            example = dataset._encode_record(longest)
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
            for record in records:
                # <BOS> + 96状態token + <MOVES>の後が第1指手である。
                position = 97 + len(record["move_tokens"])
                for source in sources:
                    if source == "token_embedding":
                        value = model.token_embedding(input_ids)[0, position]
                    elif source == "recurrent":
                        if output.recurrent_states is None:
                            raise ValueError("model has no recurrent state")
                        value = output.recurrent_states[0, position]
                    else:
                        layer = int(source.split("_", 1)[1])
                        value = output.hidden_states[layer][0, position]
                    features[source].append(value.detach().cpu())
                labels.append(int(bool(record["in_check"])))
            if group_index % 100 == 0 or group_index == len(grouped):
                print(
                    "check_feature_extract_progress games={}/{} states={}".format(
                        group_index, len(grouped), len(labels)
                    ),
                    flush=True,
                )
    return (
        {source: torch.stack(values) for source, values in features.items()},
        torch.tensor(labels, dtype=torch.long),
    )


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    prediction = logits.argmax(dim=-1).cpu()
    truth = labels.cpu()
    positive = truth == 1
    predicted_positive = prediction == 1
    true_positive = int((positive & predicted_positive).sum())
    false_positive = int((~positive & predicted_positive).sum())
    false_negative = int((positive & ~predicted_positive).sum())
    true_negative = int((~positive & ~predicted_positive).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        "samples": int(truth.numel()),
        "in_check_rate": float(positive.float().mean()),
        "accuracy": float((prediction == truth).float().mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "balanced_accuracy": (recall + specificity) / 2,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def evaluate_linear(probe: nn.Module, features: torch.Tensor, labels: torch.Tensor, batch_size: int, device):
    outputs = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            outputs.append(probe(features[start : start + batch_size].to(device)).cpu())
    logits = torch.cat(outputs)
    metrics = binary_metrics(logits, labels)
    metrics["cross_entropy"] = float(nn.functional.cross_entropy(logits, labels))
    return metrics, logits


def train_probe(train_x, train_y, validation_x, validation_y, args, device):
    probe = nn.Linear(train_x.shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        probe.train()
        order = torch.randperm(train_x.shape[0], generator=generator)
        loss_sum = 0.0
        for start in range(0, train_x.shape[0], args.batch_size):
            indices = order[start : start + args.batch_size]
            logits = probe(train_x[indices].to(device))
            loss = nn.functional.cross_entropy(logits, train_y[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        probe.eval()
        validation, _ = evaluate_linear(probe, validation_x, validation_y, args.batch_size, device)
        train_loss = loss_sum / max(train_x.shape[0], 1)
        history.append({"epoch": epoch, "training_loss": train_loss, "validation_loss": validation["cross_entropy"], "validation_f1": validation["f1"]})
        if validation["cross_entropy"] < best_loss - 1e-7:
            best_loss = validation["cross_entropy"]
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise AssertionError("check probe did not train")
    probe.load_state_dict(best_state)
    return probe.eval(), {"best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history}


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    vocabulary = load_vocabulary(args.vocab)
    model, model_type, config = load_backbone(args.checkpoint, device, False)
    has_recurrent = model_type in {"t2mlr", "t^2mlr", "t²mlr"}
    sources = expand_sources(args.sources, config.n_layers, has_recurrent)
    datasets = {
        "train": CheckProbeDataset(args.train_jsonl, vocabulary),
        "validation": CheckProbeDataset(args.validation_jsonl, vocabulary),
        "evaluation": CheckProbeDataset(args.evaluation_jsonl, vocabulary),
    }
    extracted = {}
    for name, dataset in datasets.items():
        print("check_feature_extract split={} examples={}".format(name, len(dataset)), flush=True)
        extracted[name] = extract_features(model, model_type, dataset, sources, device)

    # 学習集合の多数派を常に出す対照。均衡化された評価では通常ほぼ50%となる。
    majority = int(extracted["train"][1].mode().values)
    evaluation_labels = extracted["evaluation"][1]
    baseline_logits = torch.zeros((evaluation_labels.numel(), 2))
    baseline_logits[:, majority] = 1.0
    report: Dict[str, object] = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "model_type": model_type,
        "sources": sources,
        "settings": {"seed": args.seed, "device": str(device), "balanced_dataset_expected": True},
        "majority_baseline": binary_metrics(baseline_logits, evaluation_labels),
        "probe_results": {},
    }
    saved = {}
    for source in sources:
        probe, training = train_probe(
            extracted["train"][0][source], extracted["train"][1],
            extracted["validation"][0][source], extracted["validation"][1], args, device,
        )
        validation, _ = evaluate_linear(probe, extracted["validation"][0][source], extracted["validation"][1], args.batch_size, device)
        evaluation, _ = evaluate_linear(probe, extracted["evaluation"][0][source], extracted["evaluation"][1], args.batch_size, device)
        report["probe_results"][source] = {
            "training": training,
            "validation": validation,
            "evaluation": evaluation,
            "evaluation_minus_majority": {
                name: float(evaluation[name]) - float(report["majority_baseline"][name])
                for name in ("accuracy", "precision", "recall", "f1", "balanced_accuracy")
            },
        }
        saved[source] = {key: value.detach().cpu() for key, value in probe.state_dict().items()}

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "check_probe_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"checkpoint": str(args.checkpoint), "sources": sources, "probe_state_dicts": saved}, output / "check_linear_probes.pt")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
