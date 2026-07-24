#!/usr/bin/env python3
"""通常棋譜の次手予測とCoT-like SFTを同じdecoderで学習する。"""

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader

from cot_data import ReasoningTraceDataset, collate_reasoning_traces
from data import (
    RandomStartSequenceDataset,
    causal_lm_loss,
    collate_sequences,
    load_vocabulary,
    weighted_causal_lm_loss,
)
from models import (
    ModelConfig,
    T2MLRConfig,
    build_model,
    parameter_matched_vanilla_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将棋decoderの次手予測pretrainingまたはCoT-like SFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=("pretrain", "cot"), required=True)
    parser.add_argument("--model-type", choices=("vanilla", "t2mlr"), default="vanilla")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--max-seq-len", type=int, default=640)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--l-start", type=int, default=3)
    parser.add_argument("--l-end", type=int, default=4)
    parser.add_argument("--jacobi-depth", type=int, default=4)
    parser.add_argument(
        "--match-t2mlr",
        action="store_true",
        help="VanillaのFFN幅を同じ設定のT²MLRとparameter-matchedにする",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=40)
    parser.add_argument("--min-suffix-moves", type=int, default=40)
    parser.add_argument("--answer-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def config_from_args(args: argparse.Namespace, vocab_size: int):
    common = dict(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    if args.model_type == "t2mlr":
        return T2MLRConfig(
            **common,
            l_start=args.l_start,
            l_end=args.l_end,
            jacobi_depth=args.jacobi_depth,
        )
    if args.match_t2mlr:
        reference = T2MLRConfig(
            **common,
            l_start=args.l_start,
            l_end=args.l_end,
            jacobi_depth=args.jacobi_depth,
        )
        return parameter_matched_vanilla_config(reference)
    return ModelConfig(**common)


def load_initial_model(
    args: argparse.Namespace,
    vocab_size: int,
    device: torch.device,
):
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("init checkpoint must be a mapping")
        checkpoint_type = str(payload.get("model_type", "")).lower()
        if checkpoint_type != args.model_type:
            raise ValueError(
                "checkpoint model_type {} != requested {}".format(
                    checkpoint_type, args.model_type
                )
            )
        config_payload = payload.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError("init checkpoint requires config")
        config_dict = dict(config_payload)
        if int(config_dict["vocab_size"]) != vocab_size:
            raise ValueError(
                "checkpoint vocabulary size {} != current {}".format(
                    config_dict["vocab_size"], vocab_size
                )
            )
        config = (
            T2MLRConfig(**config_dict)
            if args.model_type == "t2mlr"
            else ModelConfig(**config_dict)
        )
        model = build_model(args.model_type, config)
        state = payload.get("model_state_dict", payload.get("state_dict"))
        if state is None:
            raise ValueError("init checkpoint requires model_state_dict")
        model.load_state_dict(state)
    else:
        if args.stage == "cot":
            raise ValueError("CoT SFT requires --init-checkpoint")
        config = config_from_args(args, vocab_size)
        model = build_model(args.model_type, config)
    return model.to(device)


def build_datasets(args: argparse.Namespace, vocabulary):
    if args.stage == "cot":
        train = ReasoningTraceDataset(
            args.train_jsonl, vocabulary, answer_weight=args.answer_weight
        )
        validation = ReasoningTraceDataset(
            args.validation_jsonl, vocabulary, answer_weight=args.answer_weight
        )
        collate = partial(
            collate_reasoning_traces,
            pad_token_id=vocabulary["<PAD>"],
            max_seq_len=args.max_seq_len,
        )
    else:
        train = RandomStartSequenceDataset(
            args.train_jsonl,
            vocabulary,
            candidate_count=args.candidate_count,
            min_suffix_moves=args.min_suffix_moves,
            seed=args.seed,
            randomize_each_epoch=True,
        )
        validation = RandomStartSequenceDataset(
            args.validation_jsonl,
            vocabulary,
            candidate_count=args.candidate_count,
            min_suffix_moves=args.min_suffix_moves,
            seed=args.seed + 1,
            randomize_each_epoch=False,
        )
        collate = partial(
            collate_sequences,
            pad_token_id=vocabulary["<PAD>"],
            max_seq_len=args.max_seq_len,
        )
    return train, validation, collate


def batch_loss(model, batch, stage: str, device: torch.device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    recurrent_mask = batch["recurrent_mask"].to(device)
    labels = batch["labels"].to(device)
    output = model(
        input_ids,
        attention_mask=attention_mask,
        recurrent_mask=recurrent_mask,
    )
    if stage == "cot":
        return weighted_causal_lm_loss(
            output.logits,
            labels,
            batch["loss_weights"].to(device),
        )
    return causal_lm_loss(output.logits, labels)


def evaluate(model, loader, stage: str, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            loss_sum += float(batch_loss(model, batch, stage, device))
            batches += 1
    if not batches:
        raise ValueError("validation loader is empty")
    return loss_sum / batches


def save_checkpoint(path: Path, model, args: argparse.Namespace, epoch: int, step: int):
    torch.save(
        {
            "model_type": args.model_type,
            "config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "stage": args.stage,
            "epoch": epoch,
            "step": step,
            "seed": args.seed,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    vocabulary = load_vocabulary(args.vocab)
    model = load_initial_model(args, len(vocabulary), device)
    # checkpointの設定を優先した場合、collateの上限も実モデルへ合わせる。
    args.max_seq_len = model.config.max_seq_len
    train_dataset, validation_dataset, collate = build_datasets(args, vocabulary)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    global_step = 0
    history = []
    stop = False

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        training_sum = 0.0
        training_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, args.stage, device)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
            optimizer.step()
            global_step += 1
            training_sum += float(loss.detach())
            training_batches += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

        validation_loss = evaluate(
            model, validation_loader, args.stage, device
        )
        training_loss = training_sum / max(training_batches, 1)
        row = {
            "epoch": epoch,
            "step": global_step,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_perplexity": math.exp(min(validation_loss, 20.0)),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        save_checkpoint(output_dir / "last.pt", model, args, epoch, global_step)
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(output_dir / "best.pt", model, args, epoch, global_step)
        if stop:
            break

    with (output_dir / "training_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "stage": args.stage,
                "model_type": args.model_type,
                "best_validation_loss": best_loss,
                "history": history,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
