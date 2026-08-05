#!/usr/bin/env python3
"""新しい状態prompt artifactだけを使うVanilla decoderの学習器。"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import IGNORE_INDEX, load_vocabulary
from models import ModelConfig, build_model
from new_prompt_data import ANNOTATION_MODES, NewPromptSequenceDataset, collate_new_prompt_sequences
from train_model import amp_context, resolve_amp, resolve_device

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # setup前の最小環境でもJSONログは利用可能にする。
    SummaryWriter = None


MODEL_SIZES = {
    "small": {"d_model": 384, "n_layers": 12, "n_heads": 12, "d_ff": 1536},
    "base": {"d_model": 576, "n_layers": 12, "n_heads": 12, "d_ff": 2304},
    "large": {"d_model": 720, "n_layers": 12, "n_heads": 12, "d_ff": 2880},
}

# VanillaのGELU FFNは2投影，SwiGLU FFNは3投影である。d_ffを2/3にして，
# LLaMA型と既存Vanillaの総パラメータ数をほぼ揃える。
LLAMA_MODEL_SIZES = {
    "small": {"d_model": 384, "n_layers": 12, "n_heads": 12, "d_ff": 1024},
    "base": {"d_model": 576, "n_layers": 12, "n_heads": 12, "d_ff": 1536},
    "large": {"d_model": 720, "n_layers": 12, "n_heads": 12, "d_ff": 1920},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="新prompt用Vanilla decoderを学習する")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--dataset-manifest", help="検証済みdataset_manifest.json。指定時はhashもrun manifestへ記録する")
    parser.add_argument("--output-dir", required=True)
    # 既存呼出しとの互換性のため，既定値は従来どおりvanillaに保つ。
    parser.add_argument("--model-type", choices=("vanilla", "llama"), default="vanilla")
    parser.add_argument("--resume", action="store_true", help="output-dir/last.ptからモデルとoptimizerを再開する")
    parser.add_argument("--model-size", choices=tuple(MODEL_SIZES), required=True)
    parser.add_argument("--annotation-mode", choices=ANNOTATION_MODES, default="vanilla")
    parser.add_argument("--annotation-probability", type=float, default=0.0)
    parser.add_argument("--hint-loss-weight", type=float, default=1.0)
    # 最大開始prompt約90 token，512指手，320個の2-token注釈を含めても
    # 90 + 512 + 2*320 = 1242 tokenであり，1280 token文脈に収まる。
    parser.add_argument("--max-hints", type=int, default=320)
    parser.add_argument("--max-moves", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=1280)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--tensorboard-dir", help="TensorBoard event出力先。既定はOUTPUT_DIR/tensorboard")
    parser.add_argument("--no-tensorboard", action="store_true", help="TensorBoard eventを書き出さない")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def loss_for_batch(model, batch, device):
    output = model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        recurrent_mask=batch["recurrent_mask"].to(device),
    )
    labels = batch["labels"].to(device)
    weights = batch["loss_weights"].to(device)
    move_mask = batch["move_target_mask"].to(device)
    hint_mask = batch["hint_target_mask"].to(device)
    per_token = F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(labels)
    active = labels != IGNORE_INDEX
    effective_weights = torch.where(active, weights.to(per_token.dtype), 0.0)
    weight_sum = effective_weights.sum()
    if not bool(weight_sum > 0):
        raise ValueError("batch has no active prediction targets")
    loss = (per_token * effective_weights).sum() / weight_sum

    # 注釈の有無が平均lossに埋もれないよう，重み付け前のCEと実トークン数を返す。
    # move_mask/hint_maskはlabelsと同じ位置で，それぞれ排他的に定義される。
    def totals(mask):
        count = mask.sum()
        return {
            "nll_sum": per_token.masked_select(mask).sum().detach(),
            "targets": count.detach(),
        }

    return loss, {
        "combined_nll_sum": (per_token * effective_weights).sum().detach(),
        "combined_weight": weight_sum.detach(),
        "move": totals(move_mask),
        "hint": totals(hint_mask),
    }


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    totals = {"combined_nll_sum": 0.0, "combined_weight": 0.0, "move_nll_sum": 0.0, "move_targets": 0, "hint_nll_sum": 0.0, "hint_targets": 0}
    with torch.inference_mode():
        for batch in loader:
            with amp_context(device, amp_dtype):
                _, metrics = loss_for_batch(model, batch, device)
            totals["combined_nll_sum"] += float(metrics["combined_nll_sum"])
            totals["combined_weight"] += float(metrics["combined_weight"])
            totals["move_nll_sum"] += float(metrics["move"]["nll_sum"])
            totals["move_targets"] += int(metrics["move"]["targets"])
            totals["hint_nll_sum"] += float(metrics["hint"]["nll_sum"])
            totals["hint_targets"] += int(metrics["hint"]["targets"])
    return {
        "loss": totals["combined_nll_sum"] / max(1.0, totals["combined_weight"]),
        "move_cross_entropy": totals["move_nll_sum"] / max(1, totals["move_targets"]),
        "hint_cross_entropy": None if not totals["hint_targets"] else totals["hint_nll_sum"] / totals["hint_targets"],
        "move_targets": totals["move_targets"],
        "hint_targets": totals["hint_targets"],
    }


def write_step_scalars(writer: Optional["SummaryWriter"], step: int, loss, metrics, batch, device) -> None:
    """高頻度ログ。注釈数と系列長をlossと同じstepに残す。"""
    if writer is None:
        return
    writer.add_scalar("train/combined_cross_entropy", float(loss.detach()), step)
    writer.add_scalar("train/move_targets_per_batch", int(metrics["move"]["targets"]), step)
    writer.add_scalar("train/hint_targets_per_batch", int(metrics["hint"]["targets"]), step)
    writer.add_scalar("train/sequence_length", int(batch["input_ids"].shape[1]), step)
    if device.type == "cuda":
        writer.add_scalar("system/memory_allocated_mib", torch.cuda.memory_allocated(device) / 2**20, step)
        writer.add_scalar("system/max_memory_allocated_mib", torch.cuda.max_memory_allocated(device) / 2**20, step)


def save_checkpoint(path, model, optimizer, args, epoch, step, best_loss):
    torch.save({
        "model_type": args.model_type, "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(), "epoch": epoch, "step": step,
        "best_validation_loss": best_loss, "new_prompt": {
            "model_size": args.model_size, "annotation_mode": args.annotation_mode,
            "annotation_probability": args.annotation_probability,
            "hint_loss_weight": args.hint_loss_weight, "max_hints": args.max_hints,
            "max_moves": args.max_moves, "seed": args.seed,
        },
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    if args.annotation_mode == "vanilla" and args.annotation_probability:
        raise ValueError("vanilla requires --annotation-probability 0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    vocabulary = load_vocabulary(args.vocab)
    device = resolve_device(args.device)
    amp_dtype, scaler, amp_name = resolve_amp(args.amp, device)
    size_table = LLAMA_MODEL_SIZES if args.model_type == "llama" else MODEL_SIZES
    config = ModelConfig(vocab_size=len(vocabulary), max_seq_len=args.max_seq_len, dropout=args.dropout, **size_table[args.model_size])
    config.validate()
    model = build_model(args.model_type, config).to(device)
    common = dict(annotation_mode=args.annotation_mode, annotation_probability=args.annotation_probability,
                  hint_loss_weight=args.hint_loss_weight, max_hints=args.max_hints,
                  max_moves=args.max_moves, max_seq_len=args.max_seq_len)
    train_dataset = NewPromptSequenceDataset(args.train_jsonl, vocabulary, seed=args.seed, randomize_each_epoch=True, **common)
    # early stoppingも運用時と同じ，注釈を除いた入力で測る。主比較の制約付き
    # 指手評価はevaluate_new_prompt_moves.pyで別に実行する。
    validation_common = dict(common, annotation_mode="vanilla", annotation_probability=0.0)
    validation_dataset = NewPromptSequenceDataset(args.validation_jsonl, vocabulary, seed=args.seed + 1, randomize_each_epoch=False, **validation_common)
    collate = partial(collate_new_prompt_sequences, pad_token_id=vocabulary["<PAD>"], max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else output / "tensorboard"
    writer = None
    if not args.no_tensorboard:
        if SummaryWriter is None:
            print(json.dumps({"event": "tensorboard_unavailable", "reason": "tensorboard is not installed"}, ensure_ascii=False), flush=True)
        else:
            # resume時も既存eventを消さず，checkpointのglobal step以降へ追記する。
            writer = SummaryWriter(log_dir=str(tensorboard_dir))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = step = 0; best_loss = float("inf")
    if args.resume:
        resume_path = output / "last.pt"
        if not resume_path.is_file():
            raise FileNotFoundError("--resume was specified but last.pt is absent: {}".format(resume_path))
        resume = torch.load(resume_path, map_location=device)
        if str(resume.get("model_type", "vanilla")).lower() != args.model_type:
            raise ValueError("resume checkpoint model_type differs from requested model_type")
        if dict(resume.get("config", {})) != config.to_dict():
            raise ValueError("resume checkpoint model config differs from requested config")
        model.load_state_dict(resume["model_state_dict"])
        if "optimizer_state_dict" in resume: optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume.get("epoch", 0)); step = int(resume.get("step", 0)); best_loss = float(resume.get("best_validation_loss", best_loss))
    run = {"format_version": 1, "args": vars(args), "model_config": config.to_dict(), "parameter_count": sum(p.numel() for p in model.parameters()), "device": str(device), "amp": amp_name, "git_commit": git_commit(), "tensorboard": None if writer is None else str(tensorboard_dir.resolve())}
    if args.dataset_manifest:
        dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
        run["dataset"] = {"manifest": str(Path(args.dataset_manifest).resolve()), "schema_version": dataset_manifest.get("schema_version"), "vocab_sha256": dataset_manifest.get("vocab_sha256"), "splits": dataset_manifest.get("splits")}
    (output / "run_manifest.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if writer is not None:
        writer.add_text("run/config", json.dumps(run, ensure_ascii=False, indent=2), 0)
    print(json.dumps({"event": "data_ready", "train": len(train_dataset), "validation": len(validation_dataset), **run}, ensure_ascii=False), flush=True)
    wait, history = 0, []
    history_path = output / "training_history.json"
    if args.resume and history_path.is_file():
        history = list(json.loads(history_path.read_text(encoding="utf-8")).get("history", []))
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_totals = {"combined_nll_sum": 0.0, "combined_weight": 0.0, "move_nll_sum": 0.0, "move_targets": 0, "hint_nll_sum": 0.0, "hint_targets": 0}
        for batch_index, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_dtype):
                loss, metrics = loss_for_batch(model, batch, device)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            else:
                loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            step += 1
            epoch_totals["combined_nll_sum"] += float(metrics["combined_nll_sum"])
            epoch_totals["combined_weight"] += float(metrics["combined_weight"])
            epoch_totals["move_nll_sum"] += float(metrics["move"]["nll_sum"])
            epoch_totals["move_targets"] += int(metrics["move"]["targets"])
            epoch_totals["hint_nll_sum"] += float(metrics["hint"]["nll_sum"])
            epoch_totals["hint_targets"] += int(metrics["hint"]["targets"])
            if args.progress_every and (step == 1 or step % args.progress_every == 0):
                write_step_scalars(writer, step, loss, metrics, batch, device)
                print(json.dumps({"event": "progress", "epoch": epoch, "step": step, "batch": batch_index, "loss": float(loss.detach()), "move_targets": int(metrics["move"]["targets"]), "hint_targets": int(metrics["hint"]["targets"]), "seq_len": int(batch["input_ids"].shape[1]), "elapsed_sec": round(time.perf_counter()-started, 1)}, ensure_ascii=False), flush=True)
        validation = evaluate(model, validation_loader, device, amp_dtype)
        training_loss = epoch_totals["combined_nll_sum"] / max(1.0, epoch_totals["combined_weight"])
        row = {
            "epoch": epoch, "step": step,
            "training_loss": training_loss,
            "training_move_cross_entropy": epoch_totals["move_nll_sum"] / max(1, epoch_totals["move_targets"]),
            "training_hint_cross_entropy": None if not epoch_totals["hint_targets"] else epoch_totals["hint_nll_sum"] / epoch_totals["hint_targets"],
            "training_move_targets": epoch_totals["move_targets"],
            "training_hint_targets": epoch_totals["hint_targets"],
            "training_hint_per_move": epoch_totals["hint_targets"] / max(1, epoch_totals["move_targets"]),
            "validation_loss": validation["loss"],
            "validation_move_cross_entropy": validation["move_cross_entropy"],
            "validation_hint_cross_entropy": validation["hint_cross_entropy"],
            "validation_move_targets": validation["move_targets"],
            "validation_hint_targets": validation["hint_targets"],
            "validation_perplexity": math.exp(min(validation["loss"], 20.0)),
            "elapsed_sec": round(time.perf_counter()-started, 1),
        }
        if writer is not None:
            writer.add_scalar("epoch/training_combined_cross_entropy", row["training_loss"], epoch)
            writer.add_scalar("epoch/training_move_cross_entropy", row["training_move_cross_entropy"], epoch)
            if row["training_hint_cross_entropy"] is not None:
                writer.add_scalar("epoch/training_hint_cross_entropy", row["training_hint_cross_entropy"], epoch)
            writer.add_scalar("epoch/training_hint_per_move", row["training_hint_per_move"], epoch)
            writer.add_scalar("epoch/training_hint_targets", row["training_hint_targets"], epoch)
            writer.add_scalar("epoch/validation_move_cross_entropy", row["validation_move_cross_entropy"], epoch)
            writer.add_scalar("epoch/validation_perplexity", row["validation_perplexity"], epoch)
            writer.flush()
        history.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        save_checkpoint(output / "last.pt", model, optimizer, args, epoch, step, best_loss)
        if validation["loss"] < best_loss - 1e-4:
            best_loss, wait = validation["loss"], 0
            save_checkpoint(output / "best.pt", model, optimizer, args, epoch, step, best_loss)
        else:
            wait += 1
            if args.early_stopping_patience and wait >= args.early_stopping_patience:
                break
    history_path.write_text(json.dumps({"best_validation_loss": best_loss, "history": history}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
