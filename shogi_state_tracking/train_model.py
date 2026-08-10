#!/usr/bin/env python3
"""通常棋譜の指手予測とCoT-like SFTを同じdecoderで学習する。"""

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # --no-tensorboardのsmoke環境を許す。
    SummaryWriter = None

from cot_data import ReasoningTraceDataset, collate_reasoning_traces
from data import (
    FIXED_SEQUENCE_OVERHEAD,
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
        description="将棋decoderの指手予測pretrainingまたはCoT-like SFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=("pretrain", "cot"), required=True)
    parser.add_argument("--model-type", choices=("vanilla", "t2mlr"), default="vanilla")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=320,
        help="固定局面99トークンを含む系列長。長い系列はwindowingする",
    )
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="最大epoch数。early stoppingが先に発火すれば途中で終了する",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="validation lossが改善しないepoch数。0なら無効",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="改善とみなすvalidation lossの最小差",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="GPUメモリに応じて調整する。語彙が大きいため小さめの値を既定にする",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--amp",
        choices=("auto", "off", "fp16", "bf16"),
        default="auto",
        help=(
            "自動混合精度。autoはCUDA/ROCmで有効化し、BF16が使えなければFP16を使う。"
            "CPU/MPSでは無効"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker数。CUDA/ROCmではCPU側の系列生成を並列化する",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="この学習stepごとに進捗を表示する。0ならstep表示をしない",
    )
    parser.add_argument(
        "--tensorboard-dir",
        help="TensorBoard event出力先。既定はOUTPUT_DIR/tensorboard",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="TensorBoard eventを書き出さない。通常は指定しない",
    )
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


def _bf16_supported(device: torch.device) -> bool:
    """BF16対応を安全に確認する。

    ROCm版PyTorchでもGPUは ``cuda`` デバイスとして公開される。
    一方、古いPyTorchやGPUでは確認APIがない／例外を投げる場合があるため、
    AMPの自動判定で実行を中断しない。
    """
    if device.type != "cuda":
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except (RuntimeError, AssertionError):
        return False


def resolve_amp(
    value: str, device: torch.device
) -> tuple[torch.dtype | None, object | None, str]:
    """AMPのdtypeとGradScalerを解決する。

    ROCmはCUDAと同じ ``torch.cuda``／``device_type="cuda"`` APIを使う。
    BF16は通常scaler不要、FP16は勾配アンダーフロー対策にscalerを使う。
    """
    if value == "off" or device.type != "cuda":
        return None, None, "off"
    if value == "auto":
        if device.type == "cuda":
            dtype = torch.bfloat16 if _bf16_supported(device) else torch.float16
    elif value == "bf16":
        dtype = torch.bfloat16
        if device.type == "cuda" and not _bf16_supported(device):
            raise RuntimeError(
                "--amp bf16 was requested, but this CUDA/ROCm device does not "
                "report BF16 support; use --amp fp16 or --amp off"
            )
    else:
        if device.type != "cuda":
            raise ValueError("--amp fp16 is supported only on CUDA/ROCm devices")
        dtype = torch.float16

    if dtype != torch.float16:
        return dtype, None, str(dtype).replace("torch.", "")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        # PyTorch旧版との互換性。ROCmでもtorch.cuda.ampを使用する。
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    return dtype, scaler, "float16"


def amp_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


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
    max_suffix_moves = args.max_seq_len - FIXED_SEQUENCE_OVERHEAD
    if max_suffix_moves <= 0:
        raise ValueError(
            "max-seq-len must be greater than {} for the fixed state prefix".format(
                FIXED_SEQUENCE_OVERHEAD
            )
        )
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
            max_suffix_moves=max_suffix_moves,
        )
        validation = RandomStartSequenceDataset(
            args.validation_jsonl,
            vocabulary,
            candidate_count=args.candidate_count,
            min_suffix_moves=args.min_suffix_moves,
            seed=args.seed + 1,
            randomize_each_epoch=False,
            max_suffix_moves=max_suffix_moves,
        )
        collate = partial(
            collate_sequences,
            pad_token_id=vocabulary["<PAD>"],
            max_seq_len=args.max_seq_len,
        )
    return train, validation, collate


def batch_loss(model, batch, stage: str, device: torch.device):
    non_blocking = device.type == "cuda"
    input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
    attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
    recurrent_mask = batch["recurrent_mask"].to(device, non_blocking=non_blocking)
    labels = batch["labels"].to(device, non_blocking=non_blocking)
    output = model(
        input_ids,
        attention_mask=attention_mask,
        recurrent_mask=recurrent_mask,
    )
    if stage == "cot":
        return weighted_causal_lm_loss(
            output.logits,
            labels,
            batch["loss_weights"].to(device, non_blocking=non_blocking),
        )
    return causal_lm_loss(output.logits, labels)


def is_out_of_memory_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "hip out of memory" in message


def evaluate(
    model,
    loader,
    stage: str,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    model.eval()
    loss_sum = 0.0
    batches = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            try:
                with amp_context(device, amp_dtype):
                    loss_sum += float(batch_loss(model, batch, stage, device))
            except RuntimeError as exc:
                if not is_out_of_memory_error(exc):
                    raise
                sequence_length = int(batch["input_ids"].shape[1])
                raise RuntimeError(
                    "GPU OOM during validation at batch {} (seq_len={}); "
                    "reduce --batch-size or --max-seq-len".format(
                        batch_index, sequence_length
                    )
                ) from exc
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
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be non-negative")
    run_started_at = time.perf_counter()
    print(
        "run_start stage={} model_type={} device={} epochs={} batch_size={} "
        "max_steps={} early_stopping_patience={} early_stopping_min_delta={}".format(
            args.stage,
            args.model_type,
            args.device,
            args.epochs,
            args.batch_size,
            args.max_steps,
            args.early_stopping_patience,
            args.early_stopping_min_delta,
        ),
        flush=True,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype, scaler, amp_name = resolve_amp(args.amp, device)
    print(
        "runtime device={} amp={} scaler={} torch={} cuda={} hip={}".format(
            device,
            amp_name,
            scaler is not None,
            torch.__version__,
            getattr(torch.version, "cuda", None),
            getattr(torch.version, "hip", None),
        ),
        flush=True,
    )
    vocabulary = load_vocabulary(args.vocab)
    model = load_initial_model(args, len(vocabulary), device)
    # checkpointの設定を優先した場合、collateの上限も実モデルへ合わせる。
    args.max_seq_len = model.config.max_seq_len
    segment_move_limit = "n/a"
    train_max_game_moves = "n/a"
    validation_max_game_moves = "n/a"
    if args.stage == "pretrain":
        segment_move_limit = args.max_seq_len - FIXED_SEQUENCE_OVERHEAD
    train_dataset, validation_dataset, collate = build_datasets(args, vocabulary)
    if args.stage == "pretrain":
        train_max_game_moves = max(
            len(record["move_tokens"]) for record in train_dataset.records
        )
        validation_max_game_moves = max(
            len(record["move_tokens"]) for record in validation_dataset.records
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print(
        "data_ready train_examples={} validation_examples={} train_batches={} "
        "validation_batches={} seq_len_limit={} train_max_game_moves={} "
        "validation_max_game_moves={} segment_move_limit={}".format(
            len(train_dataset),
            len(validation_dataset),
            len(train_loader),
            len(validation_loader),
            args.max_seq_len,
            train_max_game_moves,
            validation_max_game_moves,
            segment_move_limit,
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else output_dir / "tensorboard"
    if not args.no_tensorboard and SummaryWriter is None:
        print(json.dumps({"event": "tensorboard_unavailable", "action": "disabled"}), flush=True)
    writer = None if args.no_tensorboard or SummaryWriter is None else SummaryWriter(log_dir=str(tensorboard_dir))
    if writer is not None:
        writer.add_text("run/config", json.dumps(vars(args), ensure_ascii=False, indent=2), 0)
        writer.add_scalar("model/parameter_count", sum(parameter.numel() for parameter in model.parameters()), 0)
    best_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    history = []
    stop = False
    stop_reason = "max_epochs"
    training_started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        training_sum = 0.0
        training_batches = 0
        epoch_started_at = time.perf_counter()
        for epoch_step, batch in enumerate(train_loader, 1):
            batch_seq_len = int(batch["input_ids"].shape[1])
            batch_active_tokens = int(batch["attention_mask"].sum())
            if batch_seq_len >= max(1, int(args.max_seq_len * 0.9)):
                print(
                    "batch_length_warning epoch={} step={} seq_len={} limit={} active_tokens={}".format(
                        epoch,
                        global_step + 1,
                        batch_seq_len,
                        args.max_seq_len,
                        batch_active_tokens,
                    ),
                    flush=True,
                )
            try:
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, amp_dtype):
                    loss = batch_loss(model, batch, args.stage, device)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip
                    )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            except RuntimeError as exc:
                if not is_out_of_memory_error(exc):
                    raise
                raise RuntimeError(
                    "GPU OOM during training at epoch {} step {} "
                    "(seq_len={}, batch_size={}); reduce --batch-size or "
                    "--max-seq-len".format(
                        epoch,
                        global_step + 1,
                        batch_seq_len,
                        args.batch_size,
                    )
                ) from exc
            global_step += 1
            training_sum += float(loss.detach())
            training_batches += 1
            if writer is not None:
                writer.add_scalar("train/cross_entropy", float(loss.detach()), global_step)
                writer.add_scalar("train/batch_sequence_length", batch_seq_len, global_step)
                writer.add_scalar("train/batch_active_tokens", batch_active_tokens, global_step)
                if device.type == "cuda":
                    writer.add_scalar(
                        "system/gpu_memory_allocated_mib",
                        torch.cuda.memory_allocated(device) / (1024 ** 2),
                        global_step,
                    )
            if args.progress_every > 0 and (
                global_step == 1 or global_step % args.progress_every == 0
            ):
                elapsed = time.perf_counter() - training_started_at
                steps_per_sec = global_step / max(elapsed, 1e-9)
                remaining_steps = max(
                    len(train_loader) * args.epochs - global_step, 0
                )
                print(
                    json.dumps(
                        {
                            "progress": "train",
                            "epoch": epoch,
                            "epoch_step": epoch_step,
                            "epoch_steps": len(train_loader),
                            "step": global_step,
                            "loss": float(loss.detach()),
                            "batch_seq_len": batch_seq_len,
                            "batch_active_tokens": batch_active_tokens,
                            "elapsed_sec": round(elapsed, 1),
                            "steps_per_sec": round(steps_per_sec, 3),
                            "remaining_steps_estimate": remaining_steps,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

        print(
            "validation_start epoch={} step={} train_elapsed_sec={:.1f}".format(
                epoch, global_step, time.perf_counter() - epoch_started_at
            ),
            flush=True,
        )
        validation_loss = evaluate(
            model, validation_loader, args.stage, device, amp_dtype
        )
        training_loss = training_sum / max(training_batches, 1)
        row = {
            "epoch": epoch,
            "step": global_step,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_perplexity": math.exp(min(validation_loss, 20.0)),
        }
        improved = validation_loss < best_loss - args.early_stopping_min_delta
        row["validation_improved"] = improved
        row["early_stopping_wait"] = (
            0 if improved else epochs_without_improvement + 1
        )
        history.append(row)
        row["elapsed_sec"] = round(time.perf_counter() - run_started_at, 1)
        if writer is not None:
            writer.add_scalar("epoch/training_cross_entropy", training_loss, epoch)
            writer.add_scalar("validation/cross_entropy", validation_loss, epoch)
            writer.add_scalar("validation/perplexity", row["validation_perplexity"], epoch)
            writer.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)
        save_checkpoint(output_dir / "last.pt", model, args, epoch, global_step)
        if improved:
            best_loss = validation_loss
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, args, epoch, global_step)
            print(
                "checkpoint_best epoch={} step={} validation_loss={:.6f}".format(
                    epoch, global_step, validation_loss
                ),
                flush=True,
            )
        else:
            epochs_without_improvement += 1
            print(
                "early_stopping_wait epoch={} step={} validation_loss={:.6f} "
                "best_validation_loss={:.6f} wait={} patience={}".format(
                    epoch,
                    global_step,
                    validation_loss,
                    best_loss,
                    epochs_without_improvement,
                    args.early_stopping_patience,
                ),
                flush=True,
            )
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                stop = True
                stop_reason = "early_stopping"
                print(
                    "early_stopping_stop epoch={} step={} "
                    "best_validation_loss={:.6f} patience={}".format(
                        epoch,
                        global_step,
                        best_loss,
                        args.early_stopping_patience,
                    ),
                    flush=True,
                )
        if stop:
            if stop_reason == "max_epochs" and args.max_steps > 0:
                stop_reason = "max_steps"
            break

    with (output_dir / "training_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "stage": args.stage,
                "model_type": args.model_type,
                "device": str(device),
                "amp": amp_name,
                "amp_scaler": scaler is not None,
                "torch_version": torch.__version__,
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "torch_hip_version": getattr(torch.version, "hip", None),
                "tensorboard": None if writer is None else str(tensorboard_dir.resolve()),
                "best_validation_loss": best_loss,
                "epochs_requested": args.epochs,
                "epochs_completed": len(history),
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "stop_reason": stop_reason,
                "history": history,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    if writer is not None:
        writer.close()
    print(
        "run_complete stage={} model_type={} steps={} elapsed_sec={:.1f}".format(
            args.stage,
            args.model_type,
            global_step,
            time.perf_counter() - run_started_at,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
