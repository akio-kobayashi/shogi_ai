#!/usr/bin/env python3
"""状態prompt artifactを使うVanilla／LLaMA型decoderの共通学習器。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import resource
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from data import IGNORE_INDEX, load_vocabulary
from models import ModelConfig, build_model
from new_prompt_data import collate_new_prompt_sequences
from factorized_prompt_data import FactorizedPromptSequenceDataset
from factorized_prompt import FACTORIZED_SCHEMA_VERSION, MOVE_ENCODING, TERMINAL_ENCODING, TRAINING_OBJECTIVE
from train_model import amp_context, resolve_amp, resolve_device

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # --no-tensorboardのsmoke環境を許す。
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


class LengthBucketBatchSampler(Sampler):
    """近い系列長をまとめ，paddingによる二乗attention計算を減らす。"""

    def __init__(self, lengths, batch_size, seed, pool_batches=50):
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.pool_size = max(self.batch_size, self.batch_size * int(pool_batches))
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start : start + self.pool_size]
            pool.sort(key=self.lengths.__getitem__)
            batches.extend(pool[index : index + self.batch_size] for index in range(0, len(pool), self.batch_size))
        rng.shuffle(batches)
        yield from batches


def runtime_marker(event: str, **fields: object) -> None:
    """SIGKILL直前の段階とプロセスRSSを残すための即時ログ。"""
    payload = {"event": event, "pid": os.getpid()}
    # LinuxではKiB，macOSではbyteだが，実行環境内での相対比較に使える。
    payload["max_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def validate_output_dir_model_label(output: Path, model_type: str, model_size: str) -> None:
    """出力パス内のfamily-sizeラベルと実際の学習設定の不一致を拒否する。"""
    expected = (model_type, model_size)
    for component in output.parts:
        match = re.fullmatch(r"(vanilla|llama)-(small|base|large)", component)
        if match and match.groups() != expected:
            actual = "{}-{}".format(*expected)
            raise ValueError(
                "output directory model label conflicts with training configuration: "
                "{} contains {}, but --model-type/--model-size specify {}".format(
                    output, component, actual
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="新prompt用decoderを学習する")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--dataset-manifest", help="検証済みdataset_manifest.json。指定時はhashもrun manifestへ記録する")
    parser.add_argument("--output-dir", required=True)
    # 既存呼出しとの互換性のため，既定値は従来どおりvanillaに保つ。
    parser.add_argument("--model-type", choices=("vanilla", "llama"), default="vanilla")
    parser.add_argument(
        "--move-encoding",
        choices=(MOVE_ENCODING,),
        default=MOVE_ENCODING,
        help="factorized_v3は125語彙を用い，指手を移動先座標で終える",
    )
    parser.add_argument(
        "--state-prompt-mode", choices=("implicit_initial",), default="implicit_initial",
        help="現行の第1・第2段階はimplicit_initialだけを許可する",
    )
    parser.add_argument(
        "--start-selection", choices=("fixed_initial",), default="fixed_initial",
        help="factorized_v3では平手初期局面へ固定する",
    )
    parser.add_argument("--resume", action="store_true", help="output-dir/last.ptからモデルとoptimizerを再開する")
    parser.add_argument("--model-size", choices=tuple(MODEL_SIZES), required=True)
    parser.add_argument("--annotation-mode", choices=("vanilla", "rap", "ap"), default="vanilla")
    parser.add_argument("--annotation-probability", type=float, default=0.0)
    parser.add_argument("--hint-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--eos-loss-weight", type=float, default=1.0,
        help="完全棋譜末尾のEOSを1行動として数える重み",
    )
    # 最大開始prompt約90 token，512指手，320個の2-token注釈を含めても
    # 90 + 512 + 2*320 = 1242 tokenであり，1280 token文脈に収まる。
    parser.add_argument("--max-hints", type=int, default=512)
    parser.add_argument("--max-moves", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2560)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--fused-optimizer", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length-bucket-pool-batches", type=int, default=50)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker数。JSONLストリーミングではまず0を推奨し，必要時だけ増やす",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="学習stepの総数上限。0はepoch数だけで終了する",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=0,
        help="各validationのbatch上限。0はvalidation split全体を評価する",
    )
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--tensorboard-dir", help="TensorBoard event出力先。既定はOUTPUT_DIR/tensorboard")
    parser.add_argument("--no-tensorboard", action="store_true", help="TensorBoard eventを書き出さない")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def loss_for_batch(model, batch, device):
    non_blocking = device.type == "cuda"
    output = model(
        batch["input_ids"].to(device, non_blocking=non_blocking),
        attention_mask=batch["attention_mask"].to(device, non_blocking=non_blocking),
        recurrent_mask=batch["recurrent_mask"].to(device, non_blocking=non_blocking),
        output_hidden_states=False,
    )
    labels = batch["labels"].to(device, non_blocking=non_blocking)
    weights = batch["loss_weights"].to(device, non_blocking=non_blocking)
    move_mask = batch["move_target_mask"].to(device, non_blocking=non_blocking)
    hint_mask = batch["hint_target_mask"].to(device, non_blocking=non_blocking)
    eos_mask = batch["eos_target_mask"].to(device, non_blocking=non_blocking)
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
    if "move_unit_weight" in batch:
        move_units = batch["move_unit_weight"].to(
            device, non_blocking=non_blocking
        ).to(per_token.dtype)
        move_count = batch["move_boundary_mask"].to(
            device, non_blocking=non_blocking
        ).sum()
        if not bool(move_count > 0):
            raise ValueError("factorized batch has no move targets")
        move_nll_sum = (per_token * move_units).sum()
        eos_count = eos_mask.sum()
        eos_weights = effective_weights.masked_select(eos_mask)
        eos_weight = eos_weights[0] if eos_weights.numel() else per_token.new_zeros(())
        eos_nll_sum = per_token.masked_select(eos_mask).sum()
        primary_weight = move_count.to(per_token.dtype) + eos_weight * eos_count.to(per_token.dtype)
        weighted_hint_nll_sum = (per_token * effective_weights * hint_mask).sum()
        weighted_nll_sum = move_nll_sum + eos_weight * eos_nll_sum + weighted_hint_nll_sum
        loss = weighted_nll_sum / primary_weight
        combined_nll_sum = weighted_nll_sum.detach()
        combined_weight = primary_weight.detach()
    else:
        move_count = move_mask.sum()
        weighted_nll_sum = (per_token * effective_weights).sum()
        loss = weighted_nll_sum / weight_sum
        combined_nll_sum = weighted_nll_sum.detach()
        combined_weight = weight_sum.detach()

    # factorized系列では従来の指手単位正規化を保つ．RAP平均を独立に足すと
    # 挿入率qによらずRAPがほぼ1タスク分の重みを持つため，RAP NLLの「和」を
    # 指手損失の分子へ加える．これによりq=0の目的関数は従来と完全に同じで，
    # RAPの追加寄与だけが実際の挿入token数に比例する．

    # 注釈の有無が平均lossに埋もれないよう，重み付け前のCEと実トークン数を返す。
    # move_mask/hint_maskはlabelsと同じ位置で，それぞれ排他的に定義される。
    def totals(mask):
        count = mask.sum()
        return {
            "nll_sum": per_token.masked_select(mask).sum().detach(),
            "targets": count.detach(),
        }

    return loss, {
        "combined_nll_sum": combined_nll_sum,
        "combined_weight": combined_weight,
        "move": totals(move_mask),
        "hint": totals(hint_mask),
        "eos": totals(eos_mask),
        "move_count": move_count.detach(),
    }


def evaluate(model, loader, device, amp_dtype, max_batches: int = 0):
    model.eval()
    totals = {"combined_nll_sum": 0.0, "combined_weight": 0.0, "move_nll_sum": 0.0, "move_targets": 0, "move_count": 0, "hint_nll_sum": 0.0, "hint_targets": 0, "eos_nll_sum": 0.0, "eos_targets": 0, "batches": 0}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            with amp_context(device, amp_dtype):
                _, metrics = loss_for_batch(model, batch, device)
            totals["combined_nll_sum"] += float(metrics["combined_nll_sum"])
            totals["combined_weight"] += float(metrics["combined_weight"])
            totals["move_nll_sum"] += float(metrics["move"]["nll_sum"])
            totals["move_targets"] += int(metrics["move"]["targets"])
            totals["move_count"] += int(metrics["move_count"])
            totals["hint_nll_sum"] += float(metrics["hint"]["nll_sum"])
            totals["hint_targets"] += int(metrics["hint"]["targets"])
            totals["eos_nll_sum"] += float(metrics["eos"]["nll_sum"])
            totals["eos_targets"] += int(metrics["eos"]["targets"])
            totals["batches"] += 1
            if max_batches > 0 and batch_index >= max_batches:
                break
    return {
        "loss": totals["combined_nll_sum"] / max(1.0, totals["combined_weight"]),
        "move_cross_entropy": totals["move_nll_sum"] / max(1, totals["move_targets"]),
        "hint_cross_entropy": None if not totals["hint_targets"] else totals["hint_nll_sum"] / totals["hint_targets"],
        "eos_cross_entropy": None if not totals["eos_targets"] else totals["eos_nll_sum"] / totals["eos_targets"],
        "move_targets": totals["move_targets"],
        "move_count": totals["move_count"],
        "hint_targets": totals["hint_targets"],
        "eos_targets": totals["eos_targets"],
        "batches": totals["batches"],
    }


def write_step_scalars(writer: Optional["SummaryWriter"], step: int, loss, metrics, batch, device) -> None:
    """高頻度ログ。注釈数と系列長をlossと同じstepに残す。"""
    if writer is None:
        return
    writer.add_scalar("train/combined_cross_entropy", float(loss.detach()), step)
    writer.add_scalar("train/move_targets_per_batch", int(metrics["move"]["targets"]), step)
    writer.add_scalar("train/hint_targets_per_batch", int(metrics["hint"]["targets"]), step)
    writer.add_scalar("train/eos_targets_per_batch", int(metrics["eos"]["targets"]), step)
    writer.add_scalar("train/moves_per_batch", int(metrics["move_count"]), step)
    writer.add_scalar("train/hints_per_move", int(metrics["hint"]["targets"]) / max(1, int(metrics["move_count"])), step)
    writer.add_scalar("train/sequence_length", int(batch["input_ids"].shape[1]), step)
    if device.type == "cuda":
        writer.add_scalar("system/memory_allocated_mib", torch.cuda.memory_allocated(device) / 2**20, step)
        writer.add_scalar("system/max_memory_allocated_mib", torch.cuda.max_memory_allocated(device) / 2**20, step)


def save_checkpoint(path, model, optimizer, args, epoch, step, best_loss, include_optimizer=True):
    payload = {
        "model_type": args.model_type, "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(), "epoch": epoch, "step": step,
        "best_validation_loss": best_loss, "new_prompt": {
            "model_size": args.model_size, "move_encoding": args.move_encoding,
            "state_prompt_mode": args.state_prompt_mode,
            "start_selection": args.start_selection,
            "annotation_mode": args.annotation_mode,
            "annotation_probability": args.annotation_probability,
            "hint_loss_weight": args.hint_loss_weight, "max_hints": args.max_hints,
            "eos_loss_weight": args.eos_loss_weight,
            "terminal_encoding": TERMINAL_ENCODING,
            "training_objective": TRAINING_OBJECTIVE,
            "max_moves": args.max_moves, "seed": args.seed,
        },
    }
    # 評価用best checkpointへAdamWのmomentを複製しない。再開に使うlastだけ保持する。
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    runtime_marker("process_start", model_type=args.model_type, model_size=args.model_size)
    output = Path(args.output_dir)
    validate_output_dir_model_label(output, args.model_type, args.model_size)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if args.max_validation_batches < 0:
        raise ValueError("--max-validation-batches must be non-negative")
    if args.annotation_mode == "vanilla" and args.annotation_probability:
        raise ValueError("vanilla requires --annotation-probability 0")
    if args.annotation_mode == "ap" and args.annotation_probability != 1.0:
        raise ValueError("ap requires --annotation-probability 1")
    if args.eos_loss_weight < 0.0:
        raise ValueError("--eos-loss-weight must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    vocabulary = load_vocabulary(args.vocab)
    if args.move_encoding == MOVE_ENCODING:
        vocab_payload = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
        if vocab_payload.get("move_encoding") != MOVE_ENCODING or vocab_payload.get("terminal_encoding") != TERMINAL_ENCODING or len(vocabulary) != 125:
            raise ValueError("factorized_v3 requires its canonical 125-token vocab.json")
        if not args.dataset_manifest:
            raise ValueError("factorized_v3 requires --dataset-manifest")
        dataset_payload = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
        if dataset_payload.get("move_encoding") != MOVE_ENCODING or int(dataset_payload.get("schema_version", -1)) != FACTORIZED_SCHEMA_VERSION:
            raise ValueError("dataset manifest is not factorized_v3")
        if dataset_payload.get("stage_1_2_input_mode") != "implicit_standard_initial":
            raise ValueError(
                "dataset manifest does not declare the current implicit-standard-initial experiment"
            )
        if dataset_payload.get("terminal_encoding") != TERMINAL_ENCODING or dataset_payload.get("terminal_supervision") != "complete_game_only":
            raise ValueError("dataset manifest does not declare complete-game EOS supervision")
    runtime_marker("vocabulary_loaded", vocab_size=len(vocabulary))
    device = resolve_device(args.device)
    amp_dtype, scaler, amp_name = resolve_amp(args.amp, device)
    size_table = LLAMA_MODEL_SIZES if args.model_type == "llama" else MODEL_SIZES
    config = ModelConfig(vocab_size=len(vocabulary), max_seq_len=args.max_seq_len, dropout=args.dropout, **size_table[args.model_size])
    config.validate()
    model = build_model(args.model_type, config).to(device)
    runtime_marker("model_ready", device=str(device), parameter_count=sum(parameter.numel() for parameter in model.parameters()))
    common = dict(annotation_mode=args.annotation_mode, annotation_probability=args.annotation_probability,
                  hint_loss_weight=args.hint_loss_weight, max_hints=args.max_hints,
                  eos_loss_weight=args.eos_loss_weight,
                  max_moves=args.max_moves, max_seq_len=args.max_seq_len,
                  return_metadata=False, validate_records=False)
    dataset_class = FactorizedPromptSequenceDataset
    common.update(
        state_prompt_mode=args.state_prompt_mode,
        start_selection=args.start_selection,
    )
    train_dataset = dataset_class(args.train_jsonl, vocabulary, seed=args.seed, randomize_each_epoch=True, **common)
    runtime_marker("train_dataset_ready", **train_dataset.storage_statistics())
    # early stoppingも運用時と同じ，注釈を除いた入力で測る。主比較の制約付き
    # 指手評価はevaluate_new_prompt_moves.pyで別に実行する。
    validation_common = dict(
        common,
        annotation_mode="ap" if args.annotation_mode == "ap" else "vanilla",
        annotation_probability=1.0 if args.annotation_mode == "ap" else 0.0,
    )
    validation_dataset = dataset_class(args.validation_jsonl, vocabulary, seed=args.seed + 1, randomize_each_epoch=False, **validation_common)
    runtime_marker("validation_dataset_ready", **validation_dataset.storage_statistics())
    collate = partial(collate_new_prompt_sequences, pad_token_id=vocabulary["<PAD>"], max_seq_len=args.max_seq_len)
    loader_options = {
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        # worker数×prefetch_factor個のbatchがCPU/Pinned memoryに滞留する。
        # 長い系列を扱うため既定の2ではなく1に固定する。
        loader_options["prefetch_factor"] = 1
    bucket_sampler = None
    if getattr(train_dataset, "length_estimates", None) is not None and args.length_bucket_pool_batches > 0:
        bucket_sampler = LengthBucketBatchSampler(
            train_dataset.length_estimates, args.batch_size, args.seed,
            args.length_bucket_pool_batches,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=bucket_sampler, **loader_options)
        runtime_marker(
            "length_bucketing_enabled",
            pool_batches=args.length_bucket_pool_batches,
            indexed_records=len(train_dataset.length_estimates),
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, **loader_options)
    runtime_marker("loaders_ready", batch_size=args.batch_size, num_workers=args.num_workers)
    output.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else output / "tensorboard"
    # resume時も既存eventを消さず，checkpointのglobal step以降へ追記する。
    if not args.no_tensorboard and SummaryWriter is None:
        runtime_marker("tensorboard_unavailable", action="disabled")
    writer = None if args.no_tensorboard or SummaryWriter is None else SummaryWriter(log_dir=str(tensorboard_dir))
    optimizer_kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    use_fused = args.fused_optimizer == "on" or (
        args.fused_optimizer == "auto" and device.type == "cuda"
    )
    if use_fused:
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except (RuntimeError, TypeError) as exc:
        if args.fused_optimizer == "on":
            raise
        runtime_marker("fused_optimizer_unavailable", reason=str(exc))
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    runtime_marker("optimizer_ready", fused=bool(optimizer_kwargs.get("fused", False)))
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
        resume_objective = resume.get("new_prompt", {}).get("training_objective")
        compatible_legacy_vanilla = (
            resume_objective is None
            and args.annotation_mode == "vanilla"
            and args.annotation_probability == 0.0
        )
        if resume_objective != TRAINING_OBJECTIVE and not compatible_legacy_vanilla:
            raise ValueError(
                "resume checkpoint uses a legacy loss; start a fresh run for {}".format(
                    TRAINING_OBJECTIVE
                )
            )
        model.load_state_dict(resume["model_state_dict"])
        if "optimizer_state_dict" in resume: optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume.get("epoch", 0)); step = int(resume.get("step", 0)); best_loss = float(resume.get("best_validation_loss", best_loss))
        # load_state_dict後もcheckpoint辞書を残すとmodel/optimizer stateを二重保持する。
        del resume
        gc.collect()
    run = {"format_version": 1, "training_objective": TRAINING_OBJECTIVE, "args": vars(args), "model_config": config.to_dict(), "parameter_count": sum(p.numel() for p in model.parameters()), "device": str(device), "amp": amp_name, "git_commit": git_commit(), "tensorboard": None if writer is None else str(tensorboard_dir.resolve())}
    if args.dataset_manifest:
        dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
        run["dataset"] = {"manifest": str(Path(args.dataset_manifest).resolve()), "schema_version": dataset_manifest.get("schema_version"), "terminal_encoding": dataset_manifest.get("terminal_encoding"), "vocab_sha256": dataset_manifest.get("vocab_sha256"), "splits": dataset_manifest.get("splits")}
    (output / "run_manifest.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if writer is not None:
        writer.add_text("run/config", json.dumps(run, ensure_ascii=False, indent=2), 0)
    print(json.dumps({"event": "data_ready", "train": len(train_dataset), "validation": len(validation_dataset), **run}, ensure_ascii=False), flush=True)
    wait, history = 0, []
    history_path = output / "training_history.json"
    if args.resume and history_path.is_file():
        history = list(json.loads(history_path.read_text(encoding="utf-8")).get("history", []))
    started = time.perf_counter()
    if args.max_steps and step >= args.max_steps:
        print(json.dumps({"event": "max_steps_already_reached", "step": step, "max_steps": args.max_steps}, ensure_ascii=False), flush=True)
        if writer is not None:
            writer.close()
        if not (output / "best.pt").is_file():
            raise FileNotFoundError(
                "max_steps was already reached but best.pt is absent; "
                "use a fresh output directory or resume a checkpoint that has completed validation"
            )
        return
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(json.dumps({"event": "epoch_start", "epoch": epoch, "step": step, "max_steps": args.max_steps}, ensure_ascii=False), flush=True)
        train_dataset.set_epoch(epoch)
        if bucket_sampler is not None:
            bucket_sampler.set_epoch(epoch)
        model.train()
        epoch_totals = {"combined_nll_sum": 0.0, "combined_weight": 0.0, "move_nll_sum": 0.0, "move_targets": 0, "move_count": 0, "hint_nll_sum": 0.0, "hint_targets": 0, "eos_nll_sum": 0.0, "eos_targets": 0}
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
            epoch_totals["move_count"] += int(metrics["move_count"])
            epoch_totals["hint_nll_sum"] += float(metrics["hint"]["nll_sum"])
            epoch_totals["hint_targets"] += int(metrics["hint"]["targets"])
            epoch_totals["eos_nll_sum"] += float(metrics["eos"]["nll_sum"])
            epoch_totals["eos_targets"] += int(metrics["eos"]["targets"])
            if args.progress_every and (step == 1 or step % args.progress_every == 0):
                write_step_scalars(writer, step, loss, metrics, batch, device)
                print(json.dumps({"event": "progress", "epoch": epoch, "step": step, "batch": batch_index, "loss": float(loss.detach()), "move_targets": int(metrics["move"]["targets"]), "hint_targets": int(metrics["hint"]["targets"]), "eos_targets": int(metrics["eos"]["targets"]), "seq_len": int(batch["input_ids"].shape[1]), "elapsed_sec": round(time.perf_counter()-started, 1)}, ensure_ascii=False), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        print(json.dumps({"event": "validation_start", "epoch": epoch, "step": step, "max_validation_batches": args.max_validation_batches}, ensure_ascii=False), flush=True)
        validation = evaluate(model, validation_loader, device, amp_dtype, args.max_validation_batches)
        if not math.isfinite(validation["loss"]):
            raise RuntimeError(
                "validation loss is not finite; refusing to finish without a valid best.pt: {}".format(
                    validation["loss"]
                )
            )
        training_loss = epoch_totals["combined_nll_sum"] / max(1.0, epoch_totals["combined_weight"])
        row = {
            "epoch": epoch, "step": step,
            "training_loss": training_loss,
            "training_move_cross_entropy": epoch_totals["move_nll_sum"] / max(1, epoch_totals["move_targets"]),
            "training_hint_cross_entropy": None if not epoch_totals["hint_targets"] else epoch_totals["hint_nll_sum"] / epoch_totals["hint_targets"],
            "training_move_targets": epoch_totals["move_targets"],
            "training_hint_targets": epoch_totals["hint_targets"],
            "training_hint_per_move": epoch_totals["hint_targets"] / max(1, epoch_totals["move_count"]),
            "training_eos_cross_entropy": None if not epoch_totals["eos_targets"] else epoch_totals["eos_nll_sum"] / epoch_totals["eos_targets"],
            "training_eos_targets": epoch_totals["eos_targets"],
            "validation_loss": validation["loss"],
            "validation_move_cross_entropy": validation["move_cross_entropy"],
            "validation_hint_cross_entropy": validation["hint_cross_entropy"],
            "validation_move_targets": validation["move_targets"],
            "validation_move_count": validation["move_count"],
            "validation_hint_targets": validation["hint_targets"],
            "validation_eos_cross_entropy": validation["eos_cross_entropy"],
            "validation_eos_targets": validation["eos_targets"],
            "validation_batches": validation["batches"],
            "validation_perplexity": math.exp(min(validation["loss"], 20.0)),
            "validation_move_perplexity": math.exp(min(validation["move_cross_entropy"], 20.0)),
            "validation_subtoken_perplexity": math.exp(min(validation["move_cross_entropy"], 20.0)),
            "elapsed_sec": round(time.perf_counter()-started, 1),
        }
        if writer is not None:
            writer.add_scalar("epoch/training_combined_cross_entropy", row["training_loss"], epoch)
            writer.add_scalar("epoch/training_move_cross_entropy", row["training_move_cross_entropy"], epoch)
            if row["training_hint_cross_entropy"] is not None:
                writer.add_scalar("epoch/training_hint_cross_entropy", row["training_hint_cross_entropy"], epoch)
            if row["training_eos_cross_entropy"] is not None:
                writer.add_scalar("epoch/training_eos_cross_entropy", row["training_eos_cross_entropy"], epoch)
            writer.add_scalar("epoch/training_hint_per_move", row["training_hint_per_move"], epoch)
            writer.add_scalar("epoch/training_hint_targets", row["training_hint_targets"], epoch)
            writer.add_scalar("epoch/validation_move_cross_entropy", row["validation_move_cross_entropy"], epoch)
            writer.add_scalar("epoch/validation_perplexity", row["validation_perplexity"], epoch)
            if row["validation_eos_cross_entropy"] is not None:
                writer.add_scalar("epoch/validation_eos_cross_entropy", row["validation_eos_cross_entropy"], epoch)
            writer.flush()
        history.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation["loss"] < best_loss - 1e-4:
            best_loss, wait = validation["loss"], 0
            print(json.dumps({"event": "checkpoint_save_start", "kind": "best", "epoch": epoch, "step": step}, ensure_ascii=False), flush=True)
            save_checkpoint(output / "best.pt", model, optimizer, args, epoch, step, best_loss, include_optimizer=False)
            print(json.dumps({"event": "checkpoint_save_complete", "kind": "best", "epoch": epoch, "step": step}, ensure_ascii=False), flush=True)
        else:
            wait += 1
        # best_lossを更新してからlast.ptへ保存し，resume時にも最新の最良値を使う．
        print(json.dumps({"event": "checkpoint_save_start", "kind": "last", "epoch": epoch, "step": step}, ensure_ascii=False), flush=True)
        save_checkpoint(output / "last.pt", model, optimizer, args, epoch, step, best_loss, include_optimizer=True)
        print(json.dumps({"event": "checkpoint_save_complete", "kind": "last", "epoch": epoch, "step": step}, ensure_ascii=False), flush=True)
        if args.early_stopping_patience and wait >= args.early_stopping_patience:
            break
        if args.max_steps and step >= args.max_steps:
            print(json.dumps({"event": "max_steps_reached", "epoch": epoch, "step": step, "max_steps": args.max_steps}, ensure_ascii=False), flush=True)
            break
    history_path.write_text(json.dumps({
        "training_objective": TRAINING_OBJECTIVE,
        "best_validation_loss": best_loss,
        "history": history,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not (output / "best.pt").is_file():
        raise FileNotFoundError(
            "training ended without best.pt; refusing to report run_complete"
        )
    print(json.dumps({"event": "run_complete", "step": step, "epochs_recorded": len(history)}, ensure_ascii=False), flush=True)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
