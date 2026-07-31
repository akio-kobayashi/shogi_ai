"""Colabおよび学生実験用の小規模実験ヘルパー。

ノートブックからsubprocessやデータ形式の詳細を隠し，次の処理を同じAPIで
実行できるようにする。

* toy棋譜または既存JSONLの準備
* answer-only decoderの学習
* 指手予測token accuracyの測定
* 線形probeの実行と要約
* probe結果のSVG化
* 短いChain of Movesの生成

モデルへの入力は，既存実装と同じく開始局面96トークンと指し手列である。
cshogiはtoy棋譜生成とprobe正解状態の再生にだけ使う。
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


__all__ = [
    "DatasetPaths",
    "evaluate_next_move",
    "generate_short_cot",
    "plot_training_history",
    "prepare_csa_visualization_dataset",
    "prepare_dataset",
    "render_piece_probability_svg",
    "render_probe_svg",
    "run_probe_evaluation",
    "summarize_probe",
    "train_small_model",
]


@dataclass(frozen=True)
class DatasetPaths:
    """実験で使う3 splitと語彙のパス。"""

    train_jsonl: Path
    validation_jsonl: Path
    evaluation_jsonl: Path
    vocab_json: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "train_jsonl": str(self.train_jsonl),
            "validation_jsonl": str(self.validation_jsonl),
            "evaluation_jsonl": str(self.evaluation_jsonl),
            "vocab_json": str(self.vocab_json),
        }


def _dataset_paths(data_dir: Path) -> DatasetPaths:
    dataset_dir = data_dir / "datasets"
    return DatasetPaths(
        train_jsonl=dataset_dir / "train.jsonl",
        validation_jsonl=dataset_dir / "validation.jsonl",
        evaluation_jsonl=dataset_dir / "evaluation.jsonl",
        vocab_json=data_dir / "vocab.json",
    )


def _make_toy_record(game_index: int, move_count: int, cshogi_module):
    board = cshogi_module.Board()
    initial_sfen = board.sfen()

    from create_dataset import encode_initial_state

    initial_state_tokens = encode_initial_state(board, cshogi_module)
    move_tokens = []
    for ply in range(move_count):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = legal_moves[(game_index * 31 + ply * 7) % len(legal_moves)]
        move_tokens.append(cshogi_module.move_to_usi(move))
        board.push(move)

    return {
        "schema_version": 1,
        "game_id": "toy-{:04d}".format(game_index),
        "engine_scope": "toy",
        "initial_sfen": initial_sfen,
        "initial_state_tokens": initial_state_tokens,
        "move_tokens": move_tokens,
    }


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_csa_visualization_dataset(
    csa_path: str | Path,
    output_jsonl: str | Path,
    game_index: int = 0,
) -> tuple[Path, str]:
    """CSAの1対局を大駒・玉プローブ可視化用JSONLへ変換する。

    これは学習データ作成ではない。既存checkpointとlinear probeに対して，任意の
    CSA対局を再生・可視化するだけの最小レコードを作る。
    """
    from create_dataset import encode_initial_state, import_cshogi

    source = Path(csa_path)
    cshogi = import_cshogi()
    games = cshogi.Parser.parse_file(str(source))
    if not 0 <= int(game_index) < len(games):
        raise IndexError(
            "game_index {} is outside CSA game count {}".format(game_index, len(games))
        )
    game = games[int(game_index)]
    board = cshogi.Board(game.sfen)
    initial_sfen = board.sfen()
    move_tokens = []
    for ply, move in enumerate(game.moves, 1):
        if not board.is_legal(move):
            raise ValueError("illegal CSA move at ply {}".format(ply))
        move_tokens.append(cshogi.move_to_usi(move))
        board.push(move)

    game_id = "csa:{}:{}".format(source.stem, int(game_index))
    record = {
        "schema_version": 2,
        "game_id": game_id,
        "split": "visualization",
        "player_scope": "external_csa",
        "engine_scope": "external_csa",
        "position_scope": "external_position",
        "trajectory_scope": "external_position",
        "initial_sfen": initial_sfen,
        "initial_state_tokens": encode_initial_state(cshogi.Board(initial_sfen), cshogi),
        "move_tokens": move_tokens,
    }
    output = Path(output_jsonl)
    _write_jsonl(output, [record])
    return output, game_id


def prepare_dataset(
    data_dir: str | Path,
    use_real_data: bool = False,
    toy_games: int = 24,
    toy_moves: int = 32,
    include_mask_tokens: bool = False,
) -> DatasetPaths:
    """実データを検証するか，Colab用toy datasetを作成する。

    ``use_real_data=False``では，標準初期局面から合法手を決定的に選び，
    train/validation/evaluationへ分割する。研究用の棋譜品質を表すデータではなく，
    パイプライン確認用である。``include_mask_tokens=True``では，マスク実験用の
    v4語彙を作る。ただし，マスクされた入力例の生成は別の前処理で行う。
    """
    data_path = Path(data_dir)
    paths = _dataset_paths(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    if use_real_data:
        required = [
            paths.train_jsonl,
            paths.validation_jsonl,
            paths.evaluation_jsonl,
            paths.vocab_json,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "実データの不足ファイル: {}".format(", ".join(missing))
            )
        return paths

    if toy_games < 3 or toy_moves <= 0:
        raise ValueError("toy_games must be >= 3 and toy_moves must be positive")

    import cshogi
    from create_dataset import write_vocabulary

    records = [_make_toy_record(index, toy_moves, cshogi) for index in range(toy_games)]
    train_count = max(1, round(toy_games * 2 / 3))
    validation_count = max(1, round(toy_games / 6))
    if train_count + validation_count >= toy_games:
        validation_count = 1
        train_count = toy_games - 2
    splits = {
        "train": records[:train_count],
        "validation": records[train_count : train_count + validation_count],
        "evaluation": records[train_count + validation_count :],
    }
    for split, split_records in splits.items():
        _write_jsonl(getattr(paths, "{}_jsonl".format(split)), split_records)

    observed_moves = {
        move
        for record in records
        for move in record["move_tokens"]
    }
    write_vocabulary(
        paths.vocab_json,
        observed_moves,
        include_mask_tokens=include_mask_tokens,
    )
    return paths


def train_small_model(
    project_dir: str | Path,
    datasets: DatasetPaths,
    output_dir: str | Path,
    model_type: str = "vanilla",
    device: str = "cpu",
    max_seq_len: int = 160,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    d_ff: int = 128,
    epochs: int = 3,
    max_steps: int = 40,
    batch_size: int = 8,
    candidate_count: int = 8,
    min_suffix_moves: int = 8,
    seed: int = 20260724,
    force: bool = False,
) -> Path:
    """Colab向けの小規模answer-only学習を実行し，best checkpointを返す。"""
    project_path = Path(project_dir)
    output_path = Path(output_dir)
    checkpoint = output_path / "best.pt"
    if checkpoint.exists() and not force:
        return checkpoint

    command = [
        sys.executable,
        "-u",
        "train_model.py",
        "--stage",
        "pretrain",
        "--model-type",
        model_type,
        "--vocab",
        str(datasets.vocab_json),
        "--train-jsonl",
        str(datasets.train_jsonl),
        "--validation-jsonl",
        str(datasets.validation_jsonl),
        "--output-dir",
        str(output_path),
        "--max-seq-len",
        str(max_seq_len),
        "--d-model",
        str(d_model),
        "--n-layers",
        str(n_layers),
        "--n-heads",
        str(n_heads),
        "--d-ff",
        str(d_ff),
        "--dropout",
        "0.0",
        "--epochs",
        str(epochs),
        "--max-steps",
        str(max_steps),
        "--batch-size",
        str(batch_size),
        "--candidate-count",
        str(candidate_count),
        "--min-suffix-moves",
        str(min_suffix_moves),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if model_type == "t2mlr":
        command.extend(["--l-start", "0", "--l-end", "0", "--jacobi-depth", "1"])
    output_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=project_path, check=True)
    if not checkpoint.exists():
        raise RuntimeError("学習後にbest.ptが作成されませんでした: {}".format(checkpoint))
    return checkpoint


def plot_training_history(checkpoint_dir: str | Path):
    """training_history.jsonを読み，matplotlib Figureを返す。"""
    import matplotlib.pyplot as plt

    history_path = Path(checkpoint_dir) / "training_history.json"
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    rows = payload["history"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["epoch"] for row in rows], [row["training_loss"] for row in rows], marker="o", label="train")
    axis.plot([row["epoch"] for row in rows], [row["validation_loss"] for row in rows], marker="o", label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("causal LM loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def evaluate_next_move(
    checkpoint: str | Path,
    datasets: DatasetPaths,
    device: str = "cpu",
    batch_size: int = 8,
    candidate_count: int = 8,
    min_suffix_moves: int = 8,
    seed: int = 20260724,
) -> Dict[str, object]:
    """検証JSONLの教師指し手に対する次token一致率を返す。"""
    import torch
    from torch.utils.data import DataLoader

    from data import (
        FIXED_SEQUENCE_OVERHEAD,
        IGNORE_INDEX,
        RandomStartSequenceDataset,
        collate_sequences,
        load_vocabulary,
    )
    from evaluate_probes import load_backbone

    device_obj = torch.device(device)
    vocabulary = load_vocabulary(str(datasets.vocab_json))
    model, model_type, config = load_backbone(str(checkpoint), device_obj, False)
    dataset = RandomStartSequenceDataset(
        str(datasets.validation_jsonl),
        vocabulary,
        candidate_count=candidate_count,
        min_suffix_moves=min_suffix_moves,
        seed=seed,
        randomize_each_epoch=False,
        max_suffix_moves=config.max_seq_len - FIXED_SEQUENCE_OVERHEAD,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_sequences(
            examples, vocabulary["<PAD>"], config.max_seq_len
        ),
    )
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device_obj),
                attention_mask=batch["attention_mask"].to(device_obj),
                recurrent_mask=batch["recurrent_mask"].to(device_obj),
            )
            labels = batch["labels"].to(device_obj)
            mask = labels != IGNORE_INDEX
            correct += int((output.logits.argmax(dim=-1)[mask] == labels[mask]).sum())
            total += int(mask.sum())
    return {
        "checkpoint": str(checkpoint),
        "model_type": model_type,
        "device": str(device_obj),
        "tokens": total,
        "next_move_token_accuracy": correct / max(total, 1),
    }


def run_probe_evaluation(
    project_dir: str | Path,
    checkpoint: str | Path,
    datasets: DatasetPaths,
    output_dir: str | Path,
    device: str = "cpu",
    sources: str = "final",
    candidate_count: int = 8,
    min_suffix_moves: int = 8,
    positions_per_game: int = 8,
    probe_epochs: int = 5,
    patience: int = 2,
    batch_size: int = 128,
    seed: int = 20260724,
    force: bool = False,
) -> Path:
    """既存CLIを使って線形probeを実行し，metrics JSONのパスを返す。"""
    output_path = Path(output_dir)
    metrics_path = output_path / "probe_metrics.json"
    if metrics_path.exists() and not force:
        return metrics_path
    command = [
        sys.executable,
        "-u",
        "evaluate_probes.py",
        "--checkpoint",
        str(checkpoint),
        "--vocab",
        str(datasets.vocab_json),
        "--train-jsonl",
        str(datasets.train_jsonl),
        "--validation-jsonl",
        str(datasets.validation_jsonl),
        "--evaluation-jsonl",
        str(datasets.evaluation_jsonl),
        "--output-dir",
        str(output_path),
        "--sources",
        sources,
        "--candidate-count",
        str(candidate_count),
        "--min-suffix-moves",
        str(min_suffix_moves),
        "--positions-per-game",
        str(positions_per_game),
        "--probe-epochs",
        str(probe_epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    subprocess.run(command, cwd=Path(project_dir), check=True)
    if not metrics_path.exists():
        raise RuntimeError("probe_metrics.jsonが作成されませんでした: {}".format(metrics_path))
    return metrics_path


def summarize_probe(metrics_path: str | Path, source: str = "final") -> Dict[str, object]:
    """probe reportから主要指標だけを取り出す。"""
    report = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    results = report["probe_results"]
    if source == "final":
        layer_sources = [key for key in results if str(key).startswith("layer_")]
        if not layer_sources:
            raise ValueError("finalに対応するlayer sourceがありません")
        source = max(layer_sources, key=lambda key: int(str(key).split("_", 1)[1]))
    evaluation = results[source]["evaluation"]
    return {
        "source": source,
        "board_square_accuracy": evaluation["board_square_accuracy"],
        "board_exact_match": evaluation["board_exact_match"],
        "hand_slot_accuracy": evaluation["hand_slot_accuracy"],
        "turn_accuracy": evaluation["turn_accuracy"],
        "full_state_exact_match": evaluation["full_state_exact_match"],
    }


def render_probe_svg(
    project_dir: str | Path,
    probe_dir: str | Path,
    source: str = "final",
    output_name: str = "board_accuracy.svg",
) -> Path:
    """probe_predictions.ptから盤面SVGを生成する。"""
    probe_path = Path(probe_dir)
    output_path = probe_path / output_name
    command = [
        sys.executable,
        "-u",
        "visualize_probes.py",
        "aggregate",
        "--predictions",
        str(probe_path / "probe_predictions.pt"),
        "--source",
        source,
        "--output",
        str(output_path),
    ]
    subprocess.run(command, cwd=Path(project_dir), check=True)
    return output_path


def render_piece_probability_svg(
    project_dir: str | Path,
    checkpoint: str | Path,
    vocab: str | Path,
    evaluation_jsonl: str | Path,
    probe_path: str | Path,
    game_id: str,
    ply: int,
    piece: str,
    source: str = "final",
    device: str = "cpu",
    output: str | Path | None = None,
) -> Path:
    """指定局面の大駒・玉について，probe確率を盤面SVGとして出力する。"""
    output_path = (
        Path(output)
        if output is not None
        else Path(probe_path).parent / "{}-ply{:03d}-{}.svg".format(
            str(game_id).replace(":", "-"), int(ply), piece
        )
    )
    command = [
        sys.executable,
        "-u",
        "visualize_major_piece_probe.py",
        "--checkpoint",
        str(checkpoint),
        "--vocab",
        str(vocab),
        "--evaluation-jsonl",
        str(evaluation_jsonl),
        "--probes",
        str(probe_path),
        "--game-id",
        str(game_id),
        "--ply",
        str(int(ply)),
        "--piece",
        str(piece),
        "--source",
        str(source),
        "--device",
        str(device),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, cwd=Path(project_dir), check=True)
    return output_path


def generate_short_cot(
    project_dir: str | Path,
    checkpoint: str | Path,
    datasets: DatasetPaths,
    output_jsonl: str | Path,
    device: str = "cpu",
    positions_per_game: int = 2,
    lines: int = 2,
    line_batch_size: int = 2,
    line_length: int = 2,
    max_games: int = 4,
    seed: int = 20260724,
) -> Path:
    """少数局面から短いChain of Movesを生成する。"""
    output_path = Path(output_jsonl)
    command = [
        sys.executable,
        "-u",
        "generate_reasoning_traces.py",
        "--checkpoint",
        str(checkpoint),
        "--vocab",
        str(datasets.vocab_json),
        "--input-jsonl",
        str(datasets.evaluation_jsonl),
        "--output-jsonl",
        str(output_path),
        "--positions-per-game",
        str(positions_per_game),
        "--lines",
        str(lines),
        "--line-batch-size",
        str(line_batch_size),
        "--line-length",
        str(line_length),
        "--max-games",
        str(max_games),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    subprocess.run(command, cwd=Path(project_dir), check=True)
    return output_path
