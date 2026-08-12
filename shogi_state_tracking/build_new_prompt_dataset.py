#!/usr/bin/env python3
"""新しい駒・座標prompt用artifactを，既存JSONLから作成する。

このスクリプトだけがcshogiを必要とする。出力したJSONLには，学習時に必要な
開始候補・駒種教師・評価用の局面ラベルを保存するため，計算機側はCSAもcshogiも
必要としない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

from create_dataset import HAND_ORDER, PIECE_NAMES
from new_prompt import (
    NEW_PROMPT_SCHEMA_VERSION,
    annotate_game_moves,
    encode_state_prompt,
    move_annotation,
    new_prompt_vocabulary_tokens,
    piece_token,
    square_token,
    write_new_prompt_vocabulary,
)


EMPTY_BOARD_LABEL = "<EMPTY>"


def decisive_game_result(record: Mapping[str, object]) -> int:
    """上流で抽出済みの勝敗を検査し，中間artifactへ保持する．"""
    if "game_result" not in record:
        raise ValueError("source record is missing game_result")
    try:
        result = int(record["game_result"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid game_result: {}".format(record["game_result"])) from exc
    if result == 0:
        raise ValueError("draw game is not allowed in the decisive terminal experiment")
    return result


def import_cshogi():
    try:
        import cshogi  # type: ignore
    except ImportError as exc:
        raise RuntimeError("このデータ作成スクリプトにはcshogiが必要です") from exc
    return cshogi


def parse_start_plies(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(value < 0 for value in values):
        raise ValueError("start plies must be nonnegative")
    return list(dict.fromkeys(values))


def choose_candidate_plies(
    game_id: str,
    moves: Sequence[str],
    required: Sequence[int],
    candidate_count: int,
    min_suffix_moves: int,
    seed: int,
) -> List[int]:
    valid = [ply for ply in range(len(moves)) if len(moves) - ply >= min_suffix_moves]
    if not valid:
        return []
    result = [ply for ply in required if ply in valid]
    remaining = [ply for ply in valid if ply not in result]
    rng = random.Random("{}:{}".format(seed, game_id))
    rng.shuffle(remaining)
    result.extend(remaining[: max(0, candidate_count - len(result))])
    return sorted(result)


def board_piece_label(piece: int) -> str:
    if int(piece) == 0:
        return EMPTY_BOARD_LABEL
    color = "W" if int(piece) >= 16 else "B"
    return piece_token(color, int(piece) % 16)


def state_targets(board, cshogi_module) -> Dict[str, object]:
    """開始局面に対する，プローブと評価のための明示的教師ラベル。"""
    hands: Dict[str, int] = {}
    for color_index, color in ((cshogi_module.BLACK, "B"), (cshogi_module.WHITE, "W")):
        for hand_piece, count in zip(HAND_ORDER, board.pieces_in_hand[color_index]):
            hands["<{}_{}>".format(color, hand_piece)] = int(count)
    king_squares: Dict[str, str] = {}
    board_labels = []
    for index, piece in enumerate(board.pieces):
        label = board_piece_label(int(piece))
        board_labels.append(label)
        if label in {"<B_K>", "<W_K>"}:
            king_squares[label[1]] = square_token("{}{}".format(index // 9 + 1, "abcdefghi"[index % 9]))
    return {
        "board_labels_cshogi_order": board_labels,
        "hands": hands,
        "turn": "<TURN_BLACK>" if board.turn == cshogi_module.BLACK else "<TURN_WHITE>",
        "in_check": bool(board.is_check()),
        "king_squares": king_squares,
    }


def legal_sources_by_piece(board, cshogi_module) -> Dict[str, List[str]]:
    values: Dict[str, set[str]] = {}
    for move in board.legal_moves:
        usi = cshogi_module.move_to_usi(move)
        if "*" in usi:
            continue
        source = usi[:2]
        index = (int(source[0]) - 1) * 9 + "abcdefghi".index(source[1])
        label = board_piece_label(int(board.pieces[index]))
        values.setdefault(label, set()).add(square_token(source))
    return {piece: sorted(squares) for piece, squares in sorted(values.items())}


def normalize_position_scopes(record: Mapping[str, object], move_count: int) -> List[str]:
    """局面列のscopeを，各指手を指す直前の局面へ揃える．

    ``create_dataset.py``の現行形式は初期局面から終局面までを含むため
    ``move_count + 1``要素である．new-prompt側はtarget plyごとの入力局面だけを
    保存するので，最後の終局面を除いた``move_count``要素を用いる．
    """
    raw = [str(value) for value in record.get("position_scope_by_ply", [])]
    if len(raw) == move_count + 1:
        return raw[:-1]
    if len(raw) == move_count:
        return raw
    if not raw:
        return [str(record.get("position_scope", "unknown_position_scope"))] * move_count
    raise ValueError(
        "position_scope_by_ply has {} entries for {} moves".format(len(raw), move_count)
    )


def materialize_record(
    record: Mapping[str, object],
    cshogi_module,
    required_start_plies: Sequence[int],
    candidate_count: int,
    min_suffix_moves: int,
    seed: int,
    include_evaluation_steps: bool,
    probe_offsets: Sequence[int],
    probe_start_plies: Sequence[int],
) -> Dict[str, object]:
    game_result = decisive_game_result(record)
    moves = [str(value) for value in record["move_tokens"]]
    initial_sfen = str(record["initial_sfen"])
    annotations = annotate_game_moves(initial_sfen, moves, cshogi_module)
    candidates_at = choose_candidate_plies(
        str(record["game_id"]), moves, required_start_plies, candidate_count, min_suffix_moves, seed
    )
    wanted = set(candidates_at)
    board = cshogi_module.Board(initial_sfen)
    candidates: List[Dict[str, object]] = []
    candidate_by_ply: Dict[int, Dict[str, object]] = {}
    evaluation_steps: List[Dict[str, object]] = []
    probe_examples: List[Dict[str, object]] = []
    scopes = normalize_position_scopes(record, len(moves))
    for ply in range(len(moves)):
        if ply in wanted:
            candidate = {
                    "start_ply": ply,
                    "start_sfen": board.sfen(),
                    "state_prompt_tokens": encode_state_prompt(board, cshogi_module),
                    "probe_targets": state_targets(board, cshogi_module),
                    "position_scope": scopes[ply] if ply < len(scopes) else record.get("position_scope", "unknown_position_scope"),
            }
            candidates.append(candidate)
            candidate_by_ply[ply] = candidate
        if include_evaluation_steps:
            evaluation_steps.append(
                {
                    "ply": ply,
                    "target_move": moves[ply],
                    "legal_moves": sorted(cshogi_module.move_to_usi(move) for move in board.legal_moves),
                    "legal_sources_by_piece": legal_sources_by_piece(board, cshogi_module),
                    "probe_targets": state_targets(board, cshogi_module),
                }
            )
        # 固定開始plyだけを用いる少数のプローブ例。全plyを保存せずに，状態入力後の
        # 短・中距離の履歴表現を学習できる。ターゲットはこのplyで指す前の局面である。
        for start_ply, candidate in candidate_by_ply.items():
            if start_ply not in probe_start_plies:
                continue
            offset = ply - start_ply
            if offset in probe_offsets:
                probe_examples.append({
                    "start_ply": start_ply,
                    "ply": ply,
                    "state_prompt_tokens": candidate["state_prompt_tokens"],
                    "history_moves": moves[start_ply:ply],
                    "probe_targets": state_targets(board, cshogi_module),
                    "position_scope": candidate["position_scope"] if offset == 0 else (scopes[ply] if ply < len(scopes) else "unknown_position_scope"),
                    "trajectory_scope": str(record.get("trajectory_scope", "unknown_position_scope")),
                })
        move = board.move_from_usi(moves[ply])
        if not board.is_legal(move):
            raise ValueError("illegal move at ply {}: {}".format(ply + 1, moves[ply]))
        board.push(move)
    result = {
        "schema_version": NEW_PROMPT_SCHEMA_VERSION,
        "game_id": str(record["game_id"]),
        "game_result": game_result,
        "split": str(record.get("split", "")),
        "initial_sfen": initial_sfen,
        "move_tokens": moves,
        "move_annotations": annotations,
        "start_candidates": candidates,
        "probe_examples": probe_examples,
        "player_scope": str(record.get("player_scope", record.get("engine_scope", ""))),
        "engine_scope": str(record.get("engine_scope", record.get("player_scope", ""))),
        "position_scope": str(record.get("position_scope", "unknown_position_scope")),
        # 任意の開始ply・履歴距離で評価を層別化できるよう，全plyのscopeも保存する。
        # 旧artifactとの互換性のため，評価器側はこの列がない場合probe例へfallbackする。
        "position_scope_by_ply": [str(value) for value in scopes],
        "trajectory_scope": str(record.get("trajectory_scope", "unknown_position_scope")),
    }
    if include_evaluation_steps:
        result["evaluation_steps"] = evaluation_steps
    return result


def read_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("{}:{} invalid JSON".format(path, line_number)) from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split(args: argparse.Namespace, split: str, cshogi_module) -> Dict[str, object]:
    source = Path(args.input_dir) / (split + ".jsonl")
    destination = Path(args.output_dir) / (split + ".jsonl")
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    games = 0
    candidates = 0
    with destination.open("w", encoding="utf-8") as output:
        for record in read_jsonl(source):
            transformed = materialize_record(
                record, cshogi_module, args.required_start_plies,
                args.candidate_count, args.min_suffix_moves, args.seed,
                include_evaluation_steps=(split == "evaluation"),
                probe_offsets=args.probe_offsets,
                probe_start_plies=args.probe_start_plies,
            )
            if not transformed["start_candidates"]:
                continue
            json.dump(transformed, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")
            games += 1
            candidates += len(transformed["start_candidates"])
    return {"file": destination.name, "games": games, "candidates": candidates, "sha256": sha256(destination)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="新prompt artifactを既存JSONLから構築する")
    parser.add_argument("--input-dir", required=True, help="旧create_dataset.py buildのtrain/validation/evaluation.jsonlを置く場所")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-count", type=int, default=40)
    parser.add_argument("--required-start-plies", default="0,24,32")
    parser.add_argument("--min-suffix-moves", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--probe-offsets", default="0,8,32", help="各固定開始plyから保存するプローブ距離")
    parser.add_argument("--probe-start-plies", default="0,24,32", help="プローブ例を保存する開始ply")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_count <= 0 or args.min_suffix_moves <= 0:
        raise ValueError("candidate-count and min-suffix-moves must be positive")
    args.required_start_plies = parse_start_plies(args.required_start_plies)
    args.probe_offsets = parse_start_plies(args.probe_offsets)
    args.probe_start_plies = parse_start_plies(args.probe_start_plies)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cshogi_module = import_cshogi()
    summaries = {split: build_split(args, split, cshogi_module) for split in ("train", "validation", "evaluation")}
    vocab = write_new_prompt_vocabulary(output_dir / "vocab.json")
    manifest = {
        "schema_version": NEW_PROMPT_SCHEMA_VERSION,
        "format": "shogi_piece_coordinate_prompt",
        "candidate_count": args.candidate_count,
        "required_start_plies": args.required_start_plies,
        "min_suffix_moves": args.min_suffix_moves,
        "seed": args.seed,
        "probe_offsets": args.probe_offsets,
        "probe_start_plies": args.probe_start_plies,
        "vocab_sha256": sha256(output_dir / "vocab.json"),
        "vocab_size": len(vocab["token_to_id"]),
        "splits": summaries,
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
