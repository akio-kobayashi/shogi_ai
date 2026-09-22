#!/usr/bin/env python3
"""評価集合の全指手を対象とする教師強制の指手評価．

既存の``evaluate_factorized_moves.py``は，計算量を抑えるため各対局から
8手・32手の2点だけを抽出し，queryごとにprefillしている．そのため統合値は
「8手と32手を等比率で混ぜた平均」であり，対応する母集団を持たない．
履歴長の分析も，手数別の下限を持たないまま2点を比べる形になっている．

本スクリプトは教師強制の指標だけを対象とし，1対局を1回の前向き計算へ通して
その対局の全指手を同時に採点する．因果マスクがあるため，これは各局面を
独立に採点した場合と一致する．前向き計算は対局数分で済むので，
10,000 queryのprefillより安い．

生成が必要な指標（貪欲生成の完全指手，beam上位5，生成手の合法性）は
逐次処理が避けられないため，本スクリプトには含めない．既存の抽出評価で
測る．NLLとパープレキシティの定義は``evaluate_factorized_moves.py``に
一致させてある．
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from data import load_vocabulary
from factorized_prompt import (
    BASIC_PIECE_TOKENS, DROP_TOKEN, MOVE_ENCODING, PIECE_TOKENS, PROMOTE_TOKEN,
    TERMINAL_ENCODING, TRAINING_OBJECTIVE, annotation_piece_token, factorize_usi,
)
from models import ModelConfig, build_model
from new_prompt import square_tokens
from train_model import amp_context, resolve_amp
from provenance import write_metrics_json


# 手数の層別．手数別の多数派下限と並べられるよう，境界を固定して記録する．
PLY_BUCKETS = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**9))
# 既存の抽出評価と直接突き合わせるための点．
CROSS_CHECK_PLIES = (8, 32)

# 予測位置の種別．文法上許されるtoken集合とcanonicalマスクの有無が決まる．
SLOT_START = "start"            # 指手の第1token．移動元または<DROP>
SLOT_DROP_PIECE = "drop_piece"  # <DROP>の直後．駒種はここでは正当なのでマスクしない
SLOT_DEST_OR_PROMOTE = "dest_or_promote"
SLOT_DROP_DEST = "drop_dest"
SLOT_PROMOTE_DEST = "promote_dest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="全指手を対象とする教師強制の指手評価")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-games", type=int, default=0, help="0で全対局")
    parser.add_argument("--games-per-batch", type=int, default=8)
    parser.add_argument("--amp", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every-games", type=int, default=500)
    parser.add_argument(
        "--self-check-games", type=int, default=2,
        help="先頭N対局について，接頭辞だけを入力した場合とNLLが一致するか検査する。0で無効",
    )
    parser.add_argument(
        "--self-check-tolerance", type=float, default=1e-3,
        help="自己検査で許すNLLの絶対差",
    )
    parser.add_argument(
        "--self-check-amp", default="off",
        help=("自己検査のAMP設定。既定のoffは，採点位置の検証を数値精度から切り離すため。"
              "bfloat16では位置が正しくても1e-2程度の差が出るので判定に使えない。"
              "本番の採点は--ampの設定で行う"),
    )
    parser.add_argument(
        "--compare-with", default=None,
        help="既存の抽出評価のmove_metrics.json。8手・32手の値を突き合わせて差を記録する",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    return torch.device(value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def load_checkpoint(args: argparse.Namespace, vocabulary: dict):
    """既存の``evaluate_factorized_moves.py``と同じ検証を通す．"""
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    settings = checkpoint.get("new_prompt", {})
    if settings.get("move_encoding") != MOVE_ENCODING:
        raise ValueError(f"checkpoint is not marked as {MOVE_ENCODING}")
    if settings.get("terminal_encoding") != TERMINAL_ENCODING:
        raise ValueError("checkpoint was not trained with complete-game EOS supervision")
    config = ModelConfig(**checkpoint["config"])
    if config.vocab_size != len(vocabulary):
        raise ValueError("checkpoint and vocabulary sizes differ")
    state_prompt_mode = str(settings.get("state_prompt_mode", "explicit"))
    start_selection = str(settings.get("start_selection", "random_candidates"))
    if state_prompt_mode != "implicit_initial" or start_selection != "fixed_initial":
        raise ValueError("this evaluation accepts only implicit fixed-initial checkpoints")
    # 評価入力の注釈有無はcheckpointが決める．コマンドラインでは上書きしない．
    args.evaluation_annotation_mode = "ap" if settings.get("annotation_mode") == "ap" else "vanilla"
    args.state_prompt_mode = state_prompt_mode
    model_type = str(checkpoint.get("model_type", "vanilla"))
    model = build_model(model_type, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    return model, config, model_type, settings


def slot_tables(vocabulary: dict) -> tuple[dict, dict]:
    """種別ごとの，文法上許されるid列とcanonicalマスク対象id列を返す．"""
    squares = [vocabulary[token] for token in square_tokens()]
    pieces = [vocabulary[token] for token in PIECE_TOKENS]
    allowed = {
        SLOT_START: squares + [vocabulary[DROP_TOKEN]],
        SLOT_DROP_PIECE: [vocabulary[token] for token in BASIC_PIECE_TOKENS],
        SLOT_DEST_OR_PROMOTE: squares + [vocabulary[PROMOTE_TOKEN]],
        SLOT_DROP_DEST: squares,
        SLOT_PROMOTE_DEST: squares,
    }
    # <DROP>直後だけは駒種が正当な予測対象なので，canonical側で除外しない．
    masked = {name: ([] if name == SLOT_DROP_PIECE else pieces) for name in allowed}
    return allowed, masked


def move_kind(token_ids: list[int], vocabulary: dict) -> str:
    """指手の種類．終盤で駒打ちの比率が上がるため，手数と交絡する．"""
    if token_ids[0] == vocabulary[DROP_TOKEN]:
        return "drop"
    if len(token_ids) > 2 and token_ids[1] == vocabulary[PROMOTE_TOKEN]:
        return "promotion"
    return "normal"


def classify_slots(token_ids: list[int], vocabulary: dict) -> list[str]:
    """1指手のsub-token列に対し，各tokenの予測位置の種別を返す．"""
    drop = vocabulary[DROP_TOKEN]
    promote = vocabulary[PROMOTE_TOKEN]
    slots = [SLOT_START]
    if token_ids[0] == drop:
        slots.append(SLOT_DROP_PIECE)
        if len(token_ids) > 2:
            slots.append(SLOT_DROP_DEST)
    else:
        slots.append(SLOT_DEST_OR_PROMOTE)
        if len(token_ids) > 2 and token_ids[1] == promote:
            slots.append(SLOT_PROMOTE_DEST)
    if len(slots) != len(token_ids):
        raise ValueError(f"slot classification mismatch: {token_ids}")
    return slots


def build_game(record: dict, args: argparse.Namespace, vocabulary: dict,
               max_seq_len: int) -> dict | None:
    """1対局の全指手を，1本の系列と採点位置の対応へ展開する．"""
    candidates = [value for value in record.get("start_candidates", [])
                  if int(value.get("start_ply", -1)) == 0]
    if not candidates:
        return None
    candidate = candidates[0]
    state = [] if args.state_prompt_mode == "implicit_initial" else list(candidate["state_prompt_tokens"])
    tokens = ["<BOS>", *state, "<MOVES>"]
    moves: list[dict] = []
    truncated = 0
    total_plies = len(record["move_tokens"])
    for ply in range(total_plies):
        usi = str(record["move_tokens"][ply])
        annotation = record["move_annotations"][ply]
        subtokens = factorize_usi(usi)
        block: list[str] = []
        annotated = False
        if args.evaluation_annotation_mode == "ap" and bool(annotation.get("eligible", False)):
            block.append(annotation_piece_token(str(annotation["piece"])))
            annotated = True
        block.extend(subtokens)
        if len(tokens) + len(block) > max_seq_len:
            truncated = total_plies - ply
            break
        start = len(tokens)
        tokens.extend(block)
        ids = [vocabulary[token] for token in subtokens]
        slots = classify_slots(ids, vocabulary)
        if annotated:
            # 注釈tokenは指手本体の前に置かれる．canonical NLLには含めず，
            # AP正準NLLにだけ加える．
            moves.append({
                "ply": ply, "usi": usi,
                "annotation_position": start - 1,
                "annotation_id": vocabulary[block[0]],
                "positions": [start + 1 + index - 1 for index in range(len(ids))],
                "target_ids": ids, "slots": slots,
                "is_drop": "*" in usi, "kind": move_kind(ids, vocabulary),
            })
        else:
            moves.append({
                "ply": ply, "usi": usi,
                "annotation_position": None, "annotation_id": None,
                "positions": [start + index - 1 for index in range(len(ids))],
                "target_ids": ids, "slots": slots,
                "is_drop": "*" in usi, "kind": move_kind(ids, vocabulary),
            })
    if not moves:
        return None
    return {
        "token_ids": [vocabulary[token] for token in tokens],
        "moves": moves,
        "truncated_moves": truncated,
        "total_plies": total_plies,
    }


def ply_bucket(ply: int) -> str:
    if ply == 0:
        return "ply_0"
    for low, high in PLY_BUCKETS:
        if low <= ply <= high:
            return f"ply_{low}_{high}" if high < 10**9 else f"ply_{low}_plus"
    raise ValueError(ply)


def score_batch(model, games: list[dict], allowed, masked, device, amp_dtype,
                vocabulary: dict, annotation_mode: str, accumulators: dict) -> None:
    """対局をまとめて1回の前向き計算へ通し，全指手を採点する．"""
    lengths = [len(game["token_ids"]) for game in games]
    width = max(lengths)
    pad = vocabulary["<PAD>"]
    ids = torch.full((len(games), width), pad, dtype=torch.long)
    mask = torch.zeros((len(games), width), dtype=torch.bool)
    for row, game in enumerate(games):
        ids[row, : lengths[row]] = torch.tensor(game["token_ids"], dtype=torch.long)
        mask[row, : lengths[row]] = True
    ids = ids.to(device)
    mask = mask.to(device)
    with torch.inference_mode(), amp_context(device, amp_dtype):
        logits = model(ids, attention_mask=mask, output_hidden_states=False).logits
    logits = logits.float().cpu()

    promote_id = vocabulary[PROMOTE_TOKEN]
    square_ids = {vocabulary[token] for token in square_tokens()}
    allowed_tensors = {name: torch.tensor(value, dtype=torch.long)
                       for name, value in allowed.items()}
    masked_tensors = {name: (torch.tensor(value, dtype=torch.long) if value else None)
                      for name, value in masked.items()}

    for row, game in enumerate(games):
        for move in game["moves"]:
            raw_nll = canonical_nll = grammar_nll = 0.0
            full_top1 = full_top5 = True
            source_top1 = source_top5 = 0
            destination_top1 = destination_top5 = 0
            promotion_applicable = promotion_correct = 0
            drop_piece_applicable = drop_piece_top1 = drop_piece_top5 = 0
            for offset, (position, target_id, slot) in enumerate(
                    zip(move["positions"], move["target_ids"], move["slots"])):
                vector = logits[row, position]
                raw_nll -= float(torch.log_softmax(vector, dim=-1)[target_id])
                # canonical：評価入力に現れないRAP駒種候補をsoftmax前に除く．
                excluded = masked_tensors[slot]
                if excluded is None:
                    canonical_vector = vector
                else:
                    canonical_vector = vector.clone()
                    canonical_vector[excluded] = -math.inf
                canonical_nll -= float(torch.log_softmax(canonical_vector, dim=-1)[target_id])
                # 文法内正規化と文法制約top-k．
                candidates = allowed_tensors[slot]
                selected = vector[candidates]
                log_probabilities = torch.log_softmax(selected, dim=-1)
                local = int((candidates == target_id).nonzero()[0])
                grammar_nll -= float(log_probabilities[local])
                order = torch.topk(log_probabilities, min(5, len(candidates))).indices
                top = [int(candidates[int(index)]) for index in order]
                if offset == 0:
                    source_top1, source_top5 = int(top[0] == target_id), int(target_id in top)
                if target_id in square_ids and offset > 0:
                    destination_top1 = int(top[0] == target_id)
                    destination_top5 = int(target_id in top)
                if offset == 1 and move["target_ids"][0] in square_ids:
                    promotion_applicable = 1
                    promotion_correct = int((top[0] == promote_id) == (target_id == promote_id))
                if offset == 1 and slot == SLOT_DROP_PIECE:
                    drop_piece_applicable = 1
                    drop_piece_top1 = int(top[0] == target_id)
                    drop_piece_top5 = int(target_id in top)
                full_top1 = full_top1 and top[0] == target_id
                full_top5 = full_top5 and target_id in top
            values = {
                "queries": 1,
                "move_subtokens": len(move["target_ids"]),
                "move_nll": raw_nll,
                "canonical_move_nll": canonical_nll,
                "grammar_normalized_move_nll": grammar_nll,
                "source_top1": source_top1, "source_top5": source_top5,
                "destination_given_source_top1": destination_top1,
                "destination_given_source_top5": destination_top5,
                "promotion_decision_correct": promotion_correct,
                "promotion_decision_applicable": promotion_applicable,
                "drop_piece_correct": drop_piece_top1,
                "drop_piece_correct_top5": drop_piece_top5,
                "drop_piece_applicable": drop_piece_applicable,
                "teacher_forced_full_top1": int(full_top1),
                "teacher_forced_full_top5": int(full_top5),
                "drop_moves": int(move["is_drop"]),
            }
            if annotation_mode == "ap":
                # 注釈が付かない指手（駒打ちなど）も1標本として数える．そうしないと
                # AP正準NLLの平均の分母と分子が食い違う．
                annotation_nll = 0.0
                if move["annotation_position"] is not None:
                    annotation_vector = logits[row, move["annotation_position"]]
                    annotation_nll = -float(
                        torch.log_softmax(annotation_vector, dim=-1)[move["annotation_id"]]
                    )
                values["ap_mode_queries"] = 1
                values["ap_annotation_examples"] = int(move["annotation_position"] is not None)
                values["ap_annotated_move_nll"] = annotation_nll + canonical_nll

            bucket = ply_bucket(move["ply"])
            kind = move["kind"]
            for group in ("all", bucket, f"kind_{kind}", f"{bucket}__{kind}"):
                for key, value in values.items():
                    accumulators[group][key] += value
            if move["ply"] in CROSS_CHECK_PLIES:
                for key, value in values.items():
                    accumulators[f"cross_check_ply_{move['ply']}"][key] += value


def self_check(model, game: dict, device, amp_dtype, vocabulary: dict) -> dict:
    """採点位置の検査．接頭辞だけを入力した場合とNLLが一致するはずである．

    全系列を1回通す方式は，因果マスクがある限り各局面を独立に採点した場合と
    一致する．一致しなければ採点位置がずれている．研究本体が隠れ状態について
    行っている``causal_prefix_full_alignment``と同じ趣旨の検査である．

    ``attention_mask``を省くとSDPAが``is_causal=True``の別経路を使い，明示
    マスクを渡す採点側とカーネルが変わる．bfloat16ではそれだけで1e-2程度の差が
    出るため，両方とも全要素Trueのマスクを明示して経路を揃える．
    """
    token_ids = game["token_ids"]

    def logits_for(prefix_length: int) -> torch.Tensor:
        ids = torch.tensor([token_ids[:prefix_length]], dtype=torch.long, device=device)
        mask = torch.ones((1, prefix_length), dtype=torch.bool, device=device)
        with torch.inference_mode(), amp_context(device, amp_dtype):
            return model(ids, attention_mask=mask, output_hidden_states=False).logits[0].float().cpu()

    full_logits = logits_for(len(token_ids))
    checked = 0
    worst = 0.0
    worst_at = None
    for move in game["moves"]:
        for position, target_id in zip(move["positions"], move["target_ids"]):
            prefix_logits = logits_for(position + 1)
            from_full = -float(torch.log_softmax(full_logits[position], dim=-1)[target_id])
            from_prefix = -float(torch.log_softmax(prefix_logits[-1], dim=-1)[target_id])
            difference = abs(from_full - from_prefix)
            if difference > worst:
                worst, worst_at = difference, {"ply": move["ply"], "position": position}
            checked += 1
    return {"checked_subtokens": checked, "max_abs_nll_difference": worst,
            "max_abs_nll_difference_at": worst_at}


def compare_with_sampled(path: str, metrics: dict) -> dict:
    """既存の抽出評価と，同じ手数の値を突き合わせる．

    8手・32手は抽出評価が対象にしている手数そのものなので，教師強制の指標は
    一致するはずである。一致しなければ移植のどこかが違う．
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sampled = payload.get("metrics", {}).get("by_history_distance", {}) or {}
    shared = ("move_nll", "canonical_move_nll", "grammar_normalized_move_nll",
              "source_top1", "source_top5", "destination_given_source_top1",
              "destination_given_source_top5", "teacher_forced_full_top1",
              "teacher_forced_full_top5", "canonical_move_perplexity")
    report: dict = {"reference": path, "by_ply": {}, "max_abs_difference": 0.0}
    for ply in CROSS_CHECK_PLIES:
        ours = metrics.get(f"cross_check_ply_{ply}")
        theirs = sampled.get(str(ply))
        if not ours or not theirs:
            report["by_ply"][str(ply)] = {"available": False}
            continue
        # 対象局面が同じでなければ値が一致する理由がない。--max-gamesを
        # 絞った試運転では必ずここで弾かれる。
        comparable = int(ours["queries"]) == int(theirs.get("queries") or -1)
        entry: dict = {"available": True, "comparable": comparable,
                       "our_queries": ours["queries"],
                       "their_queries": theirs.get("queries"), "differences": {}}
        for key in shared:
            if ours.get(key) is None or theirs.get(key) is None:
                continue
            difference = abs(float(ours[key]) - float(theirs[key]))
            entry["differences"][key] = difference
            if comparable:
                report["max_abs_difference"] = max(report["max_abs_difference"], difference)
        report["by_ply"][str(ply)] = entry
    comparable_plies = [value for value in report["by_ply"].values()
                        if value.get("available") and value.get("comparable")]
    report["comparable"] = bool(comparable_plies)
    if not comparable_plies:
        report["max_abs_difference"] = None
        report["note"] = ("query counts differ, so the values are not expected to match; "
                          "run without --max-games to compare")
    return report


def summarize(total: dict) -> dict | None:
    n = int(total.get("queries", 0))
    if not n:
        return None
    subtokens = int(total["move_subtokens"])
    result: dict = {"queries": n}
    for key, value in total.items():
        if key not in {"queries", "move_subtokens", "promotion_decision_correct",
                       "promotion_decision_applicable", "drop_piece_correct",
                       "drop_piece_correct_top5", "drop_piece_applicable",
                       "ap_mode_queries", "ap_annotation_examples"}:
            result[key] = value / n
    result["move_subtokens"] = subtokens
    result["drop_move_rate"] = total.get("drop_moves", 0) / n
    result.pop("drop_moves", None)
    if total.get("drop_piece_applicable", 0):
        # 駒打ちだけが分母である．全指手で割ると通常移動に薄められる．
        result["drop_piece_top1"] = total["drop_piece_correct"] / total["drop_piece_applicable"]
        result["drop_piece_top5"] = total["drop_piece_correct_top5"] / total["drop_piece_applicable"]
        result["drop_piece_examples"] = int(total["drop_piece_applicable"])
    if total.get("promotion_decision_applicable", 0):
        result["promotion_decision_top1"] = (
            total["promotion_decision_correct"] / total["promotion_decision_applicable"]
        )
        result["promotion_decision_examples"] = int(total["promotion_decision_applicable"])
    for name, source in (("raw", "move_nll"), ("canonical", "canonical_move_nll"),
                         ("grammar_normalized", "grammar_normalized_move_nll")):
        cross_entropy = total[source] / subtokens
        result[f"{name}_token_cross_entropy"] = cross_entropy
        result[f"{name}_token_perplexity"] = math.exp(min(cross_entropy, 20.0))
    result["move_perplexity"] = math.exp(min(result["move_nll"], 20.0))
    result["canonical_move_perplexity"] = math.exp(min(result["canonical_move_nll"], 20.0))
    result["grammar_normalized_move_perplexity"] = math.exp(
        min(result["grammar_normalized_move_nll"], 20.0))
    if total.get("ap_mode_queries", 0):
        result["ap_mode_queries"] = int(total["ap_mode_queries"])
        result["ap_annotation_examples"] = int(total["ap_annotation_examples"])
        result["ap_annotated_move_perplexity"] = math.exp(min(result["ap_annotated_move_nll"], 20.0))
        result["ap_canonical_move_nll"] = result["ap_annotated_move_nll"]
        result["ap_canonical_move_perplexity"] = result["ap_annotated_move_perplexity"]
        result["ap_piece_conditioned_move_nll"] = result["canonical_move_nll"]
        result["ap_piece_conditioned_move_perplexity"] = result["canonical_move_perplexity"]
    return result


def main() -> int:
    args = parse_args()
    vocabulary = load_vocabulary(args.vocab)
    model, config, model_type, checkpoint_settings = load_checkpoint(args, vocabulary)
    device = resolve_device(args.device)
    amp_dtype, _, amp_name = resolve_amp(args.amp, device)
    model.to(device).eval()
    max_seq_len = int(config.max_seq_len)

    allowed, masked = slot_tables(vocabulary)
    accumulators: dict = defaultdict(lambda: defaultdict(float))
    scan = {"games": 0, "games_with_truncation": 0, "truncated_moves": 0, "total_plies": 0}
    started = time.perf_counter()
    pending: list[dict] = []
    # 採点位置がずれていればNLLは数nat単位で食い違うので，許容差を厳しくしても
    # 実際の誤りは捕まる。逆に精度由来の差で落ちると検査が使えなくなる。
    self_check_dtype, _, self_check_amp_name = resolve_amp(args.self_check_amp, device)
    self_check_report = {"games": 0, "checked_subtokens": 0, "max_abs_nll_difference": 0.0,
                         "max_abs_nll_difference_at": None, "amp": self_check_amp_name,
                         "tolerance": args.self_check_tolerance, "passed": None}

    def flush() -> None:
        if pending:
            score_batch(model, pending, allowed, masked, device, amp_dtype, vocabulary,
                        args.evaluation_annotation_mode, accumulators)
            pending.clear()

    with Path(args.evaluation_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if args.max_games and scan["games"] >= args.max_games:
                break
            if not line.strip():
                continue
            game = build_game(json.loads(line), args, vocabulary, max_seq_len)
            if game is None:
                continue
            scan["games"] += 1
            scan["total_plies"] += game["total_plies"]
            if game["truncated_moves"]:
                scan["games_with_truncation"] += 1
                scan["truncated_moves"] += game["truncated_moves"]
            if self_check_report["games"] < args.self_check_games:
                report = self_check(model, game, device, self_check_dtype, vocabulary)
                self_check_report["games"] += 1
                self_check_report["checked_subtokens"] += report["checked_subtokens"]
                if report["max_abs_nll_difference"] > self_check_report["max_abs_nll_difference"]:
                    self_check_report["max_abs_nll_difference"] = report["max_abs_nll_difference"]
                    self_check_report["max_abs_nll_difference_at"] = report["max_abs_nll_difference_at"]
            pending.append(game)
            if len(pending) >= args.games_per_batch:
                flush()
            if args.progress_every_games and scan["games"] % args.progress_every_games == 0:
                print(json.dumps({"event": "progress", "games": scan["games"],
                                  "moves": int(accumulators["all"]["queries"]),
                                  "seconds": round(time.perf_counter() - started, 1)},
                                 ensure_ascii=False), flush=True)
    flush()

    if not accumulators["all"]["queries"]:
        raise ValueError("no moves were scored")

    if args.self_check_games:
        self_check_report["passed"] = bool(
            self_check_report["max_abs_nll_difference"] <= args.self_check_tolerance)

    metrics = {name: summarize(total) for name, total in sorted(accumulators.items())}
    comparison = compare_with_sampled(args.compare_with, metrics) if args.compare_with else None
    payload = {
        "format_version": 1,
        "evaluation": "factorized_full_history_teacher_forced_moves_v1",
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "move_encoding": MOVE_ENCODING,
        "terminal_encoding": TERMINAL_ENCODING,
        "training_objective_expected_for_new_runs": TRAINING_OBJECTIVE,
        "training_objective": checkpoint_settings.get("training_objective"),
        "evaluation_input_annotation_mode": args.evaluation_annotation_mode,
        "evaluation_input_rap": args.evaluation_annotation_mode == "ap",
        "settings": {
            "evaluation_jsonl": args.evaluation_jsonl,
            "vocab": args.vocab,
            "max_games": args.max_games,
            "games_per_batch": args.games_per_batch,
            "state_prompt_mode": args.state_prompt_mode,
            "max_seq_len": max_seq_len,
            "amp": amp_name,
            "device": str(device),
        },
        "scan": scan,
        "self_check": self_check_report,
        "comparison_with_sampled_evaluation": comparison,
        "definitions": {
            "coverage": "every move of every scanned game, minus moves beyond max_seq_len",
            "teacher_forced_only": "no autoregressive generation; identical to per-position "
                                   "scoring because of the causal mask",
            "canonical_move_nll": "RAP annotation piece logits are masked before softmax, "
                                  "except immediately after <DROP>",
            "self_check": "prefix-only and full-sequence NLL are compared at self_check.amp, "
                          "which is independent of the amp used for scoring",
            "ply_0": "the position before the first move is reported separately; it is "
                     "identical for every game",
            "kind_groups": "drop / normal / promotion; the drop share rises with ply, so "
                           "ply-wise change confounds tracking with move composition",
            "drop_piece_top1": "which piece is dropped, given that a drop is being made; "
                               "the denominator is drop moves only",
            "cross_check_ply_8_and_32": "the same target plies the sampled evaluation uses, "
                                        "so the two scripts can be compared directly",
        },
        "metrics": metrics,
    }
    write_metrics_json(Path(args.output), payload)
    print(json.dumps({
        "event": "full_history_move_evaluation_complete",
        "output": args.output,
        "games": scan["games"],
        "moves": int(accumulators["all"]["queries"]),
        "truncated_moves": scan["truncated_moves"],
        "canonical_move_perplexity": round(metrics["all"]["canonical_move_perplexity"], 4),
        "self_check_passed": self_check_report["passed"],
        "self_check_max_abs_nll_difference": self_check_report["max_abs_nll_difference"],
        "comparison_comparable": (comparison or {}).get("comparable"),
        "comparison_max_abs_difference": (comparison or {}).get("max_abs_difference"),
        "seconds": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False))
    if self_check_report["passed"] is False:
        # 採点位置がずれている可能性がある。値を信用させないため異常終了する。
        print("SELF-CHECK FAILED: prefix-only and full-sequence NLL disagree by "
              f"{self_check_report['max_abs_nll_difference']:.6f} "
              f"(tolerance {args.self_check_tolerance})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
