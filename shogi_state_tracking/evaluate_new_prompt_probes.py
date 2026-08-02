#!/usr/bin/env python3
"""新prompt artifactから，開始局面＋履歴表現の層別線形プローブを学習する。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

from create_dataset import HAND_ORDER, PIECE_NAMES
from data import load_vocabulary
from models import ModelConfig, build_model
from new_prompt import piece_token, move_token
from probes import LinearStateProbe, ProbeTargets, linear_probe_loss, majority_predictions, predictions_from_logits, state_metrics


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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
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


def read_examples(path, limit):
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            record = json.loads(line)
            for example in record.get("probe_examples", []):
                examples.append(example)
                if len(examples) >= limit: return examples
    if not examples: raise ValueError("no probe_examples in {}".format(path))
    return examples


def extract(model, examples, vocabulary, sources, board_map, hand_names, device, max_seq_len):
    chunks = {source: [] for source in sources}; target_chunks = []; scopes = []
    model.eval()
    with torch.inference_mode():
        for example in examples:
            tokens = ["<BOS>"] + list(example["state_prompt_tokens"]) + ["<MOVES>"] + [move_token(move) for move in example["history_moves"]]
            if len(tokens) > max_seq_len: continue
            ids = torch.tensor([[vocabulary[token] for token in tokens]], device=device)
            output = model(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
            position = ids.shape[1] - 1
            for source in sources:
                layer = int(source.split("_", 1)[1])
                chunks[source].append(output.hidden_states[layer][0, position].detach().cpu())
            target_chunks.append(target_from_mapping(example["probe_targets"], board_map, hand_names))
            scopes.append(str(example.get("position_scope", "unknown_position_scope")))
    if not target_chunks: raise ValueError("all probe examples exceeded max_seq_len")
    return {source: torch.stack(values) for source, values in chunks.items()}, concat_targets(target_chunks), scopes


def fit_probe(train_x, train_y, validation_x, validation_y, d_model, args, device):
    probe = LinearStateProbe(d_model).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(train_x, train_y.board, train_y.hands, train_y.turn, train_y.in_check), batch_size=args.batch_size, shuffle=True)
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
        result["in_check_accuracy"] = float((logits.in_check.argmax(dim=-1).cpu() == targets.in_check).float().mean())
        result["in_check_positive_rate"] = float(targets.in_check.float().mean())
    return result


def metrics_by_scope(probe, x, targets, scopes, device):
    result = {}
    for scope in sorted(set(scopes)):
        indices = torch.tensor([index for index, value in enumerate(scopes) if value == scope], dtype=torch.long)
        subset = ProbeTargets(targets.board[indices], targets.hands[indices], targets.turn[indices], None if targets.in_check is None else targets.in_check[indices])
        result[scope] = metric(probe, x[indices], subset, device)
    return result


def main():
    args = parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    vocabulary = load_vocabulary(args.vocab); device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig(**checkpoint["config"])
    model = build_model("vanilla", config).to(device); model.load_state_dict(checkpoint["model_state_dict"])
    board_map, hand_names = label_maps(); sources = expand_sources(args.sources, config.n_layers)
    raw = {"train": read_examples(args.train_jsonl, args.max_train_samples), "validation": read_examples(args.validation_jsonl, args.max_validation_samples), "evaluation": read_examples(args.evaluation_jsonl, args.max_evaluation_samples)}
    extracted = {name: extract(model, examples, vocabulary, sources, board_map, hand_names, device, config.max_seq_len) for name, examples in raw.items()}
    majority = majority_predictions(extracted["train"][1], extracted["evaluation"][1].board.shape[0])
    result = {"checkpoint": args.checkpoint, "settings": vars(args), "sources": sources, "majority_baseline": state_metrics(extracted["evaluation"][1], *majority), "probe_results": {}}
    states = {}
    for source in sources:
        probe, best_loss = fit_probe(extracted["train"][0][source], extracted["train"][1], extracted["validation"][0][source], extracted["validation"][1], config.d_model, args, device)
        result["probe_results"][source] = {"best_validation_loss": best_loss, "validation": metric(probe, extracted["validation"][0][source], extracted["validation"][1], device), "evaluation": metric(probe, extracted["evaluation"][0][source], extracted["evaluation"][1], device), "evaluation_by_position_scope": metrics_by_scope(probe, extracted["evaluation"][0][source], extracted["evaluation"][1], extracted["evaluation"][2], device)}
        states[source] = probe.cpu().state_dict()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "probe_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"checkpoint": args.checkpoint, "sources": sources, "probe_state_dicts": states, "board_label_map": board_map, "hand_names": hand_names}, output / "linear_probes.pt")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
