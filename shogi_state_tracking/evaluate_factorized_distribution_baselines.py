#!/usr/bin/env python3
"""完全指手評価に対する棋譜分布の集中度・多数派指手baselineを測る．"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from create_dataset import import_cshogi, make_position_hash
from provenance import write_metrics_json


FREQUENCY_BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2_4", 2, 4),
    ("5_9", 5, 9),
    ("10_plus", 10, None),
)


def parse_args():
    parser = argparse.ArgumentParser(description="factorized_v3棋譜分布baseline")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--evaluation-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-distances", default="0,8,32")
    parser.add_argument("--primary-history-distances", default="8,32")
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--max-queries", type=int, default=30000)
    parser.add_argument("--progress-every-games", type=int, default=10000)
    return parser.parse_args()


def parse_distances(value):
    values = tuple(dict.fromkeys(int(item.strip()) for item in str(value).split(",") if item.strip()))
    if not values or min(values) < 0:
        raise ValueError("history distances must be nonnegative")
    return values


def entropy_bits(counter):
    total = sum(counter.values())
    if not total:
        return None
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def frequency_bin(count):
    for label, lower, upper in FREQUENCY_BINS:
        if count >= lower and (upper is None or count <= upper):
            return label
    raise ValueError("invalid frequency: {}".format(count))


def replay_queries(path, cshogi_module, distances, max_games, max_queries):
    wanted = set(distances)
    queries = []
    games = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if games >= max_games or len(queries) >= max_queries:
                break
            record = json.loads(line)
            candidates = [
                value for value in record.get("start_candidates", [])
                if int(value.get("start_ply", -1)) == 0
            ]
            if not candidates:
                continue
            games += 1
            moves = [str(value) for value in record["move_tokens"]]
            scopes = list(record.get("position_scope_by_ply", ()))
            board = cshogi_module.Board(str(record["initial_sfen"]))
            max_distance = min(max(wanted), len(moves) - 1)
            for ply in range(max_distance + 1):
                if ply in wanted:
                    queries.append({
                        "game_id": str(record.get("game_id", line_number)),
                        "ply": ply,
                        "position_hash": make_position_hash(board.sfen()),
                        "target": moves[ply],
                        "position_scope": str(scopes[ply]) if ply < len(scopes) else "unknown_position_scope",
                        "trajectory_scope": str(record.get("trajectory_scope", "unknown_position_scope")),
                    })
                    if len(queries) >= max_queries:
                        break
                if ply < max_distance:
                    board.push(board.move_from_usi(moves[ply]))
    return queries, games


def collect_train_counts(path, wanted_hashes, cshogi_module, progress_every):
    counters = defaultdict(Counter)
    global_moves = Counter()
    games = moves_seen = matched = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            games += 1
            board = cshogi_module.Board(str(record["initial_sfen"]))
            for move in record["move_tokens"]:
                move = str(move)
                position_hash = make_position_hash(board.sfen())
                global_moves[move] += 1
                if position_hash in wanted_hashes:
                    counters[position_hash][move] += 1
                    matched += 1
                board.push(board.move_from_usi(move))
                moves_seen += 1
            if progress_every and games % progress_every == 0:
                print(json.dumps({"event": "distribution_baseline_train_progress", "games": games, "moves": moves_seen}), flush=True)
    return counters, global_moves, {"games": games, "moves": moves_seen, "matched_state_moves": matched}


def majority(counter):
    if not counter:
        return None
    return min(counter, key=lambda move: (-counter[move], move))


def summarize_queries(queries, train_counts, global_majority):
    n = len(queries)
    if not n:
        return {"queries": 0}
    covered = position_correct = global_correct = 0
    occurrence_sum = distinct_sum = majority_share_sum = entropy_sum = 0.0
    for query in queries:
        counter = train_counts.get(query["position_hash"], Counter())
        global_correct += int(query["target"] == global_majority)
        if counter:
            covered += 1
            prediction = majority(counter)
            position_correct += int(query["target"] == prediction)
            total = sum(counter.values())
            occurrence_sum += total
            distinct_sum += len(counter)
            majority_share_sum += max(counter.values()) / total
            entropy_sum += entropy_bits(counter)
    result = {
        "queries": n,
        "global_train_move_majority_accuracy": global_correct / n,
        "train_position_majority": {
            "covered_queries": covered,
            "coverage": covered / n,
            "accuracy_all_queries_uncovered_wrong": position_correct / n,
            "accuracy_covered_queries": position_correct / covered if covered else None,
        },
    }
    if covered:
        result["train_position_distribution"] = {
            "mean_occurrences_per_query": occurrence_sum / covered,
            "mean_distinct_next_moves_per_query": distinct_sum / covered,
            "mean_majority_share_per_query": majority_share_sum / covered,
            "mean_entropy_bits_per_query": entropy_sum / covered,
        }
    return result


def evaluation_concentration(queries):
    counters = defaultdict(Counter)
    for query in queries:
        counters[query["position_hash"]][query["target"]] += 1
    n = len(queries)
    repeated = sum(sum(counter.values()) for counter in counters.values() if sum(counter.values()) > 1)
    majority_hits = sum(max(counter.values()) for counter in counters.values())
    return {
        "queries": n,
        "unique_positions": len(counters),
        "singleton_positions": sum(sum(counter.values()) == 1 for counter in counters.values()),
        "queries_in_repeated_positions": repeated,
        "repeated_position_query_rate": repeated / n if n else None,
        "in_sample_position_majority_accuracy_descriptive_only": majority_hits / n if n else None,
        "macro_distinct_next_moves": sum(len(counter) for counter in counters.values()) / len(counters) if counters else None,
        "macro_entropy_bits": sum(entropy_bits(counter) for counter in counters.values()) / len(counters) if counters else None,
        "warning": "in-sample descriptive statistic; it is not a predictive baseline",
    }


def grouped(queries, key, train_counts, global_majority, ordered_labels=None):
    values = defaultdict(list)
    for query in queries:
        values[str(query[key])].append(query)
    labels = ordered_labels or sorted(values)
    return {label: summarize_queries(values.get(label, []), train_counts, global_majority) for label in labels}


def main():
    args = parse_args()
    distances = parse_distances(args.history_distances)
    primary_distances = parse_distances(args.primary_history_distances)
    cshogi_module = import_cshogi()
    queries, evaluation_games = replay_queries(
        args.evaluation_jsonl, cshogi_module, distances, args.max_games, args.max_queries
    )
    if not queries:
        raise ValueError("no evaluation queries")
    train_counts, global_moves, train_scan = collect_train_counts(
        args.train_jsonl, {query["position_hash"] for query in queries},
        cshogi_module, args.progress_every_games,
    )
    global_majority = majority(global_moves)
    primary = [query for query in queries if query["ply"] in set(primary_distances)]
    for query in queries:
        query["train_frequency_bin"] = frequency_bin(sum(train_counts.get(query["position_hash"], {}).values()))
    output = {
        "format_version": 1,
        "evaluation": "factorized_move_distribution_baselines_v1",
        "settings": {
            "train_jsonl": args.train_jsonl,
            "evaluation_jsonl": args.evaluation_jsonl,
            "history_distances": distances,
            "primary_history_distances": primary_distances,
            "max_games": args.max_games,
            "max_queries": args.max_queries,
        },
        "scan": {"evaluation_games": evaluation_games, "evaluation_queries": len(queries), "train": train_scan},
        "global_train_move": {
            "majority_move": global_majority,
            "occurrences": global_moves[global_majority],
            "total_moves": sum(global_moves.values()),
            "share": global_moves[global_majority] / sum(global_moves.values()),
        },
        "metrics": {
            "all_reported_distances": summarize_queries(queries, train_counts, global_majority),
            "primary": summarize_queries(primary, train_counts, global_majority),
            "by_history_distance": grouped(queries, "ply", train_counts, global_majority, [str(value) for value in distances]),
            "by_position_scope": grouped(primary, "position_scope", train_counts, global_majority),
            "by_trajectory_scope": grouped(primary, "trajectory_scope", train_counts, global_majority),
            "by_train_position_frequency": grouped(
                primary, "train_frequency_bin", train_counts, global_majority,
                [label for label, _, _ in FREQUENCY_BINS],
            ),
            "evaluation_position_concentration": evaluation_concentration(primary),
        },
        "definitions": {
            "position": "normalized SFEN board, hands and side to move; move number is excluded",
            "train_position_majority": "most frequent next move for the same normalized position in the training split",
            "global_train_move_majority": "single most frequent USI move over all training positions; it ignores state",
            "strict_unseen": "trajectory_scope copied from the dataset; it is reported separately from current-position scope",
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_json(path, output)
    print(json.dumps({"event": "distribution_baseline_complete", "output": str(path), "queries": len(queries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
