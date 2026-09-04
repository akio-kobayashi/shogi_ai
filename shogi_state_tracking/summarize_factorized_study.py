#!/usr/bin/env python3
"""収集済みbundleを読み，条件×シードで集約する。

論文の数値を人手で転記しないための中間層である。ここが唯一の集約点であり，
`render_paper_tables.py`は本スクリプトの`study_summary.json`だけを読む。

シード間の分散をここで初めて計算する。対局クラスタブートストラップの信頼区間は
評価対局の変動を表すもので，学習シード間の変動とは別物なので分けて保持する。

層に依存する指標は，検証損失が最小の層を各runで選ぶ。層番号を固定しないのは，
論文が「検証集合で選択した層」を報告する方法に合わせるためである。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from provenance import write_metrics_json


CONDITIONS = (
    "vanilla-p0.0",
    "rap-p0.15-proportional-rap-v1",
    "rap-p0.25-proportional-rap-v1",
    "ap-p1.0-proportional-annotation-v1",
)
PRIMARY_CONDITIONS = CONDITIONS[:3]
ACTION_CONDITION_LAYERS = ("layer_6", "layer_9", "layer_12")
ABLATION_SCOPES = ("middle", "late", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="aggregate a factorized-v3 study")
    parser.add_argument("--bundle", required=True, help="analysis_bundleディレクトリ")
    parser.add_argument("--output", required=True, help="summary出力ディレクトリ")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default="")
    return parser.parse_args()


def dig(payload: Any, *keys: Any) -> Any:
    """Follow a key path, returning None as soon as anything is missing."""
    current = payload
    for key in keys:
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return None
    return current


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def layer_index(name: str) -> int:
    try:
        return int(str(name).rsplit("_", 1)[-1])
    except ValueError:
        return -1


def select_probe_layer(probe_results: Mapping[str, Any]) -> str | None:
    """Pick the layer with the lowest probe validation loss, as the paper does."""
    scored = [
        (value["best_validation_loss"], name)
        for name, value in probe_results.items()
        if isinstance(value, Mapping) and isinstance(value.get("best_validation_loss"), (int, float))
    ]
    if not scored:
        return None
    return min(scored)[1]


# ---------------------------------------------------------------- extractors

# 主評価の指手パープレキシティはcanonical側である。RAP注釈用の駒種トークンの
# logitを除いてから正規化した値であり，注釈が評価時に出現しない条件と揃う。
# 生のmove_perplexityはRAP条件で大きく異なる（q=0.15で4.242対3.664）ため，
# 別名で併せて残し，どちらを報告しているかを曖昧にしない。
MOVE_FIELDS = {
    "move_perplexity": "canonical_move_perplexity",
    "move_perplexity_raw": "move_perplexity",
    "move_perplexity_grammar": "grammar_normalized_move_perplexity",
    "move_nll": "canonical_move_nll",
    "move_nll_raw": "move_nll",
    "move_top1": "greedy_full_move_top1",
    "move_top5": "beam_full_move_top5",
    "move_top1_legal": "greedy_legal_rate",
    "move_legal_mass": "beam_legal_probability_lower_bound",
    "move_queries": "queries",
    # APだけに存在する正準値。canonical_move_perplexityはAPでは駒種条件付きの
    # 診断値になるため，チェス先行研究と方法上対応するのはこちらである。
    "move_perplexity_ap_canonical": "ap_annotated_move_perplexity",
}


def extract_moves(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    primary = dig(payload, "metrics", "primary") or {}
    for name, key in MOVE_FIELDS.items():
        values[f"{prefix}{name}"] = primary.get(key)
    for distance, block in (dig(payload, "metrics", "by_history_distance") or {}).items():
        if not isinstance(block, Mapping):
            continue
        for name, key in MOVE_FIELDS.items():
            values[f"{prefix}{name}_h{distance}"] = block.get(key)
    for scope, block in (dig(payload, "metrics", "by_position_scope") or {}).items():
        if not isinstance(block, Mapping):
            continue
        for name, key in MOVE_FIELDS.items():
            values[f"{prefix}{name}_{scope}"] = block.get(key)
    complete = dig(payload, "complete_move_evaluation", "primary") or {}
    values[f"{prefix}complete_move_top1"] = complete.get("complete_action_beam_top1_exact")
    values[f"{prefix}complete_move_top5"] = complete.get("complete_action_beam_top5_exact")
    values[f"{prefix}complete_move_top1_legal"] = complete.get("complete_action_beam_top1_legal")
    return values


PROBE_FIELDS = (
    "board_macro_f1",
    "hand_count_macro_f1",
    "hand_count_pooled_macro_f1",
    "board_exact_match",
    "hand_exact_match",
    "full_state_exact_match",
    "board_square_accuracy",
    "board_piece_accuracy_on_occupied",
    "hand_slot_accuracy",
    "hand_mae",
    "turn_accuracy",
    "in_check_f1",
)


def extract_probes(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    results = dig(payload, "probe_results") or {}
    selected = select_probe_layer(results)
    layers = sorted(results, key=layer_index)
    positions = {
        "input": layers[0] if layers else None,
        "selected": selected,
        "final": layers[-1] if layers else None,
    }
    values[f"{prefix}probe_selected_layer"] = layer_index(selected) if selected else None
    for position, layer in positions.items():
        block = dig(results, layer, "evaluation") if layer else None
        for field in PROBE_FIELDS:
            values[f"{prefix}{position}_{field}"] = dig(block, field)
    for field in PROBE_FIELDS:
        values[f"{prefix}majority_{field}"] = dig(payload, "majority_baseline", field)
    values[f"{prefix}probe_samples"] = dig(results, selected, "evaluation", "samples") if selected else None
    values[f"{prefix}game_splits_disjoint"] = payload.get("game_splits_disjoint")
    return values


TOKEN_PROBE_FIELDS = (
    "queries",
    "start_actual_top1", "start_actual_top5", "start_legal_r_precision",
    "start_other_top1", "start_other_probability_mass",
    "end_actual_top1", "end_actual_top5", "end_legal_r_precision",
    "end_other_top1", "end_other_probability_mass",
)


def extract_token_probe(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dig(payload, "metrics") or {}
    values = {f"token_{field}": metrics.get(field) for field in TOKEN_PROBE_FIELDS}
    return values


def extract_terminal_probe(payload: Mapping[str, Any]) -> dict[str, Any]:
    task = dig(payload, "tasks", "terminal_next") or {}
    scored = [
        (dig(block, "evaluation", "accuracy"), name)
        for name, block in task.items()
        if isinstance(block, Mapping) and dig(block, "evaluation", "accuracy") is not None
    ]
    values: dict[str, Any] = {}
    layers = sorted(task, key=layer_index)
    if scored:
        best_accuracy, best_layer = max(scored)
        values["terminal_best_accuracy"] = best_accuracy
        values["terminal_best_layer"] = layer_index(best_layer)
        values["terminal_best_macro_f1"] = dig(task, best_layer, "evaluation", "macro_f1")
    else:
        values.update(terminal_best_accuracy=None, terminal_best_layer=None,
                      terminal_best_macro_f1=None)
    values["terminal_input_accuracy"] = dig(task, layers[0], "evaluation", "accuracy") if layers else None
    values["terminal_final_accuracy"] = dig(task, layers[-1], "evaluation", "accuracy") if layers else None
    values["terminal_majority_accuracy"] = (
        dig(task, layers[0], "majority_baseline", "accuracy") if layers else None
    )
    return values


def extract_action_condition(payload: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for layer in ACTION_CONDITION_LAYERS:
        block = dig(payload, "metrics", layer, "pooled_probe_within_prefix", "difference")
        index = layer_index(layer)
        values[f"action_difference_l{index}"] = dig(block, "mean")
        values[f"action_difference_l{index}_ci_lower"] = dig(block, "clustered_95ci", "lower")
        values[f"action_difference_l{index}_ci_upper"] = dig(block, "clustered_95ci", "upper")
        values[f"action_difference_l{index}_clusters"] = dig(block, "clustered_95ci", "clusters")
        values[f"action_instances_l{index}"] = dig(
            payload, "metrics", layer, "pooled_probe_within_prefix", "instances"
        )
    values["action_evaluation_games"] = dig(payload, "split_audit", "game_counts", "evaluation")
    values["action_evaluation_pairs"] = dig(payload, "branch_summary", "evaluation", "pairs")
    return values


def extract_attention_ablation(payload: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for scope in ABLATION_SCOPES:
        for group in ("relevant", "matched_control"):
            block = dig(payload, "ablation", f"drop:{scope}:{group}:after_drop")
            base = dig(block, "baseline_probability")
            masked = dig(block, "masked_probability")
            values[f"ablation_{scope}_{group}_baseline"] = base
            values[f"ablation_{scope}_{group}_masked"] = masked
            values[f"ablation_{scope}_{group}_delta"] = (
                None if base is None or masked is None else masked - base
            )
            values[f"ablation_{scope}_{group}_examples"] = dig(block, "examples")
        # 対応付き差分とそのクラスタCI（実装後の成果物にだけ存在する）
        contrast = dig(payload, "ablation_contrasts", f"drop:{scope}:after_drop")
        values[f"ablation_{scope}_contrast_mean"] = dig(contrast, "probability_change_difference")
        values[f"ablation_{scope}_contrast_ci_lower"] = dig(
            contrast, "probability_change_difference_clustered_95ci", "lower")
        values[f"ablation_{scope}_contrast_ci_upper"] = dig(
            contrast, "probability_change_difference_clustered_95ci", "upper")
    values["ablation_matched_pairs"] = dig(payload, "matching", "matched_pairs")
    values["ablation_mask_logit_error"] = payload.get("no_mask_forward_max_absolute_logit_error")
    return values


# artifact key -> (relative path, extractor). 欠けている成果物はNoneのまま残す。
ARTIFACTS: tuple[tuple[str, str, Callable[[Mapping[str, Any]], dict[str, Any]]], ...] = (
    ("moves", "move_metrics.json", extract_moves),
    ("lishogi-moves", "lishogi-non-bot/moves/move_metrics.json",
     lambda payload: extract_moves(payload, prefix="lishogi_")),
    ("probes", "probes/probe_metrics.json", extract_probes),
    ("lishogi-probes", "lishogi-non-bot/linear-probes/probe_metrics.json",
     lambda payload: extract_probes(payload, prefix="lishogi_")),
    ("token-probe", "token_probe_metrics.json", extract_token_probe),
    ("terminal-probe", "terminal-probe/action_probe_metrics.json", extract_terminal_probe),
    ("action-condition", "action-condition/primary/action_condition_robustness.json",
     extract_action_condition),
    ("attention-ablation", "action-condition/primary/action_condition_attention_ablation.json",
     extract_attention_ablation),
)
# APはprimaryではなくoracle-native側へ保存されるため，参照先を差し替える。
ORACLE_REPLACEMENTS = {
    "action-condition": "action-condition/oracle-native/action_condition_robustness.json",
}
# APは過去の通常移動へ正解駒種注釈を含むoracle条件なので，注意遮断を実行しない。
# 欠損ではなく設計上の除外であり，missing_artifactsへ数えない。
ORACLE_EXCLUDED = ("attention-ablation",)
# oracle条件だけが持つ注釈除去プロトコル。主比較へpoolせず別prefixで保持する。
ORACLE_ONLY: tuple[tuple[str, str, Callable[[Mapping[str, Any]], dict[str, Any]]], ...] = (
    ("action-condition-no-annotation",
     "action-condition/sensitivity-no-annotation/action_condition_robustness.json",
     lambda payload: {f"noann_{key}": value
                      for key, value in extract_action_condition(payload).items()}),
)
# 指標仕様が未定義の成果物。生成されたら抽出器を追加する。
# distribution-baselinesは収集はされるが，フィールド名を実物で確認できていない。
PENDING_ARTIFACTS = ("chess-protocol", "hand-dynamics", "policy-relevance",
                     "drop-relevance", "distribution-baselines")
# 契約にあるが意図的に集約しない成果物と，その理由。
# tests/test_pipeline_contracts.pyがこの宣言と契約を突き合わせ，
# 宣言のない取りこぼしを失敗として検出する。
NOT_SUMMARIZED = {
    "action_condition_metrics.json":
        "旧h_pre probeによる予備解析である。主結果はaction_condition_robustness.jsonの"
        "pooled_probe_within_prefixであり，READMEもそちらを主結果と定めている。",
}


# 図が必要とする層別系列。CSVはスカラーだけを持つので，JSON側へ分けて保持する。
SERIES_PROBE_FIELDS = ("board_macro_f1", "hand_count_macro_f1", "full_state_exact_match")


def collect_series(run_dir: Path, condition: str) -> dict[str, Any]:
    """Per-layer series for the figures. Kept out of the CSVs, which stay scalar."""
    evaluation = run_dir / "evaluation"
    series: dict[str, Any] = {}
    probes = load_json(evaluation / "probes/probe_metrics.json")
    if probes is not None:
        results = dig(probes, "probe_results") or {}
        series["probe_by_layer"] = {
            str(layer_index(name)): {
                field: dig(results, name, "evaluation", field) for field in SERIES_PROBE_FIELDS
            }
            for name in sorted(results, key=layer_index)
        }
    terminal = load_json(evaluation / "terminal-probe/action_probe_metrics.json")
    if terminal is not None:
        task = dig(terminal, "tasks", "terminal_next") or {}
        series["terminal_by_layer"] = {
            str(layer_index(name)): dig(task, name, "evaluation", "accuracy")
            for name in sorted(task, key=layer_index)
        }
    return series


def collect_run(run_dir: Path, condition: str) -> tuple[dict[str, Any], list[str]]:
    evaluation = run_dir / "evaluation"
    values: dict[str, Any] = {}
    missing: list[str] = []
    oracle = condition not in PRIMARY_CONDITIONS
    for key, relative, extractor in ARTIFACTS:
        if oracle:
            if key in ORACLE_EXCLUDED:
                continue
            relative = ORACLE_REPLACEMENTS.get(key, relative)
        payload = load_json(evaluation / relative)
        if payload is None:
            missing.append(key)
            continue
        values.update(extractor(payload))
    if oracle:
        for key, relative, extractor in ORACLE_ONLY:
            payload = load_json(evaluation / relative)
            if payload is None:
                missing.append(key)
                continue
            values.update(extractor(payload))
    epoch = dig(load_json(run_dir / "training_history.json") or {}, "history")
    if isinstance(epoch, list) and epoch:
        values["epoch"] = max(
            (entry.get("epoch") for entry in epoch if isinstance(entry, Mapping)
             and isinstance(entry.get("epoch"), int)),
            default=None,
        )
    return values, missing


def discover_runs(results: Path, conditions: Iterable[str], seeds: Iterable[str]) -> list[tuple[str, str, Path]]:
    wanted = set(seeds)
    runs: list[tuple[str, str, Path]] = []
    for condition in conditions:
        base = results / "llama-reference" / "implicit-initial" / condition
        if not base.is_dir():
            continue
        for directory in sorted(base.glob("seed-*")):
            seed = directory.name.removeprefix("seed-")
            if wanted and seed not in wanted:
                continue
            runs.append((condition, seed, directory))
    return runs


def aggregate(rows: list[dict[str, Any]], names: list[str]) -> dict[str, dict[str, Any]]:
    """Mean and sample standard deviation across seeds, per metric."""
    summary: dict[str, dict[str, Any]] = {}
    for name in names:
        values = [row[name] for row in rows
                  if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)]
        if not values:
            summary[name] = {"mean": None, "std": None, "n": 0, "values": []}
            continue
        summary[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values),
            "values": values,
        }
    return summary


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    bundle = Path(args.bundle).expanduser().resolve()
    results = bundle / "results"
    if not results.is_dir():
        results = bundle
    output = Path(args.output).expanduser().resolve()
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    seeds = [item.strip() for item in args.seeds.split(",") if item.strip()]

    runs = discover_runs(results, conditions, seeds)
    if not runs:
        print(json.dumps({"event": "no_runs_found", "results": str(results)}, ensure_ascii=False))
        return 2

    by_run: list[dict[str, Any]] = []
    series: dict[str, Any] = {}
    metric_names: list[str] = []
    for condition, seed, run_dir in runs:
        values, missing = collect_run(run_dir, condition)
        row: dict[str, Any] = {"condition": condition, "seed": seed}
        row.update(values)
        row["missing_artifacts"] = ";".join(missing)
        by_run.append(row)
        series[f"{condition}/seed-{seed}"] = collect_series(run_dir, condition)
        for name in values:
            if name not in metric_names:
                metric_names.append(name)

    by_condition: dict[str, Any] = {}
    condition_rows: list[dict[str, Any]] = []
    for condition in conditions:
        rows = [row for row in by_run if row["condition"] == condition]
        if not rows:
            continue
        summary = aggregate(rows, metric_names)
        by_condition[condition] = {"runs": len(rows), "seeds": [row["seed"] for row in rows],
                                   "metrics": summary}
        flat: dict[str, Any] = {"condition": condition, "runs": len(rows),
                                "seeds": ";".join(row["seed"] for row in rows)}
        for name, entry in summary.items():
            flat[f"{name}_mean"] = entry["mean"]
            flat[f"{name}_std"] = entry["std"]
        condition_rows.append(flat)

    run_fields = ["condition", "seed", *metric_names, "missing_artifacts"]
    condition_fields = ["condition", "runs", "seeds"]
    for name in metric_names:
        condition_fields += [f"{name}_mean", f"{name}_std"]
    write_csv(output / "by_run.csv", run_fields, by_run)
    write_csv(output / "by_condition.csv", condition_fields, condition_rows)

    single_seed = sorted(
        condition for condition, entry in by_condition.items()
        if entry["runs"] < 3 and condition in PRIMARY_CONDITIONS
    )
    document = {
        "format_version": 1,
        "bundle": str(bundle),
        "conditions": conditions,
        "primary_conditions": list(PRIMARY_CONDITIONS),
        "metric_names": metric_names,
        "pending_artifacts": list(PENDING_ARTIFACTS),
        "single_seed_conditions": single_seed,
        "interpretation_limits": [
            "clustered_95ci fields describe evaluation-game variation, not training-seed variation.",
            "std is the sample standard deviation across training seeds and is null when a condition has one run.",
            "Conditions listed in single_seed_conditions carry no seed variance and stay exploratory.",
            "Oracle AP results come from the oracle-native protocol and are not pooled with the primary conditions.",
        ],
        "runs": by_run,
        "by_condition": by_condition,
        "series": series,
    }
    write_metrics_json(output / "study_summary.json", document)

    print(json.dumps({
        "event": "summary_complete",
        "runs": len(by_run),
        "conditions": len(by_condition),
        "metrics": len(metric_names),
        "single_seed_conditions": single_seed,
        "output": str(output),
    }, ensure_ascii=False))
    for row in by_run:
        if row["missing_artifacts"]:
            print(f"MISSING [{row['condition']}/seed-{row['seed']}] {row['missing_artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
