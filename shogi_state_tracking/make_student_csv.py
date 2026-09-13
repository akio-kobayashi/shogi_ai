#!/usr/bin/env python3
"""study_summary.jsonから学生が分析に使うCSVを作る。

`by_run.csv`は646列あり，そのままでは読めない。ここでは論文の表に対応する
20列程度へ絞り，日本語の列名を付けて出力する。

出力は3つである。
  runs.csv       1行が1モデル（条件×シード）。生の数値
  summary.csv    1行が1条件。3シードの平均と標準偏差
  columns.csv    列の意味と単位の説明
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


CONDITION_LABELS = {
    "vanilla-p0.0": "RAPなし",
    "rap-p0.15-proportional-rap-v1": "RAP q=0.15",
    "rap-p0.25-proportional-rap-v1": "RAP q=0.25",
    "ap-p1.0-proportional-annotation-v1": "AP(参考)",
}

# (日本語の列名, summary側の指標名, 単位, 説明)
# 単位 pct は0〜1の値を百分率へ直す。num はそのまま。
COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("指手パープレキシティ", "move_perplexity", "num",
     "次の指手の当てにくさ。小さいほど良い。RAP注釈用トークンを除いて正規化した値"),
    ("指手top1", "move_top1", "pct", "生成した指手が棋譜と一致した割合"),
    ("指手top5", "move_top5", "pct", "上位5候補に棋譜の指手が含まれた割合"),
    ("top1合法率", "move_top1_legal", "pct", "生成した第1候補が将棋の規則上合法だった割合"),
    ("Lishogi指手PP", "lishogi_move_perplexity", "num", "別のサイトの棋譜での指手パープレキシティ"),
    ("Lishogi指手top1", "lishogi_move_top1", "pct", "同上のtop1一致率"),

    ("選択層", "probe_selected_layer", "num", "プローブの検証損失が最小だった層。12層のうちどこか"),
    ("盤面macroF1", "selected_board_macro_f1", "pct", "選択層の隠れ状態から盤面81マスを当てた精度"),
    ("持ち駒macroF1", "selected_hand_count_macro_f1", "pct", "同じく持ち駒14項目を当てた精度"),
    ("局面完全一致", "selected_full_state_exact_match", "pct", "盤面・持ち駒・手番をすべて当てた割合"),
    ("盤面macroF1_最終層", "final_board_macro_f1", "pct", "最終層での盤面精度。選択層と比べる"),
    ("局面完全一致_最終層", "final_full_state_exact_match", "pct", "最終層での局面完全一致"),
    ("盤面macroF1_多数派", "majority_board_macro_f1", "pct", "常に最頻クラスを答えた場合の精度。比較の下限"),

    ("Start_top1", "token_start_actual_top1", "pct",
     "駒種を与えて移動元を当てる課題。RAPなしでは駒種が未知の入力になる点に注意"),
    ("Start_R精度", "token_start_legal_r_precision", "pct", "同課題で合法な移動元を上位に並べられた度合い"),
    ("End_top1", "token_end_actual_top1", "pct", "移動元を与えて移動先を当てる課題。条件間の公平な比較はこちら"),
    ("End_R精度", "token_end_legal_r_precision", "pct", "同課題のR精度"),

    ("終端復号_最良層", "terminal_best_accuracy", "pct", "棋譜の終わりかどうかを当てた精度の最大値"),
    ("終端復号_多数派", "terminal_majority_accuracy", "pct", "同上の下限（常に多数派を答えた場合）"),

    ("指手依存差_第6層", "action_difference_l6", "num",
     "駒打ちを続けたとき持ち駒を当てやすくなる度合い。正なら当てやすい"),
    ("指手依存差_第9層", "action_difference_l9", "num", "同上"),
    ("指手依存差_第12層", "action_difference_l12", "num", "同上"),

    ("遮断_関連履歴の変化", "ablation_all_relevant_delta", "pct",
     "持ち駒を変えた過去の指手への参照を切ったときの正解駒種確率の変化。負が大きいほど依存している"),
    ("遮断_対照履歴の変化", "ablation_all_matched_control_delta", "pct",
     "無関係な位置を切った場合の変化。関連履歴と比べる"),

    ("持ち駒_駒打ち後の増減正解", "hand_drop_changed_slot_delta_accuracy", "pct",
     "駒打ちで減った持ち駒の変化を当てた割合"),
    ("持ち駒_保有駒への確率", "handdrop_mean_probability_mass_on_held_pieces_given_drop", "pct",
     "駒打ちを指示したとき実際に持っている駒へ置いた確率の合計"),
)


# データセットから決まる下限。モデルの成績ではないので，条件ごとの行には入れず
# baselines.csvへ1行だけ書き出す。全runで同じ値になる。
BASELINE_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("暗記ベースライン_指手top1", "baseline_train_position_top1", "pct",
     "同じ局面を学習データから探し、そこで最も多かった指手を答えた場合の正解率。"
     "モデルの指手top1と同じ母数で比べる。これを超えた分が丸暗記では説明できない部分"),
    ("暗記ベースライン_学習データ被覆率", "baseline_train_position_coverage", "pct",
     "評価局面のうち、同じ局面が学習データにもあった割合"),
    ("暗記ベースライン_被覆局面での正解率", "baseline_train_position_top1_covered", "pct",
     "学習データにあった局面だけに限った正解率"),
    ("最頻手ベースライン_指手top1", "baseline_global_move_top1", "pct",
     "局面を見ずに、学習データ全体で最も多い指手を常に答えた場合の正解率"),
    ("同一局面の指手の種類数", "baseline_mean_distinct_next_moves", "num",
     "学習データで同じ局面から指された手が何種類あったか。多いほど1手に決まらない"),
    ("同一局面の指手のエントロピー", "baseline_mean_entropy_bits", "num",
     "同上のばらつき（ビット）。大きいほど予測が難しい局面"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="build the CSV a student analyses")
    parser.add_argument("--summary", required=True, help="study_summary.json")
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    return parser.parse_args()


def scale(value: Any, unit: str) -> Any:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    return round(value * 100, 2) if unit == "pct" else round(value, 4)


def main() -> int:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # 1行が1モデル。
    run_rows: list[dict[str, Any]] = []
    for run in summary.get("runs", []):
        row: dict[str, Any] = {
            "条件": CONDITION_LABELS.get(run["condition"], run["condition"]),
            "シード": run["seed"],
        }
        for label, metric, unit, _ in COLUMNS:
            row[label] = scale(run.get(metric), unit)
        run_rows.append(row)

    # 1行が1条件。シード間の平均と標準偏差。
    summary_rows: list[dict[str, Any]] = []
    for condition, block in (summary.get("by_condition") or {}).items():
        row: dict[str, Any] = {
            "条件": CONDITION_LABELS.get(condition, condition),
            "シード数": block.get("runs", 0),
        }
        metrics = block.get("metrics") or {}
        for label, metric, unit, _ in COLUMNS:
            record = metrics.get(metric) or {}
            row[f"{label}_平均"] = scale(record.get("mean"), unit)
            deviation = record.get("std")
            row[f"{label}_標準偏差"] = scale(deviation, unit) if deviation is not None else ""
        summary_rows.append(row)

    def write(name: str, fields: list[str], rows: list[dict[str, Any]]) -> None:
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    # 下限はどのrunでも同じ値なので，最初のrunから1行だけ取る。
    runs = summary.get("runs") or []
    first = runs[0] if runs else {}
    baseline_rows: list[dict[str, Any]] = [
        {"指標": label, "値": scale(first.get(metric), unit), "説明": note}
        for label, metric, unit, note in BASELINE_COLUMNS
    ]

    labels = [label for label, _, _, _ in COLUMNS]
    write("runs.csv", ["条件", "シード", *labels], run_rows)
    write("summary.csv",
          ["条件", "シード数", *[f"{label}_{suffix}" for label in labels
                              for suffix in ("平均", "標準偏差")]],
          summary_rows)
    write("columns.csv", ["列名", "単位", "説明"],
          [{"列名": label, "単位": "％" if unit == "pct" else "数値", "説明": note}
           for label, _, unit, note in (*COLUMNS, *BASELINE_COLUMNS)])
    write("baselines.csv", ["指標", "値", "説明"], baseline_rows)

    single = summary.get("single_seed_conditions") or []
    print(json.dumps({
        "event": "student_csv_complete",
        "runs": len(run_rows), "conditions": len(summary_rows), "columns": len(labels),
        "baselines": len(baseline_rows),
        "single_seed_conditions": single,
        "output": str(output),
    }, ensure_ascii=False))
    if single:
        print("注意: 次の条件は1シードしかないため標準偏差が空欄になります: " + ", ".join(single))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
