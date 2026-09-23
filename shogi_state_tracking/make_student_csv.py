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

# 列の定義。(列名, 指標名, 単位, 分類, 何を確認するか, 読み方, 説明)
# 単位 pct は0〜1の値を百分率へ直す。num はそのまま。
# 「何を確認するか」は必ず埋める。数値の意味が分からないまま表に載せさせないため。
COLUMNS: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    # ---- 指手の予測 ----
    ("指手パープレキシティ", "move_perplexity", "num", "指手の予測",
     "次の指手をどれだけ絞り込めているか",
     "小さいほど良い",
     "正解の指手に高い確率を与えているほど小さくなる。1に近いほど迷っていない。"
     "RAP条件でも比べられるよう、駒種を注釈するためのトークンは候補から除いてある"),
    ("指手PP_既出局面", "move_perplexity_seen_position", "num", "指手の予測",
     "学習データに同じ局面があるとき、どれだけ当たるか",
     "小さいほど良い",
     "丸暗記が効く側の値。全体の約3分の2がこちらに入るため、統合値はこの値に引っ張られる"),
    ("指手PP_未見局面", "move_perplexity_unseen_position", "num", "指手の予測",
     "学習で見ていない局面でも絞り込めるか",
     "小さいほど良い",
     "丸暗記の寄与を外した値。統合値より必ず大きくなる。"
     "チェスの先行研究と条件が近いのはこちらで、比較するならこの列を使う"),
    ("指手top1", "move_top1", "pct", "指手の予測",
     "棋譜と同じ手を第1候補に選べるか",
     "大きいほど良い。ただし強さではない",
     "棋譜の指手と一致した割合。良い手かどうかではなく、棋譜に載っていた手と同じかどうかを測っている"),
    ("指手top1_未見局面", "move_top1_unseen_position", "pct", "指手の予測",
     "学習で見ていない局面でも棋譜と一致するか",
     "大きいほど良い",
     "同じ局面が学習データに無い局面だけに限ったtop1"),
    ("指手top5", "move_top5", "pct", "指手の予測",
     "上位5候補まで広げれば棋譜の手が入るか",
     "大きいほど良い",
     "候補を5つ出したとき、そこに棋譜の指手が含まれた割合"),
    ("top1合法率", "move_top1_legal", "pct", "指手の予測",
     "ルールを教えていないのに、合法な手を作れるか",
     "大きいほど良い",
     "モデルには将棋のルールも合法手の一覧も与えていない。"
     "それでも合法な手が出るなら、その時点の盤面を追えていないと説明がつかない。"
     "この研究で最も分かりやすい証拠になる列"),

    # ---- 別の棋譜での予測 ----
    ("Lishogi指手PP", "lishogi_move_perplexity", "num", "別の棋譜での予測",
     "学習に使っていない別の棋譜でも通用するか",
     "小さいほど良い",
     "学習にはコンピュータ同士の棋譜(Floodgate)を使った。"
     "こちらは別のサイト(Lishogi)のBOTでない対局で、指し方の傾向が違う"),
    ("Lishogi指手top1", "lishogi_move_top1", "pct", "別の棋譜での予測",
     "別の棋譜で棋譜と同じ手を第1候補に選べるか", "大きいほど良い",
     "Lishogi棋譜での棋譜一致率"),

    # ---- 局面の読み出し ----
    ("選択層", "probe_selected_layer", "num", "局面の読み出し",
     "12層のうち何層目の隠れ状態を使ったか",
     "そのまま読む。良し悪しではない",
     "層は別に用意したデータで選んだ。結果を見てから都合の良い層を選んだのではないことを示す値"),
    ("盤面macroF1", "selected_board_macro_f1", "pct", "局面の読み出し",
     "盤面が隠れ状態から読み出せるか",
     "「盤面macroF1_多数派」と必ず比べる",
     "選択層の隠れ状態だけを入力にして、81マスそれぞれが空か、どちらのどの駒かを当てた成績。"
     "空マスや歩が多いので、単純な正解率ではなくクラスごとの成績を平均している"),
    ("持ち駒macroF1", "selected_hand_count_macro_f1", "pct", "局面の読み出し",
     "持ち駒が読み出せるか",
     "大きいほど良い",
     "先手後手それぞれ7種類、計14項目の枚数を当てた成績。"
     "持ち駒は盤の上に無いので、過去の駒取りと駒打ちを追えていないと当たらない"),
    ("局面完全一致", "selected_full_state_exact_match", "pct", "局面の読み出し",
     "局面をまるごと復元できるか",
     "大きいほど良い",
     "盤面81マス、持ち駒14項目、手番が全部合った局面の割合。1つでも外れると不一致"),
    ("盤面macroF1_最終層", "final_board_macro_f1", "pct", "局面の読み出し",
     "読み出しやすさが層によって変わるか",
     "「盤面macroF1」(選択層)と比べる",
     "最後の層での成績。選択層より低ければ、途中の層のほうが盤面を読み出しやすいことになる"),
    ("局面完全一致_最終層", "final_full_state_exact_match", "pct", "局面の読み出し",
     "局面まるごとの復元も層によって変わるか", "「局面完全一致」(選択層)と比べる",
     "最終層での局面完全一致"),
    ("盤面macroF1_多数派", "majority_board_macro_f1", "pct", "局面の読み出し",
     "何もしなくても届いてしまう水準はどこか",
     "比較の下限。モデルの値がこれを超えて初めて意味がある",
     "各マスについて学習データで最も多かったクラスを常に答えた場合の成績。"
     "全条件で同じ値になる"),

    # ---- 駒の位置の問い合わせ ----
    ("Start_top1", "token_start_actual_top1", "pct", "駒の位置の問い合わせ",
     "駒の種類を与えたとき、その駒の現在位置を答えられるか",
     "条件間で比べてはいけない",
     "指手の履歴の後に駒種を入れ、次に移動元の座標を出させる課題。"
     "RAPなしのモデルは学習中に駒種トークンを一度も見ていないので、"
     "この入力自体が想定外であり、低くても「追跡できない」根拠にはならない"),
    ("Start_R精度", "token_start_legal_r_precision", "pct", "駒の位置の問い合わせ",
     "正解1つではなく、合法な候補をまとめて上位に置けるか",
     "大きいほど良い",
     "合法な移動元がN個あるとき、モデルの上位N件のうち合法だったものの割合"),
    ("End_top1", "token_end_actual_top1", "pct", "駒の位置の問い合わせ",
     "移動元を与えたとき、移動先を当てられるか",
     "大きいほど良い。条件間を公平に比べられるのはこちら",
     "履歴の後に正しい移動元を入れ、次に移動先を出させる課題。"
     "全条件が学習中に座標トークンを見ているので、Startと違って公平に比べられる"),
    ("End_R精度", "token_end_legal_r_precision", "pct", "駒の位置の問い合わせ",
     "移動先について、合法な候補をまとめて上位に置けるか", "大きいほど良い",
     "合法な移動先がN個あるとき、上位N件のうち合法だったものの割合"),

    # ---- 棋譜の終わり ----
    ("終端復号_選択層", "terminal_selected_accuracy", "pct", "棋譜の終わり",
     "対局が終わる局面かどうかを隠れ状態から判定できるか",
     "「終端復号_多数派」と比べる",
     "各層で判定させ、検証データの損失が最小だった層の、評価データでの正解率。"
     "勝敗の付いた対局の終わりを見分けられるかを見ている"),
    ("終端復号_多数派", "terminal_majority_accuracy", "pct", "棋譜の終わり",
     "終わりかどうかを当てずっぽうで答えた場合はどこまで届くか", "比較の下限。50%前後になる",
     "常に多数派を答えた場合の成績"),

    # ---- 持ち駒への依存 ----
    ("指手依存差_第6層", "action_difference_l6", "num", "持ち駒への依存",
     "次に駒打ちを続けると、持ち駒を読み出しやすくなるか",
     "符号を見る。正なら読み出しやすくなっている",
     "同じ履歴から駒打ちに進んだ場合と、通常の移動に進んだ場合で、"
     "正しい持ち駒枚数に与える確率の差。0に近ければ差がない"),
    ("指手依存差_第9層", "action_difference_l9", "num", "持ち駒への依存",
     "第9層でも同じことが起きているか", "符号を見る。正なら読み出しやすくなっている",
     "第6層・第12層と並べると、層が深いほど差が大きいかが分かる"),
    ("指手依存差_第12層", "action_difference_l12", "num", "持ち駒への依存",
     "最終層でも同じことが起きているか", "符号を見る。正なら読み出しやすくなっている",
     "第6層・第9層と並べると、層が深いほど差が大きいかが分かる"),
    ("遮断_関連履歴の変化", "ablation_all_relevant_delta", "pct", "持ち駒への依存",
     "持ち駒を変えた過去の指手を、実際に見に行っているか",
     "負で大きいほど依存している。必ず「遮断_対照履歴の変化」と比べる",
     "駒を取った手・打った手の位置への参照を切ったとき、正解の駒種に与える確率が何ポイント動いたか"),
    ("遮断_対照履歴の変化", "ablation_all_matched_control_delta", "pct", "持ち駒への依存",
     "参照を切ること自体の影響はどれくらいか",
     "関連履歴より変化が小さければ、選択的に見に行っていることになる",
     "持ち駒と関係のない位置を、同じ数・同じ距離だけ切った場合の変化。"
     "切ること自体の影響を差し引くための対照"),
    ("持ち駒_駒打ち後の増減正解", "hand_drop_changed_slot_delta_accuracy", "pct", "持ち駒への依存",
     "駒を打った後、減った持ち駒を正しく追えているか",
     "大きいほど良い",
     "駒打ちによって枚数が変わった項目について、増減を当てられた割合"),
    ("持ち駒_保有駒への確率", "handdrop_mean_probability_mass_on_held_pieces_given_drop", "pct",
     "持ち駒への依存",
     "持っている駒だけを打とうとするか",
     "大きいほど良い",
     "駒打ちを指示したとき、実際に手元にある駒種へ置いた確率の合計。"
     "持っていない駒を打とうとすれば下がる"),
)


# AP（oracle条件）は評価時にも正解の駒種を与えるため、指手系の値が他条件と
# 同じ意味を持たない。主列には比較可能な値だけを置き、駒種条件付きの診断値は
# 別列へ退避する。混ぜると「APが圧倒的に良い」という誤読になる。
ORACLE_CONDITIONS = ("ap-p1.0-proportional-annotation-v1",)
# APだけ別フィールドから取る主列。正準値は注釈トークンを含めて1指手単位で集計した値。
ORACLE_CANONICAL = {
    "move_perplexity": "move_perplexity_ap_canonical",
    "move_perplexity_seen_position": "move_perplexity_ap_canonical_seen_position",
    "move_perplexity_unseen_position": "move_perplexity_ap_canonical_unseen_position",
}
# 比較可能な対応値が存在しないため、APでは主列を空にする。
ORACLE_WITHHELD = ("move_top1", "move_top1_unseen_position", "move_top5", "move_top1_legal",
                   "lishogi_move_perplexity", "lishogi_move_top1")
# 退避先。AP以外は空になる。
ORACLE_COLUMNS: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    ("AP診断_指手PP", "move_perplexity", "num", "AP専用の診断値",
     "正解の駒種を与えた後、座標だけをどれだけ絞れるか",
     "他条件と比べてはいけない",
     "APは評価のときにも正解の駒種を教えている。"
     "次が駒打ちでないことまで分かった状態での値なので、他条件より小さくて当然"),
    ("AP診断_指手top1", "move_top1", "pct", "AP専用の診断値",
     "駒種を与えた状態で棋譜と一致するか", "他条件と比べてはいけない",
     "駒種を与えた条件下でのtop1"),
    ("AP診断_指手top5", "move_top5", "pct", "AP専用の診断値",
     "駒種を与えた状態で上位5候補に入るか", "他条件と比べてはいけない",
     "駒種を与えた条件下でのtop5"),
    ("AP診断_top1合法率", "move_top1_legal", "pct", "AP専用の診断値",
     "駒種を与えた状態で合法な手を作れるか", "他条件と比べてはいけない",
     "駒種を与えた条件下での合法率"),
    ("AP診断_Lishogi指手PP", "lishogi_move_perplexity", "num", "AP専用の診断値",
     "別の棋譜でも駒種を与えれば絞れるか", "他条件と比べてはいけない",
     "Lishogi棋譜での駒種条件付きの値"),
    ("AP診断_Lishogi指手top1", "lishogi_move_top1", "pct", "AP専用の診断値",
     "別の棋譜で駒種を与えた場合のtop1", "他条件と比べてはいけない",
     "Lishogi棋譜での駒種条件付きのtop1"),
)


def metric_for(condition: str, metric: str) -> str | None:
    """条件に応じて読むフィールドを決める。Noneなら主列を空にする。"""
    if condition not in ORACLE_CONDITIONS:
        return metric
    if metric in ORACLE_WITHHELD:
        return None
    return ORACLE_CANONICAL.get(metric, metric)


# データセットから決まる下限。モデルの成績ではないので，条件ごとの行には入れず
# baselines.csvへ1行だけ書き出す。全runで同じ値になる。
BASELINE_COLUMNS: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    ("暗記ベースライン_指手top1", "baseline_train_position_top1", "pct", "比較の下限",
     "モデルを使わず丸暗記だけでどこまで届くか",
     "モデルの「指手top1」と比べる。超えた分が丸暗記では説明できない部分",
     "評価する局面と同じ局面を学習データから探し、そこで最も多かった指手を答えた場合の正解率。"
     "モデルの指手top1と同じ母数で計算してある"),
    ("暗記ベースライン_学習データ被覆率", "baseline_train_position_coverage", "pct", "比較の下限",
     "評価局面のうち、どれだけが学習でも出てきたか",
     "高いほど丸暗記が効きやすい",
     "同じ局面が学習データにもあった割合。この割合が高いほど、統合値は暗記に助けられている"),
    ("暗記ベースライン_被覆局面での正解率", "baseline_train_position_top1_covered", "pct", "比較の下限",
     "学習にもあった局面に限れば、丸暗記でどこまで当たるか",
     "参考値",
     "被覆された局面だけに限った正解率"),
    ("最頻手ベースライン_指手top1", "baseline_global_move_top1", "pct", "比較の下限",
     "局面をまったく見ない場合はどこまで届くか",
     "最も低い下限。数%にしかならない",
     "学習データ全体で最も多い指手を、局面によらず常に答えた場合の正解率"),
    ("同一局面の指手の種類数", "baseline_mean_distinct_next_moves", "num", "比較の下限",
     "同じ局面から何通りの手が指されているか",
     "少ないほど予測しやすい棋譜",
     "学習データで同じ局面から指された手の種類数の平均。"
     "コンピュータ同士の棋譜は指し方が偏るため少なくなる"),
    ("同一局面の指手のエントロピー", "baseline_mean_entropy_bits", "num", "比較の下限",
     "同じ局面での手のばらつきはどれくらいか",
     "小さいほど1手に決まっている",
     "上の種類数を確率のばらつき(ビット)で表したもの。"
     "この値が小さいこと自体が、パープレキシティが小さく出る理由になる"),
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
        for label, metric, unit, *_ in COLUMNS:
            source = metric_for(run["condition"], metric)
            row[label] = scale(run.get(source), unit) if source else ""
        for label, metric, unit, *_ in ORACLE_COLUMNS:
            row[label] = scale(run.get(metric), unit) if run["condition"] in ORACLE_CONDITIONS else ""
        run_rows.append(row)

    # 1行が1条件。シード間の平均と標準偏差。
    summary_rows: list[dict[str, Any]] = []
    for condition, block in (summary.get("by_condition") or {}).items():
        row: dict[str, Any] = {
            "条件": CONDITION_LABELS.get(condition, condition),
            "シード数": block.get("runs", 0),
        }
        metrics = block.get("metrics") or {}
        oracle = condition in ORACLE_CONDITIONS

        def put(label: str, metric: str | None, unit: str) -> None:
            record = (metrics.get(metric) or {}) if metric else {}
            row[f"{label}_平均"] = scale(record.get("mean"), unit) if metric else ""
            deviation = record.get("std") if metric else None
            row[f"{label}_標準偏差"] = scale(deviation, unit) if deviation is not None else ""

        for label, metric, unit, *_ in COLUMNS:
            put(label, metric_for(condition, metric), unit)
        for label, metric, unit, *_ in ORACLE_COLUMNS:
            put(label, metric if oracle else None, unit)
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
        {"指標": label, "値": scale(first.get(metric), unit),
         "何を確認するか": question, "読み方": reading, "説明": note}
        for label, metric, unit, _group, question, reading, note in BASELINE_COLUMNS
    ]

    labels = [label for label, *_ in (*COLUMNS, *ORACLE_COLUMNS)]
    write("runs.csv", ["条件", "シード", *labels], run_rows)
    write("summary.csv",
          ["条件", "シード数", *[f"{label}_{suffix}" for label in labels
                              for suffix in ("平均", "標準偏差")]],
          summary_rows)
    write("columns.csv", ["分類", "列名", "単位", "何を確認するか", "読み方", "説明"],
          [{"分類": group, "列名": label, "単位": "％" if unit == "pct" else "数値",
            "何を確認するか": question, "読み方": reading, "説明": note}
           for label, _, unit, group, question, reading, note
           in (*COLUMNS, *ORACLE_COLUMNS, *BASELINE_COLUMNS)])
    write("baselines.csv", ["指標", "値", "何を確認するか", "読み方", "説明"], baseline_rows)

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
