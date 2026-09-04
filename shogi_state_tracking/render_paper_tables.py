#!/usr/bin/env python3
"""study_summary.jsonから論文の表と図を生成する。

論文本文には数値を書かず，ここが出力する`paper/tables/*.tex`を`\\input{}`で
差し込む。丸め，パーセント換算，信頼区間の書式はすべて本スクリプトのTABLES
定義に集約し，表番号と1対1で対応させる。

シード間の標準偏差が得られている指標は`mean ± std`で描く。単一シードの条件では
平均だけを描き，条件名に注記を付ける。複数シードへ移行したときに本文を触らずに
表が更新される。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from provenance import provenance_record


CONDITION_LABELS = {
    "vanilla-p0.0": "RAPなし",
    "rap-p0.15-proportional-rap-v1": "RAP $q=0.15$",
    "rap-p0.25-proportional-rap-v1": "RAP $q=0.25$",
    "ap-p1.0-proportional-annotation-v1": "AP（正解注釈あり）",
}
POSITION_LABELS = {"input": "入力埋め込み", "selected": "選択層", "final": "最終層"}


@dataclass(frozen=True)
class Column:
    label: str
    metric: str
    fmt: str = "pct"     # pct | num3 | num4 | int | ci


@dataclass(frozen=True)
class Table:
    key: str
    caption: str
    columns: tuple[Column, ...]
    kind: str = "condition"          # condition | condition_position
    conditions: tuple[str, ...] | None = None   # None = summaryの全条件
    note: str = ""


TABLES: tuple[Table, ...] = (
    Table(
        key="move_prediction",
        conditions=("vanilla-p0.0", "rap-p0.15-proportional-rap-v1",
                    "rap-p0.25-proportional-rap-v1"),
        caption="指手予測（Floodgate評価集合，統合）",
        columns=(
            Column("局面数", "move_queries", "int"),
            Column("指手パープレキシティ", "move_perplexity", "num3"),
            Column("指手top-1", "move_top1"),
            Column("指手top-5", "move_top5"),
            Column("top-1合法率", "move_top1_legal"),
        ),
        note="指手パープレキシティはRAP注釈用の駒種トークンを候補から除いて正規化した値である．",
    ),
    Table(
        key="move_prediction_unseen",
        conditions=("vanilla-p0.0", "rap-p0.15-proportional-rap-v1",
                    "rap-p0.25-proportional-rap-v1"),
        caption="学習集合に同一局面がない評価局面での指手予測",
        columns=(
            Column("局面数", "move_queries_unseen_position", "int"),
            Column("指手パープレキシティ", "move_perplexity_unseen_position", "num3"),
            Column("指手top-1", "move_top1_unseen_position"),
            Column("指手top-5", "move_top5_unseen_position"),
            Column("top-1合法率", "move_top1_legal_unseen_position"),
        ),
    ),
    Table(
        key="move_prediction_lishogi",
        conditions=("vanilla-p0.0", "rap-p0.15-proportional-rap-v1",
                    "rap-p0.25-proportional-rap-v1"),
        caption="Lishogi非BOT棋譜での指手予測",
        columns=(
            Column("局面数", "lishogi_move_queries", "int"),
            Column("指手パープレキシティ", "lishogi_move_perplexity", "num3"),
            Column("指手top-1", "lishogi_move_top1"),
            Column("指手top-5", "lishogi_move_top5"),
            Column("top-1合法率", "lishogi_move_top1_legal"),
        ),
    ),
    Table(
        key="state_decoding",
        caption="線形プローブによる局面状態の復号",
        kind="condition_position",
        columns=(
            Column("盤面macro-F1", "board_macro_f1"),
            Column("持ち駒macro-F1", "hand_count_macro_f1"),
            Column("盤面完全一致", "board_exact_match"),
            Column("持ち駒完全一致", "hand_exact_match"),
            Column("局面完全一致", "full_state_exact_match"),
        ),
        note="層は検証損失が最小のものを各runで選択した．手番は完了指手数の偶奇から決まるため局面完全一致へ実質的に寄与しない．",
    ),
    Table(
        key="start_end_task",
        caption="駒種を条件とする位置予測（Start／End課題）",
        conditions=("vanilla-p0.0", "rap-p0.15-proportional-rap-v1",
                    "rap-p0.25-proportional-rap-v1"),
        columns=(
            Column("クエリ数", "token_queries", "int"),
            Column("Start top-1", "token_start_actual_top1"),
            Column("Start R-Prec", "token_start_legal_r_precision"),
            Column("End top-1", "token_end_actual_top1"),
            Column("End R-Prec", "token_end_legal_r_precision"),
        ),
        note="RAPなしモデルにとって駒種トークンは分布外入力である．条件間の公平な比較にはEnd課題を用いる．",
    ),
    Table(
        key="terminal_probe",
        caption="完全棋譜終端の線形復号",
        columns=(
            Column("入力埋め込み", "terminal_input_accuracy"),
            Column("最良層", "terminal_best_layer", "int"),
            Column("最良層の正解率", "terminal_best_accuracy"),
            Column("最終層", "terminal_final_accuracy"),
            Column("多数派", "terminal_majority_accuracy"),
        ),
        note="比較対象は多数派ベースラインのみである．棋譜長だけを用いるベースラインは未実装であり，局面理解にもとづく終端予測とは解釈できない．",
    ),
    Table(
        key="action_dependence",
        caption="持ち駒復号の指手依存差（駒打ち分岐－通常移動3分岐の平均）",
        columns=(
            Column("第6層", "action_difference_l6", "ci"),
            Column("第9層", "action_difference_l9", "ci"),
            Column("第12層", "action_difference_l12", "ci"),
        ),
        note="括弧内は評価対局を単位とするクラスタ・ブートストラップの95\\%信頼区間である．学習シード間の変動は含まない．",
    ),
    Table(
        key="ap_sensitivity",
        caption="APによる感度分析（oracle条件・主比較へpoolしない）",
        conditions=("ap-p1.0-proportional-annotation-v1",),
        columns=(
            Column("AP正準指手PP", "move_perplexity_ap_canonical", "num3"),
            Column("駒種条件付きPP", "move_perplexity", "num3"),
            Column("条件付きtop-1", "move_top1"),
            Column("条件付きtop-5", "move_top5"),
            Column("注釈あり第12層", "action_difference_l12", "ci"),
            Column("注釈除去第12層", "noann_action_difference_l12", "ci"),
        ),
        note="AP正準指手PPは注釈付き指手を1単位とする値で，チェス先行研究と方法上対応する．"
             "駒種条件付きPPは正解駒種を与えた後だけを採点する診断値であり，両者は比較できない．"
             "注釈を除くと指手依存差が消えるため，注釈のない入力へは一般化しない．",
    ),
    Table(
        key="attention_ablation",
        caption="注意接続の遮断による正解駒種確率の変化（全層遮断）",
        conditions=("vanilla-p0.0", "rap-p0.15-proportional-rap-v1",
                    "rap-p0.25-proportional-rap-v1"),
        columns=(
            Column("遮断前", "ablation_all_relevant_baseline"),
            Column("関連履歴遮断後", "ablation_all_relevant_masked"),
            Column("関連履歴の変化", "ablation_all_relevant_delta"),
            Column("対照履歴の変化", "ablation_all_matched_control_delta"),
        ),
        note="APはoracle条件のため除外する．",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="render paper tables and figures")
    parser.add_argument("--summary", required=True, help="study_summary.json")
    parser.add_argument("--output", required=True, help="paper出力ディレクトリ")
    parser.add_argument("--tables", default="", help="生成する表をカンマ区切りで限定する")
    return parser.parse_args()


# ---------------------------------------------------------------- formatting

def entry(summary: Mapping[str, Any], condition: str, metric: str) -> Mapping[str, Any] | None:
    return (summary.get("by_condition", {}).get(condition, {}).get("metrics", {}) or {}).get(metric)


def render_value(summary: Mapping[str, Any], condition: str, column: Column) -> str:
    if column.fmt == "ci":
        return render_interval(summary, condition, column.metric)
    record = entry(summary, condition, column.metric)
    if record is None or record.get("mean") is None:
        return "---"
    mean, std = record["mean"], record.get("std")
    if column.fmt == "int":
        text = f"{round(mean):,}"
        return text if std in (None, 0) else f"{text} $\\pm$ {std:.1f}"
    if column.fmt == "pct":
        text, spread = f"{mean * 100:.2f}\\%", None if std is None else f"{std * 100:.2f}"
    elif column.fmt == "num4":
        text, spread = f"{mean:.4f}", None if std is None else f"{std:.4f}"
    else:
        text, spread = f"{mean:.3f}", None if std is None else f"{std:.3f}"
    return text if spread is None else f"{text} $\\pm$ {spread}"


def render_interval(summary: Mapping[str, Any], condition: str, metric: str) -> str:
    record = entry(summary, condition, metric)
    if record is None or record.get("mean") is None:
        return "---"
    lower = entry(summary, condition, f"{metric}_ci_lower")
    upper = entry(summary, condition, f"{metric}_ci_upper")
    text = f"{record['mean']:.3f}"
    if record.get("std") is not None:
        text = f"{text} $\\pm$ {record['std']:.3f}"
    if lower and upper and lower.get("mean") is not None and upper.get("mean") is not None:
        text = f"{text} [{lower['mean']:.3f}, {upper['mean']:.3f}]"
    return text


def position_value(summary: Mapping[str, Any], condition: str, position: str, column: Column) -> str:
    return render_value(summary, condition, Column(column.label, f"{position}_{column.metric}", column.fmt))


def condition_label(summary: Mapping[str, Any], condition: str) -> str:
    label = CONDITION_LABELS.get(condition, condition)
    runs = summary.get("by_condition", {}).get(condition, {}).get("runs", 0)
    if runs == 1:
        return f"{label}\\,\\textsuperscript{{†}}"
    return label


def table_rows(summary: Mapping[str, Any], table: Table) -> list[list[str]]:
    conditions = [c for c in (table.conditions or summary.get("conditions", []))
                  if c in summary.get("by_condition", {})]
    rows: list[list[str]] = []
    if table.kind == "condition_position":
        for condition in conditions:
            for index, position in enumerate(POSITION_LABELS):
                label = condition_label(summary, condition) if index == 0 else ""
                cells = [position_value(summary, condition, position, column)
                         for column in table.columns]
                rows.append([label, POSITION_LABELS[position], *cells])
        majority = [render_value(summary, conditions[0], Column(c.label, f"majority_{c.metric}", c.fmt))
                    if conditions else "---" for c in table.columns]
        rows.append(["多数派ベースライン", "", *majority])
    else:
        for condition in conditions:
            rows.append([condition_label(summary, condition),
                         *[render_value(summary, condition, column) for column in table.columns]])
    return rows


def header_cells(table: Table) -> list[str]:
    prefix = ["条件", "位置"] if table.kind == "condition_position" else ["条件"]
    return prefix + [column.label for column in table.columns]


# ---------------------------------------------------------------- writers

def write_tex(path: Path, table: Table, header: Sequence[str], rows: Sequence[Sequence[str]],
              stamp: str, single_seed: bool) -> None:
    alignment = "l" + ("l" if table.kind == "condition_position" else "") + "r" * len(table.columns)
    lines = [
        f"% 自動生成：render_paper_tables.py（{stamp}）",
        "% 内容を変更する場合はstudy_summary.jsonを再生成すること．",
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{table.caption}}}",
        f"  \\label{{tab:{table.key}}}",
        f"  \\begin{{tabular}}{{{alignment}}}",
        "    \\toprule",
        "    " + " & ".join(header) + " \\\\",
        "    \\midrule",
    ]
    lines += ["    " + " & ".join(row) + " \\\\" for row in rows]
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    notes = [table.note] if table.note else []
    if single_seed:
        notes.append("†は単一シードの条件であり，学習シード間の変動を含まない探索的な値である．")
    for note in notes:
        lines.append(f"  \\par\\smallskip\\footnotesize {note}")
    lines += ["\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(path: Path, table: Table, header: Sequence[str], rows: Sequence[Sequence[str]],
                   stamp: str, single_seed: bool) -> None:
    def clean(text: str) -> str:
        for source, target in (("\\%", "%"), ("$\\pm$", "±"), ("\\,", ""),
                               ("\\textsuperscript{†}", "†"), ("{", ""), ("}", ""),
                               ("$", ""), ("---", "—")):
            text = text.replace(source, target)
        return text.strip()

    lines = [f"<!-- 自動生成：render_paper_tables.py（{stamp}） -->",
             f"**{clean(table.caption)}**", ""]
    lines.append("| " + " | ".join(clean(cell) for cell in header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(clean(cell) for cell in row) + " |")
    notes = [table.note] if table.note else []
    if single_seed:
        notes.append("†は単一シードの条件であり，学習シード間の変動を含まない探索的な値である．")
    if notes:
        lines.append("")
        lines += [clean(note) for note in notes]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- figures

PALETTE = ("#2c4a7c", "#8a6432", "#46664a", "#8c3a36")


def line_figure(path: Path, title: str, y_label: str,
                series: Mapping[str, Sequence[tuple[float, float]]],
                y_range: tuple[float, float]) -> None:
    """Hand-authored SVG. The project does not depend on a plotting library."""
    width, height = 720, 380
    left, right, top, bottom = 70, 200, 44, 52
    plot_width, plot_height = width - left - right, height - top - bottom
    all_x = [x for points in series.values() for x, _ in points]
    if not all_x:
        return
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = y_range
    span_x = (x_max - x_min) or 1
    span_y = (y_max - y_min) or 1

    def place(x: float, y: float) -> tuple[float, float]:
        return (left + (x - x_min) / span_x * plot_width,
                top + plot_height - (y - y_min) / span_y * plot_height)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-size="15" font-weight="600" fill="#14181c">{title}</text>',
    ]
    for step in range(5):
        value = y_min + span_y * step / 4
        _, y = place(x_min, value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
                     f'stroke="#e3e6e1" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="11" fill="#7c868f" '
                     f'text-anchor="end">{value * 100:.0f}%</text>')
    for x_value in range(int(x_min), int(x_max) + 1, max(1, int(span_x // 6))):
        x, _ = place(x_value, y_min)
        lines.append(f'<text x="{x:.1f}" y="{top + plot_height + 20}" font-size="11" '
                     f'fill="#7c868f" text-anchor="middle">{x_value}</text>')
    lines.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
                 f'y2="{top + plot_height}" stroke="#14181c" stroke-width="1.2"/>')
    lines.append(f'<text x="{left + plot_width / 2:.0f}" y="{height - 14}" font-size="12" '
                 f'fill="#4a535c" text-anchor="middle">層</text>')
    lines.append(f'<text x="18" y="{top + plot_height / 2:.0f}" font-size="12" fill="#4a535c" '
                 f'transform="rotate(-90 18 {top + plot_height / 2:.0f})" '
                 f'text-anchor="middle">{y_label}</text>')
    for index, (name, points) in enumerate(series.items()):
        color = PALETTE[index % len(PALETTE)]
        path_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (place(px, py) for px, py in points))
        lines.append(f'<polyline points="{path_points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for px, py in points:
            x, y = place(px, py)
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}"/>')
        legend_y = top + 6 + index * 20
        lines.append(f'<line x1="{left + plot_width + 16}" y1="{legend_y}" '
                     f'x2="{left + plot_width + 40}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{left + plot_width + 46}" y="{legend_y + 4}" font-size="11" '
                     f'fill="#14181c">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure_series(summary: Mapping[str, Any], field: str | None) -> dict[str, list[tuple[float, float]]]:
    """Collect a per-layer series per condition, averaging over seeds."""
    grouped: dict[str, dict[int, list[float]]] = {}
    for run_key, block in (summary.get("series") or {}).items():
        condition = run_key.split("/seed-")[0]
        source = block.get("probe_by_layer" if field else "terminal_by_layer") or {}
        for layer, value in source.items():
            if field:
                value = (value or {}).get(field)
            if not isinstance(value, (int, float)):
                continue
            grouped.setdefault(condition, {}).setdefault(int(layer), []).append(float(value))
    output: dict[str, list[tuple[float, float]]] = {}
    for condition, layers in grouped.items():
        label = CONDITION_LABELS.get(condition, condition).replace("$", "")
        output[label] = [(layer, sum(values) / len(values)) for layer, values in sorted(layers.items())]
    return output


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output = Path(args.output).expanduser().resolve()
    tables_dir, figures_dir = output / "tables", output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    wanted = {item.strip() for item in args.tables.split(",") if item.strip()}
    stamp = provenance_record()
    label = f"{stamp['git_commit'] or 'unknown'} / {stamp['generated_at']}"
    single_seed = bool(summary.get("single_seed_conditions"))

    written: list[str] = []
    for table in TABLES:
        if wanted and table.key not in wanted:
            continue
        rows = table_rows(summary, table)
        if not rows:
            continue
        header = header_cells(table)
        write_tex(tables_dir / f"{table.key}.tex", table, header, rows, label, single_seed)
        write_markdown(tables_dir / f"{table.key}.md", table, header, rows, label, single_seed)
        written.append(table.key)

    figures: list[str] = []
    board = figure_series(summary, "board_macro_f1")
    if board:
        line_figure(figures_dir / "probe_board_by_layer.svg",
                    "盤面macro-F1の層別推移", "盤面macro-F1", board, (0.4, 1.0))
        figures.append("probe_board_by_layer.svg")
    state = figure_series(summary, "full_state_exact_match")
    if state:
        line_figure(figures_dir / "probe_full_state_by_layer.svg",
                    "局面完全一致の層別推移", "局面完全一致", state, (0.0, 1.0))
        figures.append("probe_full_state_by_layer.svg")
    terminal = figure_series(summary, None)
    if terminal:
        line_figure(figures_dir / "terminal_probe_by_layer.svg",
                    "完全棋譜終端の線形復号の層別推移", "正解率", terminal, (0.4, 1.0))
        figures.append("terminal_probe_by_layer.svg")

    index = ["% 自動生成：render_paper_tables.py",
             f"% {label}",
             "% 本文からは \\input{paper/tables/<key>} で差し込む．"]
    index += [f"\\input{{tables/{key}}}" for key in written]
    (tables_dir / "all_tables.tex").write_text("\n".join(index) + "\n", encoding="utf-8")

    print(json.dumps({
        "event": "render_complete",
        "tables": written,
        "figures": figures,
        "single_seed_conditions": summary.get("single_seed_conditions", []),
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
