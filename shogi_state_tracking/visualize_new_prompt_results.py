#!/usr/bin/env python3
"""主実験の条件比較・学習曲線・層別probeを外部ライブラリなしでSVG化する。"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


WIDTH, HEIGHT = 1200, 680
COLORS = {"vanilla": "#4477aa", "partial_action": "#228833", "random_control": "#cc6677"}


def svg_header(title, width=WIDTH, height=HEIGHT):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<style>text{font-family:Arial,"Noto Sans JP",sans-serif;fill:#202020}.title{font-size:24px;font-weight:bold}.label{font-size:15px}.small{font-size:12px}</style>', f"<title>{html.escape(title)}</title>"]


def write(path, lines):
    lines.append("</svg>"); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def bar_chart(rows, metric, title, output):
    lines = svg_header(title); lines += [f'<text class="title" x="40" y="42">{html.escape(title)}</text>', '<line x1="90" y1="580" x2="1160" y2="580" stroke="#333"/>', '<line x1="90" y1="90" x2="90" y2="580" stroke="#333"/>']
    items = [
        (f'{row["model_size"]}/{row["condition"]}/p{row.get("annotation_probability", "?")}',
         float(row.get(metric, 0.0)), row["condition"])
        for row in rows
    ]
    for tick in range(6):
        value = tick / 5; y = 580 - value * 460
        lines += [f'<line x1="85" y1="{y:.1f}" x2="90" y2="{y:.1f}" stroke="#333"/>', f'<text class="small" x="45" y="{y+4:.1f}">{value:.1f}</text>']
    width = 950 / max(1, len(items))
    for index, (name, value, condition) in enumerate(items):
        x = 110 + index * width; height = max(0, min(1, value)) * 460
        lines += [f'<rect x="{x:.1f}" y="{580-height:.1f}" width="{width*0.68:.1f}" height="{height:.1f}" fill="{COLORS.get(condition,"#777")}"/>', f'<text class="small" transform="translate({x+width*0.34:.1f},625) rotate(-35)" text-anchor="end">{html.escape(name)}</text>', f'<text class="small" x="{x+width*0.34:.1f}" y="{570-height:.1f}" text-anchor="middle">{value:.3f}</text>']
    write(output, lines)


def learning_curves(result_dir, output):
    histories = []
    for path in sorted(result_dir.rglob("training_history.json")):
        label = "/".join(path.parent.relative_to(result_dir).parts)
        history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
        if history: histories.append((label, history))
    if not histories: return
    values = [float(row["validation_loss"]) for _, history in histories for row in history]; low, high = min(values), max(values)
    high = max(high, low + 1e-6); lines = svg_header("Validation loss learning curves"); lines += ['<text class="title" x="40" y="42">Validation loss</text>', '<line x1="80" y1="600" x2="1160" y2="600" stroke="#333"/>', '<line x1="80" y1="80" x2="80" y2="600" stroke="#333"/>']
    palette = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#66ccee", "#ccbb44", "#777777", "#ddcc77", "#44aa99"]
    for index, (label, history) in enumerate(histories):
        max_epoch = max(int(row["epoch"]) for row in history); points=[]
        for row in history:
            x=80+1060*(int(row["epoch"])-1)/max(1,max_epoch-1); y=600-500*(float(row["validation_loss"])-low)/(high-low); points.append(f"{x:.1f},{y:.1f}")
        color=palette[index%len(palette)]; lines += [f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>', f'<text class="small" x="90" y="{70+index*17}"><tspan fill="{color}">■</tspan> {html.escape(label)}</text>']
    write(output, lines)


def probe_heatmaps(result_dir, output):
    lines = svg_header("Layer-wise state probes", 1200, 760); lines += ['<text class="title" x="40" y="42">Layer-wise linear-probe accuracy</text>']
    entries=[]
    for path in sorted(result_dir.rglob("probes/probe_metrics.json")):
        label="/".join(path.parent.parent.relative_to(result_dir).parts); payload=json.loads(path.read_text(encoding="utf-8")); entries.append((label,payload))
    metrics=("board_piece_accuracy_on_occupied","hand_nonzero_accuracy","turn_accuracy","in_check_accuracy")
    cell_w=55; cell_h=26
    for row_index,(label,payload) in enumerate(entries):
        y=95+row_index*35; lines.append(f'<text class="small" x="10" y="{y+18}">{html.escape(label)}</text>')
        for col_index,metric in enumerate(metrics): lines.append(f'<text class="small" x="{310+col_index*220}" y="72">{html.escape(metric.replace("_accuracy", ""))}</text>')
        sources=payload.get("probe_results",{})
        for layer in range(13):
            values=sources.get(f"layer_{layer}",{}).get("evaluation",{})
            for metric_index,metric in enumerate(metrics):
                value=values.get(metric); color="#eeeeee" if not isinstance(value,(int,float)) else f"rgb({int(245*(1-value))},{int(245*value+10)},{110})"
                x=310+metric_index*220+layer*cell_w/4
                lines.append(f'<rect x="{x:.1f}" y="{y}" width="{cell_w/4-1:.1f}" height="{cell_h}" fill="{color}"/><title>{layer} {metric}: {value}</title>')
    write(output, lines)


def main():
    parser=argparse.ArgumentParser(description="新prompt主実験の自動SVG可視化")
    parser.add_argument("--results-dir", required=True); parser.add_argument("--summary-json", required=True); parser.add_argument("--output-dir", required=True)
    args=parser.parse_args(); results=Path(args.results_dir); output=Path(args.output_dir); rows=json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    bar_chart(rows,"top1_accuracy","Constrained move top-1",output/"move_top1.svg")
    bar_chart(rows,"mean_legal_probability_mass","Legal move probability mass",output/"legal_mass.svg")
    bar_chart(rows,"probe_final_full_state_exact_match","Full-state probe exact match",output/"probe_full_state.svg")
    learning_curves(results,output/"learning_curves.svg"); probe_heatmaps(results,output/"probe_layers.svg")
    print(output)


if __name__=="__main__": main()
