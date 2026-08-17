#!/usr/bin/env python3
"""駒打ち関連の信頼度・attention・遮断効果を依存なしSVGで図示する。"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="駒打ち関連評価のSVG可視化")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--attention")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def layer_number(value: str) -> int:
    return int(value.split("_", 1)[1])


def esc(value) -> str:
    return html.escape(str(value))


def color(value: float, minimum: float, maximum: float, diverging: bool = False) -> str:
    if not math.isfinite(value):
        return "#eeeeee"
    ratio = 0.5 if maximum <= minimum else max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
    if diverging:
        if ratio < .5:
            local = ratio * 2
            red, green, blue = int(50 + 205 * local), int(90 + 165 * local), 220
        else:
            local = (ratio - .5) * 2
            red, green, blue = 220, int(255 - 180 * local), int(255 - 205 * local)
    else:
        red, green, blue = int(40 + 190 * ratio), int(45 + 150 * ratio), int(110 - 70 * ratio)
    return "#{:02x}{:02x}{:02x}".format(red, green, blue)


def write_heatmap(path: Path, matrix, rows, columns, title, diverging=False):
    cell_w, cell_h, left, top = 34, 24, 120, 55
    width, height = left + cell_w * len(columns) + 30, top + cell_h * len(rows) + 70
    finite = [value for row in matrix for value in row if math.isfinite(value)]
    if diverging:
        bound = max((abs(value) for value in finite), default=1.0)
        minimum, maximum = -bound, bound
    else:
        minimum, maximum = (min(finite), max(finite)) if finite else (0.0, 1.0)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{}" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">{}</text>'.format(width / 2, esc(title)),
    ]
    for row_index, (label, values) in enumerate(zip(rows, matrix)):
        y = top + row_index * cell_h
        parts.append('<text x="{}" y="{}" text-anchor="end" font-family="sans-serif" font-size="10">{}</text>'.format(left - 7, y + 16, esc(label)))
        for column_index, value in enumerate(values):
            x = left + column_index * cell_w
            parts.append('<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="white"><title>{}: {} = {:.6g}</title></rect>'.format(
                x, y, cell_w, cell_h, color(value, minimum, maximum, diverging), esc(label), esc(columns[column_index]), value,
            ))
    tick_every = max(1, len(columns) // 16)
    for index, label in enumerate(columns):
        if index % tick_every == 0 or index == len(columns) - 1:
            x = left + index * cell_w + cell_w / 2
            parts.append('<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="9">{}</text>'.format(x, top + len(rows) * cell_h + 15, esc(label)))
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_lines(path: Path, series, title, x_label, y_label):
    width, height, left, top, right, bottom = 920, 480, 75, 50, 25, 65
    all_points = [point for _, points in series for point in points]
    xs = [point[0] for point in all_points]; ys = [point[1] for point in all_points]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    if ymax <= ymin: ymax = ymin + 1
    def px(x): return left + (x - xmin) / max(xmax - xmin, 1e-12) * (width - left - right)
    def py(y): return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
    palette = ("#1565c0", "#1565c0", "#c62828", "#c62828", "#2e7d32", "#6a1b9a")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(width, height), '<rect width="100%" height="100%" fill="white"/>']
    parts.append('<text x="{}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">{}</text>'.format(width / 2, esc(title)))
    parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>'.format(left, height-bottom, width-right, height-bottom))
    parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>'.format(left, top, left, height-bottom))
    if xmin <= 0 <= xmax:
        parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#777" stroke-dasharray="4 3"/>'.format(px(0), top, px(0), height-bottom))
    for index, (label, points) in enumerate(series):
        dash = ' stroke-dasharray="7 4"' if "control" in label else ""
        coordinates = " ".join("{:.2f},{:.2f}".format(px(x), py(y)) for x, y in points)
        parts.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2"{}/>'.format(coordinates, palette[index % len(palette)], dash))
        ly = top + 18 * index
        parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"{}/><text x="{}" y="{}" font-family="sans-serif" font-size="11">{}</text>'.format(width-220, ly, width-195, ly, palette[index % len(palette)], dash, width-190, ly+4, esc(label)))
    for index in range(6):
        value = ymin + (ymax-ymin) * index / 5
        parts.append('<text x="{}" y="{}" text-anchor="end" font-family="sans-serif" font-size="10">{:.3f}</text>'.format(left-7, py(value)+4, value))
    parts.append('<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12">{}</text>'.format((left+width-right)/2, height-15, esc(x_label)))
    parts.append('<text transform="translate(16 {}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="12">{}</text>'.format(height/2, esc(y_label)))
    parts.append('</svg>'); path.write_text("\n".join(parts)+"\n", encoding="utf-8")


def write_bars(path: Path, labels, values, title, y_label):
    width = max(760, 85 * len(labels)); height, left, top, bottom = 470, 80, 50, 125
    bound = max((abs(value) for value in values), default=1.0); ymin, ymax = -bound, bound
    plot_h = height - top - bottom
    def py(value): return top + (ymax-value)/(ymax-ymin) * plot_h
    zero = py(0); slot = (width-left-25)/max(len(labels), 1)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(width,height), '<rect width="100%" height="100%" fill="white"/>']
    parts.append('<text x="{}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">{}</text>'.format(width/2,esc(title)))
    parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>'.format(left,zero,width-25,zero))
    for index,(label,value) in enumerate(zip(labels,values)):
        x=left+index*slot+slot*.18; y=min(zero,py(value)); h=abs(py(value)-zero)
        parts.append('<rect x="{}" y="{}" width="{}" height="{}" fill="#5e81ac"><title>{}: {:.6g}</title></rect>'.format(x,y,slot*.64,h,esc(label),value))
        parts.append('<text transform="translate({} {}) rotate(-45)" text-anchor="end" font-family="sans-serif" font-size="10">{}</text>'.format(x+slot*.32,height-bottom+18,esc(label)))
    parts.append('<text transform="translate(16 {}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="12">{}</text>'.format((top+height-bottom)/2,esc(y_label)))
    parts.append('</svg>'); path.write_text("\n".join(parts)+"\n",encoding="utf-8")


def main():
    args = parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    trajectory = json.loads(Path(args.trajectory).read_text(encoding="utf-8"))
    sources = sorted(trajectory["metrics"], key=layer_number)
    offsets = sorted({int(offset) for source in sources for group in ("drop","control") for offset in trajectory["metrics"][source]["trajectory"][group]})
    difference=[]
    for source in sources:
        values=trajectory["metrics"][source]["trajectory"]; row=[]
        for offset in offsets:
            drop=values["drop"].get(str(offset),{}).get("mean_true_count_probability")
            control=values["control"].get(str(offset),{}).get("mean_true_count_probability")
            row.append(float("nan") if drop is None or control is None else drop-control)
        difference.append(row)
    write_heatmap(output/"hand_confidence_difference_heatmap.svg",difference,sources,offsets,"True hand-count probability: drop minus matched normal move",True)
    selected=list(dict.fromkeys([sources[len(sources)//2],sources[-1]])); series=[]
    for source in selected:
        for group in ("drop","control"):
            entries=trajectory["metrics"][source]["trajectory"][group]
            series.append(("{} {}".format(source,group),[(int(x),entries[x]["mean_true_count_probability"]) for x in sorted(entries,key=int)]))
    write_lines(output/"hand_confidence_trajectory.svg",series,"Hand-count confidence around a drop","ply relative to actual/matched move","probability assigned to true hand count")

    if args.attention and Path(args.attention).is_file():
        attention=json.loads(Path(args.attention).read_text(encoding="utf-8"))
        values=attention.get("attention",{}).get("drop",{}).get("after_drop",{})
        if values:
            n_layers=max(int(key.split("_")[1]) for key in values)+1; n_heads=max(int(key.split("_")[3]) for key in values)+1
            matrix=[[float("nan")]*n_heads for _ in range(n_layers)]
            for key,entry in values.items():
                parts=key.split("_"); matrix[int(parts[1])][int(parts[3])]=entry["enrichment_ratio"]
            write_heatmap(output/"drop_attention_enrichment.svg",matrix,["layer_{}".format(x) for x in range(n_layers)],list(range(n_heads)),"Attention enrichment to relevant hand events after <DROP>")
        labels=[]; changes=[]
        for key,entry in attention.get("ablation",{}).items():
            if key.startswith("drop:") and key.endswith(":after_drop"):
                labels.append(key.replace("drop:","").replace(":after_drop","")); changes.append(entry["log_probability_change"])
        if labels:
            order=sorted(range(len(labels)),key=lambda i:labels[i]); write_bars(output/"drop_attention_ablation.svg",[labels[i] for i in order],[changes[i] for i in order],"Direct attention-edge ablation after <DROP>","change in correct-piece log probability")
    print(json.dumps({"event":"drop_relevance_visualization_complete","output_dir":str(output)},ensure_ascii=False))


if __name__ == "__main__":
    main()
