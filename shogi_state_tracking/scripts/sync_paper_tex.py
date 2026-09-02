#!/usr/bin/env python3
"""PAPER_DRAFT_JA.mdから2段組LuaLaTeX原稿を生成する。"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PAPER_DRAFT_JA.md"
DESTINATION = ROOT / "PAPER_DRAFT_JA.tex"

PREAMBLE = r"""% このファイルは scripts/sync_paper_tex.py により自動生成される．
% 内容を変更する場合は PAPER_DRAFT_JA.md を編集して再生成すること．
\documentclass[a4paper,10pt,twocolumn]{ltjsarticle}

\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{fancyvrb}
\usepackage[hidelinks]{hyperref}
\usepackage[margin=20mm,columnsep=7mm]{geometry}

\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}

\title{将棋の指手系列から形成されるTransformer内部状態の分析}
\author{著者名未定}
\date{}

\begin{document}
\maketitle

"""

POSTAMBLE = r"""

\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""


def inline(text: str, footnotes: dict[str, str]) -> str:
    """Markdownのインラインコードと脚注参照だけをLaTeXへ変換する。"""
    parts = re.split(r"(`[^`]+`)", text)
    converted: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            converted.append(r"\texttt{\detokenize{" + part[1:-1] + "}}")
        else:
            # Markdownでは通常文字として使えるが，LaTeXでは構文上の意味を
            # 持つ文字をエスケープする．特に表中の % を放置すると，それ以降の
            # 列区切りと行末がコメントとして消え，alignment errorになる．
            escaped = re.sub(r"(?<!\\)%", r"\\%", part)
            escaped = re.sub(r"(?<!\\)&", r"\\&", escaped)
            converted.append(escaped)
    result = "".join(converted)
    result = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", result)
    for key, value in footnotes.items():
        result = result.replace(f"[^{key}]", r"\footnote{" + inline(value, {}) + "}")
    return result


def parse_table(lines: list[str], start: int, footnotes: dict[str, str], index: int) -> tuple[list[str], int]:
    rows: list[list[str]] = []
    position = start
    while position < len(lines) and lines[position].startswith("|"):
        cells = [cell.strip() for cell in lines[position].strip().strip("|").split("|")]
        rows.append(cells)
        position += 1
    header = rows[0]
    body = rows[2:]
    if index == 1:
        columns = r">{\raggedright\arraybackslash}p{0.13\textwidth} >{\raggedright\arraybackslash}X r"
        caption = "本実験で使用するトークン"
        label = "tab:vocabulary"
    elif index == 2:
        columns = r"l >{\raggedright\arraybackslash}X r r"
        caption = "実験データの分割"
        label = "tab:dataset"
    else:
        table_metadata = {
            3: ("Floodgate評価における指手予測", "tab:move-standard"),
            4: ("未見局面における指手予測", "tab:move-unseen"),
            5: ("Lishogi非BOT棋譜における指手予測", "tab:move-lishogi"),
            6: ("代表層からの局面状態の線形復号", "tab:state-probe"),
            7: ("持ち駒復号の指手依存差", "tab:action-condition"),
            8: ("指手分岐後の持ち駒枚数復号", "tab:action-condition-accuracy"),
            9: ("持ち駒関連履歴への注意接続の遮断", "tab:attention-ablation"),
            10: ("APによる感度分析", "tab:ap-sensitivity"),
        }
        caption, label = table_metadata.get(index, (f"実験結果{index - 2}", f"tab:result-{index - 2}"))
        columns = r">{\raggedright\arraybackslash}X " + " ".join("r" for _ in header[1:])
    environment = "table*"
    width = r"\textwidth"
    output = [
        rf"\begin{{{environment}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize" if index >= 3 else r"\small",
        rf"\begin{{tabularx}}{{{width}}}{{{columns}}}",
        r"\toprule",
        " & ".join(inline(cell, footnotes) for cell in header) + r" \\",
        r"\midrule",
    ]
    output.extend(" & ".join(inline(cell, footnotes) for cell in row) + r" \\" for row in body)
    output.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        rf"\end{{{environment}}}",
    ])
    return output, position


def convert(source: str) -> str:
    lines = source.splitlines()
    footnotes: dict[str, str] = {}
    retained: list[str] = []
    for line in lines:
        match = re.match(r"\[\^([^]]+)\]:\s*(.*)", line)
        if match:
            footnotes[match.group(1)] = match.group(2)
        else:
            retained.append(line)
    lines = retained

    output: list[str] = [PREAMBLE.rstrip()]
    in_display_math = False
    in_code = False
    table_index = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == "$$":
            output.append(r"\]" if in_display_math else r"\[")
            in_display_math = not in_display_math
            index += 1
            continue
        if line.startswith("```"):
            if in_code:
                output.append(r"\end{Verbatim}")
            else:
                output.append(r"\begin{Verbatim}[fontsize=\footnotesize]")
            in_code = not in_code
            index += 1
            continue
        if in_display_math or in_code:
            output.append(line)
            index += 1
            continue
        subsubsection = re.match(r"####\s+(?:\d+\.\d+\.\d+\s+)?(.+)", line)
        if subsubsection:
            output.extend(["", r"\subsubsection{" + subsubsection.group(1) + "}"])
            index += 1
            continue
        section = re.match(r"##\s+(?:\d+\s+)?(.+)", line)
        if section:
            output.extend(["", r"\section{" + section.group(1) + "}"])
            index += 1
            continue
        subsection = re.match(r"###\s+(?:\d+\.\d+\s+)?(.+)", line)
        if subsection:
            output.extend(["", r"\subsection{" + subsection.group(1) + "}"])
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", lines[index + 1]):
            table_index += 1
            table, index = parse_table(lines, index, footnotes, table_index)
            output.extend(["", *table, ""])
            continue
        output.append(inline(line, footnotes))
        index += 1
    output.append(POSTAMBLE.strip())
    return "\n".join(output) + "\n"


def main() -> None:
    DESTINATION.write_text(convert(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
