#!/usr/bin/env python3
"""PAPER_SHORT_JA.mdから日本語4ページ版の2段組TeXを生成する。"""

from __future__ import annotations

import re
from pathlib import Path

import sync_paper_tex as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PAPER_SHORT_JA.md"
DESTINATION = ROOT / "PAPER_SHORT_JA.tex"

base.PREAMBLE = r"""% このファイルは scripts/sync_short_paper_tex.py により自動生成される．
% 内容を変更する場合は PAPER_SHORT_JA.md を編集して再生成すること．
\documentclass[a4paper,10pt,twocolumn]{ltjsarticle}

\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{fancyvrb}
\usepackage{cite}
\usepackage[hidelinks]{hyperref}
\usepackage[margin=18mm,columnsep=7mm]{geometry}

\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\setlength{\abovedisplayskip}{4pt}
\setlength{\belowdisplayskip}{4pt}

\title{将棋の指手系列モデルにおける局面復号の指手依存性}
\author{著者名未定}
\date{}

\begin{document}
\maketitle

"""

base.POSTAMBLE = r"""

{\footnotesize
\bibliographystyle{plain}
\bibliography{references}
}

\end{document}
"""


TABLE_METADATA = {
    1: ("局面状態の線形復号", "tab:short-state"),
    2: ("局面復号の指手依存性に関する介入結果", "tab:short-dependency"),
}


def parse_short_table(
    lines: list[str],
    start: int,
    footnotes: dict[str, str],
    index: int,
) -> tuple[list[str], int]:
    rows: list[list[str]] = []
    position = start
    while position < len(lines) and lines[position].startswith("|"):
        rows.append([cell.strip() for cell in lines[position].strip().strip("|").split("|")])
        position += 1
    header = rows[0]
    body = rows[2:]
    caption, label = TABLE_METADATA.get(index, (f"実験結果{index}", f"tab:short-{index}"))
    columns = r">{\raggedright\arraybackslash}X " + " ".join("r" for _ in header[1:])
    output = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        rf"\begin{{tabularx}}{{\textwidth}}{{{columns}}}",
        r"\toprule",
        " & ".join(base.inline(cell, footnotes) for cell in header) + r" \\",
        r"\midrule",
    ]
    output.extend(" & ".join(base.inline(cell, footnotes) for cell in row) + r" \\" for row in body)
    output.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    return output, position


def main() -> None:
    base.parse_table = parse_short_table
    source = SOURCE.read_text(encoding="utf-8")
    destination = base.convert(source)
    # 短縮版ではMarkdownの1段落目を要旨として用いる。TeX上でも見出しを
    # 太字の段落として残し，独立したsectionにはしない。
    destination = re.sub(r"\n{3,}", "\n\n", destination)
    DESTINATION.write_text(destination, encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
