#!/usr/bin/env python3
"""学生向け文書（docs/*.md）を、Notion に貼る形式へ変換する。

docs/ が正本で、Notion はその写しです。docs/ を直したら、このスクリプトで
Notion 用の本文を作り、各ページの本文をそれで置き換えます。

    python scripts/docs_to_notion.py [出力先ディレクトリ]

出力先には <ページ名>.notion.md ができます（既定は tmp/notion/）。

変換の内容:
- 先頭の # 見出しはページのタイトルになるので、本文から外す
- パイプ表を Notion の <table> にする
- コードの外にある Notion の特殊文字をエスケープする
- 文書どうしのリンクを Notion のページへのリンクにする。表示がファイル名なら、ページのタイトルにする
- ../将棋.pdf へのリンクをスライドのページへのリンクにする
- リポジトリ外のファイルへのリンク（references.bib など）は、文字だけ残す
- タイトル直後の引用（用語集の案内）を Notion の callout にする
- 用語集には目次を入れる
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DOCS = PROJECT / "docs"


def notion_url(page_id: str) -> str:
    return f"https://app.notion.com/p/{page_id}"


# docs/ のファイル -> Notion のページ。親ページは「将棋の状態追跡：学生向け資料」。
PAGES = {
    "research_overview.md": "3e502671ae4c814d8255fa14c5a71af6",
    "student_analysis_guide.md": "3e502671ae4c81b89b2ad838c1b6469c",
    "csv_computation.md": "3e502671ae4c815aa0abf74c107ea1c6",
    "student_research_questions.md": "3e502671ae4c81ee8a08ffb0c66b263d",
    "glossary.md": "3e502671ae4c81b2a493d8e9a8e7808a",
}
# リポジトリに入れていないファイルのうち、Notion に置いてあるもの。
EXTERNAL = {"将棋.pdf": "3e502671ae4c81ea8b28feb62c3d2eb3"}

SPECIAL = re.compile(r"([\\~$<>{}|^])")
INLINE = re.compile(r"(`[^`]*`)|\[([^\]]+)\]\(([^)]+)\)")


def escape(text: str) -> str:
    return SPECIAL.sub(r"\\\1", text)


def titles() -> dict[str, str]:
    return {
        name: (DOCS / name).read_text(encoding="utf-8").splitlines()[0].removeprefix("# ").strip()
        for name in PAGES
    }


def convert_inline(text: str, page_titles: dict[str, str]) -> str:
    out, pos = [], 0
    for match in INLINE.finditer(text):
        out.append(escape(text[pos:match.start()]))
        if match.group(1):
            out.append(match.group(1))
        else:
            label, target = match.group(2), match.group(3)
            name = target.split("/")[-1]
            if target.startswith("http"):
                out.append(f"[{convert_inline(label, page_titles)}]({target})")
            elif name in PAGES:
                shown = page_titles[name] if label == name else label
                out.append(f"[{convert_inline(shown, page_titles)}]({notion_url(PAGES[name])})")
            elif name in EXTERNAL:
                out.append(f"[{convert_inline(label, page_titles)}]({notion_url(EXTERNAL[name])})")
            else:
                out.append(convert_inline(label, page_titles))
        pos = match.end()
    out.append(escape(text[pos:]))
    return "".join(out)


def table(rows: list[str], page_titles: dict[str, str]) -> list[str]:
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    lines = ['<table header-row="true">']
    for row in [cells[0]] + cells[2:]:
        lines.append("\t<tr>")
        lines.extend(f"\t\t<td>{convert_inline(cell, page_titles)}</td>" for cell in row)
        lines.append("\t</tr>")
    lines.append("</table>")
    return lines


def convert(name: str, page_titles: dict[str, str]) -> str:
    lines = (DOCS / name).read_text(encoding="utf-8").splitlines()
    out, index, in_code, toc_done = [], 1, False, name != "glossary.md"
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            if not in_code and line.strip() == "```":
                line = "```plain text"
            in_code = not in_code
            out.append(line)
        elif in_code:
            out.append(line)
        elif line.lstrip().startswith("|"):
            block = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                block.append(lines[index])
                index += 1
            out.extend(table(block, page_titles))
            continue
        elif line.strip() == "---":
            out.append("---")
        elif not any(out) and line.startswith("> ") and "用語集" in line:
            out.append('<callout icon="📖">')
            out.append("\t" + convert_inline(line[2:], page_titles))
            out.append("</callout>")
        else:
            if not toc_done and line.startswith("## "):
                out.extend(["<table_of_contents/>", ""])
                toc_done = True
            prefix = re.match(r"^(#{1,6} |> |- |\d+\. )?", line).group(0)
            out.append(prefix + convert_inline(line[len(prefix):].rstrip(), page_titles))
        index += 1
    return "\n".join(out).strip() + "\n"


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else PROJECT / "tmp" / "notion"
    target.mkdir(parents=True, exist_ok=True)
    page_titles = titles()
    for name, page_id in PAGES.items():
        path = target / f"{Path(name).stem}.notion.md"
        path.write_text(convert(name, page_titles), encoding="utf-8")
        print(f"{page_id}  {page_titles[name]}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
