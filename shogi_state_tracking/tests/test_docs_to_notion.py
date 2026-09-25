"""docs/ から Notion 用の本文への変換。

docs/ が正本なので、文書どうしのリンクがすべて Notion のページに結び付き、
用語集へのリンクが切れていないことを確かめる。
"""

import importlib.util
import re
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("docs_to_notion", PROJECT / "scripts" / "docs_to_notion.py")
docs_to_notion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs_to_notion)


class DocsToNotionTest(unittest.TestCase):
    def setUp(self):
        self.titles = docs_to_notion.titles()

    def test_every_doc_has_a_notion_page(self):
        self.assertEqual(
            {path.name for path in docs_to_notion.DOCS.glob("*.md")},
            set(docs_to_notion.PAGES),
        )

    def test_links_between_docs_resolve_to_notion_pages(self):
        for name in docs_to_notion.PAGES:
            text = (docs_to_notion.DOCS / name).read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if target.startswith("http"):
                    continue
                file_name = target.split("/")[-1]
                with self.subTest(doc=name, target=target):
                    self.assertTrue(
                        file_name in docs_to_notion.PAGES
                        or file_name in docs_to_notion.EXTERNAL
                        or file_name == "references.bib",
                        f"{name} links to {target}, which has no Notion page",
                    )

    def test_converted_pages_contain_no_relative_links(self):
        for name in docs_to_notion.PAGES:
            converted = docs_to_notion.convert(name, self.titles)
            with self.subTest(doc=name):
                self.assertNotIn(".md)", converted)
                self.assertNotIn("../", converted)

    def test_tables_and_file_name_labels(self):
        converted = docs_to_notion.convert("student_analysis_guide.md", self.titles)
        self.assertIn('<table header-row="true">', converted)
        self.assertNotIn("|---|", converted)
        self.assertIn(f"[{self.titles['research_overview.md']}](", converted)
        self.assertTrue(converted.startswith('<callout icon="📖">'))

    def test_glossary_has_table_of_contents(self):
        converted = docs_to_notion.convert("glossary.md", self.titles)
        self.assertIn("<table_of_contents/>", converted)


if __name__ == "__main__":
    unittest.main()
