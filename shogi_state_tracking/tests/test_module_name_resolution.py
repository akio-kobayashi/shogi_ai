"""トップレベルの各スクリプトに，未定義名の参照がないことを検査する。

importの書き漏れは，importした時点では気づけない。実行がその行へ到達して
初めて``NameError``になるため，checkpointを読んだ後や評価の途中で落ちる。
評価は時間がかかるので、その位置で落ちると損失が大きい。

テストは静的にのみ判定する。``import *``を含むファイルは判定できないため
除外し、除外されたファイル数も検査して、除外が静かに増えないようにする。
"""

import ast
import builtins
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
# 判定を諦めるファイル（import *を含むもの）。増えたらテストが知らせる。
EXPECTED_STAR_IMPORT_FILES = 0


def analyse(path: Path) -> list[str] | None:
    """未定義の可能性がある名前を返す。判定不能ならNone。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return None
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.Global):
            defined.update(node.names)
    used = {node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return sorted(used - defined)


class ModuleNameResolutionTest(unittest.TestCase):
    def scripts(self) -> list[Path]:
        return sorted(MODULE_DIR.glob("*.py"))

    def test_scripts_exist(self):
        self.assertGreater(len(self.scripts()), 10)

    def test_no_script_references_an_undefined_name(self):
        problems = {}
        for path in self.scripts():
            names = analyse(path)
            if names:
                problems[path.name] = names
        self.assertEqual(problems, {}, f"undefined names: {problems}")

    def test_star_import_exemptions_do_not_grow(self):
        undecidable = [path.name for path in self.scripts() if analyse(path) is None]
        self.assertEqual(len(undecidable), EXPECTED_STAR_IMPORT_FILES,
                         f"import * makes these undecidable: {undecidable}")
