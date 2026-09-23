"""Keep the student's paste-ready method synchronized with the starter API."""

from __future__ import annotations

import ast
from pathlib import Path
import textwrap
import unittest


class PasteCopyTests(unittest.TestCase):
    def test_estimate_cameras_matches_starter_method(self) -> None:
        assignment = Path(__file__).resolve().parents[1]
        module = ast.parse((assignment / "modules/colmap/api.py").read_text())
        api = next(node for node in module.body if isinstance(node, ast.ClassDef)
                   and node.name == "ColmapAPI")
        method = next(node for node in api.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_estimate_cameras")
        copied_module = ast.parse(textwrap.dedent(
            (assignment.parent / "estimate_cameras.py").read_text()
        ))
        copied = next(node for node in copied_module.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_estimate_cameras")
        # Ignore file offsets/indentation but compare the decorator, arguments,
        # docstring, implementation, and nested helpers in their entirety.
        self.assertEqual(ast.dump(method, include_attributes=False),
                         ast.dump(copied, include_attributes=False))


if __name__ == "__main__":
    unittest.main()
