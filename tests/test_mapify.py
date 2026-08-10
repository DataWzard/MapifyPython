import tempfile
import unittest
from pathlib import Path

import mapify


SAMPLE = '''import os
VALUE = 2

def greet(name: str = "world") -> str:
    if name:
        return f"Hello {name}"
    return "Hello"

for item in range(VALUE):
    print(greet(str(item)))
'''


class MapifyTests(unittest.TestCase):
    def test_catalog_and_flow(self):
        result = mapify.map_python_source(SAMPLE, "sample.py")
        self.assertEqual(result["catalog"]["functions"][0]["name"], "greet")
        self.assertEqual([n["kind"] for n in result["flow"]["children"]], ["import", "assignment", "function", "loop"])
        function_body = result["flow"]["children"][2]["children"][0]
        self.assertEqual(function_body["execution"], "deferred")
        self.assertEqual(function_body["children"][0]["kind"], "condition")

    def test_html_is_self_contained_and_escapes_script_data(self):
        result = mapify.map_python_source('x = "</script>"', "unsafe.py")
        rendered = mapify.map_to_html(result)
        self.assertIn("<svg", rendered)
        self.assertNotIn('x = "</script>"', rendered)
        self.assertIn("\\u003c/script>", rendered)

    def test_package_keeps_syntax_errors_as_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "good.py").write_text("x = 1", encoding="utf-8")
            (root / "bad.py").write_text("if:", encoding="utf-8")
            result = mapify.map_python_package(root)
        self.assertEqual(len(result["modules"]), 2)
        self.assertTrue(any("error" in module for module in result["modules"]))


if __name__ == "__main__":
    unittest.main()
