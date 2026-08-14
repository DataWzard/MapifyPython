import re
import unittest

from scripts.security_audit import ROOT, audit_local


class SecurityTests(unittest.TestCase):
    def test_repository_security_invariants(self) -> None:
        result = audit_local(ROOT)
        self.assertEqual(result.errors, [], "\n".join(result.errors))

    def test_browser_analyzer_asset_exists(self) -> None:
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        match = re.search(r'fetch\(["\']\./([^"\']+\.py)["\']\)', index)
        self.assertIsNotNone(match, "index.html must load a Python analyzer asset")

        analyzer = match.group(1)
        self.assertTrue(
            (ROOT / analyzer).is_file(),
            f"index.html references missing analyzer asset: {analyzer}",
        )
        module = analyzer.removesuffix(".py")
        self.assertIn(
            f'import {module}',
            index,
            f"index.html must import the analyzer module {module}",
        )


if __name__ == "__main__":
    unittest.main()
