import unittest

from scripts.security_audit import ROOT, audit_local


class SecurityTests(unittest.TestCase):
    def test_repository_security_invariants(self) -> None:
        result = audit_local(ROOT)
        self.assertEqual(result.errors, [], "\n".join(result.errors))


if __name__ == "__main__":
    unittest.main()
