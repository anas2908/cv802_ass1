"""Unit tests for the read-only VGGSfM environment audit."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_environment.py"
SPEC = importlib.util.spec_from_file_location("audit_environment", SCRIPT)
audit_environment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_environment)


class EnvironmentAuditTests(unittest.TestCase):
    def test_accepts_only_exact_reviewed_headless_metadata_warning(self) -> None:
        result = {
            "returncode": 1,
            "stdout": audit_environment.KNOWN_HEADLESS_METADATA_WARNING + "\n",
            "stderr": "",
        }
        classified = audit_environment.classify_pip_check(result)
        self.assertEqual(
            classified["classification"],
            "known_headless_distribution_metadata_warning_only",
        )
        self.assertTrue(classified["only_known_headless_distribution_metadata_warning"])

    def test_rejects_any_additional_dependency_problem(self) -> None:
        result = {
            "returncode": 1,
            "stdout": audit_environment.KNOWN_HEADLESS_METADATA_WARNING + "\nother is broken\n",
            "stderr": "",
        }
        with self.assertRaisesRegex(RuntimeError, "unreviewed dependency"):
            audit_environment.classify_pip_check(result)

    def test_clean_pip_check_is_valid(self) -> None:
        classified = audit_environment.classify_pip_check({
            "returncode": 0, "stdout": "", "stderr": "",
        })
        self.assertEqual(classified["classification"], "clean")

    def test_atomic_inventory_write_replaces_complete_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inventory.txt"
            audit_environment.atomic_write_text(path, "first\n")
            audit_environment.atomic_write_text(path, "second\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "second\n")


if __name__ == "__main__":
    unittest.main()
