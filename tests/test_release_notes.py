from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from roller.version import __version__

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "release_notes", ROOT / "scripts" / "release_notes.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
release_notes = module.release_notes

CHANGELOG = """# Changelog

## Unreleased

### Added

- Next thing.

## 1.2.0 - 2026-01-02

### Fixed

- A fix.

## 1.1.0

First.
"""


class ReleaseNotesTests(unittest.TestCase):
    def test_returns_the_section_of_one_version(self) -> None:
        self.assertEqual(release_notes(CHANGELOG, "1.2.0"), "### Fixed\n\n- A fix.\n")
        self.assertEqual(release_notes(CHANGELOG, "1.1.0"), "First.\n")

    def test_a_version_without_a_section_stops_the_release(self) -> None:
        for version in ("1.3.0", "1.2", "Unreleased "):
            with self.subTest(version=version), self.assertRaises(SystemExit):
                release_notes(CHANGELOG, version)

    def test_the_changelog_covers_the_package_version(self) -> None:
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertTrue(release_notes(changelog, __version__).strip())
