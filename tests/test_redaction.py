from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roller.redaction import redact_file_literals


class RedactFileLiteralsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "trace.jsonl"

    def test_secrets_are_replaced_across_chunk_boundaries_without_reading_the_file_whole(
        self,
    ) -> None:
        raw = "prefix\r\nsk-secret-value\nmiddle sk-secret tail sk-secret-value" * 3
        for size in (1, 2, 3, 7, 8, 64, 4096):
            with self.subTest(chunk_size=size):
                self.path.write_bytes(raw.encode())
                with patch.object(
                    Path, "read_text", side_effect=AssertionError("whole file read")
                ):
                    redact_file_literals(
                        self.path, ["sk-secret-value", "sk-secret"], chunk_size=size
                    )
                expected = raw.replace("sk-secret-value", "[REDACTED]").replace(
                    "sk-secret", "[REDACTED]"
                )
                self.assertEqual(self.path.read_bytes().decode(), expected)
        self.assertEqual([entry.name for entry in self.root.iterdir()], ["trace.jsonl"])

    def test_repeated_prefixes_next_to_a_boundary_match_like_one_pass(self) -> None:
        raw = "aaab" * 5 + "aab" + "ab" * 4
        for size in range(1, 9):
            with self.subTest(chunk_size=size):
                self.path.write_text(raw)
                redact_file_literals(self.path, ["aab", "ab"], chunk_size=size)
                self.assertEqual(
                    self.path.read_text(),
                    raw.replace("aab", "[REDACTED]").replace("ab", "[REDACTED]"),
                )

    def test_without_a_secret_the_file_is_not_rewritten(self) -> None:
        self.path.write_text("unchanged")
        before = os.stat(self.path)
        redact_file_literals(self.path, ["", ""])
        after = os.stat(self.path)
        self.assertEqual(
            (before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns)
        )

    def test_undecodable_file_stays_intact_unless_errors_are_replaced(self) -> None:
        raw = b"key=secret \xff tail"
        self.path.write_bytes(raw)
        with self.assertRaises(UnicodeDecodeError):
            redact_file_literals(self.path, ["secret"])
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertEqual([entry.name for entry in self.root.iterdir()], ["trace.jsonl"])
        redact_file_literals(self.path, ["secret"], errors="replace")
        self.assertEqual(self.path.read_text(), "key=[REDACTED] � tail")


if __name__ == "__main__":
    unittest.main()
