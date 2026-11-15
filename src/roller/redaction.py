"""Literal secret redaction of trace and log files in bounded memory."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

REDACTED = "[REDACTED]"


def redact_file_literals(
    path: Path,
    secrets: Iterable[str],
    *,
    errors: str = "strict",
    chunk_size: int = 64 * 1024,
) -> None:
    """Replace every occurrence of ``secrets`` in ``path`` with a marker.

    The file is streamed through a temporary sibling and swapped in at the
    end, so memory stays at one chunk plus the longest secret and a decoding
    or I/O failure leaves the original file in place. A secret that straddles
    two chunks is still replaced.
    """
    values = sorted({value for value in secrets if value}, key=len, reverse=True)
    if not values:
        return
    pattern = re.compile("|".join(re.escape(value) for value in values))
    # A secret that starts in the last ``overlap`` characters may be incomplete.
    overlap = len(values[0]) - 1
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as destination:
            with path.open(encoding="utf-8", errors=errors, newline="") as source:
                pending = ""
                while chunk := source.read(chunk_size):
                    pending += chunk
                    boundary = max(0, len(pending) - overlap)
                    for match in pattern.finditer(pending):
                        if match.start() < boundary < match.end():
                            boundary = match.start()
                            break
                    destination.write(pattern.sub(REDACTED, pending[:boundary]))
                    pending = pending[boundary:]
                destination.write(pattern.sub(REDACTED, pending))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
