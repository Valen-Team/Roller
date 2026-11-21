"""One copy of each image of a sample's recorded request bodies.

Every model request resends the images of the conversation so far. The request
recording proxy writes the bytes of each base64 data URI once, to
``blobs/<sha256>.<subtype>`` beside ``wire_trace.jsonl``, and records
``data:<mime>;base64,roller-blob:<sha256>.<subtype>`` in its place in
``wire_trace.jsonl`` and in the captured Calls. ``load_blobs`` puts the
original data URI back, character for character.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

BLOBS_DIR = "blobs"

_DATA_URI = re.compile(r"data:[\w.+-]+/([\w.+-]+);base64,([A-Za-z0-9+/]*={0,2})")
_REFERENCE = re.compile(
    r"(data:[\w.+-]+/[\w.+-]+;base64,)roller-blob:([0-9a-f]{64}\.[\w.+-]+)"
)


def store_blobs(value: Any, directory: Path) -> Any:
    """``value`` with each base64 data URI written to ``directory`` and referenced."""

    if isinstance(value, dict):
        return {key: store_blobs(item, directory) for key, item in value.items()}
    if isinstance(value, list):
        return [store_blobs(item, directory) for item in value]
    if not isinstance(value, str):
        return value
    match = _DATA_URI.fullmatch(value)
    if match is None or len(match[2]) % 4:
        return value
    data = base64.b64decode(match[2])
    # Only canonical base64 is restored as it was sent.
    if base64.b64encode(data).decode() != match[2]:
        return value
    name = f"{hashlib.sha256(data).hexdigest()}.{match[1]}"
    path = directory / name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    return f"{value[: match.start(2)]}roller-blob:{name}"


def load_blobs(value: Any, directory: Path) -> Any:
    """``value`` with each blob reference replaced by its original data URI."""

    if isinstance(value, dict):
        return {key: load_blobs(item, directory) for key, item in value.items()}
    if isinstance(value, list):
        return [load_blobs(item, directory) for item in value]
    if isinstance(value, str) and (match := _REFERENCE.fullmatch(value)):
        data = (directory / match[2]).read_bytes()
        return match[1] + base64.b64encode(data).decode()
    return value
