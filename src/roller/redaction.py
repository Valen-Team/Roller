"""Literal and JSON-key secret redaction for traces, logs, and run locks."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

REDACTED = "[REDACTED]"
REDACTED_VALUE = "<redacted>"


def normalize_anthropic_base_url(value: str) -> str:
    """Normalize an Anthropic endpoint or API base to the SDK base URL."""

    try:
        parsed = urlsplit(value.strip())
        parsed.port  # noqa: B018 - reading it raises for an invalid port
    except ValueError as exc:
        raise ValueError("Anthropic API base must be an absolute http(s) URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Anthropic API base must be an absolute http(s) URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/messages"):
        path = path[: -len("/v1/messages")]
    elif path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def redact_anthropic_base_url(value: str) -> str:
    """Return a credential-free endpoint suitable for provenance metadata."""

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    netloc = hostname if port is None else f"{hostname}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


SENSITIVE_KEY_NAMES = {
    "authorization",
    "password",
    "secret",
    "token",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
}


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def redact_sensitive_values(value: Any) -> Any:
    """Return a JSON-compatible copy with credentials replaced by a marker."""
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            normalized = _normalized_key(key)
            if (
                normalized in SENSITIVE_KEY_NAMES
                or normalized.endswith("apikey")
                or normalized.endswith("token")
            ):
                redacted[key] = REDACTED_VALUE
            else:
                redacted[key] = redact_sensitive_values(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_values(item) for item in value]
    return value


def redact_jsonl_file(path: Path) -> None:
    """Atomically redact credentials in a JSONL file, preserving malformed lines."""
    temporary = path.with_name(f".{path.name}.redacted.tmp")
    try:
        with (
            path.open("r", encoding="utf-8", errors="replace") as source,
            temporary.open("w", encoding="utf-8") as destination,
        ):
            for line in source:
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    destination.write(line)
                    continue
                destination.write(
                    json.dumps(redact_sensitive_values(parsed), ensure_ascii=False)
                    + "\n"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
