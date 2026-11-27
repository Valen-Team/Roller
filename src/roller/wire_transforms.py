"""Request/response transformations and extraction helpers for WireTraceProxy."""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from roller.blobs import BLOBS_DIR, load_blobs

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}

BODY_KEYS = {"messages", "tools", "input"}

THINKING_MESSAGE_KEYS = ("reasoning_content", "reasoning", "reasoning_text")
THINKING_BLOCK_TYPES = ("thinking", "reasoning")

ENCRYPTED_CONTENT_ERROR = "invalid_encrypted_content"
ENCRYPTED_CONTENT_RETRY_DELAY_SECONDS = 0.5
ERROR_INSPECTION_LIMIT = 64 * 1024


def _is_encrypted_content_rejection(payload: bytes, content_encoding: str) -> bool:
    """Inspect a bounded copy; keep the original response intact for forwarding."""
    if len(payload) > ERROR_INSPECTION_LIMIT:
        return False
    encoding = content_encoding.strip().lower()
    try:
        if encoding in {"gzip", "deflate"}:
            decoder = zlib.decompressobj(31 if encoding == "gzip" else 15)
            payload = decoder.decompress(payload, ERROR_INSPECTION_LIMIT + 1)
            if (
                len(payload) > ERROR_INSPECTION_LIMIT
                or not decoder.eof
                or decoder.unused_data
            ):
                return False
        elif encoding not in {"", "identity"}:
            return False
        body = json.loads(payload)
    except (ValueError, UnicodeError, RecursionError, zlib.error):
        return False
    error = body.get("error") if isinstance(body, dict) else None
    return isinstance(error, dict) and error.get("code") == ENCRYPTED_CONTENT_ERROR


def normalize_chat_completion_reasoning_response(body: Any) -> Any:
    """Normalize visible Chat Completions reasoning to ``reasoning_content``.

    OpenClaw 2026.7.1 records ``reasoning``/``reasoning_text`` response deltas as
    thinking blocks but only replays ``reasoning_content`` for a generic custom
    provider. Normalizing those equivalent response fields before OpenClaw sees
    them preserves the text in later request history without inventing content.
    """
    if not isinstance(body, dict):
        return body
    choices = body.get("choices")
    if not isinstance(choices, list):
        return body
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for container_name in ("delta", "message"):
            message = choice.get(container_name)
            if not isinstance(message, dict):
                continue
            current = message.get("reasoning_content")
            if not isinstance(current, str) or not current:
                for source_name in ("reasoning", "reasoning_text"):
                    source = message.get(source_name)
                    if isinstance(source, str) and source:
                        message["reasoning_content"] = source
                        break
            message.pop("reasoning", None)
            message.pop("reasoning_text", None)
    return body


def strip_thinking_from_body(body: Any) -> Any:
    """Drop thinking/reasoning content from request history.

    Applied before forwarding so the model never receives prior-turn thinking;
    the current turn's reasoning streams back untouched. Handles both Chat
    Completions ``messages`` and Responses API ``input`` items.
    """
    if not isinstance(body, dict):
        return body
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for key in THINKING_MESSAGE_KEYS:
                message.pop(key, None)
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [
                    block
                    for block in content
                    if not (
                        isinstance(block, dict)
                        and block.get("type") in THINKING_BLOCK_TYPES
                    )
                ]
    responses_input = body.get("input")
    if isinstance(responses_input, list):
        body["input"] = [
            item
            for item in responses_input
            if not (isinstance(item, dict) and item.get("type") in THINKING_BLOCK_TYPES)
        ]
    return body


RESPONSES_IMAGE_DETAILS = ("auto", "low", "high", "original")


def fill_responses_image_detail(body: Any, detail: str) -> bool:
    """Set ``detail`` on Responses ``input_image`` blocks that omit it.

    Covers message content and tool outputs in ``input``. Returns whether
    the body changed; blocks that already carry a detail are left alone.
    """
    items = body.get("input") if isinstance(body, dict) else None
    changed = False
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        for blocks in (item.get("content"), item.get("output")):
            for block in blocks if isinstance(blocks, list) else []:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "input_image"
                    and "detail" not in block
                ):
                    block["detail"] = detail
                    changed = True
    return changed


def move_responses_tool_images(body: Any) -> bool:
    """Attach tool images as user images for backends that ignore them in outputs."""
    items = body.get("input") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return False
    rewritten = []
    changed = False
    for item in items:
        output = item.get("output") if isinstance(item, dict) else None
        if not (
            isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and isinstance(output, list)
        ):
            rewritten.append(item)
            continue
        images = [
            b for b in output if isinstance(b, dict) and b.get("type") == "input_image"
        ]
        if not images:
            rewritten.append(item)
            continue
        remaining = [b for b in output if b not in images]
        call_id = item["call_id"]
        notice = (
            f"Images from tool call {call_id} are attached in the following message."
        )
        if all(
            isinstance(b, dict) and b.get("type") == "input_text" for b in remaining
        ):
            tool_output = "\n".join([b["text"] for b in remaining] + [notice])
        else:
            tool_output = remaining + [{"type": "input_text", "text": notice}]
        rewritten.extend(
            [
                {**item, "output": tool_output},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Image output from tool call {call_id}:",
                        },
                        *images,
                    ],
                },
            ]
        )
        changed = True
    if changed:
        body["input"] = rewritten
    return changed


def _wire_image_block(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    block_type = value.get("type")
    if block_type in {"image_url", "input_image"}:
        image_url = value.get("image_url", value.get("url"))
        if isinstance(image_url, Mapping):
            image_url = image_url.get("url")
        if isinstance(image_url, str) and image_url:
            return {"type": "image", "data": image_url}
        return None
    if block_type != "image":
        return None
    source = value.get("source")
    if isinstance(source, Mapping):
        return {"type": "image", "source": deepcopy(dict(source))}
    data = value.get("data")
    if not isinstance(data, str) or not data:
        return None
    block: dict[str, Any] = {"type": "image", "data": data}
    mime_type = value.get("mimeType", value.get("mime_type"))
    if isinstance(mime_type, str) and mime_type:
        block["mimeType"] = mime_type
    return block


def _request_image_inputs(body: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Return a request's image blocks and the text sent in the same messages."""

    if not isinstance(body, Mapping):
        return [], []
    messages = body.get("messages", body.get("input"))
    if not isinstance(messages, list):
        return [], []
    images: list[dict[str, Any]] = []
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        direct = _wire_image_block(message)
        if direct is not None:
            images.append(direct)
        content = message.get("content")
        if not isinstance(content, list):
            continue
        found = [
            image
            for block in content
            if (image := _wire_image_block(block)) is not None
        ]
        if found:
            images.extend(found)
            texts.extend(
                block["text"]
                for block in content
                if isinstance(block, Mapping) and isinstance(block.get("text"), str)
            )
    return images, texts


def _image_digest(image: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(image, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def wire_text_digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def extract_wire_image_requests(path: Path) -> list[dict[str, Any]]:
    """Return image-bearing model requests in wire order.

    Full traces expose the original request body, with its images read back
    from ``blobs/``. Compact ACTF-mode traces keep
    ``image_inputs`` (each distinct image's bytes once, later occurrences as a
    ``sha256`` reference) and digests of the accompanying text, so image bytes
    survive long enough to enrich ACTF without retaining prompt text.
    """

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return []
    entries: list[Mapping[str, Any]] = []
    stored: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, Mapping):
            continue
        entries.append(entry)
        compact = entry.get("image_inputs")
        if isinstance(compact, list):
            for item in compact:
                if isinstance(item, Mapping) and "sha256" not in item:
                    stored[_image_digest(item)] = dict(item)
    requests: list[dict[str, Any]] = []
    for entry in entries:
        compact = entry.get("image_inputs")
        body = entry.get("body")
        if isinstance(compact, list):
            images = [
                deepcopy(stored[digest])
                for item in compact
                if isinstance(item, Mapping)
                and (digest := item.get("sha256") or _image_digest(item)) in stored
            ]
            text_digests = entry.get("image_text_digests")
            if not isinstance(text_digests, list):
                text_digests = []
        else:
            images, texts = _request_image_inputs(
                load_blobs(body, path.parent / BLOBS_DIR)
            )
            text_digests = [wire_text_digest(text) for text in texts]
        if images:
            body_tools = body.get("tools") if isinstance(body, Mapping) else None
            tool_count = (
                len(body_tools)
                if isinstance(body_tools, list)
                else entry.get("n_tools")
            )
            requests.append(
                {
                    "seq": entry.get("seq"),
                    "ts": entry.get("ts"),
                    "tool_count": tool_count,
                    "images": images,
                    "text_digests": text_digests,
                }
            )
    return requests


def extract_wire_tools(path: Path) -> list[Any] | None:
    """Merge every tools schema recorded across wire-trace requests."""
    try:
        # JSONL records are separated by LF.  Unicode line separators can
        # legally occur inside string values and must not split a record.
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return None
    merged: dict[str, Any] = {}
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") == "tools":
            tools = entry.get("tools")
        else:
            body = entry.get("body")
            tools = body.get("tools") if isinstance(body, dict) else None
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            source = function if isinstance(function, dict) else tool
            name = source.get("name")
            if isinstance(name, str) and name:
                merged[name] = tool
    return list(merged.values()) or None


def _rewrite_json(payload: bytes, rewrite: Callable[[Any], Any]) -> bytes:
    """A JSON body after ``rewrite``, which changes it in place.

    The bytes are returned as they came when they are not JSON or ``rewrite``
    returns nothing true.
    """

    try:
        body = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return payload
    if not rewrite(body):
        return payload
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _rewrite_event(line: bytes, rewrite: Callable[[Any], Any]) -> bytes:
    """A stream line after ``rewrite`` of the JSON in its ``data`` field."""

    content = line.rstrip(b"\r\n")
    if not content.startswith(b"data:"):
        return line
    try:
        event = json.loads(content[5:])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return line
    if not rewrite(event):
        return line
    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return b"data: " + data.encode("utf-8") + line[len(content) :]


def _sse(event: dict[str, Any]) -> bytes:
    data = json.dumps(event, ensure_ascii=False)
    return f"event: {event['type']}\ndata: {data}\n\n".encode()
