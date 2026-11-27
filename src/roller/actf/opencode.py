"""Project OpenCode's native session export and request journal into ACTF."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roller.actf import openclaw_events_to_actf
from roller.opencode_capture import native_messages, root_session_ids
from roller.storage import atomic_write_json
from roller.usage import ZERO_USAGE


def read_opencode_events(path: Path) -> list[dict[str, Any]]:
    # JSON Lines records are delimited by LF, not every character recognized by
    # str.splitlines().  In particular, U+2028/U+2029 are legal inside a JSON
    # string and must remain part of the record.
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    if not all(isinstance(e, dict) for e in events):
        raise ValueError("OpenCode journal must contain JSON object events")
    return events


def _timestamp(milliseconds):
    if isinstance(milliseconds, (int, float)):
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()
    return None


def _attachment_metadata(value: Any) -> Any:
    """Keep attachment provenance without copying large inline payloads to ACTF.

    OpenCode's ``read`` tool returns images as data URLs in ``attachments``.
    The native JSONL remains the lossless source, while ACTF/SFT gets a bounded,
    auditable description instead of duplicating multi-megabyte base64 strings.
    """

    if isinstance(value, dict):
        return {str(key): _attachment_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_attachment_metadata(item) for item in value]
    if isinstance(value, str) and len(value) > 4096:
        summary: dict[str, Any] = {
            "omitted": True,
            "characters": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
        if value.startswith("data:"):
            header, separator, payload = value.partition(",")
            summary.update(
                {
                    "scheme": "data",
                    "header": header,
                    "payload_characters": len(payload) if separator else 0,
                }
            )
        return summary
    return deepcopy(value)


def _attachment_placeholder(attachments: list[Any]) -> str:
    metadata = _attachment_metadata(attachments)
    return (
        "[OpenCode attachments omitted from text-only export: "
        + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        + "]"
    )


def _image_block(file: dict[str, Any]) -> dict[str, Any] | None:
    """ACTF image block for an OpenCode file reference, None for other media."""

    mime = file.get("mime")
    if not isinstance(mime, str) or not mime.lower().startswith("image/"):
        return None
    image_url = file.get("url")
    if not isinstance(image_url, str) or not image_url:
        raise ValueError("OpenCode image attachment must contain a URL")
    return {"type": "image", "data": image_url, "mimeType": mime.lower()}


def _tool_result_content(result: str, attachments: list[Any]) -> list[dict[str, Any]]:
    content = [{"type": "text", "text": result}]
    non_image_attachments: list[Any] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            raise ValueError("OpenCode tool attachment must be an object")
        image = _image_block(attachment)
        if image is not None:
            content.append(image)
        else:
            non_image_attachments.append(attachment)
    if non_image_attachments:
        content.append(
            {
                "type": "text",
                "text": _attachment_placeholder(non_image_attachments),
            }
        )
    return content


def _json_type(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        return {"type": "array"}
    if isinstance(value, dict):
        return {"type": "object"}
    return {}


def opencode_events_to_actf(
    events: Iterable[dict[str, Any]],
    *,
    keep_thinking: bool = True,
    require_captured_requests: bool = True,
) -> dict[str, Any]:
    events = list(events)
    schemas, systems = {}, []
    has_requests = False
    roots = root_session_ids(events)
    for event in events:
        if event.get("type") != "request":
            continue
        if (
            event.get("native_session_id")
            and roots
            and event["native_session_id"] not in roots
        ):
            continue
        has_requests = True
        body = event["body"]
        for tool in body.get("tools", []):
            name = tool.get("function", {}).get("name")
            if not name:
                raise ValueError("OpenCode tool schema has no function name")
            if name in schemas and schemas[name] != tool:
                raise ValueError(f"OpenCode tool schema changed: {name}")
            schemas[name] = deepcopy(tool)
        for message in body.get("messages", []):
            if message.get("role") == "system":
                text = message.get("content")
                if not isinstance(text, str):
                    raise ValueError("OpenCode system prompt must be text")
                if text not in systems:
                    systems.append(text)
    if not systems and require_captured_requests:
        raise ValueError("OpenCode journal has no captured system prompt")
    rows, calls_seen = [], set()
    inferred_tools: dict[str, dict[str, dict[str, Any]]] = {}
    sft_unavailable_reason = None
    for native in native_messages(events, root_only=True):
        info, parts = native["info"], native["parts"]
        role = info["role"]
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported OpenCode message role: {role}")
        if error := info.get("error"):
            if not (
                role == "assistant"
                and isinstance(error, dict)
                and error.get("name") == "ContextOverflowError"
            ):
                raise ValueError("OpenCode native assistant contains an error")
            info = {**info, "finish": info.get("finish") or "error"}
            sft_unavailable_reason = "OpenCode recorded a context overflow"
        content, results = [], []
        for part in parts:
            kind = part["type"]
            if kind == "text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("OpenCode text part must contain text")
                content.append({"type": "text", "text": part["text"]})
            elif (
                kind == "file"
                and role == "user"
                and (image := _image_block(part)) is not None
            ):
                content.append(image)
            elif kind == "reasoning" and role == "assistant":
                content.append({"type": "thinking", "thinking": part["text"]})
            elif kind == "tool" and role == "assistant":
                state = part["state"]
                if state["status"] not in {"completed", "error", "running"}:
                    raise ValueError("OpenCode tool did not complete")
                call_id, name, arguments = part["callID"], part["tool"], state["input"]
                if not isinstance(arguments, dict):
                    raise ValueError("OpenCode tool arguments must be a JSON object")
                if not call_id or call_id in calls_seen:
                    raise ValueError("OpenCode tool call ID is missing or repeated")
                if name not in schemas:
                    if require_captured_requests:
                        raise ValueError(
                            f"OpenCode tool call has no captured schema: {name}"
                        )
                    fields = inferred_tools.setdefault(name, {})
                    for key, value in arguments.items():
                        if isinstance(key, str):
                            fields.setdefault(key, _json_type(value))
                calls_seen.add(call_id)
                content.append(
                    {
                        "type": "toolCall",
                        "id": call_id,
                        "name": name,
                        "arguments": arguments,
                    }
                )
                if state["status"] == "running":
                    sft_unavailable_reason = (
                        "OpenCode journal ended with a running tool call"
                    )
                    continue
                result = (
                    state.get("error")
                    if state["status"] == "error"
                    else state.get("output")
                )
                if not isinstance(result, str):
                    raise ValueError("OpenCode tool result must be text")
                attachments = state.get("attachments")
                message: dict[str, Any] = {
                    "role": "toolResult",
                    "toolCallId": call_id,
                    "toolName": name,
                    "content": result,
                    "isError": state["status"] == "error",
                }
                if attachments:
                    if not isinstance(attachments, list):
                        raise ValueError("OpenCode tool attachments must be a list")
                    metadata = _attachment_metadata(attachments)
                    message["content"] = _tool_result_content(result, attachments)
                    message["details"] = {"attachments": metadata}
                results.append(
                    {
                        "type": "message",
                        "timestamp": _timestamp(state.get("time", {}).get("end")),
                        "message": message,
                    }
                )
            elif kind in {"step-start", "step-finish", "snapshot", "patch", "retry"}:
                continue  # Execution metadata remains in the native journal.
            else:
                raise ValueError(
                    f"OpenCode part would lose data in text-only SFT: {kind}"
                )
        visible_content = any(
            part.get("type") == "toolCall"
            or (part.get("type") == "text" and bool(part.get("text")))
            or (
                keep_thinking
                and part.get("type") == "thinking"
                and bool(part.get("thinking"))
            )
            for part in content
        )
        if role == "assistant" and not visible_content and not results:
            # Some providers emit a completed reasoning-only turn before the
            # next assistant message. Once private reasoning is stripped that
            # turn cannot be represented as valid SFT. Keep it in the native
            # journal, but omit it from ACTF so a successful rollout does not
            # become an invalid empty assistant message.
            continue
        if (
            role == "assistant"
            and not content
            and not results
            and not info.get("finish")
            and not info.get("time", {}).get("completed")
        ):
            # OpenCode creates the next assistant record before sending the
            # provider request. If Roller reaches its wall-clock timeout while
            # that request is still in flight, the native export can end with
            # this completely empty, unfinished placeholder. Preserve it in
            # the raw journal for diagnosis, but do not turn it into an invalid
            # empty training message.
            continue
        tokens = info.get("tokens") or {}
        cache = tokens.get("cache") or {}
        message = {
            "role": role,
            "content": content,
            "usage": {
                "input": tokens.get("input") or 0,
                "output": tokens.get("output") or 0,
                "cacheRead": cache.get("read") or 0,
                "cacheWrite": cache.get("write") or 0,
            },
            "stopReason": info.get("finish", ""),
        }
        rows.append(
            {
                "type": "message",
                "timestamp": _timestamp(info.get("time", {}).get("created")),
                "message": message,
            }
        )
        rows.extend(results)
    all_tools = list(schemas.values()) + [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Tool invoked by OpenCode.",
                "parameters": {
                    "type": "object",
                    "properties": fields,
                    "additionalProperties": True,
                },
            },
        }
        for name, fields in inferred_tools.items()
    ]
    trajectory = openclaw_events_to_actf(
        rows,
        system_prompts=["\n\n".join(systems)] if systems else [],
        tools=all_tools,
        keep_thinking=keep_thinking,
    )
    if schemas and inferred_tools:
        tools_capture = {"source": "requests", "status": "captured_merged"}
    elif inferred_tools:
        tools_capture = {"source": "session", "status": "inferred"}
    elif schemas or has_requests:
        tools_capture = {"source": "requests", "status": "captured"}
    else:
        tools_capture = {"source": "none", "status": "missing"}
    trajectory["tools_capture"] = tools_capture
    if sft_unavailable_reason:
        trajectory["sft_unavailable_reason"] = sft_unavailable_reason
    return trajectory


def convert_opencode_trace(
    source: Path,
    destination: Path,
    *,
    keep_thinking: bool = True,
    require_captured_requests: bool = True,
) -> dict[str, Any]:
    trajectory = opencode_events_to_actf(
        read_opencode_events(source),
        keep_thinking=keep_thinking,
        require_captured_requests=require_captured_requests,
    )
    atomic_write_json(destination, trajectory)
    return trajectory


def extract_opencode_usage(source: Path) -> dict[str, int | float]:
    totals = dict(ZERO_USAGE)
    for message in native_messages(read_opencode_events(source)):
        info = message["info"]
        if info.get("role") != "assistant":
            continue
        tokens = info.get("tokens") or {}
        cache = tokens.get("cache") or {}
        values = {
            "input_tokens": tokens.get("input", 0),
            "output_tokens": tokens.get("output", 0),
            "cache_read_tokens": cache.get("read", 0),
            "cache_write_tokens": cache.get("write", 0),
            "cost_usd": info.get("cost", 0),
        }
        totals["request_count"] += 1
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
        total = tokens.get("total")
        totals["total_tokens"] += (
            total
            if isinstance(total, (int, float))
            else sum(
                values[k] or 0
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                )
            )
        )
    return totals
