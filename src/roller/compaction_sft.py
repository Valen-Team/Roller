"""Reconstruct main-agent SFT windows from captured model Calls.

One window is one SFT record: its first Call supplies the full prompt, and
assistant outputs from its Calls are targets. Later Calls join that record only
when their prompts extend the verified record exactly. Native events identify
known compaction boundaries; native request rows detect possible rewrites.
Captured Calls establish the actual prompt and response for export.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roller.actf import OPENCLAW_ANNOUNCEMENT_SFT_REASON
from roller.actf.claude_code import (
    CLAUDE_COMPACTION_SFT_REASON,
)
from roller.actf.claude_code import (
    _read_events as read_claude_events,
)
from roller.actf.pi import PI_COMPACTION_SFT_REASON
from roller.actf.stirrup import (
    STIRRUP_COMPACTION_SFT_REASON,
    kept_call_ids,
    summary_call_ids,
)
from roller.blobs import BLOBS_DIR, load_blobs
from roller.capture import _safe_file, read_jsonl
from roller.compaction_parsers import (
    _prompt_from_call,
    _target_messages_from_call,
    _time,
)
from roller.harness_registry import compaction_mode
from roller.opencode_capture import root_session_ids
from roller.sft import (
    SFTConversionError,
    _arguments_json,
    validate_sft_record,
)
from roller.storage import atomic_write_json

SFT_SEGMENTS_DIR = "sft_segments"
# These only reject the linear ACTF projection; captured Calls supersede them.
_CONTEXT_ONLY_REASONS = {
    CLAUDE_COMPACTION_SFT_REASON,
    PI_COMPACTION_SFT_REASON,
    STIRRUP_COMPACTION_SFT_REASON,
    OPENCLAW_ANNOUNCEMENT_SFT_REASON,
}


def _is_main_compaction(row: Mapping[str, Any], mode: str) -> bool:
    if mode == "claude":
        return row.get("subtype") == "compact_boundary" and not row.get("isSidechain")
    return mode == "codex" and row.get("type") == "compacted"


def _native_rows(path: Path, harness: str) -> list[dict[str, Any]]:
    return (
        read_claude_events(path)
        if compaction_mode(harness) == "claude"
        else read_jsonl(path)
    )


def _comparable(value: Any) -> Any:
    """Ignore cache hints while comparing the *same* native representation."""
    if isinstance(value, Mapping):
        return {
            key: _comparable(item)
            for key, item in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [_comparable(item) for item in value]
    return value


def _context_changed(contexts: list[Mapping[str, Any]]) -> bool:
    previous_messages: list[Any] | None = None
    previous_tools: Any = None
    for context in contexts:
        messages = context.get("messages")
        if not isinstance(messages, list):
            continue
        current = _comparable(messages)
        tools = _comparable(context.get("tools", []))
        if previous_messages is not None and (
            current[: len(previous_messages)] != previous_messages
            or tools != previous_tools
        ):
            return True
        previous_messages, previous_tools = current, tools
    return False


def _single_opencode_session(rows: list[dict[str, Any]], roots: set[str]) -> bool:
    """Allow untagged requests only when native evidence shows one root session."""
    if len(roots) != 1:
        return False
    sessions = set()
    for row in rows:
        info = None
        parts = []
        if row.get("type") == "session_export":
            data = row.get("data") or {}
            info = data.get("info")
            parts = [
                part
                for message in data.get("messages", [])
                for part in message.get("parts", [])
            ]
        elif row.get("type") == "event":
            event = row.get("event") or {}
            properties = event.get("properties") or {}
            if event.get("type") in {"session.created", "session.updated"}:
                info = properties.get("info")
            elif event.get("type") == "message.part.updated":
                parts = [properties.get("part") or {}]
        if isinstance(info, Mapping) and isinstance(info.get("id"), str):
            sessions.add(info["id"])
        if any(
            part.get("type") == "tool" and part.get("tool") == "task" for part in parts
        ):
            return False
    return sessions == roots


def _opencode_request_scope(
    rows: list[dict[str, Any]], requests: list[dict[str, Any]]
) -> tuple[set[str], bool]:
    roots = root_session_ids(rows)
    session_ids = {
        row.get("native_session_id")
        for row in requests
        if isinstance(row.get("native_session_id"), str)
    }
    if not roots:
        return session_ids, len(session_ids) > 1
    if any(row.get("native_session_id") is None for row in requests):
        ambiguous = not _single_opencode_session(rows, roots) or bool(
            session_ids - roots
        )
        return (set() if not ambiguous else roots), ambiguous
    return roots, False


def has_main_compaction(
    path: Path, harness: str | None, rows: list[dict[str, Any]] | None = None
) -> bool:
    """Find native reset evidence without converting request content to SFT."""
    rows = rows if rows is not None else _native_rows(path, harness or "")
    mode = compaction_mode(harness)
    if mode in {"claude", "codex"}:
        return any(_is_main_compaction(row, mode) for row in rows)
    if mode == "openclaw":
        return any(row.get("type") == "compaction" for row in rows)
    if mode == "stirrup":
        return any(row.get("type") == "summary" for row in rows)
    if mode == "pi" and any(
        row.get("type") == "event"
        and (row.get("event") or {}).get("type") == "compaction_end"
        for row in rows
    ):
        return True
    requests = [row for row in rows if row.get("type") == "request"]
    roots, ambiguous = (
        _opencode_request_scope(rows, requests)
        if mode == "opencode"
        else (set(), False)
    )
    if ambiguous:
        return True  # A request may belong to a child; do not assume linearity.
    contexts = []
    for row in requests:
        if roots and row.get("native_session_id") not in roots:
            continue
        body = row.get("body", row)
        if isinstance(body, Mapping):
            contexts.append(body)
    return _context_changed(contexts)


def _raw_context_changed(calls: list[dict[str, Any]]) -> bool:
    """Detect a prompt rewrite without requiring SFT conversion of its content."""
    contexts = []
    for call in calls:
        context = call.get("context")
        if isinstance(context, Mapping):
            contexts.append(context)
    return _context_changed(contexts)


def _assistant_signature(
    message: Mapping[str, Any],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Compare model outputs with ACTF main steps without harness-specific tool IDs."""
    content = message.get("content")
    calls = message.get("tool_calls") or []
    if not isinstance(content, str) or not isinstance(calls, list):
        raise SFTConversionError("ACTF assistant output cannot be matched to a Call")
    tools = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise SFTConversionError("ACTF tool call cannot be matched to a Call")
        kind = call.get("type")
        if kind == "function":
            function = call.get("function")
        elif kind == "custom":
            function = call.get("custom")
        else:
            function = call
        if not isinstance(function, Mapping):
            raise SFTConversionError("ACTF tool call cannot be matched to a Call")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise SFTConversionError("ACTF tool call lacks a name")
        if kind == "custom" or call.get("native_type") == "custom_tool_call":
            arguments = function.get("input")
            if not isinstance(arguments, str):
                raise SFTConversionError("ACTF custom tool input is not text")
            tools.append((name, arguments))
        else:
            encoded = _arguments_json(
                function.get("arguments", function.get("input", {})),
                "ACTF tool arguments",
            )
            tools.append(
                (
                    name,
                    json.dumps(json.loads(encoded), ensure_ascii=False, sort_keys=True),
                )
            )
    return content, tuple(tools)


def _tool_arguments_match(observed: str, native: str) -> bool:
    """A Harness may record a tool input after applying its schema defaults.

    Claude Code stores ``Edit`` with ``replace_all: false`` in its native
    session even when the model never emitted the key, so the native step
    may carry keys the wire response does not. Every key the model did send
    must still be present and equal; keys only the Harness added are ignored.
    Non-JSON inputs (Codex custom tools) keep comparing exactly.
    """

    if observed == native:
        return True
    try:
        sent, recorded = json.loads(observed), json.loads(native)
    except (TypeError, ValueError):
        return False
    if not isinstance(sent, dict) or not isinstance(recorded, dict):
        return False
    return not set(sent) - set(recorded) and all(
        recorded[key] == value for key, value in sent.items()
    )


def _signature_matches(
    observed: tuple[str, tuple[tuple[str, str], ...]],
    native: tuple[str, tuple[tuple[str, str], ...]],
) -> bool:
    """Bind a captured Call to a native step; text and tool order stay exact."""

    if observed == native:
        return True
    observed_content, observed_tools = observed
    native_content, native_tools = native
    if observed_content != native_content or len(observed_tools) != len(native_tools):
        return False
    return all(
        observed_name == native_name
        and _tool_arguments_match(observed_args, native_args)
        for (observed_name, observed_args), (native_name, native_args) in zip(
            observed_tools, native_tools
        )
    )


def _is_summary_call(
    call: Mapping[str, Any],
    messages: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
) -> bool:
    """The next native boundary recorded this Call's text as its summary."""
    started = _time(call.get("started_at"))
    boundary = next(
        (item for item in boundaries if _time(item["timestamp"]) > started), None
    )
    summary = boundary["summary"] if boundary else None
    return (
        isinstance(summary, str)
        and len(messages) == 1
        and not messages[0]["tool_calls"]
        and bool(messages[0]["content"])
        and messages[0]["content"] in summary
    )


def _call_context(call: Mapping[str, Any]) -> str:
    """Where a Call sits in the capture, for error messages."""
    return f"request_seq {call.get('request_seq')}, protocol {call.get('protocol')}"


def _describe_call(call: Mapping[str, Any]) -> str:
    return f"Call {call.get('call_id')} ({_call_context(call)})"


def _step_mismatch(
    messages: list[dict[str, Any]],
    index: int,
    targets: list[tuple[str, tuple[tuple[str, str], ...]]],
) -> str:
    """Why a Call's output does not bind to the next ACTF main step."""
    if index >= len(targets):
        return f"all {len(targets)} ACTF main steps are already bound"
    expected = f"expected ACTF main step {index + 1} of {len(targets)}"
    if len(messages) != 1:
        return f"{expected}, but the Call has {len(messages)} assistant messages"
    content, tools = _assistant_signature(messages[0])
    native_content, native_tools = targets[index]
    if content != native_content:
        return f"{expected}, but the assistant text differs"
    if [name for name, _ in tools] != [name for name, _ in native_tools]:
        return (
            f"{expected}, but the tool calls differ: Call has "
            f"{[name for name, _ in tools]}, step has "
            f"{[name for name, _ in native_tools]}"
        )
    return f"{expected}, but the tool call arguments differ"


def _calls_matching_main_steps(
    sample_dir: Path,
    calls: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    summaries: set[str] | None = None,
    native_summaries: frozenset[str] = frozenset(),
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Bind root Calls to ACTF main steps in order; only native summaries are skipped.

    Codex and OpenClaw send compaction requests under the main identity. A
    Call is skipped only when it is named in ``native_summaries`` or it does
    not match the next main step and its text is recorded verbatim in the
    following native compaction row. With ``summaries``, such a Call is kept
    as a target and its ID added there.
    """
    trajectory = json.loads(
        (sample_dir / "trace_ACTF.json").read_text(encoding="utf-8")
    )
    reason = (
        trajectory.get("sft_unavailable_reason")
        if isinstance(trajectory, Mapping)
        else None
    )
    if reason and reason not in _CONTEXT_ONLY_REASONS:
        raise SFTConversionError(str(reason))
    steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else None
    if (
        not isinstance(steps, list)
        or not steps
        or any(not isinstance(step, Mapping) for step in steps)
    ):
        raise SFTConversionError("main ACTF steps are required to bind captured Calls")
    targets = [
        _assistant_signature(step.get("assistant_content", {})) for step in steps
    ]
    verified = []
    for call in calls:
        if call.get("export_status") != "available":
            raise SFTConversionError(
                f"main Call {call.get('call_id')} is incomplete: "
                f"{_describe_call(call)} has export_status "
                f"{call.get('export_status')}"
            )
        messages = _target_messages_from_call(call)
        index = len(verified) - len(summaries or ())
        if (
            index < len(steps)
            and len(messages) == 1
            and _signature_matches(_assistant_signature(messages[0]), targets[index])
        ):
            # ACTF applies keep_thinking and retains visible Codex reasoning. Use
            # that canonical value while the Call verifies the output and prompt.
            reasoning = steps[index]["assistant_content"].get("reasoning_content")
            if isinstance(reasoning, str):
                messages[0]["reasoning_content"] = reasoning
            verified.append((call, messages))
        elif not (
            call["call_id"] in native_summaries
            or _is_summary_call(call, messages, boundaries)
        ):
            raise SFTConversionError(
                f"root Calls do not correspond one-to-one with ACTF main steps; Call "
                f"{call.get('call_id')} matches neither the next step nor a native summary "
                f"({_call_context(call)}; {_step_mismatch(messages, index, targets)})"
            )
        elif summaries is not None:
            summaries.add(call["call_id"])
            verified.append((call, messages))
    if len(verified) - len(summaries or ()) != len(steps):
        raise SFTConversionError(
            "root Calls do not correspond one-to-one with ACTF main steps; "
            f"{len(steps) - len(verified)} steps have no Call "
            f"({len(verified) - len(summaries or ())} of {len(steps)} ACTF main "
            f"steps bound by {len(calls)} root Calls)"
        )
    return verified


def _compaction_boundaries(
    harness: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    mode = compaction_mode(harness)
    boundaries = []
    for row in rows:
        if not (
            _is_main_compaction(row, mode)
            or mode == "openclaw"
            and row.get("type") == "compaction"
            and isinstance(row.get("id"), str)
            and isinstance(row.get("summary"), str)
            or mode == "stirrup"
            and row.get("type") == "summary"
        ):
            continue
        if mode == "claude":
            event_id, summary = row.get("uuid"), None
        elif mode in {"openclaw", "stirrup"}:
            event_id, summary = row["id"], row["summary"]
        else:
            payload = row.get("payload") or {}
            event_id, summary = payload.get("window_id"), payload.get("message")
        boundaries.append(
            {
                "event_id": event_id,
                "timestamp": row.get("timestamp"),
                "summary": summary,
            }
        )
    return boundaries


def _root_calls(
    calls: list[dict[str, Any]], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    participants = manifest.get("participants")
    if not isinstance(participants, list):
        raise SFTConversionError("capture manifest lacks participants")
    roots = {
        p.get("participant_id")
        for p in participants
        if isinstance(p, Mapping) and p.get("role") == "root"
    }
    if len(roots) != 1 or not all(isinstance(root, str) and root for root in roots):
        raise SFTConversionError("compaction requires one observed root participant")
    return [
        call
        for call in sorted(calls, key=lambda c: c.get("request_seq", 0))
        if call.get("participant_id") in roots
    ]


def _without_tool_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenClaw rewrites tool-call IDs when replaying history to some providers."""
    return [
        {
            **message,
            "tool_calls": [{**call, "id": ""} for call in message["tool_calls"]],
        }
        if message["role"] == "assistant"
        else {**message, "tool_call_id": ""}
        if message["role"] == "tool"
        else message
        for message in messages
    ]


def _replayed_prefix_matches(
    replayed: list[dict[str, Any]], stored: list[dict[str, Any]]
) -> bool:
    """A window's own messages against the same messages in a later prompt.

    A stored window ends with the target as the model sent it, while the next
    prompt replays that turn as the Harness recorded it: the same arguments
    can come back with the schema defaults applied and the keys in another
    order. Compare tool arguments by value, so that only a real context
    rewrite ends the window.
    """

    normalized_replayed = _without_tool_ids(replayed)
    normalized_stored = _without_tool_ids(stored)
    if len(normalized_replayed) != len(normalized_stored):
        return False
    for replayed_message, stored_message in zip(normalized_replayed, normalized_stored):
        if replayed_message == stored_message:
            continue
        replayed_calls = replayed_message.get("tool_calls") or []
        stored_calls = stored_message.get("tool_calls") or []
        if len(replayed_calls) != len(stored_calls) or _without_tool_calls(
            replayed_message
        ) != _without_tool_calls(stored_message):
            return False
        for replayed_call, stored_call in zip(replayed_calls, stored_calls):
            if _without_arguments(replayed_call) != _without_arguments(
                stored_call
            ) or not _tool_arguments_match(
                _call_arguments(stored_call), _call_arguments(replayed_call)
            ):
                return False
    return True


def _without_tool_calls(message: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in message.items() if key != "tool_calls"}


def _without_arguments(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return dict(call)
    return {
        **call,
        "function": {
            key: value for key, value in function.items() if key != "arguments"
        },
    }


def _call_arguments(call: Mapping[str, Any]) -> str:
    function = call.get("function")
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    return arguments if isinstance(arguments, str) else ""


def _segment_records(
    calls: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    boundaries: list[dict[str, Any]],
    keep_thinking: bool,
    summaries: set[str] = frozenset(),
    *,
    images_inline: bool = False,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Use one continuation rule; window_index counts emitted SFT records.

    A summary Call in ``summaries`` is a record of its own, so the summary
    never extends the window before it or the one after it.
    """
    times = [_time(boundary["timestamp"]) for boundary in boundaries]
    if any(
        not isinstance(boundary["event_id"], str) or not boundary["event_id"]
        for boundary in boundaries
    ):
        raise SFTConversionError("compaction boundary lacks a native event ID")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise SFTConversionError("compaction boundaries are not in chronological order")
    exported: list[tuple[dict[str, Any], dict[str, Any]]] = []
    native_index = -1
    for call, target in calls:
        when = _time(call.get("started_at")) if boundaries else None
        if when in times:
            raise SFTConversionError(
                "main Call starts at an ambiguous compaction boundary: "
                f"{_describe_call(call)} starts at {call.get('started_at')}"
            )
        current_native = sum(when > boundary for boundary in times) if when else 0
        prompt, tools = _prompt_from_call(call, images_inline=images_inline)
        if not prompt:
            raise SFTConversionError(
                "compaction segment lacks its actual input: "
                f"{_describe_call(call)} has an empty prompt"
            )
        if not keep_thinking:
            for message in [*prompt, *target]:
                if message["role"] == "assistant":
                    message["reasoning_content"] = ""
        previous = exported[-1][1] if exported else None
        prefix = previous["messages"] if previous else []
        summary = call["call_id"] in summaries
        continuation = (
            previous is not None
            and not summary
            and exported[-1][0]["first_call_id"] not in summaries
            and current_native == native_index
            and tools == previous["tools"]
            and _replayed_prefix_matches(prompt[: len(prefix)], prefix)
            and not any(item["role"] == "assistant" for item in prompt[len(prefix) :])
        )
        if continuation:
            # The newest prompt carries the IDs the harness actually replayed.
            previous["messages"] = prompt + target
            exported[-1][0]["last_call_id"] = call["call_id"]
        else:
            boundary = (
                boundaries[current_native - 1]
                if current_native > native_index and current_native
                else None
            )
            metadata = {
                "path": f"{SFT_SEGMENTS_DIR}/{len(exported) + 1:04d}.json",
                "window_index": len(exported),
                "boundary_source": (
                    "summary_request"
                    if summary
                    else "native_boundary"
                    if boundary
                    else "observed_context_change"
                    if exported
                    else "initial"
                ),
                "boundary_event_id": boundary["event_id"] if boundary else None,
                "first_call_id": call["call_id"],
                "last_call_id": call["call_id"],
            }
            record = {
                "messages": prompt + target,
                "tools": tools,
                "target_start_index": len(prompt),
            }
            # Only assistant messages at/after this index enter the loss;
            # intervening user/tool messages are context for later targets.
            exported.append((metadata, record))
        native_index = current_native
    if not exported:
        raise SFTConversionError("compaction has no complete main-agent Calls")
    for _, record in exported:
        validate_sft_record(record)
    return exported


def build_compaction_segments(
    sample_dir: Path,
    harness: str,
    manifest_relative: str,
    *,
    keep_thinking: bool = True,
    images_inline: bool = False,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Build verified main-agent windows without changing the sample directory.

    ``keep_thinking=False`` clears reasoning from prompts as well as targets,
    matching the linear SFT export. ``images_inline`` says where the collection
    put the images a tool returned, so a rebuilt window holds the arrangement
    the model was actually served.
    """
    rows = _native_rows(sample_dir / "trace.jsonl", harness)
    mode = compaction_mode(harness)
    boundaries = _compaction_boundaries(harness, rows)
    native_reset = has_main_compaction(sample_dir / "trace.jsonl", harness, rows)
    if (
        mode == "opencode"
        and _opencode_request_scope(
            rows, [row for row in rows if row.get("type") == "request"]
        )[1]
    ):
        raise SFTConversionError(
            "OpenCode request lacks a unique main-session identity"
        )
    if mode in {"claude", "codex"} and not boundaries:
        return []
    if (
        mode == "openclaw"
        and boundaries
        and not (sample_dir / "trace_ACTF.json").is_file()
    ):
        raise SFTConversionError("OpenClaw compaction requires ACTF main-step evidence")
    manifest_path = _safe_file(sample_dir, manifest_relative)
    capture_dir = manifest_path.parent
    calls = load_blobs(read_jsonl(capture_dir / "calls.jsonl"), sample_dir / BLOBS_DIR)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if boundaries:
        # Compaction after the last captured Call changed no model prompt.
        first = _time(boundaries[0]["timestamp"])
        if not any(_time(call.get("started_at")) > first for call in calls):
            return []
    elif not native_reset and not _raw_context_changed(calls):
        return []
    main = _root_calls(calls, manifest)
    summaries: set[str] | None = None
    native_summaries: frozenset[str] = frozenset()
    if mode == "stirrup":
        # Drop the calls whose replies the agent discarded: retried transport
        # faults, context overflows and summaries without text.
        kept = kept_call_ids(rows)
        main = [
            call
            for call in main
            if (call.get("native_identity") or {}).get("run_id") in kept
        ]
        # The journal names the reply that replaced the history. Stirrup takes
        # the summary from its text and ignores any tool call beside it, so the
        # text alone does not identify the Call.
        replaced = summary_call_ids(rows)
        native_summaries = frozenset(
            call["call_id"]
            for call in main
            if (call.get("native_identity") or {}).get("run_id") in replaced
        )
        summaries = set()
    if not boundaries:
        unsupported = next(
            (
                call
                for call in main
                if call.get("protocol") != "openai-chat-completions"
            ),
            None,
        )
        if not main or unsupported is not None:
            if native_reset:
                raise SFTConversionError(
                    "native context change lacks usable root Calls: "
                    + (
                        f"{_describe_call(unsupported)} is not openai-chat-completions"
                        if unsupported
                        else f"no usable root Call among {len(calls)} captured Calls"
                    )
                )
            return []
        if not native_reset and not _raw_context_changed(main):
            return []  # Preserve the existing linear SFT path, including images.
    main = _calls_matching_main_steps(
        sample_dir, main, boundaries, summaries, native_summaries
    )
    return _segment_records(
        main,
        boundaries,
        keep_thinking,
        summaries or set(),
        images_inline=images_inline,
    )


def export_compaction_segments(
    sample_dir: Path,
    harness: str,
    manifest_relative: str,
    *,
    keep_thinking: bool = True,
    images_inline: bool = False,
) -> list[dict[str, Any]]:
    """Write SFT windows only after every window has passed validation."""
    segments = build_compaction_segments(
        sample_dir,
        harness,
        manifest_relative,
        keep_thinking=keep_thinking,
        images_inline=images_inline,
    )
    for metadata, record in segments:
        atomic_write_json(sample_dir / metadata["path"], record)
    return [metadata for metadata, _ in segments]
