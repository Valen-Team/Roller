"""ACTF projection and capture evidence of Roller's Stirrup journal.

The journal is written by ``roller.stirrup_agent`` in the collecting process.
A ``request`` row holds the messages a model call added to the shared prefix,
a ``response`` row its outcome, a ``tool`` row one tool result as sent to the
model, and a ``summary`` row each history summary. Reading it needs no
Stirrup installation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from roller.actf import append_user_content
from roller.capture import native_directory, read_jsonl
from roller.capture_protocol import NativeCapture
from roller.storage import atomic_write_json
from roller.usage import ZERO_USAGE

STIRRUP_COMPACTION_SFT_REASON = (
    "Stirrup history summaries require captured per-call contexts"
)
# Outcomes of the calls whose replies the agent kept: a step of the
# trajectory, or the text that replaced the history.
KEPT_OUTCOMES = frozenset({"step", "summary"})


def _blocks(content: Any) -> list[dict[str, Any]]:
    """Chat Completions content as ACTF blocks; images keep their data URI."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for part in content or []:
        if part.get("type") == "image_url":
            blocks.append({"type": "image", "data": part["image_url"]["url"]})
        else:
            blocks.append({"type": "text", "text": part["text"]})
    return blocks


def _user_content(blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if all(block["type"] == "text" for block in blocks):
        return "\n".join(block["text"] for block in blocks)
    return blocks


def _text(content: Any) -> str:
    return "".join(block.get("text", "") for block in _blocks(content))


def stirrup_events_to_actf(
    rows: list[dict[str, Any]], *, keep_thinking: bool = True
) -> dict[str, Any]:
    system_prompt, tools = "", None
    steps: list[dict[str, Any]] = []
    by_call: dict[str, dict[str, Any]] = {}
    requests: dict[str, dict[str, Any]] = {}
    pending_user: str | list[Any] = ""
    compacted = False
    for row in rows:
        kind = row.get("type")
        if kind == "request":
            requests[row["id"]] = row
            if row["purpose"] != "main":
                continue
            messages = row["messages"]
            if row["base"] == 0 and messages and messages[0]["role"] == "system":
                system_prompt = _text(messages[0]["content"])
            if tools is None and "tools" in row:
                tools = deepcopy(row["tools"])
            for message in messages:
                if message["role"] == "user":
                    pending_user = append_user_content(
                        pending_user, _user_content(_blocks(message["content"]))
                    )
        elif kind == "response" and row.get("outcome") == "step":
            message = row["message"]
            calls = [
                {
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
                for call in message.get("tool_calls", [])
            ]
            usage = row.get("usage") or {}
            started = requests[row["id"]]["timestamp"]
            step = {
                "step_id": len(steps) + 1,
                "assistant_content": {
                    "content": _text(message.get("content")),
                    "reasoning_content": message.get("reasoning_content", "")
                    if keep_thinking
                    else "",
                    "tool_calls": calls,
                },
                "metric": {
                    "prompt_tokens_len": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens_len": int(usage.get("completion_tokens") or 0),
                    "llm_infer_ms": None,
                    "env_action_ms": None,
                    "stop_reason": row.get("finish_reason"),
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                },
                "user_content": pending_user,
                "tools": deepcopy(calls),
                "observation": [],
                "started_at": started,
                "finished_at": row["timestamp"],
                "extra": {"call_id": row["id"]},
            }
            pending_user = ""
            steps.append(step)
            for call in calls:
                by_call[call["id"]] = step
        elif kind == "tool" and row["tool_call_id"] in by_call:
            step = by_call[row["tool_call_id"]]
            step["observation"].append(
                {
                    "role": "toolResult",
                    "toolCallId": row["tool_call_id"],
                    "toolName": row["name"],
                    "content": _blocks(row["content"]),
                    "isError": not row["success"],
                    "timestamp": row["timestamp"],
                }
            )
            step["finished_at"] = row["timestamp"]
        elif kind == "summary":
            compacted = True
    trajectory: dict[str, Any] = {
        "schema_version": "ACTF_v1.0",
        "system_prompt": system_prompt,
        "tools": tools,
        "steps": steps,
        "started_at": rows[0]["timestamp"] if rows else "",
        "finished_at": rows[-1]["timestamp"] if rows else "",
    }
    if compacted:
        trajectory["sft_unavailable_reason"] = STIRRUP_COMPACTION_SFT_REASON
    return trajectory


def convert_stirrup_trace(
    source: Path, destination: Path, *, keep_thinking: bool = True
) -> dict[str, Any]:
    trajectory = stirrup_events_to_actf(read_jsonl(source), keep_thinking=keep_thinking)
    atomic_write_json(destination, trajectory)
    return trajectory


def extract_stirrup_usage(source: Path) -> dict[str, int | float]:
    totals = dict(ZERO_USAGE)
    for row in read_jsonl(source):
        if row.get("type") == "request":
            totals["request_count"] += 1
        usage = row.get("usage") if row.get("type") == "response" else None
        if usage:
            totals["input_tokens"] += usage.get("prompt_tokens") or 0
            totals["output_tokens"] += usage.get("completion_tokens") or 0
            totals["total_tokens"] += usage.get("total_tokens") or 0
    return totals


def kept_call_ids(rows: list[dict[str, Any]]) -> set[str]:
    """The calls whose replies became steps or summaries."""
    return {
        row["id"]
        for row in rows
        if row.get("type") == "response" and row.get("outcome") in KEPT_OUTCOMES
    }


def summary_call_ids(rows: list[dict[str, Any]]) -> set[str]:
    """The calls whose replies replaced the history."""
    return {
        row["id"]
        for row in rows
        if row.get("type") == "response" and row.get("outcome") == "summary"
    }


def read_stirrup_native(directory: Path) -> NativeCapture:
    native = NativeCapture(
        "stirrup",
        "openai-chat-completions",
        capabilities={
            "request_response": "full",
            "identity": "native_session",
            "native_output_coverage": "assistant_count",
            "subagents": "not_available_in_bundled_profile",
            "background_drain": False,
        },
    )
    for path in sorted(native_directory(directory).glob("*.jsonl")):
        rows = read_jsonl(path)
        session = next((row for row in rows if row.get("type") == "session"), {})
        sid = session.get("session_id")
        if not isinstance(sid, str) or not sid:
            native.errors.append(f"{path.name}: missing native session identity")
            continue
        pid = "session:" + sid
        native.participants[pid] = {
            "participant_id": pid,
            "native_session_id": sid,
            "role": "root",
            "identity_status": "observed",
            "native_files": [path.relative_to(directory).as_posix()],
            "call_ids": [],
        }
        native.session_owners.setdefault(sid, set()).add(pid)
        # Calls that finished normally, as the capture counts complete ones.
        native.expected_call_counts[pid] = sum(
            row.get("type") == "response"
            and row.get("finish_reason") in {"stop", "tool_calls"}
            for row in rows
        )
    return native
