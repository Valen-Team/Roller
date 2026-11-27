"""Translate native Harness journals to the message format expected by graders."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from roller.harness_registry import convert_native_trace


def session_trajectory(session: Any) -> dict[str, Any]:
    harness, name = session.harness, session.harness.name
    if name in {"openclaw", "codex", "claude-code"}:
        kwargs = {"state_dir": session.state_dir} if name == "openclaw" else {}
        source = harness._collect_trace(
            session.container,
            session.output_dir,
            session.attempt,
            **kwargs,
        )
        if source is None:
            raise RuntimeError(f"{name}: no native transcript available for grading")
    else:
        source = session.output_dir / "rollout" / "bench-native.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        copied = harness.runtime.copy_out(
            session.container,
            session.transcript_container_path,
            source,
            timeout=60,
        )
        if copied.returncode:
            raise RuntimeError(f"{name}: transcript copy failed")
    converter_kwargs = {
        "keep_thinking": getattr(
            getattr(harness, "config", None), "keep_thinking", True
        )
    }
    if name in {"openclaw", "codex", "claude-code"}:
        converter_kwargs["user_messages"] = session.messages
    return convert_native_trace(
        name,
        Path(source),
        session.output_dir / "rollout" / "bench_ACTF.json",
        **converter_kwargs,
    )


def grader_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve text, reasoning, tool arguments/results and IDs without fabricating actions."""
    events = []
    for step in trajectory["steps"]:
        if step.get("user_content"):
            user_content = step["user_content"]
            events.append(
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": (
                            deepcopy(user_content)
                            if isinstance(user_content, list)
                            else [{"type": "text", "text": user_content}]
                        ),
                    },
                }
            )
        assistant, blocks = step["assistant_content"], []
        if assistant.get("reasoning_content"):
            blocks.append(
                {"type": "thinking", "thinking": assistant["reasoning_content"]}
            )
        if assistant.get("content"):
            blocks.append({"type": "text", "text": assistant["content"]})
        for call in assistant.get("tool_calls", []):
            function = call.get("function") or call
            arguments = deepcopy(function.get("arguments", {}))
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": function["name"],
                    "input": arguments,
                    "arguments": arguments,
                }
            )
        events.append(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": blocks,
                    "stopReason": step.get("metric", {}).get("stop_reason", ""),
                },
            }
        )
        for observation in step.get("observation", []):
            events.append({"type": "message", "message": deepcopy(observation)})
    return events
