"""Serve OpenAI Responses requests from a Chat Completions endpoint.

Codex speaks the Responses API. The wire proxy translates each Responses
request on its way out and the response stream on its way back, with the
conversion vLLM applies in its own Responses endpoint, so the prompt is the
same either way. Codex streams every call, so only streams are translated.
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any

# The path of the API this module translates, below the API base.
ROUTE = "/responses"


def _functions(tools: list[dict[str, Any]]) -> dict[str, tuple[str | None, dict]]:
    """The function tools of a request, by the name the model sees.

    A function inside a namespace goes by ``<namespace>__<name>``. Each value
    is the namespace, if any, and the tool.
    """

    functions: dict[str, tuple[str | None, dict]] = {}
    for tool in tools:
        if tool["type"] == "function":
            functions[tool["name"]] = (None, tool)
        elif tool["type"] == "namespace":
            for inner in tool["tools"]:
                if inner["type"] == "function":
                    functions[f"{tool['name']}__{inner['name']}"] = (
                        tool["name"],
                        inner,
                    )
    return functions


def _content(parts: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Chat content for the text and image parts of one message."""

    if isinstance(parts, str):
        return parts
    return [
        {"type": "image_url", "image_url": {"url": part["image_url"]}}
        if part["type"] == "input_image"
        else {"type": "text", "text": part["text"]}
        for part in parts
        if part["type"] in ("input_text", "output_text", "input_image")
    ]


def _assistant(messages: list[dict[str, Any]], key: str, value: Any) -> None:
    """Add one output item of the model to the assistant message it belongs to.

    The reasoning, text and tool calls of one model response are separate
    input items, and one assistant message in Chat Completions.
    """

    last = messages[-1] if messages and messages[-1]["role"] == "assistant" else None
    if key == "tool_calls" and last is not None:
        last.setdefault("tool_calls", []).append(value)
    elif key == "tool_calls":
        messages.append({"role": "assistant", "tool_calls": [value]})
    elif last is None or key in last:
        messages.append({"role": "assistant", key: value})
    else:
        last[key] = value


def chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """The Chat Completions request for one Responses request."""

    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    for item in body["input"]:
        kind = item.get("type", "message")
        if kind == "message" and item["role"] == "assistant":
            _assistant(messages, "content", item["content"][0]["text"])
        elif kind == "message":
            messages.append(
                {"role": item["role"], "content": _content(item["content"])}
            )
        elif kind == "reasoning":
            text = (item.get("content") or item["summary"])[0]["text"]
            _assistant(messages, "reasoning", text)
        elif kind == "function_call":
            name = item["name"]
            if item.get("namespace"):
                name = f"{item['namespace']}__{name}"
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": name, "arguments": item["arguments"]},
            }
            _assistant(messages, "tool_calls", call)
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": _content(item["output"]),
                }
            )
    request: dict[str, Any] = {
        "model": body["model"],
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        request["reasoning_effort"] = effort
    functions = _functions(body.get("tools", []))
    if functions:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    **{key: value for key, value in tool.items() if key != "type"},
                    "name": name,
                },
            }
            for name, (_, tool) in functions.items()
        ]
        request["tool_choice"] = body["tool_choice"]
        request["parallel_tool_calls"] = body["parallel_tool_calls"]
    return request


def error(status: int | None, payload: bytes) -> dict[str, Any]:
    """The Responses error body, which has the shape of the Chat Completions one."""

    return json.loads(payload)


class Stream:
    """Turn the chunks of a Chat Completions stream into Responses stream events."""

    def __init__(self, request: dict[str, Any]) -> None:
        """``request`` is the Responses request the stream answers."""

        self.functions = _functions(request.get("tools", []))
        self.response: dict[str, Any] | None = None
        # The open output item, and what it is: "reasoning", "message" or a
        # tool call's index.
        self.item: dict[str, Any] | None = None
        self.key: str | int | None = None
        self.usage: dict[str, Any] = {}

    def _close(self) -> list[dict[str, Any]]:
        if self.item is None:
            return []
        output = self.response["output"]
        output.append(self.item)
        self.item = None
        return [
            {
                "type": "response.output_item.done",
                "output_index": len(output) - 1,
                "item": output[-1],
            }
        ]

    def _open(self, key: str | int, item: dict[str, Any]) -> list[dict[str, Any]]:
        if self.item is not None and self.key == key:
            return []
        events = self._close()
        self.key, self.item = key, item
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": len(self.response["output"]),
                "item": copy.deepcopy(item),
            }
        )
        return events

    def _delta(self, kind: str, text: str) -> dict[str, Any]:
        return {
            "type": kind,
            "item_id": self.item["id"],
            "output_index": len(self.response["output"]),
            "content_index": 0,
            "delta": text,
        }

    def feed(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        if self.response is None:
            self.response = {
                "id": f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "model": chunk.get("model"),
                "status": "in_progress",
                "output": [],
            }
            events.append(
                {"type": "response.created", "response": copy.deepcopy(self.response)}
            )
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                events += self._open(
                    "reasoning",
                    {
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex}",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": ""}],
                    },
                )
                self.item["content"][0]["text"] += reasoning
                events.append(self._delta("response.reasoning_text.delta", reasoning))
            if delta.get("content"):
                events += self._open(
                    "message",
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "", "annotations": []}
                        ],
                    },
                )
                self.item["content"][0]["text"] += delta["content"]
                events.append(
                    self._delta("response.output_text.delta", delta["content"])
                )
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                if self.item is None or self.key != call["index"]:
                    # A name outside the request's tools goes back as it came.
                    namespace, tool = self.functions.get(function["name"]) or (
                        None,
                        function,
                    )
                    events += self._open(
                        call["index"],
                        {
                            "type": "function_call",
                            "id": f"fc_{uuid.uuid4().hex}",
                            "call_id": call["id"],
                            "name": tool["name"],
                            **({"namespace": namespace} if namespace else {}),
                            "arguments": "",
                            "status": "completed",
                        },
                    )
                if function.get("arguments"):
                    self.item["arguments"] += function["arguments"]
                    events.append(
                        self._delta(
                            "response.function_call_arguments.delta",
                            function["arguments"],
                        )
                    )
        return events

    def finish(self) -> list[dict[str, Any]]:
        events = self._close()
        self.response.update(
            status="completed",
            usage={
                "input_tokens": self.usage.get("prompt_tokens") or 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": self.usage.get("completion_tokens") or 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": self.usage.get("total_tokens") or 0,
            },
        )
        return [*events, {"type": "response.completed", "response": self.response}]
