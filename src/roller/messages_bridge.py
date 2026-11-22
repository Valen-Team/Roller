"""Serve Anthropic Messages requests from a Chat Completions endpoint.

Claude Code speaks the Messages API. The wire proxy translates each Messages
request on its way out and the response on its way back, with the conversion
vLLM applies in its own Messages endpoint, so the prompt is the same either way.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

# The path of the API this module translates, below the API base.
ROUTE = "/messages"

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}

# Request fields both APIs spell the same way.
SHARED_PARAMETERS = ("model", "max_tokens", "temperature", "top_p", "top_k", "stream")


def _blocks(content: Any) -> list[dict[str, Any]]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def _system(content: Any) -> str:
    """System text without the header Claude Code adds for its own billing.

    The header carries a per-request hash, which would change the prompt
    prefix on every call.
    """

    return "".join(
        block["text"]
        for block in _blocks(content)
        if block["type"] == "text"
        and not block["text"].startswith("x-anthropic-billing-header")
    )


def _image(block: dict[str, Any]) -> dict[str, Any]:
    source = block["source"]
    url = (
        source["url"]
        if source["type"] == "url"
        else f"data:{source['media_type']};base64,{source['data']}"
    )
    return {"type": "image_url", "image_url": {"url": url}}


def _content(blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]] | None:
    """Chat content for the text and image blocks of one message."""

    parts = [
        {"type": "text", "text": block["text"]}
        if block["type"] == "text"
        else _image(block)
        for block in blocks
        if (block["type"] == "text" and block["text"]) or block["type"] == "image"
    ]
    if not parts:
        return None
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def _assistant(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    content = _content(blocks)
    if content is not None:
        message["content"] = content
    thinking = "".join(
        block["thinking"] for block in blocks if block["type"] == "thinking"
    )
    if thinking:
        message["reasoning"] = thinking
    calls = [
        {
            "id": block["id"],
            "type": "function",
            "function": {
                "name": block["name"],
                "arguments": json.dumps(block["input"]),
            },
        }
        for block in blocks
        if block["type"] == "tool_use"
    ]
    if calls:
        message["tool_calls"] = calls
    return message


def _user(
    blocks: list[dict[str, Any]], *, tool_images_inline: bool = False
) -> list[dict[str, Any]]:
    """A user message, its tool results first as ``tool`` messages.

    Chat Completions wants every tool result directly after the assistant
    message that called it. A tool message carries text only, so by default the
    images a tool returned follow it in a user message of their own, as vLLM's
    own Messages endpoint arranges them.

    ``tool_images_inline`` keeps each image where the tool returned it, as a
    part of the tool message itself. GLM-5.3's chat template renders an image
    part inside a tool message in place, so a collected trace then tokenizes
    the same way in training as the request that produced it; a served endpoint
    that ignores images outside a user message does not see them at all, which
    is why the split remains the default.
    """

    messages: list[dict[str, Any]] = []
    for block in blocks:
        if block["type"] != "tool_result":
            continue
        result = _blocks(block.get("content") or "")
        if tool_images_inline:
            inline = _content(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": "" if inline is None else inline,
                }
            )
            continue
        text = "\n".join(part["text"] for part in result if part["type"] == "text")
        messages.append(
            {"role": "tool", "tool_call_id": block["tool_use_id"], "content": text}
        )
        images = [_image(part) for part in result if part["type"] == "image"]
        if images:
            messages.append({"role": "user", "content": images})
    content = _content(blocks)
    if content is not None:
        messages.append({"role": "user", "content": content})
    return messages


def chat_request(
    body: dict[str, Any], *, tool_images_inline: bool = False
) -> dict[str, Any]:
    """The Chat Completions request for one Messages request."""

    messages = []
    system = _system(body.get("system") or "")
    if system:
        messages.append({"role": "system", "content": system})
    for message in body["messages"]:
        blocks = _blocks(message["content"])
        if message["role"] == "assistant":
            messages.append(_assistant(blocks))
        elif message["role"] == "user":
            messages += _user(blocks, tool_images_inline=tool_images_inline)
        elif _system(blocks):
            messages.append({"role": "system", "content": _system(blocks)})
    request = {key: body[key] for key in SHARED_PARAMETERS if key in body}
    request["messages"] = messages
    if body.get("stream"):
        request["stream_options"] = {"include_usage": True}
    if body.get("stop_sequences"):
        request["stop"] = body["stop_sequences"]
    effort = body.get("output_config", {}).get("effort")
    if effort:
        request["reasoning_effort"] = effort
    # A tool with a ``type`` is one the Anthropic API runs itself.
    tools = [tool for tool in body.get("tools", []) if "input_schema" in tool]
    if tools:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]
    choice = body.get("tool_choice")
    if choice:
        request["tool_choice"] = {
            "auto": "auto",
            "none": "none",
            "any": "required",
            "tool": {"type": "function", "function": {"name": choice.get("name")}},
        }[choice["type"]]
    return request


def _usage(usage: dict[str, Any] | None, prompt_tokens: int = 0) -> dict[str, int]:
    usage = usage or {}
    return {
        "input_tokens": usage.get("prompt_tokens") or prompt_tokens,
        "output_tokens": usage.get("completion_tokens") or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def response(chat: dict[str, Any]) -> dict[str, Any]:
    """The Messages response for one complete Chat Completions response."""

    choice = chat["choices"][0]
    message = choice["message"]
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        content.append(
            {"type": "thinking", "thinking": reasoning, "signature": uuid.uuid4().hex}
        )
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        content.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": json.loads(call["function"]["arguments"] or "{}"),
            }
        )
    return {
        "id": chat["id"],
        "type": "message",
        "role": "assistant",
        "model": chat["model"],
        "content": content,
        "stop_reason": STOP_REASONS.get(choice["finish_reason"], "end_turn"),
        "stop_sequence": None,
        "usage": _usage(chat.get("usage")),
    }


def error(status: int | None, payload: bytes) -> dict[str, Any]:
    """The Messages error body for a failed Chat Completions response."""

    try:
        error = json.loads(payload).get("error", {})
        message = error.get("message") if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        message = None
    kind = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        529: "overloaded_error",
    }.get(status, "api_error")
    return {
        "type": "error",
        "error": {
            "type": kind,
            "message": message or payload.decode("utf-8", errors="replace")[:2000],
        },
    }


class Stream:
    """Turn the chunks of a Chat Completions stream into Messages stream events."""

    def __init__(self, request: dict[str, Any]) -> None:
        """``request`` is the Messages request, which the events do not depend on."""

        self.started = False
        self.index = -1
        # The open content block: "thinking", "text" or a tool call's index.
        self.block: str | int | None = None
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.prompt_tokens = 0

    def _close(self) -> list[dict[str, Any]]:
        if self.block is None:
            return []
        events = []
        if self.block == "thinking":
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.index,
                    "delta": {"type": "signature_delta", "signature": uuid.uuid4().hex},
                }
            )
        events.append({"type": "content_block_stop", "index": self.index})
        self.block = None
        return events

    def _open(self, block: str | int, content: dict[str, Any]) -> list[dict[str, Any]]:
        if self.block == block:
            return []
        events = self._close()
        self.index += 1
        self.block = block
        events.append(
            {
                "type": "content_block_start",
                "index": self.index,
                "content_block": content,
            }
        )
        return events

    def _delta(self, delta: dict[str, Any]) -> dict[str, Any]:
        return {"type": "content_block_delta", "index": self.index, "delta": delta}

    def feed(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        if not self.started:
            self.started = True
            self.prompt_tokens = len(chunk.get("prompt_token_ids") or [])
            events.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": chunk.get("id"),
                        "type": "message",
                        "role": "assistant",
                        "model": chunk.get("model"),
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": _usage(self.usage, self.prompt_tokens),
                    },
                }
            )
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                events += self._open(
                    "thinking", {"type": "thinking", "thinking": "", "signature": ""}
                )
                events.append(
                    self._delta({"type": "thinking_delta", "thinking": reasoning})
                )
            if delta.get("content"):
                events += self._open("text", {"type": "text", "text": ""})
                events.append(
                    self._delta({"type": "text_delta", "text": delta["content"]})
                )
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                events += self._open(
                    call["index"],
                    {
                        "type": "tool_use",
                        "id": call.get("id"),
                        "name": function.get("name"),
                        "input": {},
                    },
                )
                if function.get("arguments"):
                    events.append(
                        self._delta(
                            {
                                "type": "input_json_delta",
                                "partial_json": function["arguments"],
                            }
                        )
                    )
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return events

    def finish(self) -> list[dict[str, Any]]:
        return [
            *self._close(),
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": STOP_REASONS.get(self.finish_reason, "end_turn"),
                    "stop_sequence": None,
                },
                "usage": _usage(self.usage, self.prompt_tokens),
            },
            {"type": "message_stop"},
        ]
