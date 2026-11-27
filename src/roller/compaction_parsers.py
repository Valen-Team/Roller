"""Protocol-specific message parsers converting captured Calls into SFT messages."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any

from roller.sft import (
    SFTConversionError,
    _image_data_uri,
    _normalize_tool_call,
)


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SFTConversionError("compaction or call lacks a timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SFTConversionError(f"invalid compaction timestamp: {value}") from exc
    if timestamp.tzinfo is None:
        raise SFTConversionError(f"compaction timestamp lacks timezone: {value}")
    return timestamp


def _text_blocks(content: Any, *, kinds: set[str], separator: str = "") -> str:
    """The text blocks of one message, joined the way the endpoint joins them.

    A chat template emits the text parts of a message back to back, so a
    separator here would put a character in the window that the model was never
    served and the prompt would no longer tokenize to what it answered.
    """

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise SFTConversionError("model input contains unsupported content")
    texts = []
    for block in content:
        if (
            not isinstance(block, Mapping)
            or block.get("type") not in kinds
            or not isinstance(block.get("text"), str)
        ):
            raise SFTConversionError("model input contains non-text content")
        texts.append(block["text"])
    return separator.join(texts)


def _image_part(block: Mapping[str, Any], field: str) -> dict[str, Any]:
    """One Anthropic image block as the multimodal part an SFT record carries."""

    return {
        "type": "image_url",
        "image_url": {"url": _image_data_uri(dict(block), field)},
    }


def _joined(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Text-only parts as the plain string they were; anything else as parts.

    The text is concatenated, not separated: see :func:`_text_blocks`.
    """

    if all(part["type"] == "text" for part in parts):
        return "".join(part["text"] for part in parts)
    return list(parts)


def _anthropic_tool_result(
    identifier: str, content: Any, *, images_inline: bool
) -> list[dict[str, Any]]:
    """One Anthropic tool result as the messages the served prompt held.

    An image a tool returned is a part of the tool message itself, which is
    where a chat template that renders tool content in place expects it.
    ``images_inline=False`` puts those images in a user message after it
    instead, as vLLM's own Messages endpoint arranges such a turn: a window
    rebuilt from captured Calls has to hold whichever arrangement the model was
    served, or a trainer tokenizes a prompt that never existed.
    """

    blocks = (
        [{"type": "text", "text": content}] if isinstance(content, str) else content
    )
    if not isinstance(blocks, list):
        raise SFTConversionError("Anthropic tool result content is invalid")
    texts: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            raise SFTConversionError("Anthropic tool result block is invalid")
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            images.append(_image_part(block, "Anthropic tool result image"))
        else:
            raise SFTConversionError(f"unsupported Anthropic tool result block: {kind}")
    if images_inline:
        return [
            {
                "role": "tool",
                "tool_call_id": identifier,
                "content": _joined(texts + images),
            }
        ]
    messages: list[dict[str, Any]] = [
        {"role": "tool", "tool_call_id": identifier, "content": _joined(texts)}
    ]
    if images:
        messages.append({"role": "user", "content": images})
    return messages


def _tool_call(
    identifier: Any, name: Any, arguments: Any, kind: str = "function"
) -> dict[str, Any]:
    if (
        not isinstance(identifier, str)
        or not identifier
        or not isinstance(name, str)
        or not name
    ):
        raise SFTConversionError("captured tool call lacks an ID or name")
    if kind == "custom":
        if not isinstance(arguments, str):
            raise SFTConversionError("captured custom tool input is not text")
        return {
            "id": identifier,
            "type": "custom",
            "custom": {"name": name, "input": arguments},
        }
    return _normalize_tool_call(
        {"id": identifier, "name": name, "arguments": arguments}, "captured tool call"
    )


def _anthropic_message(
    message: Mapping[str, Any], *, images_inline: bool = False
) -> list[dict[str, Any]]:
    role = message.get("role")
    content = message.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise SFTConversionError("Anthropic message content is invalid")
    if role == "system":
        return [{"role": "system", "content": _text_blocks(content, kinds={"text"})}]
    if role == "assistant":
        visible, reasoning, calls = [], [], []
        for block in content:
            if not isinstance(block, Mapping):
                raise SFTConversionError("Anthropic assistant block is invalid")
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                visible.append(block["text"])
            elif kind == "thinking" and isinstance(block.get("thinking"), str):
                reasoning.append(block["thinking"])
            elif kind == "tool_use":
                calls.append(
                    _tool_call(
                        block.get("id"), block.get("name"), block.get("input", {})
                    )
                )
            elif kind == "redacted_thinking":
                raise SFTConversionError(
                    "redacted Anthropic thinking cannot be represented in SFT"
                )
            else:
                raise SFTConversionError(
                    f"unsupported Anthropic assistant block: {kind}"
                )
        return [
            {
                "role": "assistant",
                "reasoning_content": "\n\n".join(reasoning),
                "content": "\n\n".join(visible),
                "tool_calls": calls,
            }
        ]
    if role != "user":
        raise SFTConversionError(f"unsupported Anthropic role: {role}")
    result: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []

    def flush() -> None:
        if parts:
            result.append({"role": "user", "content": _joined(parts)})
            parts.clear()

    for block in content:
        if not isinstance(block, Mapping):
            raise SFTConversionError("Anthropic user block is invalid")
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            parts.append(_image_part(block, "Anthropic user image"))
        elif kind == "tool_result":
            flush()
            identifier = block.get("tool_use_id")
            if not isinstance(identifier, str) or not identifier:
                raise SFTConversionError("Anthropic tool result lacks an ID")
            result.extend(
                _anthropic_tool_result(
                    identifier,
                    block.get("content", ""),
                    images_inline=images_inline,
                )
            )
        else:
            raise SFTConversionError(f"unsupported Anthropic user block: {kind}")
    flush()
    return result


def _responses_item(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    kind = item.get("type")
    if kind == "message":
        role = item.get("role")
        if role not in {"system", "developer", "user", "assistant"}:
            raise SFTConversionError(f"unsupported Responses role: {role}")
        value = _text_blocks(
            item.get("content", []), kinds={"input_text", "output_text"}, separator=""
        )
        if role == "assistant":
            return [
                {
                    "role": role,
                    "reasoning_content": "",
                    "content": value,
                    "tool_calls": [],
                }
            ]
        return [{"role": role, "content": value}]
    if kind in {"function_call", "custom_tool_call"}:
        tool_kind = "custom" if kind == "custom_tool_call" else "function"
        call = _tool_call(
            item.get("call_id"),
            item.get("name"),
            item.get("input") if tool_kind == "custom" else item.get("arguments", {}),
            tool_kind,
        )
        return [
            {
                "role": "assistant",
                "reasoning_content": "",
                "content": "",
                "tool_calls": [call],
            }
        ]
    if kind in {"function_call_output", "custom_tool_call_output"}:
        identifier = item.get("call_id")
        if not isinstance(identifier, str) or not identifier:
            raise SFTConversionError("Responses tool output lacks a call ID")
        output = item.get("output")
        if not isinstance(output, str):
            raise SFTConversionError("Responses tool output is not text")
        return [{"role": "tool", "tool_call_id": identifier, "content": output}]
    if kind == "reasoning":
        # An empty placeholder contributes no prompt content. Any populated
        # reasoning item still needs a representation we do not have.
        if (
            item.get("summary") in (None, [])
            and item.get("content") in (None, [])
            and item.get("encrypted_content") in (None, "")
        ):
            return []
        raise SFTConversionError(
            "Responses reasoning item cannot be represented in SFT"
        )
    raise SFTConversionError(f"unsupported Responses input item: {kind}")


def _chat_content(content: Any) -> str | list[dict[str, Any]]:
    """OpenAI multimodal content list when images are present, plain text otherwise."""
    if not isinstance(content, list):
        return _text_blocks(content, kinds={"text"})
    parts: list[dict[str, Any]] = []
    has_image = False
    for block in content:
        image = block.get("image_url") if isinstance(block, Mapping) else None
        if (
            isinstance(block, Mapping)
            and block.get("type") == "image_url"
            and isinstance(image, Mapping)
        ):
            uri = _image_data_uri({"data": image.get("url")}, "image_url")
            parts.append({"type": "image_url", "image_url": {"url": uri}})
            has_image = True
        else:
            parts.append(
                {"type": "text", "text": _text_blocks([block], kinds={"text"})}
            )
    if not has_image:
        return "\n".join(part["text"] for part in parts)
    return parts


def _chat_message(message: Mapping[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise SFTConversionError(f"unsupported Chat Completions role: {role}")
    content = message.get("content")
    if content is None and role == "assistant":
        content = ""
    value = (
        _text_blocks(content, kinds={"text"})
        if role in {"system", "developer", "assistant"}
        else _chat_content(content)
    )
    if role == "tool":
        identifier = message.get("tool_call_id")
        if not isinstance(identifier, str) or not identifier:
            raise SFTConversionError("Chat Completions tool result lacks a call ID")
        return {"role": role, "tool_call_id": identifier, "content": value}
    if role == "assistant":
        reasoning = message.get("reasoning_content") or ""
        if not isinstance(reasoning, str):
            raise SFTConversionError("Chat Completions reasoning is not text")
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise SFTConversionError("Chat Completions tool calls are invalid")
        calls = []
        for call in raw_calls:
            if not isinstance(call, Mapping) or not isinstance(
                call.get("function"), Mapping
            ):
                raise SFTConversionError("Chat Completions tool call is invalid")
            function = call["function"]
            calls.append(
                _tool_call(
                    call.get("id"), function.get("name"), function.get("arguments", {})
                )
            )
        return {
            "role": role,
            "reasoning_content": reasoning,
            "content": value,
            "tool_calls": calls,
        }
    return {"role": role, "content": value}


def _merge_response_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Codex stores one response as separate text and tool items; ACTF keeps one step."""
    merged: list[dict[str, Any]] = []
    for message in messages:
        previous = merged[-1] if merged else None
        if previous is not None and previous["role"] == message["role"] == "assistant":
            previous["content"] = "\n\n".join(
                text for text in (previous["content"], message["content"]) if text
            )
            previous["tool_calls"].extend(message["tool_calls"])
        else:
            merged.append(message)
    return merged


def _prompt_from_call(
    call: Mapping[str, Any], *, images_inline: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = call.get("context")
    if not isinstance(context, Mapping):
        raise SFTConversionError("captured Call lacks context")
    protocol = call.get("protocol")
    messages: list[dict[str, Any]] = []
    tools_value = context.get("tools", [])
    if not isinstance(tools_value, list):
        raise SFTConversionError("captured tools are invalid")
    if protocol == "anthropic-messages":
        system = context.get("system", "")
        if system:
            messages.append(
                {"role": "system", "content": _text_blocks(system, kinds={"text"})}
            )
        items = context.get("messages", [])
        if not isinstance(items, list):
            raise SFTConversionError("Anthropic request messages are invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise SFTConversionError("Anthropic request has an invalid message")
            messages.extend(_anthropic_message(item, images_inline=images_inline))
        tools = []
        for tool in tools_value:
            if not isinstance(tool, Mapping) or not isinstance(tool.get("name"), str):
                raise SFTConversionError("Anthropic tool schema is invalid")
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": deepcopy(tool.get("input_schema", {})),
                    },
                }
            )
    elif protocol == "openai-responses":
        instructions = context.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.append({"role": "system", "content": instructions})
        items = context.get("input", [])
        if not isinstance(items, list):
            raise SFTConversionError("Responses request input is invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise SFTConversionError("Responses request has an invalid item")
            messages.extend(_responses_item(item))
        messages = _merge_response_items(messages)
        tools = deepcopy(tools_value)
    elif protocol == "openai-chat-completions":
        items = context.get("messages", [])
        if not isinstance(items, list):
            raise SFTConversionError("Chat Completions request messages are invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise SFTConversionError(
                    "Chat Completions request has an invalid message"
                )
            messages.append(_chat_message(item))
        tools = deepcopy(tools_value)
    else:
        raise SFTConversionError(f"unsupported compaction Call protocol: {protocol}")
    if not isinstance(tools, list):
        raise SFTConversionError("captured tools are invalid")
    return messages, tools


def _target_messages_from_call(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = call.get("output")
    if not isinstance(output, Mapping):
        raise SFTConversionError("captured Call lacks output")
    if call.get("protocol") == "anthropic-messages":
        return _anthropic_message(
            {"role": "assistant", "content": output.get("content", [])}
        )
    if call.get("protocol") == "openai-responses":
        result: list[dict[str, Any]] = []
        items = output.get("output", [])
        if not isinstance(items, list):
            raise SFTConversionError("Responses output is invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise SFTConversionError("Responses output contains an invalid item")
            if item.get("type") == "reasoning":
                # The Responses API may return encrypted reasoning. It is not
                # a supervised text target. A later request containing it is
                # rejected because the actual prompt cannot be represented.
                continue
            result.extend(_responses_item(item))
        result = _merge_response_items(result)
        if len(result) != 1 or result[0]["role"] != "assistant":
            raise SFTConversionError(
                "Responses output has no representable assistant target"
            )
        return result
    if call.get("protocol") == "openai-chat-completions":
        choices = output.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise SFTConversionError("Chat Completions output lacks one choice")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise SFTConversionError(
                "Chat Completions output lacks an assistant target"
            )
        target = _chat_message(message)
        # vLLM returns the thinking as ``reasoning``; a main step takes it
        # from ACTF instead, a kept summary from here.
        if not target["reasoning_content"] and isinstance(
            message.get("reasoning"), str
        ):
            target["reasoning_content"] = message["reasoning"]
        return [target]
    raise SFTConversionError("unsupported compaction response protocol")
