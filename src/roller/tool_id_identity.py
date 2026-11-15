"""Keep Anthropic tool-use IDs unique within one rollout.

Some Anthropic-compatible endpoints issue the same ``tool_use`` ID in more
than one response. ``ToolIdRegistry`` repairs the IDs before Claude Code sees
them; ``scope_claude_tool_ids`` separates them in a trace that was collected
without the repair.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import uuid


class ToolIdRegistry:
    """Replace tool-use IDs the endpoint has already issued in this rollout."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def repair(self, event: Any) -> list[dict[str, str]]:
        """Rewrite repeated IDs of one Messages response or stream event in place.

        Returns one ``native_id``/``delivered_id`` pair per replaced ID.
        """

        if not isinstance(event, dict):
            return []
        if event.get("type") == "content_block_start":
            blocks = [event.get("content_block")]
        elif event.get("type") == "message":
            blocks = event.get("content")
        else:
            return []
        mappings: list[dict[str, str]] = []
        for block in blocks if isinstance(blocks, list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            native = block.get("id")
            if not isinstance(native, str) or not native:
                continue
            if native in self._seen:
                block["id"] = f"toolu_roller_{uuid.uuid4().hex}"
                mappings.append({"native_id": native, "delivered_id": block["id"]})
            self._seen.add(block["id"])
        return mappings


def scope_claude_tool_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give each reuse of a tool-use ID in root session rows its own ID.

    The first call keeps the native ID. A later assistant message that reuses
    it after the earlier result arrived gets ``<id>__occurrence_<n>``, and its
    result follows. Blocks of one assistant message share ``message.id`` and
    stay one call. Renamed blocks keep the native ID in
    ``roller_original_tool_id``; rows that need no change are returned as is.
    Reuse while the earlier call still waits for its result cannot be paired
    and raises ``ValueError``.
    """

    native_ids = {
        block["id"]
        for row in rows
        for block in _blocks(row)
        if block.get("type") == "tool_use" and isinstance(block.get("id"), str)
    }
    used: set[str] = set()
    calls: dict[tuple[Any, str], str] = {}
    pending: dict[str, str] = {}
    scoped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        renames: list[tuple[int, str, str]] = []
        message = row.get("message")
        identity = (
            (message.get("id") if isinstance(message, dict) else None)
            or row.get("uuid")
            or ("event", index)
        )
        for position, block in enumerate(_blocks(row)):
            if row.get("type") == "assistant" and block.get("type") == "tool_use":
                native = block.get("id")
                if not isinstance(native, str) or not native:
                    continue
                key = (identity, native)
                if key not in calls:
                    if native in pending:
                        raise ValueError(
                            f"Claude tool ID is reused before its result: {native}"
                        )
                    unique, count = native, 2
                    while unique in used or (unique != native and unique in native_ids):
                        unique = f"{native}__occurrence_{count}"
                        count += 1
                    calls[key] = pending[native] = unique
                    used.add(unique)
                if calls[key] != native:
                    renames.append((position, "id", calls[key]))
            elif row.get("type") == "user" and block.get("type") == "tool_result":
                native = block.get("tool_use_id")
                unique = pending.pop(native, None) if isinstance(native, str) else None
                if unique is not None and unique != native:
                    renames.append((position, "tool_use_id", unique))
        if renames:
            row = deepcopy(row)
            for position, field, unique in renames:
                block = row["message"]["content"][position]
                block["roller_original_tool_id"] = block[field]
                block[field] = unique
        scoped.append(row)
    return scoped


def _blocks(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Content blocks of a root session row, with list positions preserved."""

    message = row.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if row.get("isSidechain") or not isinstance(content, list):
        return []
    return [block if isinstance(block, dict) else {} for block in content]
