"""Dependency-free readers for recorded sample and trajectory metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def trajectory_conversion_unavailable(
    metadata: Mapping[str, Any] | None,
) -> str | None:
    """Return the unavailable trajectory target explicitly recorded in metadata.

    A warning string is deliberately insufficient: callers must provide a
    structured, non-empty conversion error. Older OpenCode results emitted no
    target and are interpreted as ACTF failures.
    """

    if not isinstance(metadata, Mapping):
        return None
    conversion = metadata.get("trajectory_conversion")
    valid = (
        isinstance(conversion, Mapping)
        and conversion.get("status") == "unavailable"
        and isinstance(conversion.get("error"), str)
        and bool(conversion["error"].strip())
    )
    if not valid:
        return None
    target = conversion.get("target", "actf")
    return target if target in {"actf", "sft"} else None


def keeps_thinking(metadata: Mapping[str, Any] | None) -> bool:
    """Read the recorded ``--keep-thinking`` choice; bridge harnesses nest it in settings."""
    if not isinstance(metadata, Mapping):
        return True
    settings = metadata.get("settings")
    settings = settings if isinstance(settings, Mapping) else {}
    return bool(metadata.get("keep_thinking", settings.get("keep_thinking", True)))


def has_legacy_claude_sidechain(trajectory: Mapping[str, Any]) -> bool:
    """Identify older ACTF records that mixed child steps into the root."""

    steps = trajectory.get("steps")
    return isinstance(steps, list) and any(
        isinstance(step, Mapping)
        and isinstance(step.get("extra"), Mapping)
        and step["extra"].get("is_sidechain") is True
        for step in steps
    )
