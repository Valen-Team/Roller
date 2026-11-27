"""Dependency-free Harness capabilities shared by parsing and artifact checks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HarnessCapabilities:
    label: str
    cache_trace: bool = False
    capture_reader: str = ""
    capture_reader_kwargs: tuple[tuple[str, str], ...] = ()
    trace_converter: str = ""
    usage_extractor: str = ""
    compaction_mode: str = "request"
    allows_raw_only_actf: bool = False
    # The CLI or SDK accepts ``files.attach_images`` on a user message.
    attach_images: bool = False
    # From the API base to the Chat Completions endpoint whose calls
    # ``--token-ids`` records.
    chat_completions: str = "/chat/completions"


HARNESSES = {
    "openclaw": HarnessCapabilities(
        "OpenClaw",
        cache_trace=True,
        capture_reader="roller.openclaw_capture:read_openclaw_native",
        trace_converter="roller.actf.openclaw:convert_openclaw_trace",
        usage_extractor="roller.usage:extract_openclaw_usage",
        compaction_mode="openclaw",
        allows_raw_only_actf=True,
        attach_images=True,
    ),
    "codex": HarnessCapabilities(
        "Codex",
        capture_reader="roller.codex_capture:read_codex_native",
        trace_converter="roller.actf.codex:convert_codex_trace",
        usage_extractor="roller.usage:extract_codex_usage",
        compaction_mode="codex",
        attach_images=True,
    ),
    "claude-code": HarnessCapabilities(
        "Claude Code",
        capture_reader="roller.claude_capture:read_claude_native",
        trace_converter="roller.actf.claude_code:convert_claude_code_trace",
        usage_extractor="roller.usage:extract_claude_code_usage",
        compaction_mode="claude",
        attach_images=True,
        # Its API base has no version segment.
        chat_completions="/v1/chat/completions",
    ),
    "hermes": HarnessCapabilities(
        "Hermes",
        capture_reader="roller.bridge_capture:read_bridge_native",
        capture_reader_kwargs=(("harness", "hermes"),),
        trace_converter="roller.actf.hermes:convert_hermes_trace",
        usage_extractor="roller.actf.hermes:extract_hermes_usage",
    ),
    "opencode": HarnessCapabilities(
        "OpenCode",
        capture_reader="roller.opencode_capture:read_opencode_native",
        trace_converter="roller.actf.opencode:convert_opencode_trace",
        usage_extractor="roller.actf.opencode:extract_opencode_usage",
        compaction_mode="opencode",
        allows_raw_only_actf=True,
        attach_images=True,
    ),
    "deepseek": HarnessCapabilities(
        "DeepSeek Harness",
        capture_reader="roller.bridge_capture:read_bridge_native",
        capture_reader_kwargs=(("harness", "deepseek"),),
        trace_converter="roller.actf.deepseek:convert_deepseek_trace",
        usage_extractor="roller.actf.deepseek:extract_deepseek_usage",
    ),
    "pi": HarnessCapabilities(
        "Pi Agent",
        capture_reader="roller.bridge_capture:read_bridge_native",
        capture_reader_kwargs=(("harness", "pi"),),
        trace_converter="roller.actf.pi:convert_pi_trace",
        usage_extractor="roller.actf.pi:extract_pi_usage",
        compaction_mode="pi",
        attach_images=True,
    ),
    "stirrup": HarnessCapabilities(
        "Stirrup",
        capture_reader="roller.actf.stirrup:read_stirrup_native",
        trace_converter="roller.actf.stirrup:convert_stirrup_trace",
        usage_extractor="roller.actf.stirrup:extract_stirrup_usage",
        compaction_mode="stirrup",
    ),
}
HARNESS_NAMES = tuple(HARNESSES)


def requires_cache_trace(harness: object) -> bool:
    """Unknown/legacy metadata retains the collector's strict capture policy."""
    capability = HARNESSES.get(harness) if isinstance(harness, str) else None
    return capability.cache_trace if capability is not None else True


def compaction_mode(harness: object) -> str:
    capability = HARNESSES.get(harness) if isinstance(harness, str) else None
    return capability.compaction_mode if capability is not None else "request"


def allows_raw_only_actf(harness: object) -> bool:
    capability = HARNESSES.get(harness) if isinstance(harness, str) else None
    return capability.allows_raw_only_actf if capability is not None else False


def read_native_capture(harness: str, directory: Path) -> Any:
    """Load a harness's native evidence reader without importing adapters at startup."""
    capability = HARNESSES[harness]
    module_name, reader_name = capability.capture_reader.split(":", 1)
    reader = getattr(import_module(module_name), reader_name)
    return reader(directory, **dict(capability.capture_reader_kwargs))


def convert_native_trace(harness: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke a harness's ACTF converter without coupling harness adapters to ACTF."""
    module_name, func_name = HARNESSES[harness].trace_converter.split(":", 1)
    converter = getattr(import_module(module_name), func_name)
    return converter(*args, **kwargs)


def extract_native_usage(harness: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke a harness's usage extractor without coupling harness adapters to ACTF."""
    module_name, func_name = HARNESSES[harness].usage_extractor.split(":", 1)
    extractor = getattr(import_module(module_name), func_name)
    return extractor(*args, **kwargs)
