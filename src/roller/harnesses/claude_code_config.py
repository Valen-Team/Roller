"""Claude Code configuration, effort mapping, and CLI option declarations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from roller.harnesses.container import PROXY_ENV_NAMES
from roller.redaction import normalize_anthropic_base_url
from roller.sandbox import SandboxRun
from roller.thinking import normalize_thinking

COMMON_ENV_NAMES = PROXY_ENV_NAMES
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

DEFAULT_CLAUDE_CODE_IMAGE = "roller-claude-code:2.1.222"
CLAUDE_HOME_CONTAINER_PATH = "/roller/claude-home"
# Native and delivered IDs of every tool call --claude-unique-tool-ids repaired.
TOOL_ID_MAPPINGS_NAME = "tool_id_mappings.jsonl"
# Stream-json user message of a turn that carries images.
CLAUDE_INPUT_CONTAINER_PATH = "/roller/claude-input.jsonl"
# Normalized conversation a bench grader reads; see
# ``ClaudeCodeSession.prepare_transcript_for_grading``.
GRADING_TRANSCRIPT_CONTAINER_PATH = "/tmp/roller-claude-transcript.json"

CLAUDE_PERMISSION_MODES = (
    "default",
    "acceptEdits",
    "plan",
    "auto",
    "dontAsk",
    "bypassPermissions",
)

CLAUDE_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
    "ultra": "ultracode",
}


def actf_assistant_messages(
    trajectory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """An ACTF trajectory's visible assistant turns, as a grader transcript.

    The order is the trajectory's, so the last entry is the step
    :meth:`ClaudeCodeSession._latest_reply` returns as the run's reply. A run
    whose turns all ended in a tool call yields an empty list, which a bench
    grader scores by its own protocol rather than reading as an error.
    """

    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise RuntimeError("Claude Code produced an invalid ACTF trajectory")
    messages: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        assistant = step.get("assistant_content")
        if not isinstance(assistant, Mapping):
            continue
        content = assistant.get("content")
        if isinstance(content, str) and content.strip():
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": content.strip()}],
                }
            )
    return messages


def claude_effort(value: str | None) -> str | None:
    """Translate Roller's logical thinking level to Claude Code's ``--effort``."""

    normalized = normalize_thinking(value, field="Claude Code reasoning effort")
    if normalized is None:
        return None
    mapped = CLAUDE_EFFORT_MAP.get(normalized)
    if mapped is None:
        raise ValueError(
            "Claude Code reasoning effort must be low|medium|high|xhigh|max|ultra; "
            f"{normalized!r} has no exact Claude Code mapping"
        )
    return mapped


@dataclass(frozen=True)
class ClaudeCodeConfig:
    docker_image: str = DEFAULT_CLAUDE_CODE_IMAGE
    container_user: str | None = "0"
    network: str | None = "host"
    workspace_container_path: str = "/roller/workspace"
    settings_path: Path | None = None
    config_sha256: str | None = None
    api_base: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    pass_env: tuple[str, ...] = COMMON_ENV_NAMES
    reasoning_effort: str | None = None
    max_thinking_tokens: int | None = None
    max_turns: int | None = None
    permission_mode: str = "bypassPermissions"
    wire_trace_mode: str = "actf"
    keep_thinking: bool = True
    keep_container_on_error: bool = False
    docker_binary: str = "docker"
    preflight_image_digest: str | None = None
    capture_calls: bool = False
    capture_debug: str = "on-error"
    # Let the proxy replace tool-use IDs the endpoint issues more than once.
    unique_tool_ids: bool = False
    pdf_as_images: bool = False
    # The proxy sends Messages requests to the endpoint's Chat Completions API.
    chat_completions: bool = False
    # The proxy records the token ids and logprobs of each call.
    token_ids: bool = False
    # The CLI's tool results keep their images in the tool message itself.
    tool_images_inline: bool = False
    # Set for a remote sandbox provider: the CLI then runs in one of its
    # instances instead of a local Docker container.
    sandbox_run: SandboxRun | None = None
    # Where that sandbox's template mounts the pinned CLI release.
    sandbox_binary: str | None = None
    # Served context length told to the CLI, which compacts its history before
    # the server would refuse the request.
    context_window: int | None = None

    @property
    def sandbox(self) -> bool:
        return self.sandbox_run is not None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reasoning_effort",
            normalize_thinking(
                self.reasoning_effort, field="Claude Code reasoning effort"
            ),
        )
        if self.api_base:
            object.__setattr__(
                self, "api_base", normalize_anthropic_base_url(self.api_base)
            )
        if not self.docker_image.strip():
            raise ValueError("Claude Code docker image must be non-empty")
        if self.container_user is not None and not self.container_user.strip():
            raise ValueError("Claude Code container user must be non-empty or None")
        if not self.workspace_container_path.startswith("/"):
            raise ValueError("Claude Code workspace container path must be absolute")
        if not ENV_NAME_RE.fullmatch(self.api_key_env):
            raise ValueError(
                f"invalid Claude Code API key environment name: {self.api_key_env!r}"
            )
        claude_effort(self.reasoning_effort)
        if self.max_thinking_tokens is not None and self.max_thinking_tokens <= 0:
            raise ValueError(
                "Claude Code max thinking tokens must be greater than zero"
            )
        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError("Claude Code max turns must be greater than zero")
        if self.permission_mode not in CLAUDE_PERMISSION_MODES:
            raise ValueError(
                "Claude Code permission mode must be "
                + "|".join(CLAUDE_PERMISSION_MODES)
            )
        if self.wire_trace_mode not in {"off", "actf", "full"}:
            raise ValueError("Claude Code wire trace mode must be off|actf|full")
        if self.wire_trace_mode == "full" and not self.api_base:
            raise ValueError("Claude Code --wire-trace full requires --api-base")
        # A remote sandbox reaches the proxy through the provider's relay, not
        # through the host's network namespace.
        if self.capture_calls and (
            not self.api_base or (self.network != "host" and not self.sandbox)
        ):
            raise ValueError(
                "Claude call capture requires --claude-api-base and --network host"
            )
        if self.capture_calls and self.wire_trace_mode == "off":
            raise ValueError("Claude call capture requires wire tracing")
        if self.unique_tool_ids and (
            not self.api_base or self.wire_trace_mode == "off"
        ):
            raise ValueError(
                "Claude tool ID repair requires --claude-api-base and wire tracing"
            )
        if self.chat_completions and (
            not self.api_base or self.wire_trace_mode == "off"
        ):
            raise ValueError(
                "Claude Chat Completions translation requires --claude-api-base "
                "and wire tracing"
            )
        if self.tool_images_inline and not self.chat_completions:
            raise ValueError(
                "Claude inline tool images apply to the Chat Completions "
                "translation; add --claude-chat-completions"
            )
        if self.sandbox_binary is not None and not self.sandbox_binary.startswith("/"):
            raise ValueError("Claude Code sandbox binary path must be absolute")
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError("Claude Code context window must be greater than zero")
        if self.token_ids and not (self.chat_completions and self.capture_calls):
            raise ValueError(
                "Claude Code --token-ids requires --claude-chat-completions "
                "and --capture-calls"
            )
        if self.pdf_as_images:
            if not self.api_base or self.wire_trace_mode == "off":
                raise ValueError(
                    "Claude PDF conversion requires --claude-api-base and wire tracing"
                )
            if find_spec("pypdfium2") is None or find_spec("PIL") is None:
                raise ValueError(
                    "Claude PDF conversion requires the roller-agent[pdf] extra, "
                    "which needs Python 3.11 or later"
                )
        if (
            self.wire_trace_mode != "off"
            and self.api_base
            and self.network != "host"
            and not self.sandbox
        ):
            raise ValueError(
                "Claude Code wire tracing requires --network host so the container "
                "can reach the rollout-local proxy"
            )
        if self.settings_path is not None and not self.settings_path.is_file():
            raise FileNotFoundError(
                f"Claude Code settings file not found: {self.settings_path}"
            )


def _claude_code_cli_options():
    from roller.harnesses.claude_code_sandbox import DEFAULT_BINARY
    from roller.option_specs import OptionSpec

    return (
        OptionSpec(
            "--claude-capture-calls",
            yaml_key="capture_calls",
            yaml_kind="switch",
            action="store_true",
            help="save Claude main/subagent native files, full HTTP calls and participant relations",
        ),
        OptionSpec(
            "--claude-settings",
            yaml_key="settings",
            help="optional Claude Code settings.json copied into the disposable container",
        ),
        OptionSpec(
            "--claude-max-turns",
            yaml_key="max_turns",
            type=int,
            help="optional Claude Code --max-turns limit for each predefined user message",
        ),
        OptionSpec(
            "--claude-max-thinking-tokens",
            yaml_key="max_thinking_tokens",
            type=int,
            help="optional MAX_THINKING_TOKENS value passed to Claude Code",
        ),
        OptionSpec(
            "--claude-permission-mode",
            yaml_key="permission_mode",
            choices=CLAUDE_PERMISSION_MODES,
            default="bypassPermissions",
        ),
        OptionSpec(
            "--claude-unique-tool-ids",
            yaml_key="unique_tool_ids",
            yaml_kind="switch",
            action="store_true",
            help=(
                "replace tool-use IDs the Anthropic-compatible endpoint issues more "
                "than once before Claude Code receives them; needs the request "
                "recording proxy"
            ),
        ),
        OptionSpec(
            "--claude-chat-completions",
            yaml_key="chat_completions",
            yaml_kind="switch",
            action="store_true",
            help=(
                "translate Claude Code's Messages requests to Chat Completions at the "
                "request recording proxy, for endpoints that serve only that API"
            ),
        ),
        OptionSpec(
            "--claude-tool-images-inline",
            yaml_key="tool_images_inline",
            yaml_kind="switch",
            action="store_true",
            help=(
                "keep an image a tool returned in the tool message itself instead of "
                "the user message that follows it, for endpoints and chat templates "
                "that render it there; needs --claude-chat-completions"
            ),
        ),
        OptionSpec(
            "--claude-context-window",
            yaml_key="context_window",
            type=int,
            default=None,
            help=(
                "served context length told to Claude Code, which compacts its "
                "history before the endpoint would refuse the request"
            ),
        ),
        OptionSpec(
            "--claude-sandbox-binary",
            yaml_key="sandbox_binary",
            default=None,
            help=(
                "absolute path of the Claude Code executable inside a remote sandbox; "
                f"defaults to {DEFAULT_BINARY}"
            ),
        ),
        OptionSpec(
            "--claude-pdf-as-images",
            yaml_key="pdf_as_images",
            yaml_kind="switch",
            action="store_true",
            help="render inline PDF documents as page text and images for Messages backends without document support",
        ),
    )


CLI_OPTIONS = _claude_code_cli_options()
