"""Codex configuration, effort mapping, and CLI option declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from roller.harnesses.config_overrides import (
    ConfigAssignment,
    config_path,
    validate_config_assignments,
)
from roller.harnesses.container import PROXY_ENV_NAMES
from roller.thinking import normalize_thinking
from roller.wire_trace import RESPONSES_IMAGE_DETAILS

COMMON_ENV_NAMES = PROXY_ENV_NAMES
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

DEFAULT_CODEX_IMAGE = "roller-codex:0.149.0"
CODEX_HOME_CONTAINER_PATH = "/roller/codex-home"
# Codex always replays reasoning.encrypted_content with store=false; give
# relayed endpoints a few chances before one rejected replay ends the rollout.
CODEX_ENCRYPTED_CONTENT_RETRIES = 3
# Provider Roller defines for --api-base. Its other keys (stream_idle_timeout_ms,
# request_max_retries, ...) stay open to --codex-config-set.
CODEX_PROVIDER_ID = "roller_api"
# Config paths Roller passes on every ``codex exec``, with the option that
# controls each. ``--codex-config-set`` may not touch them.
ROLLER_MANAGED_CODEX_CONFIG: dict[str, str] = {
    "model": "use --model",
    "model_provider": "use --api-base / --codex-api-base",
    f"model_providers.{CODEX_PROVIDER_ID}.name": "use --api-base / --codex-api-base",
    f"model_providers.{CODEX_PROVIDER_ID}.base_url": "use --api-base / --codex-api-base",
    f"model_providers.{CODEX_PROVIDER_ID}.env_key": "use --api-key-env / --codex-api-key-env",
    f"model_providers.{CODEX_PROVIDER_ID}.wire_api": "Roller uses the Responses API",
    f"model_providers.{CODEX_PROVIDER_ID}.supports_websockets": (
        "use --codex-websockets / --no-codex-websockets"
    ),
    "disable_response_storage": "Roller disables response storage",
    "model_reasoning_effort": "use --thinking",
    "model_reasoning_summary": "use --reasoning-summary",
    "web_search": "use --codex-web-search",
    "features.unified_exec": "Roller enables it for every rollout",
    "sandbox_mode": "Roller bypasses the sandbox inside the disposable container",
    "approval_policy": "Roller bypasses approvals inside the disposable container",
}


def codex_effort(value: str | None) -> str | None:
    """Validate a logical effort using the same rules as Codex execution."""
    effort = normalize_thinking(value, field="Codex reasoning effort")
    if effort == "off":
        raise ValueError(
            "Codex does not expose a harness-level reasoning-off mode; omit "
            "--thinking to use the model default or choose an explicit effort"
        )
    if effort == "adaptive":
        raise ValueError(
            "Codex does not accept the OpenClaw-specific adaptive thinking profile"
        )
    return effort


def redact_api_base(value: str) -> str:
    """Return an API endpoint safe to retain in run metadata."""

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    netloc = hostname
    if port is not None:
        netloc = f"{hostname}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


@dataclass(frozen=True)
class CodexConfig:
    docker_image: str = DEFAULT_CODEX_IMAGE
    container_user: str | None = "0"
    network: str | None = "host"
    workspace_container_path: str = "/roller/workspace"
    config_path: Path | None = None
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    pass_env: tuple[str, ...] = COMMON_ENV_NAMES
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    web_search: str = "disabled"
    supports_websockets: bool = False
    capture_calls: bool = False
    capture_debug: str = "on-error"
    wire_trace_mode: str = "actf"
    keep_thinking: bool = True
    keep_container_on_error: bool = False
    docker_binary: str = "docker"
    preflight_image_digest: str | None = None
    # Extra ``codex exec -c KEY=VALUE`` overrides, after Roller's own.
    config_set: tuple[ConfigAssignment, ...] = ()
    # ``detail`` the proxy sets on Responses ``input_image`` blocks without one.
    responses_image_detail: str | None = None
    responses_tool_images_as_user: bool = False
    # Serve the Responses calls from the endpoint's Chat Completions API.
    chat_completions: bool = False
    token_ids: bool = False

    def __post_init__(self) -> None:
        managed = ROLLER_MANAGED_CODEX_CONFIG
        if self.capture_calls:
            managed = {
                **managed,
                "features.multi_agent": "Roller enables it for --capture-calls",
            }
        object.__setattr__(
            self,
            "config_set",
            validate_config_assignments(
                self.config_set,
                managed,
                flag="--codex-config-set",
            ),
        )
        if not self.api_base:
            for key, _ in self.config_set:
                if config_path(key)[:2] == ("model_providers", CODEX_PROVIDER_ID):
                    raise ValueError(
                        f"--codex-config-set cannot set {key!r}: Roller defines the "
                        f"{CODEX_PROVIDER_ID} provider only with --api-base / --codex-api-base"
                    )
        object.__setattr__(
            self,
            "reasoning_effort",
            normalize_thinking(self.reasoning_effort, field="Codex reasoning effort"),
        )
        if not self.docker_image.strip():
            raise ValueError("Codex docker image must be non-empty")
        if self.container_user is not None and not self.container_user.strip():
            raise ValueError("Codex container user must be non-empty or None")
        if not self.workspace_container_path.startswith("/"):
            raise ValueError("Codex workspace container path must be absolute")
        if not ENV_NAME_RE.fullmatch(self.api_key_env):
            raise ValueError(
                f"invalid Codex API key environment name: {self.api_key_env!r}"
            )
        if self.api_base:
            try:
                parsed = urlsplit(self.api_base)
                parsed.port  # noqa: B018 - reading it raises for an invalid port
            except ValueError as exc:
                raise ValueError(
                    "Codex API base must be an absolute http(s) URL"
                ) from exc
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Codex API base must be an absolute http(s) URL")
        if self.reasoning_summary not in {None, "auto", "concise", "detailed", "none"}:
            raise ValueError(
                "Codex reasoning summary must be auto|concise|detailed|none"
            )
        if self.web_search not in {"disabled", "cached", "live"}:
            raise ValueError("Codex web search must be disabled|cached|live")
        if self.capture_calls and (
            not self.api_base
            or self.network != "host"
            or self.supports_websockets
            or self.wire_trace_mode == "off"
        ):
            raise ValueError(
                "Codex call capture requires --api-base, host network, wire tracing and --no-codex-websockets"
            )
        if self.chat_completions and (
            not self.api_base
            or self.supports_websockets
            or self.wire_trace_mode == "off"
        ):
            raise ValueError(
                "Codex Chat Completions translation requires --api-base, wire "
                "tracing and --no-codex-websockets"
            )
        if self.token_ids and not (self.chat_completions and self.capture_calls):
            raise ValueError(
                "Codex --token-ids requires --codex-chat-completions and --capture-calls"
            )
        if self.wire_trace_mode not in {"off", "actf", "full"}:
            raise ValueError("Codex wire trace mode must be off|actf|full")
        if self.responses_image_detail is not None:
            if self.responses_image_detail not in RESPONSES_IMAGE_DETAILS:
                raise ValueError(
                    f"Codex image detail must be {'|'.join(RESPONSES_IMAGE_DETAILS)}"
                )
            if (
                not self.api_base
                or self.network != "host"
                or self.supports_websockets
                or self.wire_trace_mode == "off"
            ):
                raise ValueError(
                    "Codex image detail requires --api-base, host network, wire tracing and --no-codex-websockets"
                )
        if self.wire_trace_mode == "full" and not self.api_base:
            raise ValueError("Codex --wire-trace full requires --api-base")
        if self.wire_trace_mode == "full" and self.supports_websockets:
            raise ValueError("Codex tool capture requires --no-codex-websockets")
        if (
            self.wire_trace_mode != "off"
            and self.api_base
            and not self.supports_websockets
            and self.network != "host"
        ):
            raise ValueError(
                "Codex tool capture requires "
                "--network host so the container can reach the rollout-local proxy"
            )
        if self.config_path is not None and not self.config_path.is_file():
            raise FileNotFoundError(f"Codex config not found: {self.config_path}")


def _codex_cli_options():
    import argparse

    from roller.harnesses.config_overrides import parse_config_assignment_arg
    from roller.option_specs import OptionSpec

    return (
        OptionSpec(
            "--codex-config",
            yaml_key="config",
            help="optional Codex config.toml copied into the disposable container",
        ),
        OptionSpec(
            "--codex-config-set",
            yaml_key="config_set",
            yaml_kind="repeat",
            action="append",
            default=(),
            type=parse_config_assignment_arg,
            metavar="KEY=VALUE",
            help=(
                "pass `-c KEY=VALUE` to every codex exec after Roller's own overrides; "
                "repeatable. Paths Roller manages are rejected"
            ),
        ),
        OptionSpec(
            "--reasoning-summary",
            yaml_key="reasoning_summary",
            choices=("auto", "concise", "detailed", "none"),
            help="Codex model_reasoning_summary override",
        ),
        OptionSpec(
            "--codex-web-search",
            yaml_key="web_search",
            choices=("disabled", "cached", "live"),
            default="disabled",
        ),
        OptionSpec(
            "--codex-websockets",
            yaml_key="websockets",
            yaml_kind="bool",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "allow the custom Codex provider to use Responses WebSockets; "
                "disabled by default for OpenAI-compatible HTTP proxies"
            ),
        ),
        OptionSpec(
            "--codex-image-detail",
            yaml_key="image_detail",
            choices=RESPONSES_IMAGE_DETAILS,
            help=(
                "set this detail on Responses input_image blocks that Codex sends "
                "without one; needs the request recording proxy"
            ),
        ),
        OptionSpec(
            "--codex-tool-images-as-user",
            yaml_key="tool_images_as_user",
            yaml_kind="switch",
            action="store_true",
            help="attach tool-returned images as user images for Responses backends that ignore multimodal tool outputs",
        ),
        OptionSpec(
            "--codex-chat-completions",
            yaml_key="chat_completions",
            yaml_kind="switch",
            action="store_true",
            help=(
                "translate Codex's Responses requests to Chat Completions at the "
                "request recording proxy, for endpoints that serve only that API"
            ),
        ),
    )


CLI_OPTIONS = _codex_cli_options()
