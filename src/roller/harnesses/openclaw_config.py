"""OpenClaw configuration, model-config preparation, and CLI option declarations."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from roller.harnesses.config_overrides import (
    ConfigAssignment,
    validate_config_assignments,
)
from roller.harnesses.container import PROXY_ENV_NAMES
from roller.thinking import normalize_thinking
from roller.wire_trace import WireTraceProxy, rewrite_model_config

DEFAULT_OPENCLAW_IMAGE = "ghcr.io/openclaw/openclaw:2026.11.10"
OPENCLAW_STATE_CONTAINER_PATH = "/tmp/roller-openclaw-state"
# The workspace of an OpenClaw install. Roller links it to the task workspace
# and runs the agent in it, so a session records the working directory that
# OpenClaw sessions have everywhere else.
OPENCLAW_WORKSPACE_LINK = "/root/.openclaw/workspace"
# OpenClaw diagnostics.cacheTrace output; records the assembled system prompt
# (and full request messages) for every LLM call made by the agent.
CACHE_TRACE_CONTAINER_PATH = "/tmp/roller-cache-trace.jsonl"
TRANSCRIPT_EXPORT_CONTAINER_PATH = "/tmp/roller-session-trace.jsonl"
TASK_ENVIRONMENT_CONTAINER_PATH = "/roller/task/environment"

COMMON_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "BRAVE_API_KEY",
    "TAVILY_API_KEY",
    *PROXY_ENV_NAMES,
)

_OPENCLAW_2026_3_11_UNSAFE_THINKING = frozenset({"xhigh", "adaptive", "max", "ultra"})


def validate_openclaw_thinking(version: str, thinking: str | None) -> None:
    """Guard the legacy downgrade; modern model profiles belong to OpenClaw.

    A version string or CLI help text is not a model capability table. Modern
    OpenClaw validates explicit levels against its provider/model/runtime
    profile, including custom catalog compatibility settings.
    """

    if thinking in _OPENCLAW_2026_3_11_UNSAFE_THINKING and re.search(
        r"(?:^|\s)2026\.3\.11(?:\s|$)", version
    ):
        raise RuntimeError(
            f"OpenClaw {version!r} cannot faithfully apply --thinking {thinking!r}; "
            "the bundled 2026.3.11 runtime predates the current advanced thinking "
            "profiles and can silently downgrade the request. Use off|minimal|low|"
            "medium|high, or select a newer OpenClaw image with --docker-image."
        )


def expand_env_placeholders(
    value: Any, environ: Mapping[str, str] | None = None
) -> Any:
    """Expand ${NAME} in model config values without persisting expanded secrets."""

    environment = os.environ if environ is None else environ
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def expand_string(text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in environment or environment[name] == "":
                raise ValueError(
                    f"model config references missing environment variable {name}"
                )
            return environment[name]

        return pattern.sub(replace, text)

    if isinstance(value, str):
        return expand_string(value)
    if isinstance(value, list):
        return [expand_env_placeholders(item, environment) for item in value]
    if isinstance(value, dict):
        return {
            key: expand_env_placeholders(item, environment)
            for key, item in value.items()
        }
    return value


def load_model_config(
    path: Path, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model config JSON {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"model config must be a JSON object: {path}")
    return expand_env_placeholders(payload, environ)


def prepare_model_config(
    config: Mapping[str, Any],
    model_name: str,
    *,
    proxy: WireTraceProxy | None = None,
    enable_reasoning: bool = False,
    request_timeout_seconds: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Copy a model config and apply rollout-local transport/model settings."""
    prepared = (
        rewrite_model_config(config, proxy)
        if proxy is not None
        else deepcopy(dict(config))
    )
    providers = prepared.get("providers")
    if not isinstance(providers, dict):
        return prepared
    for provider in providers.values():
        models = provider.get("models") if isinstance(provider, dict) else None
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, dict):
                continue
            # Earlier local configs used tags=["reasoning"]. OpenClaw 2026.3.11
            # rejects the tags key and gates reasoning_effort on this boolean.
            tags = model.pop("tags", None)
            if isinstance(tags, list) and "reasoning" in tags:
                model.setdefault("reasoning", True)
    if "/" in model_name:
        provider_name, model_id = model_name.split("/", 1)
        provider = providers.get(provider_name)
        models = provider.get("models") if isinstance(provider, dict) else None
        if isinstance(provider, dict) and request_timeout_seconds is not None:
            provider["timeoutSeconds"] = request_timeout_seconds
        if isinstance(models, list):
            for model in models:
                if isinstance(model, dict) and model.get("id") == model_id:
                    if enable_reasoning:
                        # Preserve explicit capabilities and provider mappings.
                        model.setdefault("reasoning", True)
                    if max_tokens is not None:
                        model["maxTokens"] = max_tokens
                    break
    return prepared


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<invalid-url>"
    if not parsed.scheme or not parsed.netloc:
        return value.split("?", 1)[0]
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _model_base_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {
                "baseurl",
                "base_url",
                "apiurl",
                "api_url",
            } and isinstance(item, str):
                urls.append(_redact_url(item))
            else:
                urls.extend(_model_base_urls(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_model_base_urls(item))
    return list(dict.fromkeys(urls))


# Config paths Roller writes into every OpenClaw container, with the option that
# controls each. ``--openclaw-config-set`` may not touch them.
ROLLER_MANAGED_OPENCLAW_CONFIG: dict[str, str] = {
    "gateway.port": "use --gateway-port",
    "gateway.mode": "Roller runs a local gateway",
    "gateway.bind": "Roller binds the gateway to loopback",
    "gateway.auth": "Roller issues a token per rollout",
    "gateway.remote": "Roller issues a token per rollout",
    "models": "use --model-config",
    "agents.defaults.model": "use --model",
    "agents.defaults.models": "use --model",
    "agents.defaults.workspace": "Roller owns the task workspace",
    "agents.defaults.thinkingDefault": "use --thinking",
    "agents.defaults.imageModel": "use --set-image-model / --no-set-image-model",
    "tools.web.search.provider": "use --search-provider",
    "diagnostics.cacheTrace": "use --cache-trace / --no-cache-trace",
    "logging.redactSensitive": "Roller keeps task text verbatim in session transcripts",
    "agents.defaults.heartbeat": "Roller turns off heartbeat polls, which are model calls outside the task",
}


@dataclass(frozen=True)
class OpenClawConfig:
    docker_image: str = DEFAULT_OPENCLAW_IMAGE
    container_user: str | None = "0"
    gateway_port_start: int = 18789
    gateway_port_step: int = 10
    gateway_startup_timeout: float = 240.0
    network: str = "host"
    workspace_container_path: str = "/tmp_workspace"
    session_id: str = "roller"
    model_config: Mapping[str, Any] | None = None
    model_request_timeout_seconds: int | None = None
    model_max_tokens: int | None = None
    pass_env: tuple[str, ...] = COMMON_ENV_NAMES
    search_provider: str | None = None
    thinking: str | None = None
    capture_cache_trace: bool = True
    capture_calls: bool = False
    token_ids: bool = False
    capture_debug: str = "on-error"
    wire_trace_mode: str = "actf"
    keep_thinking: bool = True
    collect_paths: tuple[str, ...] = ()
    keep_container_on_error: bool = False
    set_image_model: bool = True
    docker_binary: str = "docker"
    preflight_image_digest: str | None = None
    # Extra ``openclaw config set`` calls, applied after Roller's own settings.
    config_set: tuple[ConfigAssignment, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "config_set",
            validate_config_assignments(
                self.config_set,
                ROLLER_MANAGED_OPENCLAW_CONFIG,
                flag="--openclaw-config-set",
            ),
        )
        if not self.docker_image.strip():
            raise ValueError("docker_image cannot be empty")
        if self.container_user is not None and not self.container_user.strip():
            raise ValueError("container_user must be non-empty or None")
        if not (1 <= self.gateway_port_start <= 65535):
            raise ValueError("gateway_port_start must be between 1 and 65535")
        if self.gateway_port_step < 1:
            raise ValueError("gateway_port_step must be positive")
        if self.gateway_startup_timeout <= 0:
            raise ValueError("gateway_startup_timeout must be positive")
        if not self.workspace_container_path.startswith("/"):
            raise ValueError("workspace_container_path must be absolute")
        if not self.session_id or "/" in self.session_id:
            raise ValueError("session_id must be a non-empty path-safe value")
        if (
            self.model_request_timeout_seconds is not None
            and self.model_request_timeout_seconds <= 0
        ):
            raise ValueError("model_request_timeout_seconds must be positive")
        if self.model_max_tokens is not None and self.model_max_tokens <= 0:
            raise ValueError("model_max_tokens must be positive")
        for path in self.collect_paths:
            if not path.startswith("/"):
                raise ValueError(f"collected container path must be absolute: {path}")
        object.__setattr__(self, "thinking", normalize_thinking(self.thinking))
        if self.capture_calls and (
            self.model_config is None
            or self.network != "host"
            or self.wire_trace_mode == "off"
            or not self.capture_cache_trace
        ):
            raise ValueError(
                "OpenClaw call capture requires --model-config, host network, wire tracing and cache trace"
            )
        if self.token_ids and not self.capture_calls:
            raise ValueError("OpenClaw --token-ids requires --capture-calls")
        if self.wire_trace_mode not in ("off", "actf", "full"):
            raise ValueError(
                f"wire_trace_mode must be off|actf|full: {self.wire_trace_mode}"
            )


def _openclaw_cli_options():
    import argparse

    from roller.harnesses.config_overrides import parse_config_assignment_arg
    from roller.harnesses.gateway_ports import DEFAULT_GATEWAY_PORT
    from roller.option_specs import OptionSpec

    return (
        OptionSpec(
            "--model-config",
            yaml_key="model_config",
            help="OpenClaw models JSON; supports ${ENV_NAME}",
        ),
        OptionSpec(
            "--gateway-port",
            yaml_key="gateway_port",
            type=int,
            help=(
                "first OpenClaw gateway port (default: the first range from "
                f"{DEFAULT_GATEWAY_PORT} that no other run on this host holds)"
            ),
        ),
        OptionSpec(
            "--gateway-port-step", yaml_key="gateway_port_step", type=int, default=10
        ),
        OptionSpec(
            "--gateway-startup-timeout",
            yaml_key="gateway_startup_timeout",
            type=float,
            default=240.0,
        ),
        OptionSpec(
            "--openclaw-model-request-timeout",
            yaml_key="model_request_timeout",
            type=int,
            metavar="SECONDS",
            help=(
                "per-provider model request and stream-idle timeout; overrides "
                "models.providers.<id>.timeoutSeconds"
            ),
        ),
        OptionSpec(
            "--openclaw-max-tokens",
            yaml_key="max_tokens",
            type=int,
            metavar="TOKENS",
            help="override maxTokens for the selected OpenClaw model",
        ),
        OptionSpec(
            "--openclaw-config-set",
            yaml_key="config_set",
            yaml_kind="repeat",
            action="append",
            default=(),
            type=parse_config_assignment_arg,
            metavar="KEY=VALUE",
            help=(
                "run `openclaw config set KEY VALUE` in every OpenClaw container after "
                "Roller's own settings; repeatable. Paths Roller manages are rejected"
            ),
        ),
        OptionSpec("--session-id", yaml_key="session_id", default="roller"),
        OptionSpec("--search-provider", yaml_key="search_provider"),
        OptionSpec(
            "--collect-path",
            yaml_key="collect_paths",
            yaml_kind="repeat",
            action="append",
            default=(),
            metavar="CONTAINER_PATH",
            help="legacy: copy an extra container path into results/legacy",
        ),
        OptionSpec(
            "--set-image-model",
            yaml_key="set_image_model",
            yaml_kind="bool",
            action=argparse.BooleanOptionalAction,
            default=True,
        ),
        OptionSpec(
            "--cache-trace",
            yaml_key="cache_trace",
            yaml_kind="bool",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="record the assembled system prompt per LLM call (diagnostics.cacheTrace)",
        ),
    )


CLI_OPTIONS = _openclaw_cli_options()
