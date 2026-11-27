"""Harness configuration assembly and run-info metadata for ``roller collect``."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roller.benches import missing_grader_env
from roller.cli import preflight as _preflight
from roller.cli.preflight import (
    TaskResolution,
    _deduplicate,
    _preflight_environment,
    _validate_openclaw_models,
    _validate_thinking_requests,
)
from roller.harness_registry import HARNESS_NAMES, HARNESSES
from roller.harnesses.claude_code_config import (
    DEFAULT_CLAUDE_CODE_IMAGE,
    ClaudeCodeConfig,
    claude_effort,
)
from roller.harnesses.codex_config import (
    DEFAULT_CODEX_IMAGE,
    CodexConfig,
    codex_effort,
    redact_api_base,
)
from roller.harnesses.config_overrides import recorded_config_assignments
from roller.harnesses.container import PROXY_ENV_NAMES
from roller.harnesses.deepseek import (
    DEEPSEEK_VERSION,
    DEFAULT_DEEPSEEK_IMAGE,
    DeepSeekConfig,
    deepseek_effort,
)
from roller.harnesses.gateway_ports import DEFAULT_GATEWAY_PORT
from roller.harnesses.hermes import (
    DEFAULT_HERMES_IMAGE,
    HERMES_REVISION,
    HermesConfig,
    hermes_effort,
)
from roller.harnesses.openclaw_config import (
    COMMON_ENV_NAMES as OPENCLAW_COMMON_ENV_NAMES,
)
from roller.harnesses.openclaw_config import (
    DEFAULT_OPENCLAW_IMAGE,
    OpenClawConfig,
    load_model_config,
)
from roller.harnesses.opencode import (
    DEFAULT_OPENCODE_IMAGE,
    OPENCODE_VERSION,
    OpenCodeConfig,
    opencode_effort,
)
from roller.harnesses.pi import (
    DEFAULT_PI_IMAGE,
    PI_TOOLS,
    PI_VERSION,
    PiConfig,
    pi_effort,
)
from roller.harnesses.stirrup import DEFAULT_STIRRUP_IMAGE, STIRRUP_TOOLS, StirrupConfig
from roller.redaction import redact_anthropic_base_url
from roller.thinking import normalize_thinking
from roller.thinking_capabilities import (
    ThinkingCapabilities,
    load_thinking_capabilities,
)


@dataclass
class HarnessSetup:
    configs: dict[str, Any]
    images: dict[str, str]
    model_config: dict[str, Any] | None
    model_config_path: Path | None
    codex_config_path: Path | None
    codex_config_digest: str | None
    claude_settings_path: Path | None
    claude_settings_content: bytes | None
    claude_settings_sha256: str | None
    thinking_capabilities_path: Path | None
    thinking_capabilities_digest: str | None
    api_bases: dict[str, str | None]
    api_key_envs: dict[str, str]


def build_harness_setup(
    args: argparse.Namespace, resolution: TaskResolution
) -> HarnessSetup:
    selected_harnesses = resolution.selected_harnesses
    cli_harness = resolution.cli_harness
    tasks = resolution.tasks
    requests = resolution.requests
    remote = resolution.remote

    requested_api_base = args.api_base
    if args.api_base is None:
        if selected_harnesses == {"codex"}:
            args.api_base = os.environ.get("OPENAI_BASE_URL")
        elif selected_harnesses == {"claude-code"}:
            args.api_base = os.environ.get("ANTHROPIC_BASE_URL")
    api_bases = {
        "hermes": args.hermes_api_base
        or requested_api_base
        or os.environ.get("OPENAI_BASE_URL"),
        "opencode": args.opencode_api_base
        or requested_api_base
        or os.environ.get("OPENAI_BASE_URL"),
        "deepseek": args.deepseek_api_base
        or requested_api_base
        or os.environ.get("DEEPSEEK_BASE_URL"),
        "pi": args.pi_api_base
        or requested_api_base
        or os.environ.get("OPENAI_BASE_URL"),
        "stirrup": args.stirrup_api_base
        or requested_api_base
        or os.environ.get("OPENAI_BASE_URL"),
        "codex": args.codex_api_base
        or requested_api_base
        or os.environ.get("OPENAI_BASE_URL"),
        "claude-code": args.claude_api_base
        or requested_api_base
        or os.environ.get("ANTHROPIC_BASE_URL"),
    }
    requested_api_key_env = args.api_key_env
    if args.api_key_env is None:
        args.api_key_env = (
            "ANTHROPIC_API_KEY"
            if selected_harnesses == {"claude-code"}
            else "OPENAI_API_KEY"
        )
    api_key_envs = {
        "hermes": args.hermes_api_key_env or requested_api_key_env or "OPENAI_API_KEY",
        "opencode": args.opencode_api_key_env
        or requested_api_key_env
        or "OPENAI_API_KEY",
        "deepseek": args.deepseek_api_key_env
        or requested_api_key_env
        or "DEEPSEEK_API_KEY",
        "pi": args.pi_api_key_env or requested_api_key_env or "OPENAI_API_KEY",
        "stirrup": args.stirrup_api_key_env
        or requested_api_key_env
        or "OPENAI_API_KEY",
        "codex": args.codex_api_key_env or requested_api_key_env or "OPENAI_API_KEY",
        "claude-code": args.claude_api_key_env
        or requested_api_key_env
        or "ANTHROPIC_API_KEY",
    }
    defaults = {
        "hermes": os.environ.get("ROLLER_HERMES_IMAGE", DEFAULT_HERMES_IMAGE),
        "opencode": os.environ.get("ROLLER_OPENCODE_IMAGE", DEFAULT_OPENCODE_IMAGE),
        "deepseek": os.environ.get("ROLLER_DEEPSEEK_IMAGE", DEFAULT_DEEPSEEK_IMAGE),
        "pi": os.environ.get("ROLLER_PI_IMAGE", DEFAULT_PI_IMAGE),
        "stirrup": os.environ.get("ROLLER_STIRRUP_IMAGE", DEFAULT_STIRRUP_IMAGE),
        "openclaw": os.environ.get("ROLLER_OPENCLAW_IMAGE", DEFAULT_OPENCLAW_IMAGE),
        "codex": os.environ.get("ROLLER_CODEX_IMAGE", DEFAULT_CODEX_IMAGE),
        "claude-code": os.environ.get(
            "ROLLER_CLAUDE_CODE_IMAGE", DEFAULT_CLAUDE_CODE_IMAGE
        ),
    }
    image_overrides = {
        "hermes": args.hermes_image,
        "opencode": args.opencode_image,
        "deepseek": args.deepseek_image,
        "pi": args.pi_image,
        "stirrup": args.stirrup_image,
        "openclaw": args.openclaw_image,
        "codex": args.codex_image,
        "claude-code": args.claude_code_image,
    }
    images = {
        name: (image_overrides[name] or args.docker_image or defaults[name])
        for name in selected_harnesses
    }
    args.docker_image = args.docker_image or images[args.harness]
    if args.collect_path and "openclaw" not in selected_harnesses:
        raise ValueError("--collect-path requires an OpenClaw participant")

    model_config = None
    model_config_path: Path | None = None
    if args.model_config:
        if "openclaw" not in selected_harnesses:
            raise ValueError(
                "--model-config is only valid when an OpenClaw participant is selected"
            )
        model_config_path = Path(args.model_config).expanduser().resolve()
        if not model_config_path.is_file():
            raise FileNotFoundError(f"model config not found: {model_config_path}")
        model_config = load_model_config(model_config_path)
    if args.openclaw_model_request_timeout is not None and model_config is None:
        raise ValueError("--openclaw-model-request-timeout requires --model-config")
    if args.openclaw_max_tokens is not None and model_config is None:
        raise ValueError("--openclaw-max-tokens requires --model-config")
    if args.openclaw_config_set and "openclaw" not in selected_harnesses:
        raise ValueError(
            "--openclaw-config-set is only valid when an OpenClaw participant is selected"
        )
    if args.codex_config_set and "codex" not in selected_harnesses:
        raise ValueError(
            "--codex-config-set is only valid when a Codex participant is selected"
        )

    thinking_capabilities: ThinkingCapabilities | None = None
    thinking_capabilities_path: Path | None = None
    thinking_capabilities_digest: str | None = None
    if args.thinking_capabilities:
        thinking_capabilities_path = (
            Path(args.thinking_capabilities).expanduser().resolve()
        )
        if not thinking_capabilities_path.is_file():
            raise FileNotFoundError(
                f"thinking capabilities not found: {thinking_capabilities_path}"
            )
        thinking_capabilities = load_thinking_capabilities(thinking_capabilities_path)
        thinking_capabilities_digest = hashlib.sha256(
            thinking_capabilities_path.read_bytes()
        ).hexdigest()

    codex_config_path: Path | None = None
    codex_config_digest: str | None = None
    if args.codex_config:
        if "codex" not in selected_harnesses:
            raise ValueError(
                "--codex-config is only valid when a Codex participant is selected"
            )
        codex_config_path = Path(args.codex_config).expanduser().resolve()
        if not codex_config_path.is_file():
            raise FileNotFoundError(f"Codex config not found: {codex_config_path}")
        codex_config_digest = hashlib.sha256(codex_config_path.read_bytes()).hexdigest()

    claude_settings_path: Path | None = None
    claude_settings_content: bytes | None = None
    claude_settings_sha256: str | None = None
    if args.claude_settings:
        if "claude-code" not in selected_harnesses:
            raise ValueError(
                "--claude-settings is only valid when a Claude Code participant is selected"
            )
        claude_settings_path = Path(args.claude_settings).expanduser().resolve()
        if not claude_settings_path.is_file():
            raise FileNotFoundError(
                f"Claude Code settings file not found: {claude_settings_path}"
            )
        claude_settings_content = claude_settings_path.read_bytes()
        claude_settings_sha256 = hashlib.sha256(claude_settings_content).hexdigest()

    configs: dict[str, Any] = {name: None for name in HARNESS_NAMES}
    container_user = None if args.container_user == "image" else args.container_user

    if "openclaw" in selected_harnesses:
        configs["openclaw"] = OpenClawConfig(
            docker_image=images["openclaw"],
            container_user=container_user,
            gateway_port_start=(
                DEFAULT_GATEWAY_PORT if args.gateway_port is None else args.gateway_port
            ),
            gateway_port_step=args.gateway_port_step,
            gateway_startup_timeout=args.gateway_startup_timeout,
            network=args.network,
            session_id=args.session_id,
            model_config=model_config,
            model_request_timeout_seconds=args.openclaw_model_request_timeout,
            model_max_tokens=args.openclaw_max_tokens,
            pass_env=_deduplicate((*OPENCLAW_COMMON_ENV_NAMES, *args.pass_env)),
            search_provider=args.search_provider,
            thinking=args.thinking,
            capture_cache_trace=args.cache_trace,
            wire_trace_mode=args.wire_trace,
            capture_calls=args.capture_calls,
            token_ids=args.token_ids,
            capture_debug=args.capture_debug,
            keep_thinking=args.keep_thinking,
            collect_paths=tuple(args.collect_path),
            keep_container_on_error=args.keep_container_on_error,
            set_image_model=args.set_image_model,
            docker_binary=args.docker_binary,
            config_set=tuple(args.openclaw_config_set),
        )

        def openclaw_endpoint(model: str) -> str | None:
            providers = (model_config or {}).get("providers", {})
            provider = (
                providers.get(model.split("/", 1)[0], {})
                if isinstance(providers, dict)
                else {}
            )
            endpoint = provider.get("baseUrl") if isinstance(provider, dict) else None
            return endpoint if isinstance(endpoint, str) else None

        _validate_thinking_requests(
            requests,
            cli_harness=cli_harness,
            harness="openclaw",
            api_base=openclaw_endpoint,
            default_effort=args.thinking,
            effort_validator=normalize_thinking,
            capabilities=thinking_capabilities,
        )
        if model_config is not None:
            _validate_openclaw_models(
                requests, cli_harness=cli_harness, model_config=model_config
            )

    if "codex" in selected_harnesses:
        configs["codex"] = CodexConfig(
            docker_image=images["codex"],
            container_user=container_user,
            network=args.network,
            config_path=codex_config_path,
            api_base=api_bases["codex"],
            api_key_env=api_key_envs["codex"],
            pass_env=_deduplicate((*PROXY_ENV_NAMES, *args.pass_env)),
            reasoning_effort=args.thinking,
            reasoning_summary=args.reasoning_summary,
            web_search=args.codex_web_search,
            supports_websockets=args.codex_websockets,
            responses_image_detail=args.codex_image_detail,
            responses_tool_images_as_user=args.codex_tool_images_as_user,
            chat_completions=args.codex_chat_completions,
            token_ids=args.token_ids,
            wire_trace_mode=args.wire_trace,
            capture_calls=args.capture_calls,
            capture_debug=args.capture_debug,
            keep_thinking=args.keep_thinking,
            keep_container_on_error=args.keep_container_on_error,
            docker_binary=args.docker_binary,
            config_set=tuple(args.codex_config_set),
        )
        _validate_thinking_requests(
            requests,
            cli_harness=cli_harness,
            harness="codex",
            api_base=api_bases["codex"],
            default_effort=args.thinking,
            effort_validator=codex_effort,
            capabilities=thinking_capabilities,
        )

    if "claude-code" in selected_harnesses:
        configs["claude-code"] = ClaudeCodeConfig(
            docker_image=images["claude-code"],
            container_user=container_user,
            network=args.network,
            settings_path=claude_settings_path,
            config_sha256=claude_settings_sha256,
            api_base=api_bases["claude-code"],
            api_key_env=api_key_envs["claude-code"],
            pass_env=_deduplicate((*PROXY_ENV_NAMES, *args.pass_env)),
            reasoning_effort=args.thinking,
            max_thinking_tokens=args.claude_max_thinking_tokens,
            max_turns=args.claude_max_turns,
            capture_calls=args.claude_capture_calls or args.capture_calls,
            capture_debug=args.capture_debug,
            permission_mode=args.claude_permission_mode,
            unique_tool_ids=args.claude_unique_tool_ids,
            pdf_as_images=args.claude_pdf_as_images,
            chat_completions=args.claude_chat_completions,
            tool_images_inline=args.claude_tool_images_inline,
            token_ids=args.token_ids,
            sandbox_binary=args.claude_sandbox_binary,
            context_window=args.claude_context_window,
            wire_trace_mode=args.wire_trace,
            keep_thinking=args.keep_thinking,
            keep_container_on_error=args.keep_container_on_error,
            docker_binary=args.docker_binary,
        )
        _validate_thinking_requests(
            requests,
            cli_harness=cli_harness,
            harness="claude-code",
            api_base=configs["claude-code"].api_base,
            default_effort=args.thinking,
            effort_validator=claude_effort,
            capabilities=thinking_capabilities,
        )

    def bridge_options(name: str) -> dict[str, Any]:
        return dict(
            docker_image=images[name],
            container_user=container_user,
            network=args.network,
            api_base=api_bases[name],
            api_key_env=api_key_envs[name],
            pass_env=_deduplicate((*PROXY_ENV_NAMES, *args.pass_env)),
            reasoning_effort=args.thinking,
            wire_trace_mode=args.wire_trace,
            capture_calls=args.capture_calls,
            token_ids=args.token_ids,
            capture_debug=args.capture_debug,
            keep_thinking=args.keep_thinking,
            keep_container_on_error=args.keep_container_on_error,
            docker_binary=args.docker_binary,
        )

    if "hermes" in selected_harnesses:
        configs["hermes"] = HermesConfig(
            **bridge_options("hermes"),
            max_iterations=args.hermes_max_iterations,
            context_length=args.hermes_context_length,
            vision=args.hermes_vision,
            toolsets=tuple(args.hermes_toolset) if args.hermes_toolset else None,
        )
    if "opencode" in selected_harnesses:
        configs["opencode"] = OpenCodeConfig(
            **bridge_options("opencode"),
            max_steps=args.opencode_max_steps,
            context_length=args.opencode_context_length,
            output_tokens=args.opencode_output_tokens,
            vision=args.opencode_vision,
            native_tools=args.opencode_native_tools,
            deny_tools=tuple(args.opencode_deny_tool),
        )
    if "deepseek" in selected_harnesses:
        configs["deepseek"] = DeepSeekConfig(
            **bridge_options("deepseek"),
            remote_sandbox=remote,
            max_steps=args.deepseek_max_steps,
            context_length=args.deepseek_context_length,
            output_tokens=args.deepseek_output_tokens,
            profile=args.deepseek_profile,
            permission_mode=args.deepseek_permission_mode,
            input_modalities=tuple(args.deepseek_input_modalities.split(",")),
        )
    if "pi" in selected_harnesses:
        configs["pi"] = PiConfig(
            **bridge_options("pi"),
            max_steps=args.pi_max_steps,
            context_length=args.pi_context_length,
            output_tokens=args.pi_output_tokens,
            input_modalities=tuple(args.pi_input_modalities.split(",")),
        )
    if "stirrup" in selected_harnesses:
        if not _preflight._stirrup_installed():
            raise ValueError(
                "the Stirrup Harness needs Python 3.12 or later with the stirrup "
                "extra: pip install 'roller-agent[stirrup]'"
            )
        configs["stirrup"] = StirrupConfig(
            **{
                **bridge_options("stirrup"),
                "pass_env": _deduplicate(args.pass_env),
            },
            remote_sandbox=remote,
            max_steps=args.stirrup_max_steps,
            context_length=args.stirrup_context_length,
            output_tokens=args.stirrup_output_tokens,
            command_timeout=args.stirrup_command_timeout,
            temperature=args.stirrup_temperature,
            top_p=args.stirrup_top_p,
            seed=args.stirrup_seed,
            request_fields=tuple(args.stirrup_request_set),
        )

    for name, effort_validator, default_api_base in (
        ("hermes", hermes_effort, "https://api.openai.com/v1"),
        ("opencode", opencode_effort, "https://api.openai.com/v1"),
        ("deepseek", deepseek_effort, "https://api.deepseek.com"),
        ("pi", pi_effort, "https://api.openai.com/v1"),
        ("stirrup", normalize_thinking, "https://api.openai.com/v1"),
    ):
        bridge_config = configs[name]
        if bridge_config is not None:
            _validate_thinking_requests(
                requests,
                cli_harness=cli_harness,
                harness=name,
                api_base=bridge_config.api_base or default_api_base,
                default_effort=bridge_config.reasoning_effort,
                effort_validator=effort_validator,
                capabilities=thinking_capabilities,
            )

    for task in tasks:
        if not task.attach_images:
            continue
        harness = cli_harness or task.harness
        if not HARNESSES[harness].attach_images:
            supported = ", ".join(
                name for name, item in HARNESSES.items() if item.attach_images
            )
            raise ValueError(
                f"task {task.id!r}: files.attach_images is not supported by the "
                f"{HARNESSES[harness].label} Harness; use {supported}"
            )
        if harness == "opencode" and not args.opencode_vision:
            raise ValueError(
                f"task {task.id!r}: files.attach_images requires --opencode-vision"
            )
        if harness == "pi" and args.pi_input_modalities == "text":
            raise ValueError(
                f"task {task.id!r}: files.attach_images requires --pi-input-modalities text,image"
            )

    _preflight_environment(requests, cli_harness, configs)
    missing_env = missing_grader_env(task.raw for task in tasks)
    if missing_env:
        raise ValueError(
            "benchmark graders need environment variables that are not set: "
            + "; ".join(missing_env)
        )

    return HarnessSetup(
        configs=configs,
        images=images,
        model_config=model_config,
        model_config_path=model_config_path,
        codex_config_path=codex_config_path,
        codex_config_digest=codex_config_digest,
        claude_settings_path=claude_settings_path,
        claude_settings_content=claude_settings_content,
        claude_settings_sha256=claude_settings_sha256,
        thinking_capabilities_path=thinking_capabilities_path,
        thinking_capabilities_digest=thinking_capabilities_digest,
        api_bases=api_bases,
        api_key_envs=api_key_envs,
    )


def build_harness_configurations(
    args: argparse.Namespace, setup: HarnessSetup, sandbox_run: Any
) -> dict[str, dict[str, object]]:
    configs = setup.configs
    images = setup.images
    api_bases = setup.api_bases
    api_key_envs = setup.api_key_envs
    harness_configurations: dict[str, dict[str, object]] = {}

    openclaw_config = configs["openclaw"]
    if openclaw_config is not None:
        harness_configurations["openclaw"] = {
            "docker_image": images["openclaw"],
            "container_user": args.container_user,
            "model_config_path": (
                str(setup.model_config_path) if setup.model_config_path else None
            ),
            "model_request_timeout_seconds": args.openclaw_model_request_timeout,
            "model_max_tokens": args.openclaw_max_tokens,
            "gateway_port_start": openclaw_config.gateway_port_start,
            "gateway_port_step": args.gateway_port_step,
            "network": args.network,
            "session_id": args.session_id,
            "search_provider": args.search_provider,
            "thinking": args.thinking,
            "keep_thinking": args.keep_thinking,
            "cache_trace": args.cache_trace,
            "wire_trace": args.wire_trace,
            "pass_env_names": list(openclaw_config.pass_env),
            "collect_paths": args.collect_path,
            **(
                {"capture_calls": True, "capture_debug": args.capture_debug}
                if openclaw_config.capture_calls
                else {}
            ),
            **(
                {"config_set": recorded_config_assignments(openclaw_config.config_set)}
                if openclaw_config.config_set
                else {}
            ),
        }

    codex_config = configs["codex"]
    if codex_config is not None:
        codex_api_base = api_bases["codex"]
        harness_configurations["codex"] = {
            "docker_image": images["codex"],
            "container_user": args.container_user,
            "config_path": (
                str(setup.codex_config_path) if setup.codex_config_path else None
            ),
            "config_sha256": setup.codex_config_digest,
            "network": args.network,
            "api_base": redact_api_base(codex_api_base) if codex_api_base else None,
            "api_key_env": api_key_envs["codex"],
            "reasoning_effort": args.thinking,
            "reasoning_summary": args.reasoning_summary,
            "wire_trace": args.wire_trace,
            "keep_thinking": args.keep_thinking,
            "web_search": args.codex_web_search,
            "supports_websockets": args.codex_websockets,
            **(
                {"responses_tool_images_as_user": True}
                if codex_config.responses_tool_images_as_user
                else {}
            ),
            **({"chat_completions": True} if codex_config.chat_completions else {}),
            "pass_env_names": list(codex_config.pass_env),
            **(
                {"capture_calls": True, "capture_debug": args.capture_debug}
                if codex_config.capture_calls
                else {}
            ),
            **(
                {"config_set": recorded_config_assignments(codex_config.config_set)}
                if codex_config.config_set
                else {}
            ),
            **(
                {"responses_image_detail": codex_config.responses_image_detail}
                if codex_config.responses_image_detail
                else {}
            ),
        }

    claude_code_config = configs["claude-code"]
    if claude_code_config is not None:
        claude_api_base = api_bases["claude-code"]
        harness_configurations["claude-code"] = {
            "docker_image": images["claude-code"],
            "container_user": args.container_user,
            "settings_path": (
                str(setup.claude_settings_path) if setup.claude_settings_path else None
            ),
            "config_sha256": setup.claude_settings_sha256,
            "network": args.network,
            "api_base": (
                redact_anthropic_base_url(claude_api_base) if claude_api_base else None
            ),
            "api_key_env": api_key_envs["claude-code"],
            "reasoning_effort": args.thinking,
            "max_thinking_tokens": args.claude_max_thinking_tokens,
            "max_turns": args.claude_max_turns,
            "permission_mode": args.claude_permission_mode,
            "wire_trace": args.wire_trace,
            "keep_thinking": args.keep_thinking,
            "pass_env_names": list(claude_code_config.pass_env),
            **(
                {"capture_calls": True, "capture_debug": args.capture_debug}
                if claude_code_config.capture_calls
                else {}
            ),
            **({"unique_tool_ids": True} if claude_code_config.unique_tool_ids else {}),
            **({"pdf_as_images": True} if claude_code_config.pdf_as_images else {}),
            **(
                {"chat_completions": True}
                if claude_code_config.chat_completions
                else {}
            ),
        }

    hermes_config = configs["hermes"]
    if hermes_config is not None:
        harness_configurations["hermes"] = {
            "docker_image": images["hermes"],
            "container_user": args.container_user,
            "network": args.network,
            "api_base": api_bases["hermes"] or "https://api.openai.com/v1",
            "api_key_env": api_key_envs["hermes"],
            "reasoning_effort": args.thinking,
            "max_iterations": args.hermes_max_iterations,
            "hermes_revision": HERMES_REVISION,
            "context_length": args.hermes_context_length,
            "vision": args.hermes_vision,
            "toolsets": hermes_config.enabled_toolsets,
            "api_mode": "chat_completions",
            **(
                {"disabled_toolsets": hermes_config.disabled_toolsets}
                if hermes_config.toolsets is not None
                else {}
            ),
            "wire_trace": args.wire_trace,
            "keep_thinking": args.keep_thinking,
            "pass_env_names": list(hermes_config.pass_env),
            **(
                {"capture_calls": True, "capture_debug": args.capture_debug}
                if hermes_config.capture_calls
                else {}
            ),
        }

    opencode_config = configs["opencode"]
    if opencode_config is not None:
        harness_configurations["opencode"] = {
            "docker_image": images["opencode"],
            "container_user": args.container_user,
            "network": args.network,
            "api_base": api_bases["opencode"] or "https://api.openai.com/v1",
            "api_key_env": api_key_envs["opencode"],
            "reasoning_effort": args.thinking,
            "max_steps": args.opencode_max_steps,
            "opencode_version": OPENCODE_VERSION,
            "context_length": args.opencode_context_length,
            "output_tokens": args.opencode_output_tokens,
            "vision": args.opencode_vision,
            **opencode_config.tool_selection,
            "api_mode": "chat_completions",
            "wire_trace": args.wire_trace,
            "keep_thinking": args.keep_thinking,
            "pass_env_names": list(opencode_config.pass_env),
            "compaction": {"auto": False, "prune": False},
            **(
                {
                    "capture_calls": True,
                    "capture_debug": args.capture_debug,
                    "subagent_profile_policy": "explore-tool-restrictions/v1",
                }
                if opencode_config.capture_calls
                else {}
            ),
        }

    deepseek_config = configs["deepseek"]
    if deepseek_config is not None:
        harness_configurations["deepseek"] = {
            **(
                sandbox_run.target.provenance()
                if sandbox_run is not None
                else {
                    "docker_image": images["deepseek"],
                    "container_user": args.container_user,
                    "network": args.network,
                }
            ),
            "api_base": api_bases["deepseek"] or "https://api.deepseek.com",
            "api_key_env": api_key_envs["deepseek"],
            "reasoning_effort": args.thinking,
            "max_steps": args.deepseek_max_steps,
            "deepseek_version": DEEPSEEK_VERSION,
            "context_length": args.deepseek_context_length,
            "output_tokens": args.deepseek_output_tokens,
            "tools": deepseek_config.tools,
            "api_mode": "chat_completions",
            "wire_trace": args.wire_trace,
            "keep_thinking": args.keep_thinking,
            "pass_env_names": list(deepseek_config.pass_env),
            "profile": deepseek_config.profile,
            "retry_count": 0,
            **(
                {"permission_mode": deepseek_config.native_permission_mode}
                if deepseek_config.native_permission_mode
                else {}
            ),
            **(
                {"input_modalities": list(deepseek_config.input_modalities)}
                if deepseek_config.input_modalities != ("text",)
                else {}
            ),
            **(
                {"capture_calls": True, "capture_debug": args.capture_debug}
                if deepseek_config.capture_calls
                else {}
            ),
        }

    pi_config = configs["pi"]
    if pi_config is not None:
        harness_configurations["pi"] = {
            "docker_image": images["pi"],
            "container_user": args.container_user,
            "network": args.network,
            "api_base": api_bases["pi"] or "https://api.openai.com/v1",
            "api_key_env": api_key_envs["pi"],
            "reasoning_effort": pi_config.reasoning_effort,
            "max_steps": args.pi_max_steps,
            "pi_version": PI_VERSION,
            "context_length": args.pi_context_length,
            "output_tokens": args.pi_output_tokens,
            "input_modalities": list(pi_config.input_modalities),
            "tools": list(PI_TOOLS),
            "api_mode": "chat_completions",
            "wire_trace": args.wire_trace,
            "keep_thinking": args.keep_thinking,
            "pass_env_names": list(pi_config.pass_env),
            "compaction": {"auto": False},
            "retry_count": 0,
            **(
                {"capture_calls": True, "capture_debug": args.capture_debug}
                if pi_config.capture_calls
                else {}
            ),
        }

    stirrup_config = configs["stirrup"]
    if stirrup_config is not None:
        harness_configurations["stirrup"] = {
            **(
                sandbox_run.target.provenance()
                if sandbox_run is not None
                else {
                    "docker_image": images["stirrup"],
                    "container_user": args.container_user,
                    "network": args.network,
                }
            ),
            "api_base": api_bases["stirrup"] or "https://api.openai.com/v1",
            "api_key_env": api_key_envs["stirrup"],
            "reasoning_effort": args.thinking,
            "max_steps": args.stirrup_max_steps,
            "context_length": args.stirrup_context_length,
            "output_tokens": args.stirrup_output_tokens,
            "command_timeout": args.stirrup_command_timeout,
            "temperature": args.stirrup_temperature,
            "top_p": args.stirrup_top_p,
            "seed": args.stirrup_seed,
            "request_fields": [list(item) for item in args.stirrup_request_set],
            "tools": list(STIRRUP_TOOLS),
            "api_mode": "chat_completions",
            "wire_trace": args.wire_trace,
            "keep_thinking": args.keep_thinking,
            "pass_env_names": list(stirrup_config.pass_env),
            **(
                {"capture_calls": True, "capture_debug": args.capture_debug}
                if stirrup_config.capture_calls
                else {}
            ),
        }

    return harness_configurations
