"""Stirrup: the agent loop runs in the collecting process, the task container executes its code.

Stirrup's ``code_exec``, ``view_image`` and ``finish`` tools act on the
sample's Docker container or remote sandbox through one code backend. The
model client sends Chat Completions requests from this process, so a local
recording proxy sees them for Docker and remote sandboxes alike.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar

from roller.harness_registry import convert_native_trace, extract_native_usage
from roller.harnesses.base import TurnResult
from roller.harnesses.bridge import (
    BridgeHarness,
    BridgeSession,
    positive_int,
    validate_bridge_config,
)
from roller.harnesses.container import PROXY_ENV_NAMES as COMMON_ENV_NAMES
from roller.sandbox import SandboxRun
from roller.thinking import normalize_thinking, resolve_thinking

STIRRUP_VERSION = "0.2.1"
DEFAULT_STIRRUP_IMAGE = "python:3.13-slim"
STIRRUP_TOOLS = ("code_exec", "view_image", "finish")


def sample_seed(seed: int, task_id: str, sample_index: int) -> int:
    """A sample's seed, stable across retries and distinct across samples."""

    value = json.dumps([seed, task_id, sample_index]).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") % (2**31)


@dataclass(frozen=True)
class StirrupConfig:
    agent_in_container: ClassVar[bool] = False

    docker_image: str = DEFAULT_STIRRUP_IMAGE
    container_user: str | None = "0"
    network: str | None = "host"
    workspace_container_path: str = "/roller/workspace"
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    pass_env: tuple[str, ...] = COMMON_ENV_NAMES
    reasoning_effort: str | None = None
    max_steps: int = 30
    context_length: int = 128000
    output_tokens: int = 8192
    command_timeout: int = 300
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    # Further top-level request fields, in order, after Roller's own.
    request_fields: tuple[tuple[str, Any], ...] = ()
    request_timeout: float = 3600
    request_attempts: int = 3
    capture_calls: bool = False
    token_ids: bool = False
    capture_debug: str = "on-error"
    wire_trace_mode: str = "actf"
    keep_thinking: bool = True
    keep_container_on_error: bool = False
    docker_binary: str = "docker"
    preflight_image_digest: str | None = None
    remote_sandbox: bool = False
    sandbox_run: SandboxRun | None = None

    def __post_init__(self) -> None:
        validate_bridge_config(self, "Stirrup")
        for name in ("max_steps", "context_length", "output_tokens"):
            if not positive_int(getattr(self, name)):
                raise ValueError(f"Stirrup {name} must be a positive integer")
        if self.output_tokens > self.context_length:
            raise ValueError("Stirrup output tokens must not exceed context length")


class StirrupSession(BridgeSession):
    label = "Stirrup"
    state_dir = "/roller/stirrup-state"
    raw_trace_schema = "roller.stirrup-journal/v1"
    agent_in_container = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_base = ""
        self.stop_reason: str | None = None

    def effort(self) -> str | None:
        return normalize_thinking(
            resolve_thinking(
                self.request.task.thinking, self.harness.config.reasoning_effort
            )
        )

    def harness_environment(self, api_base: str) -> dict[str, str]:
        self.model_base = api_base  # the client's base; the container needs none
        return {}

    def request_body(self) -> dict[str, Any]:
        """Every request field except the conversation, in the order sent."""

        config, task = self.harness.config, self.request.task
        effort = self.effort()
        body: dict[str, Any] = {}
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.top_p is not None:
            body["top_p"] = config.top_p
        body["max_tokens"] = config.output_tokens
        if effort is not None:
            body["chat_template_kwargs"] = {"reasoning_effort": effort}
        body["model"] = self.request.model
        if config.seed is not None:
            body["seed"] = sample_seed(config.seed, task.id, self.request.sample_index)
        body.update(config.request_fields)
        return body

    def start(self) -> None:
        super().start()
        self.version = version("stirrup")

    def execute(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess:
        limit = min(timeout, self.remaining())
        if self.sandbox is None:
            runtime = self.harness.runtime
            return runtime.run(runtime.exec_args(self.container, argv), timeout=limit)
        return self.sandbox.run(argv, env=self.environment, timeout=limit)

    def send(self, message: str, timeout_seconds: float) -> TurnResult:
        if not self.container_started or self._result is not None:
            raise RuntimeError("Stirrup session is not active")
        if self.turns:
            raise ValueError("Stirrup runs one prompt per task")
        from roller.stirrup_agent import run_episode

        config = self.harness.config
        started = time.monotonic()
        deadline = started + min(timeout_seconds, self.remaining())

        def remaining() -> float:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("Stirrup turn timed out")
            return left

        self.turns += 1
        trace = self.output_dir / "rollout" / "trace.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        outcome = run_episode(
            journal_path=trace,
            session={
                "session_id": self.container,
                "model": self.request.model,
                "settings": self.settings(),
            },
            prompt=message,
            run=self.execute,
            upload=lambda source, target: self.upload(
                source, target, "Stirrup file write"
            ),
            workdir=config.workspace_container_path,
            command_timeout=config.command_timeout,
            max_turns=config.max_steps,
            client_options={
                "api_base": self.model_base,
                "api_key": os.environ[config.api_key_env],
                "body": self.request_body(),
                "context_window": config.context_length,
                "timeout": config.request_timeout,
                "attempts": config.request_attempts,
                "session_id": self.container if self.capture else None,
                "remaining": remaining,
            },
        )
        self.stop_reason = outcome["stop"]
        reason = (outcome.get("finish") or {}).get("reason")
        return TurnResult(
            reason or f"Stirrup stopped: {self.stop_reason}",
            time.monotonic() - started,
        )

    def fetch_trace(self, trace: Path) -> bool:
        return trace.is_file()  # written here, not in the container

    def convert_trace(self, trace: Path, destination: Path) -> dict[str, Any]:
        return convert_native_trace(
            "stirrup",
            trace,
            destination,
            keep_thinking=self.harness.config.keep_thinking,
        )

    def extract_usage(self, trace: Path) -> dict[str, int | float]:
        return extract_native_usage("stirrup", trace)

    def version_fields(self) -> dict[str, Any]:
        return {"stirrup_version": self.version, "api_mode": "chat_completions"}

    def settings(self) -> dict[str, Any]:
        config = self.harness.config
        body = {k: v for k, v in self.request_body().items() if k != "model"}
        return {
            "max_steps": config.max_steps,
            "context_length": config.context_length,
            "command_timeout": config.command_timeout,
            "tools": list(STIRRUP_TOOLS),
            "request_body": body,
        }

    def result_metadata(self) -> dict[str, Any]:
        return {"stop_reason": self.stop_reason}


class StirrupHarness(BridgeHarness):
    """Run Stirrup's agent here with the task container as its code backend."""

    name = "stirrup"
    session_class = StirrupSession


def _stirrup_cli_options():
    from roller.option_specs import OptionSpec, parse_request_field

    return (
        OptionSpec(
            "--stirrup-max-steps",
            yaml_key="max_steps",
            type=int,
            default=30,
            help="Stirrup max_turns: model turns per task, the finish turn included (default: 30)",
        ),
        OptionSpec(
            "--stirrup-context-length",
            yaml_key="context_length",
            type=int,
            default=128000,
            help=(
                "Stirrup model context window; history is summarized at 70%% of it "
                "(default: 128000 tokens)"
            ),
        ),
        OptionSpec(
            "--stirrup-output-tokens",
            yaml_key="output_tokens",
            type=int,
            default=8192,
            help="max_tokens of every Stirrup request (default: 8192)",
        ),
        OptionSpec(
            "--stirrup-command-timeout",
            yaml_key="command_timeout",
            type=int,
            default=300,
            help="seconds before a code_exec command is stopped (default: 300)",
        ),
        OptionSpec(
            "--stirrup-temperature",
            yaml_key="temperature",
            type=float,
            help="temperature of every request",
        ),
        OptionSpec(
            "--stirrup-top-p",
            yaml_key="top_p",
            type=float,
            help="top_p of every request",
        ),
        OptionSpec(
            "--stirrup-seed",
            yaml_key="seed",
            type=int,
            help=(
                "base seed; each sample's request seed is derived from it, the task ID "
                "and the sample index"
            ),
        ),
        OptionSpec(
            "--stirrup-request-set",
            yaml_key="request_set",
            yaml_kind="repeat",
            action="append",
            default=(),
            type=parse_request_field,
            metavar="KEY=JSON",
            help=(
                "add a top-level request field after Roller's own, such as n=1 or "
                "stream=false; repeatable, in order"
            ),
        ),
    )


CLI_OPTIONS = _stirrup_cli_options()
