from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roller.deepseek_actf import convert_deepseek_trace, extract_deepseek_usage
from roller.harnesses.bridge import (
    BridgeHarness,
    BridgeSession,
    validate_bridge_config,
    validate_step_limits,
)
from roller.harnesses.container import PROXY_ENV_NAMES as COMMON_ENV_NAMES
from roller.sandbox import SandboxRun
from roller.thinking import normalize_thinking, resolve_thinking

DEEPSEEK_VERSION = "0.1.5rc1"
DEFAULT_DEEPSEEK_IMAGE = "roller-deepseek:0.1.5rc1"
RUNNER_PATH = Path(__file__).resolve().parents[1] / "assets" / "deepseek" / "runner.py"
STATE = "/roller/deepseek-state"
RUNNER = "/roller/deepseek_runner.py"
PLUGIN = "/roller/deepseek_journal.mjs"
DEEPSEEK_TOOLS = ("bash",)
DEEPSEEK_PERMISSION_MODES = ("read-only", "workspace-write", "danger-full-access")


def deepseek_effort(value: str | None) -> str | None:
    value = normalize_thinking(value, field="DeepSeek reasoning effort")
    if value not in {None, "off", "low", "high", "max"}:
        raise ValueError(
            "DeepSeek reasoning effort must be off|low|high|max; provider support is required"
        )
    return value


@dataclass(frozen=True)
class DeepSeekConfig:
    docker_image: str = DEFAULT_DEEPSEEK_IMAGE
    container_user: str | None = "0"
    network: str | None = "host"
    workspace_container_path: str = "/roller/workspace"
    api_base: str | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    pass_env: tuple[str, ...] = COMMON_ENV_NAMES
    reasoning_effort: str | None = None
    profile: str = "sdk-minimal"
    permission_mode: str | None = None
    max_steps: int = 12
    context_length: int = 128000
    output_tokens: int = 8192
    input_modalities: tuple[str, ...] = ("text",)
    capture_calls: bool = False
    capture_debug: str = "on-error"
    wire_trace_mode: str = "actf"
    keep_thinking: bool = True
    keep_container_on_error: bool = False
    docker_binary: str = "docker"
    preflight_image_digest: str | None = None
    remote_sandbox: bool = False
    sandbox_run: SandboxRun | None = None

    @property
    def native_permission_mode(self) -> str | None:
        """Permission mode exported to the SDK.

        Profiles other than ``sdk-minimal`` default to a file sandbox that has
        no backend inside the task container, so every shell command would be
        refused. The container is the sandbox; run them unconfined in it.
        """

        if self.permission_mode is not None or self.profile == "sdk-minimal":
            return self.permission_mode
        return "danger-full-access"

    @property
    def tools(self) -> list[str] | None:
        """The fixed tool list of ``sdk-minimal``; other profiles discover tools natively."""

        return list(DEEPSEEK_TOOLS) if self.profile == "sdk-minimal" else None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reasoning_effort", deepseek_effort(self.reasoning_effort)
        )
        validate_bridge_config(self, "DeepSeek")
        if not isinstance(self.profile, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.profile
        ):
            raise ValueError("DeepSeek profile must be a native profile name")
        if self.input_modalities not in (("text",), ("text", "image")):
            raise ValueError("DeepSeek input modalities must be text or text,image")
        if (
            self.permission_mode is not None
            and self.permission_mode not in DEEPSEEK_PERMISSION_MODES
        ):
            raise ValueError(
                "DeepSeek permission mode must be read-only|workspace-write|danger-full-access"
            )
        validate_step_limits(self, "DeepSeek")


class DeepSeekSession(BridgeSession):
    label = "DeepSeek"
    state_dir = STATE
    runner = RUNNER
    runner_source = RUNNER_PATH
    raw_trace_schema = "roller.deepseek-events/v1"
    api_key_name = "DEEPSEEK_API_KEY"
    default_api_base = "https://api.deepseek.com"
    assets = (
        ("journal.mjs", PLUGIN, "journal plugin copy"),
        ("roller.patch.yml", STATE + "/roller.patch.yml", "profile patch copy"),
    )
    redact_trace = True

    def effort(self) -> str | None:
        return deepseek_effort(
            resolve_thinking(
                self.request.task.thinking, self.harness.config.reasoning_effort
            )
        )

    def harness_environment(self, api_base: str) -> dict[str, str]:
        config = self.harness.config
        return dict(
            DEEPSEEK_BASE_URL=api_base,
            HOME=self.home,
            DSH_HOME=self.home,
            DSH_RUNTIME_MODE="exe",
            ROLLER_DEEPSEEK_JOURNAL=STATE + "/trace.jsonl",
            **(
                {"DSH_PERMISSION_MODE": config.native_permission_mode}
                if config.native_permission_mode
                else {}
            ),
        )

    def turn_options(self, timeout: float) -> dict[str, Any]:
        # Context is present as explicit workspace files; the selected profile
        # owns tool discovery inside the isolated DSH_HOME.
        config = self.harness.config
        return {
            "turn": self.turns,
            "profile": config.profile,
            "max_steps": config.max_steps,
            "context_length": config.context_length,
            "output_tokens": config.output_tokens,
            "input_modalities": list(config.input_modalities),
            "tool_env_names": sorted(
                set((*config.pass_env, *self.request.task.env_names))
            ),
        }

    def convert_trace(self, trace: Path, destination: Path) -> dict[str, Any]:
        return convert_deepseek_trace(
            trace, destination, keep_thinking=self.harness.config.keep_thinking
        )

    def extract_usage(self, trace: Path) -> dict[str, int | float]:
        return extract_deepseek_usage(trace)

    def version_fields(self) -> dict[str, Any]:
        config = self.harness.config
        return {
            "deepseek_version": self.version,
            "profile": config.profile,
            **(
                {"permission_mode": config.native_permission_mode}
                if config.native_permission_mode
                else {}
            ),
            **(
                {"input_modalities": list(config.input_modalities)}
                if config.input_modalities != ("text",)
                else {}
            ),
        }

    def settings(self) -> dict[str, Any]:
        config = self.harness.config
        return {
            "max_steps": config.max_steps,
            "context_length": config.context_length,
            "output_tokens": config.output_tokens,
            "tools": config.tools,
        }


class DeepSeekHarness(BridgeHarness):
    """Run an official DeepSeek SDK profile in a task Docker container."""

    name = "deepseek"
    session_class = DeepSeekSession
