from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roller.harness_registry import convert_native_trace, extract_native_usage
from roller.harnesses.bridge import (
    BridgeHarness,
    BridgeSession,
    positive_int,
    validate_bridge_config,
)
from roller.harnesses.container import PROXY_ENV_NAMES as COMMON_ENV_NAMES
from roller.thinking import normalize_thinking, resolve_thinking
from roller.usage import ZERO_USAGE

HERMES_REVISION = "8fc0374447313e276f6b9a2764bfc750190c966a"
DEFAULT_HERMES_IMAGE = "roller-hermes:8fc037444731"
RUNNER_PATH = Path(__file__).resolve().parents[1] / "assets" / "hermes" / "runner.py"
STATE = "/roller/hermes-state"
RUNNER = "/roller/hermes_runner.py"
NATIVE_TOOLSETS = "native"


def hermes_effort(value: str | None) -> str | None:
    value = normalize_thinking(value, field="Hermes reasoning effort")
    if value not in {None, "off", "minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError(
            "Hermes reasoning effort must be off|minimal|low|medium|high|xhigh; provider support is required"
        )
    return value


@dataclass(frozen=True)
class HermesConfig:
    docker_image: str = DEFAULT_HERMES_IMAGE
    container_user: str | None = "0"
    network: str | None = "host"
    workspace_container_path: str = "/roller/workspace"
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    pass_env: tuple[str, ...] = COMMON_ENV_NAMES
    reasoning_effort: str | None = None
    max_iterations: int = 12
    context_length: int | None = None
    vision: bool = True
    toolsets: tuple[str, ...] | None = None
    capture_calls: bool = False
    token_ids: bool = False
    capture_debug: str = "on-error"
    wire_trace_mode: str = "actf"
    keep_thinking: bool = True
    keep_container_on_error: bool = False
    docker_binary: str = "docker"
    preflight_image_digest: str | None = None

    @property
    def enabled_toolsets(self) -> list[str] | None:
        """Toolsets passed to Hermes; ``None`` leaves discovery to the native registry."""

        if self.toolsets is None:
            return [
                "terminal",
                "file",
                *(["delegation"] if self.capture_calls else []),
                *(["vision"] if self.vision else []),
            ]
        return None if self.toolsets == (NATIVE_TOOLSETS,) else list(self.toolsets)

    @property
    def disabled_toolsets(self) -> list[str]:
        """Toolsets withheld from an explicit selection; sub-agents run only under call capture."""

        if self.toolsets is None:
            return []
        return [
            *([] if self.capture_calls else ["delegation"]),
            *([] if self.vision else ["vision"]),
        ]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reasoning_effort", hermes_effort(self.reasoning_effort)
        )
        validate_bridge_config(self, "Hermes")
        if self.toolsets is not None:
            if not self.toolsets or any(
                not isinstance(name, str) or not name.strip() for name in self.toolsets
            ):
                raise ValueError("Hermes toolsets must contain non-empty names")
            if NATIVE_TOOLSETS in self.toolsets and len(self.toolsets) > 1:
                raise ValueError(
                    "Hermes toolset 'native' cannot be combined with named toolsets"
                )
            if "delegation" in self.toolsets and not self.capture_calls:
                raise ValueError("Hermes delegation toolset requires --capture-calls")
        if not positive_int(self.max_iterations):
            raise ValueError("Hermes max iterations must be a positive integer")
        if self.context_length is not None and not (
            positive_int(self.context_length) and self.context_length >= 64000
        ):
            raise ValueError("Hermes context length must be at least 64000 tokens")


class HermesSession(BridgeSession):
    label = "Hermes"
    state_dir = STATE
    runner = RUNNER
    runner_source = RUNNER_PATH
    raw_trace_schema = "roller.hermes-events/v1"
    python = "python"
    version_check = "revision and hook check"

    def effort(self) -> str | None:
        return hermes_effort(
            resolve_thinking(
                self.request.task.thinking, self.harness.config.reasoning_effort
            )
        )

    def harness_environment(self, api_base: str) -> dict[str, str]:
        return dict(
            OPENAI_BASE_URL=api_base,
            HERMES_HOME=self.home,
            HERMES_DISABLE_LAZY_INSTALLS="1",
            TERMINAL_ENV="local",
            TERMINAL_CWD=self.harness.config.workspace_container_path,
        )

    def turn_options(self, timeout: float) -> dict[str, Any]:
        # Context is present as regular workspace files for every role; Hermes'
        # implicit host/profile discovery is disabled in the bridge.
        config = self.harness.config
        return {
            "max_iterations": config.max_iterations,
            "context_length": config.context_length,
            "vision": config.vision,
            **(
                {
                    "toolsets": config.enabled_toolsets,
                    "disabled_toolsets": config.disabled_toolsets,
                }
                if config.toolsets is not None
                else {}
            ),
        }

    def convert_trace(self, trace: Path, destination: Path) -> dict[str, Any]:
        return convert_native_trace(
            "hermes",
            trace,
            destination,
            keep_thinking=self.harness.config.keep_thinking,
        )

    def extract_usage(self, trace: Path) -> dict[str, int | float]:
        return extract_native_usage("hermes", trace)

    def version_fields(self) -> dict[str, Any]:
        return {"hermes_revision": HERMES_REVISION, "hermes_version": self.version}

    def settings(self) -> dict[str, Any]:
        config = self.harness.config
        return {
            "max_iterations": config.max_iterations,
            "context_length": config.context_length,
            "vision": config.vision,
            "toolsets": config.enabled_toolsets,
            **(
                {"disabled_toolsets": config.disabled_toolsets}
                if config.toolsets is not None
                else {}
            ),
        }

    def collect_native_evidence(self, usage: dict[str, int | float]) -> None:
        assert self.capture is not None
        self.capture.copy_native(
            self.harness.runtime,
            self.container,
            STATE + "/capture-native/.",
            "children",
        )
        children = self.capture.recorder.directory / "native" / "children"
        if not children.is_dir():
            return
        for child in children.glob("*.jsonl"):
            child_trace = children.parent / ("child-" + child.name)
            child.rename(child_trace)
            # Count child model responses once, independently of the root
            # ACTF, before compact capture removes raw logs.
            try:
                child_usage = extract_native_usage("hermes", child_trace)
            except Exception as exc:
                self.capture.errors.append(
                    "Hermes child usage extraction failed: " + self.redact(str(exc))
                )
            else:
                for key in ZERO_USAGE:
                    usage[key] += child_usage[key]
        children.rmdir()


class HermesHarness(BridgeHarness):
    """Run the pinned Hermes Python agent with local tools inside task Docker."""

    name = "hermes"
    session_class = HermesSession


def _hermes_cli_options():
    import argparse

    from roller.option_specs import OptionSpec

    return (
        OptionSpec(
            "--hermes-toolset",
            yaml_key="toolsets",
            yaml_kind="repeat",
            action="append",
            metavar="NAME",
            help="Hermes toolset to enable instead of terminal/file/vision; repeat to select several, or pass 'native' for native discovery",
        ),
        OptionSpec(
            "--hermes-max-iterations",
            yaml_key="max_iterations",
            type=int,
            default=12,
            help="maximum model iterations per Hermes user turn (default: 12)",
        ),
        OptionSpec(
            "--hermes-context-length",
            yaml_key="context_length",
            type=int,
            help="override Hermes model context window (at least 64000 tokens)",
        ),
        OptionSpec(
            "--hermes-vision",
            yaml_key="vision",
            yaml_kind="bool",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="enable Hermes vision_analyze and declare native image support (default: enabled)",
        ),
    )


CLI_OPTIONS = _hermes_cli_options()
