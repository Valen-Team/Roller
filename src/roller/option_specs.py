"""Declarative option specifications shared by the CLI parser and bench run configs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from roller.thinking import THINKING_LEVELS, normalize_thinking

GRADER_RETRY_MAX_SECONDS = 600.0


def parse_request_field(text: str) -> tuple[str, object]:
    key, separator, value = text.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=JSON, got {text!r}")
    try:
        return key.strip(), json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{key}: {exc}") from None


@dataclass(frozen=True)
class OptionSpec:
    """Single source of truth for one ``roller collect`` / ``bench run`` option."""

    flag: str
    yaml_key: str | None = None
    yaml_kind: str = "value"  # "value" | "bool" | "switch" | "repeat"
    dest: str | None = None
    aliases: tuple[str, ...] = ()
    action: Any = None
    type: Any = None
    choices: tuple[Any, ...] | None = None
    default: Any = None
    metavar: str | None = None
    help: str | None = None

    @property
    def resolved_dest(self) -> str:
        return self.dest or self.flag.lstrip("-").replace("-", "_")

    def default_value(self) -> Any:
        if self.action == "append":
            return [] if self.default == () else self.default
        if self.action == "store_true" and self.default is None:
            return False
        return self.default

    def add_to(self, target: Any) -> None:
        kwargs: dict[str, Any] = {}
        if self.dest is not None:
            kwargs["dest"] = self.dest
        if self.action is not None:
            kwargs["action"] = self.action
        if self.type is not None:
            kwargs["type"] = self.type
        if self.choices is not None:
            kwargs["choices"] = self.choices
        if self.action == "append":
            if self.default == ():
                kwargs["default"] = []
            elif self.default is not None:
                kwargs["default"] = self.default
        elif self.default is not None or self.action is argparse.BooleanOptionalAction:
            kwargs["default"] = self.default
        if self.metavar is not None:
            kwargs["metavar"] = self.metavar
        if self.help is not None:
            kwargs["help"] = self.help
        target.add_argument(self.flag, *self.aliases, **kwargs)

    def coerce_yaml(self, raw: Any, path: str) -> Any:
        if self.yaml_kind in {"bool", "switch"}:
            if not isinstance(raw, bool):
                raise ValueError(f"{path} must be a boolean")
            return raw
        if self.yaml_kind == "repeat":
            if not isinstance(raw, list) or not all(
                isinstance(item, str) and item for item in raw
            ):
                raise ValueError(f"{path} must be a string list")
            if self.type is None:
                return list(raw)
            coerced: list[Any] = []
            for item in raw:
                try:
                    coerced.append(self.type(item))
                except (argparse.ArgumentTypeError, ValueError) as exc:
                    raise ValueError(f"{path}: {exc}") from None
            return coerced
        if raw is None:
            return self.default_value()
        if isinstance(raw, bool):
            raise ValueError(f"{path} must not be a boolean")
        text = str(raw)
        if self.type is not None:
            try:
                value = self.type(text)
            except (argparse.ArgumentTypeError, ValueError) as exc:
                raise ValueError(f"{path}: {exc}") from None
        else:
            value = text
        if self.choices is not None and value not in self.choices:
            allowed = ", ".join(str(item) for item in self.choices)
            raise ValueError(f"{path} must be one of {allowed}; got {value!r}")
        return value


def register_options(target: Any, options: Iterable[OptionSpec]) -> None:
    for option in options:
        option.add_to(target)


def default_namespace(options: Iterable[OptionSpec]) -> dict[str, Any]:
    return {option.resolved_dest: option.default_value() for option in options}


BENCH_RUNS_PER_TASK_OPTION = OptionSpec(
    "--bench-runs-per-task",
    yaml_key="runs_per_task",
    type=int,
    default=1,
    help="duplicate each selected benchmark task before collection (default: 1)",
)

BENCH_SELECTION_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec(
        "--bench-root",
        help="path to the checked-out benchmark (required with --bench)",
    ),
    OptionSpec(
        "--bench-variant",
        help="benchmark variant/condition (for example with-skill or no-skill)",
    ),
    OptionSpec(
        "--bench-category",
        action="append",
        default=(),
        help="restrict benchmark categories (repeatable)",
    ),
    OptionSpec(
        "--bench-task-id",
        action="append",
        default=(),
        help="restrict benchmark task ids (repeatable)",
    ),
    OptionSpec(
        "--bench-timeout-multiplier",
        type=float,
        default=1.0,
        help="multiply benchmark task timeouts (default: 1.0)",
    ),
    BENCH_RUNS_PER_TASK_OPTION,
)

EXECUTION_CORE_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec(
        "--timeout",
        yaml_key="timeout",
        type=int,
        default=600,
        help="per-trajectory timeout in seconds",
    ),
    OptionSpec("--parallel", yaml_key="parallel", type=int, default=1),
    OptionSpec(
        "--launch-interval",
        yaml_key="launch_interval",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "minimum seconds between two rollout launches, so --parallel workers "
            "do not all start at once (default: 0)"
        ),
    ),
    OptionSpec(
        "--grader-parallel",
        yaml_key="grader_parallel",
        type=int,
        help="maximum concurrent benchmark graders (default: --parallel)",
    ),
    OptionSpec(
        "--grader-retries",
        yaml_key="grader_retries",
        type=int,
        default=2,
        metavar="N",
        help=(
            "run a failed benchmark grader again in the same task container, up "
            "to N times, before the sample is left unscored; a score is never "
            "retried, nor is a grader that failed after more than "
            f"{GRADER_RETRY_MAX_SECONDS / 60:g} minutes (default: 2)"
        ),
    ),
    OptionSpec(
        "--samples",
        yaml_key="samples",
        type=int,
        default=None,
        help=(
            "independent runs per task/model pair "
            "(default: 3 for SkillsBench, 1 otherwise)"
        ),
    ),
    OptionSpec("--retries", yaml_key="retries", type=int, default=0),
    OptionSpec(
        "--retry-backoff",
        yaml_key="retry_backoff",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "wait before the first retry of a sample, doubled before each "
            "further retry (default: 0, retry immediately)"
        ),
    ),
    OptionSpec(
        "--max-consecutive-errors",
        yaml_key="max_consecutive_errors",
        type=int,
        default=10,
        metavar="N",
        help=(
            "stop the run when N rollouts in a row end in error after their "
            "retries; timeouts do not count (default: 10, 0 never stops)"
        ),
    ),
    OptionSpec(
        "--start-index",
        yaml_key="start_index",
        type=int,
        default=0,
        help="first normalized task index",
    ),
    OptionSpec("--limit", yaml_key="limit", type=int, help="maximum number of tasks"),
    OptionSpec(
        "--output",
        yaml_key="output",
        default="runs",
        help="root directory for collections",
    ),
    OptionSpec(
        "--run-name",
        yaml_key="run_name",
        help="run directory name; defaults to a UTC timestamp",
    ),
    OptionSpec(
        "--resume",
        yaml_key="resume",
        yaml_kind="switch",
        action="store_true",
        help="skip completed task/model pairs",
    ),
    OptionSpec(
        "--resume-verify-outputs",
        yaml_key="resume_verify_outputs",
        yaml_kind="switch",
        action="store_true",
        help=(
            "with --resume, reread recorded output files and rerun a sample whose "
            "content no longer matches its SHA-256 (default: check path and size only)"
        ),
    ),
    OptionSpec(
        "--resume-unscored",
        yaml_key="resume_unscored",
        yaml_kind="switch",
        action="store_true",
        help=(
            "with --resume, also rerun benchmark samples whose agent finished but "
            "whose grading failed; the agent runs again (default: keep them)"
        ),
    ),
    OptionSpec(
        "--max-input-file-bytes",
        yaml_key="max_input_file_bytes",
        type=int,
        default=10 * 1024 * 1024 * 1024,
        help="maximum size of one snapshotted input/context file",
    ),
    OptionSpec(
        "--max-task-input-bytes",
        yaml_key="max_task_input_bytes",
        type=int,
        default=50 * 1024 * 1024 * 1024,
        help="maximum total input bytes for one task",
    ),
)

EXECUTION_RUNTIME_OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec(
        "--container-user",
        yaml_key="container_user",
        default="0",
        help="Docker user for the disposable task container (default: root); use 'image' for the image default",
    ),
    OptionSpec("--docker-binary", yaml_key="docker_binary", default="docker"),
    OptionSpec(
        "--sandbox-provider",
        yaml_key="sandbox_provider",
        default="docker",
        help="task execution provider (default: docker; remote providers are installed separately)",
    ),
    OptionSpec(
        "--sandbox-template",
        yaml_key="sandbox_template",
        help="prebuilt environment used by the remote sandbox provider",
    ),
    OptionSpec(
        "--rebuild-task-images",
        yaml_key="rebuild_task_images",
        yaml_kind="switch",
        action="store_true",
        help="rebuild task-specific benchmark images even when their tags exist locally",
    ),
    OptionSpec("--network", yaml_key="network", default="host"),
    OptionSpec(
        "--capture-debug",
        yaml_key="capture_debug",
        choices=("off", "on-error", "full"),
        default="on-error",
        help="retain call capture debug evidence: on-error (default), off, or full",
    ),
    OptionSpec(
        "--capture-calls",
        yaml_key="capture_calls",
        yaml_kind="switch",
        action="store_true",
        help="save full model calls and native subagent relations for all harness participants",
    ),
    OptionSpec(
        "--token-ids",
        yaml_key="token_ids",
        yaml_kind="switch",
        action="store_true",
        help=(
            "save the prompt and completion token ids of every model call with "
            "the logprobs of the sampled tokens in token_ids.jsonl; needs "
            "--capture-calls and a Chat Completions endpoint that returns token "
            "ids, such as vLLM; Claude Code also needs --claude-chat-completions "
            "and Codex --codex-chat-completions"
        ),
    ),
    OptionSpec(
        "--training-files-only",
        yaml_key="training_files_only",
        yaml_kind="switch",
        action="store_true",
        help=(
            "once a sample has finished and been graded, keep only trace_SFT.json, "
            "sft_segments/, token_ids.jsonl, info.json and score.json, with each "
            "image stored once in blobs/; also removes what --wire-trace full and "
            "--capture-calls keep. The run-level sft.jsonl is left to roller "
            "export --format sft. Such samples cannot be viewed, exported to "
            "Hugging Face or ATIF, or have their SFT rebuilt"
        ),
    ),
    OptionSpec(
        "--thinking",
        type=normalize_thinking,
        choices=THINKING_LEVELS,
        metavar="LEVEL",
        help=(
            "Harness reasoning/thinking level; 'none' aliases 'off'. Each Harness "
            "applies its own validation and effort mapping"
        ),
    ),
    OptionSpec(
        "--thinking-capabilities",
        yaml_key="thinking_capabilities",
        metavar="PATH",
        help=(
            "optional versioned JSON capability declarations keyed by harness, "
            "model, and endpoint; unsupported efforts fail before rollout"
        ),
    ),
    OptionSpec(
        "--pass-env",
        yaml_key="pass_env",
        yaml_kind="repeat",
        action="append",
        default=(),
        metavar="NAME",
        help="additional host environment variable to pass into the Harness container",
    ),
    OptionSpec(
        "--wire-trace",
        yaml_key="wire_trace",
        choices=("off", "actf", "full"),
        default="actf",
        help=(
            "wire-level recording through a local reverse proxy: 'actf' (default) embeds "
            "the tools schema into trace_ACTF.json and keeps no extra file, 'full' "
            "additionally saves wire_trace.jsonl with every request body, 'off' disables "
            "(OpenClaw uses model-config baseUrl; Codex uses --api-base)"
        ),
    ),
    OptionSpec(
        "--keep-thinking",
        yaml_key="keep_thinking",
        yaml_kind="bool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep visible reasoning in ACTF/SFT. For OpenClaw, --no-keep-thinking also "
            "strips reasoning from request history at the wire layer; for Codex and "
            "Claude Code/Hermes/OpenCode it controls export and leaves native raw/resume context unchanged"
        ),
    ),
    OptionSpec(
        "--keep-container-on-error",
        yaml_key="keep_container_on_error",
        yaml_kind="switch",
        action="store_true",
    ),
)

EXECUTION_OPTIONS: tuple[OptionSpec, ...] = (
    *EXECUTION_CORE_OPTIONS,
    *(opt for opt in EXECUTION_RUNTIME_OPTIONS if opt.yaml_key is not None),
)

# ``collect`` options mapped directly by ``bench run`` top-level YAML sections
# (``benchmark``, ``model``, ``harness``) rather than ``execution`` / ``extra``.
BENCH_STRUCTURED_FLAGS = frozenset(
    {
        "--bench",
        "--bench-root",
        "--bench-variant",
        "--bench-category",
        "--bench-task-id",
        "--bench-timeout-multiplier",
        "--model",
        "--model-label",
        "--harness",
        "--docker-image",
        "--api-base",
        "--api-key-env",
        "--thinking",
    }
)

# ``collect`` options a run configuration does not set.
COLLECT_ONLY_FLAGS = frozenset(
    {
        "--help",
        # ``bench run --dry-run`` toggles dry_run directly.
        "--dry-run",
        # A run configuration always selects tasks through ``benchmark``.
        "--tasks",
        "--prompt-key",
        "--id-key",
        "--workspace",
        # ``--bench`` always honours task-declared timeouts.
        "--task-timeouts",
        # Logging is configured from the ``bench run`` arguments before the file is read.
        "--log-level",
        # Per-Harness overrides for mixed-Harness task files. A run configuration has one
        # Harness, set through ``harness.image``, ``model.base_url`` and ``model.api_key_env``.
        "--openclaw-image",
        "--codex-image",
        "--claude-code-image",
        "--hermes-image",
        "--opencode-image",
        "--deepseek-image",
        "--pi-image",
        "--stirrup-image",
        "--codex-api-base",
        "--claude-api-base",
        "--hermes-api-base",
        "--opencode-api-base",
        "--deepseek-api-base",
        "--pi-api-base",
        "--stirrup-api-base",
        "--codex-api-key-env",
        "--claude-api-key-env",
        "--hermes-api-key-env",
        "--opencode-api-key-env",
        "--deepseek-api-key-env",
        "--pi-api-key-env",
        "--stirrup-api-key-env",
    }
)


def option_map(options: Sequence[OptionSpec]) -> dict[str, OptionSpec]:
    return {opt.yaml_key: opt for opt in options if opt.yaml_key is not None}
