from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roller import hf_dataset
from roller.atif import export_run
from roller.benches import (
    binding_from_task_row,
    format_report,
    get_bench,
    list_benches,
    mean_aggregate,
)
from roller.benches.grading import GradedHarness
from roller.bundles import BundlePolicy, TaskBundleStore
from roller.collector import CollectionConfig, Collector, check_retired_cot_samples
from roller.conversation import ConversationManager
from roller.harness_registry import HARNESS_NAMES, HARNESSES
from roller.harnesses.claude_code import (
    CLAUDE_PERMISSION_MODES,
    DEFAULT_CLAUDE_CODE_IMAGE,
    ClaudeCodeConfig,
    ClaudeCodeHarness,
    claude_effort,
    redact_anthropic_base_url,
)
from roller.harnesses.codex import (
    DEFAULT_CODEX_IMAGE,
    CodexConfig,
    CodexHarness,
    codex_effort,
    redact_api_base,
)
from roller.harnesses.config_overrides import (
    parse_config_assignment,
    recorded_config_assignments,
)
from roller.harnesses.container import PROXY_ENV_NAMES
from roller.harnesses.deepseek import (
    DEEPSEEK_PERMISSION_MODES,
    DEEPSEEK_VERSION,
    DEFAULT_DEEPSEEK_IMAGE,
    DeepSeekConfig,
    DeepSeekHarness,
    deepseek_effort,
)
from roller.harnesses.gateway_ports import (
    DEFAULT_GATEWAY_PORT,
    GatewayPortReservation,
    reserve_gateway_ports,
)
from roller.harnesses.hermes import (
    DEFAULT_HERMES_IMAGE,
    HERMES_REVISION,
    HermesConfig,
    HermesHarness,
    hermes_effort,
)
from roller.harnesses.openclaw import (
    COMMON_ENV_NAMES as OPENCLAW_COMMON_ENV_NAMES,
)
from roller.harnesses.openclaw import (
    DEFAULT_OPENCLAW_IMAGE,
    OpenClawConfig,
    OpenClawHarness,
    load_model_config,
)
from roller.harnesses.opencode import (
    DEFAULT_OPENCODE_IMAGE,
    OPENCODE_VERSION,
    OpenCodeConfig,
    OpenCodeHarness,
    opencode_effort,
)
from roller.harnesses.pi import (
    DEFAULT_PI_IMAGE,
    PI_TOOLS,
    PI_VERSION,
    PiConfig,
    PiHarness,
    pi_effort,
)
from roller.image_build import prepare_task_images
from roller.models import TaskBundle
from roller.run_lock import (
    RUN_LOCK_SCHEMA,
    build_run_lock,
    complete_legacy_adoption,
    ensure_run_lock,
)
from roller.storage import atomic_write_json, safe_component
from roller.tasks import TaskFormatError, build_requests, load_tasks
from roller.thinking import THINKING_LEVELS, normalize_thinking, resolve_thinking
from roller.thinking_capabilities import (
    ThinkingCapabilities,
    load_thinking_capabilities,
)
from roller.trajectory_audit import training_summary
from roller.version import __version__
from roller.viewer import DEFAULT_PORT as VIEWER_DEFAULT_PORT
from roller.viewer import serve as serve_viewer
from roller.wire_trace import RESPONSES_IMAGE_DETAILS

logger = logging.getLogger(__name__)


def _jsonl_equivalent(left: bytes, right: bytes) -> bool:
    """Compare JSONL snapshots by value, ignoring object key order."""

    try:
        return [json.loads(line) for line in left.splitlines()] == [
            json.loads(line) for line in right.splitlines()
        ]
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roller",
        description="Collect agentic trajectories from a task list.",
    )
    parser.add_argument("--version", action="version", version=f"Roller {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser(
        "init", help="write the example tasks and configuration templates"
    )
    init.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="target directory (default: current directory)",
    )

    view = subparsers.add_parser("view", help="browse a Roller run in a local web UI")
    view.add_argument(
        "run_dir", help="Roller run directory containing info.json/samples"
    )
    view.add_argument(
        "--host", default="127.0.0.1", help="listen host (default: loopback)"
    )
    view.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"listen port (default: {VIEWER_DEFAULT_PORT}, or a free port when it is busy)",
    )

    export = subparsers.add_parser(
        "export", help="export the trajectories of a Roller run to another format"
    )
    export.add_argument(
        "run_dir", help="Roller run directory containing info.json/samples"
    )
    export.add_argument(
        "--format",
        choices=("atif", "hf"),
        default="atif",
        help=(
            "target format: atif, the trajectory format Harbor reads (default), "
            "or hf, a Hugging Face dataset folder"
        ),
    )
    export.add_argument(
        "--output", "-o", help="directory to write (default: RUN_DIR/<format>)"
    )
    export.add_argument(
        "--redact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "with --format hf, replace the model endpoints, NO_PROXY hosts, "
            "machine name, host paths and credential values of the run in the "
            "exported files (default: on)"
        ),
    )
    export.add_argument(
        "--include-runs",
        action="store_true",
        help=(
            "with --format hf, also write runs/<harness>-<model>/, a Roller run "
            "directory with every file of the samples, which `roller view` opens"
        ),
    )
    export.add_argument(
        "--redact-rules",
        help=(
            "JSON file of further replace/exclude/forbid rules for --format hf, "
            "applied whether or not --redact is on"
        ),
    )

    bench = subparsers.add_parser("bench", help="work with benchmark adapters")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)

    bench_sub.add_parser("list", help="list registered benchmark adapters")

    bench_run = bench_sub.add_parser(
        "run", help="run a benchmark from layered YAML/JSON configuration"
    )
    bench_run.add_argument(
        "--config",
        required=True,
        help="base benchmark run configuration (YAML or JSON)",
    )
    bench_run.add_argument(
        "--extra-config",
        action="append",
        default=[],
        help=(
            "YAML/JSON overlay path or inline JSON object; repeatable and applied "
            "from left to right"
        ),
    )
    bench_run.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate the complete plan without starting containers",
    )

    bench_export = bench_sub.add_parser(
        "export", help="convert a checked-out bench into a Roller task list"
    )
    bench_export.add_argument(
        "name", help="bench adapter name (see 'roller bench list')"
    )
    bench_export.add_argument("root", help="path to the checked-out benchmark")
    bench_export.add_argument(
        "--output", "-o", required=True, help="task JSONL to write"
    )
    bench_export.add_argument(
        "--category",
        action="append",
        default=[],
        help="restrict to a bench category (repeatable)",
    )
    bench_export.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="restrict to a task id (repeatable)",
    )
    bench_export.add_argument(
        "--variant",
        help="bench variant/condition (for example with-skill or no-skill)",
    )
    bench_export.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="multiply every exported task timeout (default: 1.0)",
    )
    bench_export.add_argument(
        "--runs-per-task",
        type=int,
        default=1,
        help="duplicate each selected task for repeated benchmark runs (default: 1)",
    )

    bench_report = bench_sub.add_parser(
        "report", help="aggregate bench scores from a finished run"
    )
    bench_report.add_argument("run_dir", help="Roller run directory")
    bench_report.add_argument(
        "--output", "-o", help="also write the report JSON to this path"
    )

    collect = subparsers.add_parser("collect", help="collect trajectories")
    task_source = collect.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--tasks", help="JSON or JSONL task list")
    task_source.add_argument(
        "--bench",
        choices=tuple(adapter.name for adapter in list_benches()),
        help="load, run, grade, and report a checked-out benchmark in one command",
    )
    collect.add_argument(
        "--bench-root",
        help="path to the checked-out benchmark (required with --bench)",
    )
    collect.add_argument(
        "--bench-variant",
        help="benchmark variant/condition (for example with-skill or no-skill)",
    )
    collect.add_argument(
        "--bench-category",
        action="append",
        default=[],
        help="restrict benchmark categories (repeatable)",
    )
    collect.add_argument(
        "--bench-task-id",
        action="append",
        default=[],
        help="restrict benchmark task ids (repeatable)",
    )
    collect.add_argument(
        "--bench-timeout-multiplier",
        type=float,
        default=1.0,
        help="multiply benchmark task timeouts (default: 1.0)",
    )
    collect.add_argument(
        "--bench-runs-per-task",
        type=int,
        default=1,
        help="duplicate each selected benchmark task before collection (default: 1)",
    )
    collect.add_argument(
        "--model",
        action="append",
        default=[],
        help="model identifier; repeat to run every task on multiple models",
    )
    collect.add_argument(
        "--model-label",
        help=(
            "safe local label used in sample directories and container names; "
            "requires exactly one resolved model and does not change the model ID sent upstream"
        ),
    )
    collect.add_argument(
        "--harness",
        choices=HARNESS_NAMES,
        default=None,
        help="default/override task-agent harness (task 'harness' is used when omitted)",
    )
    collect.add_argument("--prompt-key", default="prompt")
    collect.add_argument("--id-key", default="id")
    collect.add_argument(
        "--workspace", help="default host workspace copied into each container"
    )
    collect.add_argument(
        "--timeout", type=int, default=600, help="per-trajectory timeout in seconds"
    )
    collect.add_argument("--parallel", type=int, default=1)
    collect.add_argument(
        "--launch-interval",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "minimum seconds between two rollout launches, so --parallel workers "
            "do not all start at once (default: 0)"
        ),
    )
    collect.add_argument(
        "--grader-parallel",
        type=int,
        help="maximum concurrent benchmark graders (default: --parallel)",
    )
    collect.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "independent runs per task/model pair "
            "(default: 3 for SkillsBench, 1 otherwise)"
        ),
    )
    collect.add_argument("--retries", type=int, default=0)
    collect.add_argument(
        "--retry-backoff",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "wait before the first retry of a sample, doubled before each "
            "further retry (default: 0, retry immediately)"
        ),
    )
    collect.add_argument(
        "--start-index", type=int, default=0, help="first normalized task index"
    )
    collect.add_argument("--limit", type=int, help="maximum number of tasks")
    collect.add_argument(
        "--output", default="runs", help="root directory for collections"
    )
    collect.add_argument(
        "--run-name", help="run directory name; defaults to a UTC timestamp"
    )
    collect.add_argument(
        "--resume", action="store_true", help="skip completed task/model pairs"
    )
    collect.add_argument(
        "--resume-verify-outputs",
        action="store_true",
        help="with --resume, reread recorded output files and rerun a sample whose "
        "content no longer matches its SHA-256 (default: check path and size only)",
    )
    collect.add_argument(
        "--dry-run", action="store_true", help="validate and print the execution plan"
    )
    collect.add_argument(
        "--max-input-file-bytes",
        type=int,
        default=10 * 1024 * 1024 * 1024,
        help="maximum size of one snapshotted input/context file",
    )
    collect.add_argument(
        "--max-task-input-bytes",
        type=int,
        default=50 * 1024 * 1024 * 1024,
        help="maximum total input bytes for one task",
    )
    collect.add_argument(
        "--task-timeouts",
        action="store_true",
        help=(
            "honor each task row's own 'timeout_seconds', falling back to "
            "--timeout when a row does not declare one (used by bench runs)"
        ),
    )
    collect.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    runtime = collect.add_argument_group("Harness runtime and trajectory")
    runtime.add_argument(
        "--docker-image",
        help=(
            "harness image; defaults to the corresponding ROLLER_*_IMAGE "
            "environment variable or Roller's pinned Harness image"
        ),
    )
    runtime.add_argument(
        "--openclaw-image", help="OpenClaw image override (for mixed Harness runs)"
    )
    runtime.add_argument(
        "--codex-image", help="Codex image override (for mixed Harness runs)"
    )
    runtime.add_argument(
        "--hermes-image", help="Hermes image override (for mixed Harness runs)"
    )
    runtime.add_argument(
        "--opencode-image", help="OpenCode image override (for mixed Harness runs)"
    )
    runtime.add_argument(
        "--deepseek-image", help="DeepSeek image override (for mixed Harness runs)"
    )
    runtime.add_argument(
        "--pi-image", help="Pi image override (for mixed Harness runs)"
    )
    runtime.add_argument(
        "--claude-code-image",
        help="Claude Code image override (for mixed Harness runs)",
    )
    runtime.add_argument(
        "--container-user",
        default="0",
        help="Docker user for the disposable task container (default: root); use 'image' for the image default",
    )
    runtime.add_argument("--docker-binary", default="docker")
    runtime.add_argument(
        "--rebuild-task-images",
        action="store_true",
        help="rebuild task-specific benchmark images even when their tags exist locally",
    )
    runtime.add_argument("--network", default="host")
    runtime.add_argument(
        "--capture-debug",
        choices=["off", "on-error", "full"],
        default="on-error",
        help="retain call capture debug evidence: on-error (default), off, or full",
    )
    runtime.add_argument(
        "--capture-calls",
        action="store_true",
        help="save full model calls and native subagent relations for all harness participants",
    )
    runtime.add_argument(
        "--thinking",
        type=normalize_thinking,
        choices=THINKING_LEVELS,
        metavar="LEVEL",
        help=(
            "Harness reasoning/thinking level; 'none' aliases 'off'. Each Harness "
            "applies its own validation and effort mapping"
        ),
    )
    runtime.add_argument(
        "--thinking-capabilities",
        metavar="PATH",
        help=(
            "optional versioned JSON capability declarations keyed by harness, "
            "model, and endpoint; unsupported efforts fail before rollout"
        ),
    )
    runtime.add_argument(
        "--pass-env",
        action="append",
        default=[],
        metavar="NAME",
        help="additional host environment variable to pass into the Harness container",
    )
    runtime.add_argument(
        "--wire-trace",
        choices=("off", "actf", "full"),
        default="actf",
        help=(
            "wire-level recording through a local reverse proxy: 'actf' (default) embeds "
            "the tools schema into trace_ACTF.json and keeps no extra file, 'full' "
            "additionally saves wire_trace.jsonl with every request body, 'off' disables "
            "(OpenClaw uses model-config baseUrl; Codex uses --api-base)"
        ),
    )
    runtime.add_argument(
        "--keep-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep visible reasoning in ACTF/SFT. For OpenClaw, --no-keep-thinking also "
            "strips reasoning from request history at the wire layer; for Codex and "
            "Claude Code/Hermes/OpenCode it controls export and leaves native raw/resume context unchanged"
        ),
    )
    runtime.add_argument("--keep-container-on-error", action="store_true")

    openclaw = collect.add_argument_group("OpenClaw")
    openclaw.add_argument(
        "--model-config", help="OpenClaw models JSON; supports ${ENV_NAME}"
    )
    openclaw.add_argument(
        "--gateway-port",
        type=int,
        help=(
            "first OpenClaw gateway port (default: the first range from "
            f"{DEFAULT_GATEWAY_PORT} that no other run on this host holds)"
        ),
    )
    openclaw.add_argument("--gateway-port-step", type=int, default=10)
    openclaw.add_argument("--gateway-startup-timeout", type=float, default=240.0)
    openclaw.add_argument(
        "--openclaw-model-request-timeout",
        type=int,
        metavar="SECONDS",
        help=(
            "per-provider model request and stream-idle timeout; overrides "
            "models.providers.<id>.timeoutSeconds"
        ),
    )
    openclaw.add_argument(
        "--openclaw-max-tokens",
        type=int,
        metavar="TOKENS",
        help="override maxTokens for the selected OpenClaw model",
    )
    openclaw.add_argument(
        "--openclaw-config-set",
        action="append",
        default=[],
        type=_config_assignment,
        metavar="KEY=VALUE",
        help=(
            "run `openclaw config set KEY VALUE` in every OpenClaw container after "
            "Roller's own settings; repeatable. Paths Roller manages are rejected"
        ),
    )
    openclaw.add_argument("--session-id", default="roller")
    openclaw.add_argument("--search-provider")
    openclaw.add_argument(
        "--collect-path",
        action="append",
        default=[],
        metavar="CONTAINER_PATH",
        help="legacy: copy an extra container path into results/legacy",
    )
    openclaw.add_argument(
        "--set-image-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    openclaw.add_argument(
        "--cache-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record the assembled system prompt per LLM call (diagnostics.cacheTrace)",
    )

    api = collect.add_argument_group("Harness API")
    api.add_argument(
        "--api-base",
        "--anthropic-base-url",
        dest="api_base",
        help=(
            "model API base: OpenAI-compatible base for Codex/Hermes/OpenCode/DeepSeek/Pi or Anthropic base/"
            "full /v1/messages endpoint for Claude Code; defaults to the Harness env"
        ),
    )
    api.add_argument(
        "--hermes-api-base",
        help="Hermes-only OpenAI-compatible Chat Completions API base",
    )
    api.add_argument(
        "--hermes-api-key-env", help="Hermes-only API key environment variable"
    )
    api.add_argument(
        "--opencode-api-base", help="OpenCode-only Chat Completions API base"
    )
    api.add_argument(
        "--deepseek-api-base", help="DeepSeek-only Chat Completions API base"
    )
    api.add_argument("--pi-api-base", help="Pi-only Chat Completions API base")
    api.add_argument(
        "--opencode-api-key-env", help="OpenCode-only API key environment variable"
    )
    api.add_argument(
        "--deepseek-api-key-env", help="DeepSeek-only API key environment variable"
    )
    api.add_argument("--pi-api-key-env", help="Pi-only API key environment variable")
    api.add_argument(
        "--codex-api-base", help="Codex-only API base override (for mixed Harness runs)"
    )
    api.add_argument(
        "--claude-api-base",
        help="Claude Code-only API base override (for mixed Harness runs)",
    )
    api.add_argument(
        "--api-key-env",
        help=(
            "host environment variable containing the model API key; defaults to "
            "OPENAI_API_KEY for Codex/Hermes/OpenCode and ANTHROPIC_API_KEY for Claude Code"
        ),
    )
    api.add_argument(
        "--codex-api-key-env", help="Codex-only API key environment variable"
    )
    api.add_argument(
        "--claude-api-key-env", help="Claude Code-only API key environment variable"
    )

    hermes = collect.add_argument_group("Hermes")
    hermes.add_argument(
        "--hermes-toolset",
        action="append",
        metavar="NAME",
        help="Hermes toolset to enable instead of terminal/file/vision; repeat to select several, or pass 'native' for native discovery",
    )
    hermes.add_argument(
        "--hermes-max-iterations",
        type=int,
        default=12,
        help="maximum model iterations per Hermes user turn (default: 12)",
    )
    hermes.add_argument(
        "--hermes-context-length",
        type=int,
        help="override Hermes model context window (at least 64000 tokens)",
    )
    hermes.add_argument(
        "--hermes-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Hermes vision_analyze and declare native image support (default: enabled)",
    )

    opencode = collect.add_argument_group("OpenCode")
    opencode.add_argument(
        "--opencode-native-tools",
        action="store_true",
        help="permit every tool of the native OpenCode registry instead of the fixed tool list",
    )
    opencode.add_argument(
        "--opencode-deny-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="deny a native OpenCode tool or permission under --opencode-native-tools; repeat as needed",
    )
    opencode.add_argument(
        "--opencode-max-steps",
        type=int,
        default=None,
        help="maximum model steps per OpenCode user turn (default: 100 for SkillsBench, 12 otherwise)",
    )
    opencode.add_argument(
        "--opencode-context-length",
        type=int,
        default=128000,
        help="OpenCode model context window (default: 128000 tokens)",
    )
    opencode.add_argument(
        "--opencode-output-tokens",
        type=int,
        default=None,
        help=(
            "OpenCode maximum output tokens "
            "(default: 32768 for SkillsBench, 8192 otherwise)"
        ),
    )
    opencode.add_argument(
        "--opencode-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="declare image input support to OpenCode (default: enabled)",
    )

    deepseek = collect.add_argument_group("DeepSeek")
    deepseek.add_argument(
        "--deepseek-profile",
        default="sdk-minimal",
        metavar="NAME",
        help="native DeepSeek SDK profile (default: sdk-minimal; sdk selects the full native tool set)",
    )
    deepseek.add_argument(
        "--deepseek-permission-mode",
        choices=DEEPSEEK_PERMISSION_MODES,
        help="native DeepSeek permission mode inside the task container (default: the profile's own)",
    )
    deepseek.add_argument(
        "--deepseek-max-steps",
        type=int,
        default=12,
        help="maximum model steps per DeepSeek user turn (default: 12)",
    )
    deepseek.add_argument(
        "--deepseek-input-modalities",
        choices=("text", "text,image"),
        default="text",
        help="DeepSeek model input capabilities (default: text; text,image lets read_image send images to the model)",
    )
    deepseek.add_argument(
        "--deepseek-context-length",
        type=int,
        default=128000,
        help="DeepSeek model context window (default: 128000 tokens)",
    )
    deepseek.add_argument(
        "--deepseek-output-tokens",
        type=int,
        default=8192,
        help="DeepSeek maximum output tokens (default: 8192)",
    )

    pi = collect.add_argument_group("Pi")
    pi.add_argument(
        "--pi-max-steps",
        type=int,
        default=12,
        help="maximum model steps per Pi user turn (default: 12)",
    )
    pi.add_argument(
        "--pi-input-modalities",
        choices=("text", "text,image"),
        default="text,image",
        help="Pi model input capabilities (default: text,image)",
    )
    pi.add_argument(
        "--pi-context-length",
        type=int,
        default=128000,
        help="Pi model context window (default: 128000 tokens)",
    )
    pi.add_argument(
        "--pi-output-tokens",
        type=int,
        default=8192,
        help="Pi maximum output tokens (default: 8192)",
    )

    codex = collect.add_argument_group("Codex")
    codex.add_argument(
        "--codex-config",
        help="optional Codex config.toml copied into the disposable container",
    )
    codex.add_argument(
        "--codex-config-set",
        action="append",
        default=[],
        type=_config_assignment,
        metavar="KEY=VALUE",
        help=(
            "pass `-c KEY=VALUE` to every codex exec after Roller's own overrides; "
            "repeatable. Paths Roller manages are rejected"
        ),
    )
    codex.add_argument(
        "--reasoning-summary",
        choices=("auto", "concise", "detailed", "none"),
        help="Codex model_reasoning_summary override",
    )
    codex.add_argument(
        "--codex-web-search",
        choices=("disabled", "cached", "live"),
        default="disabled",
    )
    codex.add_argument(
        "--codex-websockets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "allow the custom Codex provider to use Responses WebSockets; "
            "disabled by default for OpenAI-compatible HTTP proxies"
        ),
    )
    codex.add_argument(
        "--codex-image-detail",
        choices=RESPONSES_IMAGE_DETAILS,
        help=(
            "set this detail on Responses input_image blocks that Codex sends "
            "without one; needs the request recording proxy"
        ),
    )

    claude = collect.add_argument_group("Claude Code")
    claude.add_argument(
        "--claude-capture-calls",
        action="store_true",
        help="save Claude main/subagent native files, full HTTP calls and participant relations",
    )
    claude.add_argument(
        "--claude-settings",
        help="optional Claude Code settings.json copied into the disposable container",
    )
    claude.add_argument(
        "--claude-max-turns",
        type=int,
        help="optional Claude Code --max-turns limit for each predefined user message",
    )
    claude.add_argument(
        "--claude-max-thinking-tokens",
        type=int,
        help="optional MAX_THINKING_TOKENS value passed to Claude Code",
    )
    claude.add_argument(
        "--claude-permission-mode",
        choices=CLAUDE_PERMISSION_MODES,
        default="bypassPermissions",
    )
    claude.add_argument(
        "--claude-unique-tool-ids",
        action="store_true",
        help=(
            "replace tool-use IDs the Anthropic-compatible endpoint issues more "
            "than once before Claude Code receives them; needs the request "
            "recording proxy"
        ),
    )
    return parser


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _config_assignment(text: str) -> tuple[str, str]:
    try:
        return parse_config_assignment(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _docker_preflight(binary: str, image: str) -> str:
    if shutil.which(binary) is None:
        raise RuntimeError(f"Docker executable not found: {binary}")
    try:
        result = subprocess.run(
            [binary, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Docker preflight failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Docker daemon is unavailable: {detail}")
    try:
        inspected = subprocess.run(
            [binary, "image", "inspect", image, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Docker image preflight failed for {image}: {exc}") from exc
    image_digest = inspected.stdout.strip()
    if inspected.returncode != 0 or not image_digest:
        detail = (inspected.stderr or inspected.stdout).strip()
        raise RuntimeError(
            f"Docker image is not available locally: {image}; pull or load it first"
            + (f" ({detail})" if detail else "")
        )
    return image_digest


def _run_name(harness: str, value: str | None) -> str:
    if value:
        return safe_component(value, max_length=100)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{harness}_{timestamp}"


def _dry_run_payload(
    args: argparse.Namespace, requests: list, bundles: Mapping[str, TaskBundle]
) -> dict:
    cli_harness = args._cli_harness if hasattr(args, "_cli_harness") else args.harness
    agent_harnesses = {
        cli_harness or getattr(request.task, "harness", "openclaw")
        for request in requests
    }
    payload = {
        "dry_run": True,
        "harness": next(iter(agent_harnesses))
        if len(agent_harnesses) == 1
        else "mixed",
        "docker_image": args.docker_image,
        "tasks": len({request.task.id for request in requests}),
        "models": list(dict.fromkeys(request.model for request in requests)),
        "model_labels": list(
            dict.fromkeys(request.display_model for request in requests)
        ),
        "trajectories": len(requests),
        "parallel": args.parallel,
        "launch_interval": args.launch_interval,
        "retry_backoff": args.retry_backoff,
        "grader_parallel": args.grader_parallel or args.parallel,
        "samples": args.samples,
        "opencode_max_steps": args.opencode_max_steps,
        "opencode_output_tokens": args.opencode_output_tokens,
        "opencode_vision": args.opencode_vision,
        "timeout_seconds": args.timeout,
        "plan": [
            {
                "request_index": request.index,
                "sample_index": request.sample_index,
                "sample_count": request.sample_count,
                "task_id": request.task.id,
                "harness": cli_harness or request.task.harness,
                "model": request.model,
                "model_label": request.display_model,
                "timeout_seconds": request.timeout_seconds,
                "turns": (
                    {
                        "mode": "user_simulator",
                        "max": request.task.user_simulator.max_turns,
                    }
                    if request.task.user_simulator is not None
                    else {"mode": "predefined", "count": len(request.task.prompts)}
                ),
                "user_simulator": (
                    {
                        "harness": request.task.user_simulator.harness,
                        "model": request.task.user_simulator.model or request.model,
                        "max_turns": request.task.user_simulator.max_turns,
                    }
                    if request.task.user_simulator is not None
                    else None
                ),
                "workspace": str(request.task.workspace)
                if request.task.workspace
                else None,
                "inputs": len(request.task.inputs),
                "context_files": len(request.task.context_files),
                "output_dir": request.task.output_dir,
                "input_mode": request.task.input_mode,
                "attach_images": list(request.task.attach_images),
                "docker_image": request.task.docker_image,
                "image_build": request.task.image_build is not None,
                "workspace_container_path": request.task.workspace_container_path,
                "output_collection": "all",
                "legacy_workspace": request.task.legacy_workspace,
                # The digests a real run records and compares on Resume.
                "task_bundle_digest": bundles[request.task.id].manifest_digest,
                "input_tree_digest": bundles[request.task.id].input_tree_digest,
                "context_manifest_digest": bundles[
                    request.task.id
                ].context_manifest_digest,
            }
            for request in requests
        ],
    }
    bench_run_config = getattr(args, "_bench_run_config", None)
    if isinstance(bench_run_config, dict):
        payload["bench_run_config"] = bench_run_config
    return payload


def _with_task_timeout(request):
    """Apply a task's own ``timeout_seconds`` when the row declares one.

    The request already carries the global ``--timeout``, so a row without its
    own value is returned untouched.
    """

    raw = getattr(request.task, "raw", None)
    declared = raw.get("timeout_seconds") if isinstance(raw, dict) else None
    if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
        return request
    if declared == request.timeout_seconds:
        return request
    return dataclasses.replace(request, timeout_seconds=declared)


def _repeat_bench_rows(
    rows: list[dict[str, Any]], runs_per_task: int, adapter_name: str
) -> list[dict[str, Any]]:
    """Duplicate benchmark rows while preserving upstream run provenance."""

    if runs_per_task < 1:
        raise TaskFormatError("runs per task must be positive")
    if runs_per_task == 1:
        return rows
    repeated: list[dict[str, Any]] = []
    width = max(2, len(str(runs_per_task)))
    for row in rows:
        task_id = row.get("id")
        bench = row.get("bench")
        if not isinstance(task_id, str) or not task_id:
            raise TaskFormatError(f"{adapter_name}: exported task has no valid id")
        if not isinstance(bench, dict):
            raise TaskFormatError(
                f"{adapter_name}: exported task {task_id!r} has no bench block"
            )
        metadata = bench.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TaskFormatError(
                f"{adapter_name}: exported task {task_id!r} has invalid metadata"
            )
        upstream_task_id = metadata.get("upstream_task_id") or task_id
        for run_index in range(1, runs_per_task + 1):
            clone = json.loads(json.dumps(row))
            clone["id"] = f"{task_id}__run_{run_index:0{width}d}"
            clone_metadata = clone["bench"].setdefault("metadata", {})
            clone_metadata.update(
                {
                    "upstream_task_id": upstream_task_id,
                    "run_index": run_index,
                    "runs_per_task": runs_per_task,
                }
            )
            repeated.append(clone)
    return repeated


def _validate_openclaw_models(
    requests: Sequence[Any], *, cli_harness: str | None, model_config: Mapping[str, Any]
) -> None:
    """Check OpenClaw models against the providers ``--model-config`` declares.

    A provider in the file must list the requested model. A provider missing
    from the file may be one OpenClaw ships, so it only earns a warning.
    """
    providers = model_config.get("providers")
    if not isinstance(providers, dict):
        return
    models: set[str] = set()
    for request in requests:
        if (cli_harness or request.task.harness) == "openclaw":
            models.add(request.model)
        spec = request.task.user_simulator
        if spec is not None and spec.harness == "openclaw":
            models.add(spec.model or request.model)
    for model in sorted(models):
        provider_name, _, model_id = model.partition("/")
        provider = providers.get(provider_name)
        if not isinstance(provider, dict):
            logger.warning(
                "model %s: provider %r is not in --model-config, so OpenClaw has to "
                "know it already and its requests are not traced",
                model,
                provider_name,
            )
            continue
        listed = provider.get("models")
        declared = (
            [
                item["id"]
                for item in listed
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
            if isinstance(listed, list)
            else []
        )
        if model_id not in declared:
            raise ValueError(
                f"model {model!r} is not declared for provider {provider_name!r} in "
                f"--model-config; declared models: {', '.join(declared) or 'none'}"
            )


def _validate_thinking_requests(
    requests: Sequence[Any],
    *,
    cli_harness: str | None,
    harness: str,
    api_base: str | None | Callable[[str], str | None],
    default_effort: str | None,
    effort_validator: Callable[[str | None], str | None],
    capabilities: ThinkingCapabilities | None,
) -> None:
    def validate(model: str, selected: str | None) -> None:
        effort = resolve_thinking(selected, default_effort)
        effort_validator(effort)
        if capabilities is not None:
            capabilities.validate(
                harness=harness,
                model=model,
                api_base=api_base(model) if callable(api_base) else api_base,
                effort=effort,
            )

    for request in requests:
        if (cli_harness or request.task.harness) == harness:
            validate(request.model, request.task.thinking)
        spec = request.task.user_simulator
        if spec is not None and spec.harness == harness:
            validate(spec.model or request.model, spec.thinking)


def _collect(args: argparse.Namespace) -> int:
    if args.samples is None:
        args.samples = 3 if args.bench == "skillsbench" else 1
    if args.opencode_max_steps is None:
        args.opencode_max_steps = 100 if args.bench == "skillsbench" else 12
    if args.opencode_output_tokens is None:
        args.opencode_output_tokens = 32768 if args.bench == "skillsbench" else 8192
    if args.parallel < 1:
        raise ValueError("--parallel must be at least 1")
    if not math.isfinite(args.launch_interval) or args.launch_interval < 0:
        raise ValueError("--launch-interval must be a finite number of seconds >= 0")
    if args.grader_parallel is not None and args.grader_parallel < 1:
        raise ValueError("--grader-parallel must be at least 1")
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    if args.resume_verify_outputs and not args.resume:
        raise ValueError("--resume-verify-outputs requires --resume")
    if not math.isfinite(args.retry_backoff) or args.retry_backoff < 0:
        raise ValueError("--retry-backoff must be a finite number of seconds >= 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if args.bench_runs_per_task < 1:
        raise ValueError("--bench-runs-per-task must be at least 1")
    bundle_policy = BundlePolicy(
        max_file_bytes=args.max_input_file_bytes,
        max_task_input_bytes=args.max_task_input_bytes,
    )

    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else None
    if workspace is not None and not workspace.is_dir():
        raise FileNotFoundError(f"workspace is not a directory: {workspace}")

    bench_context: dict[str, Any] | None = None
    bench_manifest: bytes | None = None
    bench_temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.bench:
        if not args.bench_root:
            raise ValueError("--bench-root is required with --bench")
        root = Path(args.bench_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"bench root is not a directory: {root}")
        adapter = get_bench(args.bench)
        rows = adapter.load(
            root,
            categories=args.bench_category or None,
            task_ids=args.bench_task_id or None,
            variant=args.bench_variant,
            timeout_multiplier=args.bench_timeout_multiplier,
        )
        if not rows:
            raise TaskFormatError(f"{adapter.name}: selection matched no tasks")
        rows = _repeat_bench_rows(rows, args.bench_runs_per_task, adapter.name)
        bench_manifest = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
        bench_temporary = tempfile.TemporaryDirectory(prefix="roller-bench-")
        tasks_path = Path(bench_temporary.name) / "tasks.jsonl"
        tasks_path.write_bytes(bench_manifest)
        bench_context = {
            "name": adapter.name,
            "root": str(root),
            "variant": args.bench_variant or adapter.default_variant,
            "categories": list(args.bench_category),
            "task_ids": list(args.bench_task_id),
            "timeout_multiplier": args.bench_timeout_multiplier,
            "runs_per_task": args.bench_runs_per_task,
            "exported_task_count": len(rows),
        }
    else:
        if (
            args.bench_root
            or args.bench_variant
            or args.bench_category
            or args.bench_task_id
        ):
            raise ValueError("--bench-* selection options require --bench")
        if args.bench_timeout_multiplier != 1.0:
            raise ValueError("--bench-timeout-multiplier requires --bench")
        if args.bench_runs_per_task != 1:
            raise ValueError("--bench-runs-per-task requires --bench")
        tasks_path = Path(args.tasks).expanduser().resolve()
    tasks = load_tasks(
        tasks_path,
        prompt_key=args.prompt_key,
        id_key=args.id_key,
        global_workspace=workspace,
    )
    end_index = args.start_index + args.limit if args.limit is not None else None
    tasks = tasks[args.start_index : end_index]
    requests = build_requests(
        tasks, args.model, args.timeout, samples_per_task=args.samples
    )
    if not requests:
        raise TaskFormatError("task list contains no tasks")
    if args.model_label is not None:
        resolved_models = {request.model for request in requests}
        if len(resolved_models) != 1:
            raise ValueError("--model-label requires exactly one resolved model")
        requests = [
            dataclasses.replace(request, model_label=args.model_label)
            for request in requests
        ]
    if args.bench or getattr(args, "task_timeouts", False):
        # Benches size their own timeouts per task; a single global value would
        # either cut long tasks short or let short ones idle.
        requests = [_with_task_timeout(request) for request in requests]

    agent_harnesses = {
        args.harness or getattr(request.task, "harness", "openclaw")
        for request in requests
    }
    simulator_harnesses = {
        request.task.user_simulator.harness
        for request in requests
        if request.task.user_simulator is not None
    }
    selected_harnesses = agent_harnesses | simulator_harnesses
    if args.claude_capture_calls and "claude-code" not in selected_harnesses:
        raise ValueError("--claude-capture-calls requires a Claude Code participant")
    cli_harness = args.harness
    args._cli_harness = cli_harness
    # Resolve the historical global default after loading tasks.  This keeps
    # ``--harness`` backward compatible while allowing a task-level harness.
    args.harness = next(name for name in HARNESS_NAMES if name in agent_harnesses)
    requested_api_base = args.api_base
    if args.api_base is None:
        if selected_harnesses == {"codex"}:
            args.api_base = os.environ.get("OPENAI_BASE_URL")
        elif selected_harnesses == {"claude-code"}:
            args.api_base = os.environ.get("ANTHROPIC_BASE_URL")
    hermes_api_base = (
        args.hermes_api_base or requested_api_base or os.environ.get("OPENAI_BASE_URL")
    )
    opencode_api_base = (
        args.opencode_api_base
        or requested_api_base
        or os.environ.get("OPENAI_BASE_URL")
    )
    deepseek_api_base = (
        args.deepseek_api_base
        or requested_api_base
        or os.environ.get("DEEPSEEK_BASE_URL")
    )
    pi_api_base = (
        args.pi_api_base or requested_api_base or os.environ.get("OPENAI_BASE_URL")
    )
    codex_api_base = (
        args.codex_api_base or requested_api_base or os.environ.get("OPENAI_BASE_URL")
    )
    claude_api_base = (
        args.claude_api_base
        or requested_api_base
        or os.environ.get("ANTHROPIC_BASE_URL")
    )
    requested_api_key_env = args.api_key_env
    if args.api_key_env is None:
        args.api_key_env = (
            "ANTHROPIC_API_KEY"
            if selected_harnesses == {"claude-code"}
            else "OPENAI_API_KEY"
        )
    hermes_api_key_env = (
        args.hermes_api_key_env or requested_api_key_env or "OPENAI_API_KEY"
    )
    opencode_api_key_env = (
        args.opencode_api_key_env or requested_api_key_env or "OPENAI_API_KEY"
    )
    deepseek_api_key_env = (
        args.deepseek_api_key_env or requested_api_key_env or "DEEPSEEK_API_KEY"
    )
    pi_api_key_env = args.pi_api_key_env or requested_api_key_env or "OPENAI_API_KEY"
    codex_api_key_env = (
        args.codex_api_key_env or requested_api_key_env or "OPENAI_API_KEY"
    )
    claude_api_key_env = (
        args.claude_api_key_env or requested_api_key_env or "ANTHROPIC_API_KEY"
    )
    defaults = {
        "hermes": os.environ.get("ROLLER_HERMES_IMAGE", DEFAULT_HERMES_IMAGE),
        "opencode": os.environ.get("ROLLER_OPENCODE_IMAGE", DEFAULT_OPENCODE_IMAGE),
        "deepseek": os.environ.get("ROLLER_DEEPSEEK_IMAGE", DEFAULT_DEEPSEEK_IMAGE),
        "pi": os.environ.get("ROLLER_PI_IMAGE", DEFAULT_PI_IMAGE),
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

    hermes_config: HermesConfig | None = None
    opencode_config: OpenCodeConfig | None = None
    deepseek_config: DeepSeekConfig | None = None
    pi_config: PiConfig | None = None
    openclaw_config: OpenClawConfig | None = None
    codex_config: CodexConfig | None = None
    claude_code_config: ClaudeCodeConfig | None = None
    if "openclaw" in selected_harnesses:
        openclaw_config = OpenClawConfig(
            docker_image=images["openclaw"],
            container_user=(
                None if args.container_user == "image" else args.container_user
            ),
            # Placeholder until the run reserves its range (host network only).
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
        if args.collect_path and "openclaw" not in selected_harnesses:
            raise ValueError("--collect-path requires an OpenClaw participant")
        codex_config = CodexConfig(
            docker_image=images["codex"],
            container_user=(
                None if args.container_user == "image" else args.container_user
            ),
            network=args.network,
            config_path=codex_config_path,
            api_base=codex_api_base,
            api_key_env=codex_api_key_env,
            pass_env=_deduplicate((*PROXY_ENV_NAMES, *args.pass_env)),
            reasoning_effort=args.thinking,
            reasoning_summary=args.reasoning_summary,
            web_search=args.codex_web_search,
            supports_websockets=args.codex_websockets,
            responses_image_detail=args.codex_image_detail,
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
            api_base=codex_api_base,
            default_effort=args.thinking,
            effort_validator=codex_effort,
            capabilities=thinking_capabilities,
        )
    if "claude-code" in selected_harnesses:
        if args.collect_path and "openclaw" not in selected_harnesses:
            raise ValueError("--collect-path requires an OpenClaw participant")
        claude_code_config = ClaudeCodeConfig(
            docker_image=images["claude-code"],
            container_user=(
                None if args.container_user == "image" else args.container_user
            ),
            network=args.network,
            settings_path=claude_settings_path,
            config_sha256=claude_settings_sha256,
            api_base=claude_api_base,
            api_key_env=claude_api_key_env,
            pass_env=_deduplicate((*PROXY_ENV_NAMES, *args.pass_env)),
            reasoning_effort=args.thinking,
            max_thinking_tokens=args.claude_max_thinking_tokens,
            max_turns=args.claude_max_turns,
            capture_calls=args.claude_capture_calls or args.capture_calls,
            capture_debug=args.capture_debug,
            permission_mode=args.claude_permission_mode,
            unique_tool_ids=args.claude_unique_tool_ids,
            wire_trace_mode=args.wire_trace,
            keep_thinking=args.keep_thinking,
            keep_container_on_error=args.keep_container_on_error,
            docker_binary=args.docker_binary,
        )
        _validate_thinking_requests(
            requests,
            cli_harness=cli_harness,
            harness="claude-code",
            api_base=claude_code_config.api_base,
            default_effort=args.thinking,
            effort_validator=claude_effort,
            capabilities=thinking_capabilities,
        )

    def bridge_options(
        name: str, api_base: str | None, api_key_env: str
    ) -> dict[str, Any]:
        """Config fields shared by the Hermes, OpenCode, DeepSeek and Pi bridges."""

        return dict(
            docker_image=images[name],
            container_user=None
            if args.container_user == "image"
            else args.container_user,
            network=args.network,
            api_base=api_base,
            api_key_env=api_key_env,
            pass_env=_deduplicate((*PROXY_ENV_NAMES, *args.pass_env)),
            reasoning_effort=args.thinking,
            wire_trace_mode=args.wire_trace,
            capture_calls=args.capture_calls,
            capture_debug=args.capture_debug,
            keep_thinking=args.keep_thinking,
            keep_container_on_error=args.keep_container_on_error,
            docker_binary=args.docker_binary,
        )

    if "hermes" in selected_harnesses:
        hermes_config = HermesConfig(
            **bridge_options("hermes", hermes_api_base, hermes_api_key_env),
            max_iterations=args.hermes_max_iterations,
            context_length=args.hermes_context_length,
            vision=args.hermes_vision,
            toolsets=tuple(args.hermes_toolset) if args.hermes_toolset else None,
        )
    if "opencode" in selected_harnesses:
        opencode_config = OpenCodeConfig(
            **bridge_options("opencode", opencode_api_base, opencode_api_key_env),
            max_steps=args.opencode_max_steps,
            context_length=args.opencode_context_length,
            output_tokens=args.opencode_output_tokens,
            vision=args.opencode_vision,
            native_tools=args.opencode_native_tools,
            deny_tools=tuple(args.opencode_deny_tool),
        )
    if "deepseek" in selected_harnesses:
        deepseek_config = DeepSeekConfig(
            **bridge_options("deepseek", deepseek_api_base, deepseek_api_key_env),
            max_steps=args.deepseek_max_steps,
            context_length=args.deepseek_context_length,
            output_tokens=args.deepseek_output_tokens,
            profile=args.deepseek_profile,
            permission_mode=args.deepseek_permission_mode,
            input_modalities=tuple(args.deepseek_input_modalities.split(",")),
        )
    if "pi" in selected_harnesses:
        pi_config = PiConfig(
            **bridge_options("pi", pi_api_base, pi_api_key_env),
            max_steps=args.pi_max_steps,
            context_length=args.pi_context_length,
            output_tokens=args.pi_output_tokens,
            input_modalities=tuple(args.pi_input_modalities.split(",")),
        )
    for name, bridge_config, effort_validator, default_api_base in (
        ("hermes", hermes_config, hermes_effort, "https://api.openai.com/v1"),
        ("opencode", opencode_config, opencode_effort, "https://api.openai.com/v1"),
        ("deepseek", deepseek_config, deepseek_effort, "https://api.deepseek.com"),
        ("pi", pi_config, pi_effort, "https://api.openai.com/v1"),
    ):
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

    if args.dry_run:
        # Run every path, hash, collision and size check without writing a file.
        bundles = TaskBundleStore(
            Path(args.output), bundle_policy, materialize=False
        ).prepare_all(tasks)
        payload = _dry_run_payload(args, requests, bundles)
        if bench_context is not None:
            payload["bench"] = bench_context
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    docker_image_digests = {
        harness: _docker_preflight(args.docker_binary, image)
        for harness, image in images.items()
    }
    task_image_digests = prepare_task_images(
        args.docker_binary,
        requests,
        images,
        cli_harness=cli_harness,
        rebuild=args.rebuild_task_images,
    )
    run_label = next(iter(agent_harnesses)) if len(agent_harnesses) == 1 else "mixed"
    output_root = Path(args.output).expanduser().resolve()
    run_dir = output_root / _run_name(run_label, args.run_name)
    existing_run = run_dir.exists() and any(run_dir.iterdir())
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"run directory already exists; use --resume or another --run-name: {run_dir}"
        )
    if existing_run:
        check_retired_cot_samples(run_dir, requests)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks_snapshot: Path | None = None
    if bench_manifest is not None:
        tasks_snapshot = run_dir / "tasks.jsonl"
        if tasks_snapshot.is_file():
            recorded_manifest = tasks_snapshot.read_bytes()
            if recorded_manifest != bench_manifest and not _jsonl_equivalent(
                recorded_manifest, bench_manifest
            ):
                raise TaskFormatError(
                    "benchmark selection differs from the existing run's tasks.jsonl; "
                    "use a new --run-name"
                )
        elif existing_run:
            raise TaskFormatError(
                "cannot safely resume this benchmark run: tasks.jsonl is missing"
            )

    previous_info: dict[str, object] = {}
    previous_info_path = run_dir / "info.json"
    if args.resume and previous_info_path.is_file():
        try:
            candidate = json.loads(previous_info_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                previous_info = candidate
        except (OSError, json.JSONDecodeError):
            previous_info = {}

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    harness_configurations: dict[str, dict[str, object]] = {}
    if openclaw_config is not None:
        harness_configurations["openclaw"] = {
            "docker_image": images["openclaw"],
            "container_user": args.container_user,
            "model_config_path": (
                str(model_config_path) if model_config_path else None
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
            # Absent when unused so locks written before the option still match.
            **(
                {"config_set": recorded_config_assignments(openclaw_config.config_set)}
                if openclaw_config.config_set
                else {}
            ),
        }
    if codex_config is not None:
        harness_configurations["codex"] = {
            "docker_image": images["codex"],
            "container_user": args.container_user,
            "config_path": str(codex_config_path) if codex_config_path else None,
            "config_sha256": codex_config_digest,
            "network": args.network,
            "api_base": redact_api_base(codex_api_base) if codex_api_base else None,
            "api_key_env": codex_api_key_env,
            "reasoning_effort": args.thinking,
            "reasoning_summary": args.reasoning_summary,
            "wire_trace": args.wire_trace,
            "keep_thinking": args.keep_thinking,
            "web_search": args.codex_web_search,
            "supports_websockets": args.codex_websockets,
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
    if claude_code_config is not None:
        harness_configurations["claude-code"] = {
            "docker_image": images["claude-code"],
            "container_user": args.container_user,
            "settings_path": (
                str(claude_settings_path) if claude_settings_path else None
            ),
            "config_sha256": claude_settings_sha256,
            "network": args.network,
            "api_base": (
                redact_anthropic_base_url(claude_api_base) if claude_api_base else None
            ),
            "api_key_env": claude_api_key_env,
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
        }

    if hermes_config is not None:
        harness_configurations["hermes"] = {
            "docker_image": images["hermes"],
            "container_user": args.container_user,
            "network": args.network,
            "api_base": hermes_api_base or "https://api.openai.com/v1",
            "api_key_env": hermes_api_key_env,
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

    if opencode_config is not None:
        harness_configurations["opencode"] = {
            "docker_image": images["opencode"],
            "container_user": args.container_user,
            "network": args.network,
            "api_base": opencode_api_base or "https://api.openai.com/v1",
            "api_key_env": opencode_api_key_env,
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

    if deepseek_config is not None:
        harness_configurations["deepseek"] = {
            "docker_image": images["deepseek"],
            "container_user": args.container_user,
            "network": args.network,
            "api_base": deepseek_api_base or "https://api.deepseek.com",
            "api_key_env": deepseek_api_key_env,
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

    if pi_config is not None:
        harness_configurations["pi"] = {
            "docker_image": images["pi"],
            "container_user": args.container_user,
            "network": args.network,
            "api_base": pi_api_base or "https://api.openai.com/v1",
            "api_key_env": pi_api_key_env,
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

    # OpenClaw user simulators run in the worker slots after the agents', so
    # their gateways get ports of their own.
    simulator_gateway_slot_offset = (
        args.parallel if "openclaw" in simulator_harnesses else None
    )
    run_info = {
        "schema_version": "roller.run/v1",
        "roller_version": __version__,
        "configuration": {
            "harness": run_label,
            "agent_harnesses": sorted(agent_harnesses),
            "simulator_harnesses": sorted(simulator_harnesses),
            "tasks": str(tasks_snapshot or tasks_path),
            "bench": (
                {**bench_context, "tasks_snapshot": "tasks.jsonl"}
                if bench_context is not None
                else None
            ),
            "bench_run_config": getattr(args, "_bench_run_config", None),
            "models": args.model,
            "trajectory_count": len(requests),
            "workspace": str(workspace) if workspace else None,
            "timeout_seconds": args.timeout,
            "parallel": args.parallel,
            "launch_interval": args.launch_interval,
            "grader_parallel": args.grader_parallel or args.parallel,
            "samples": args.samples,
            "retries": args.retries,
            "retry_backoff": args.retry_backoff,
            "start_index": args.start_index,
            "limit": args.limit,
            "resume": args.resume,
            "task_schema": "roller.task/v1",
            "thinking_capabilities_path": (
                str(thinking_capabilities_path) if thinking_capabilities_path else None
            ),
            "thinking_capabilities_sha256": thinking_capabilities_digest,
            "conversation": {
                "user_simulator_tasks": sum(
                    task.user_simulator is not None for task in tasks
                ),
                "simulator_gateway_slot_offset": simulator_gateway_slot_offset,
            },
            "bundle_policy": {
                "max_input_file_bytes": args.max_input_file_bytes,
                "max_task_input_bytes": args.max_task_input_bytes,
            },
            "task_images": {
                task.id: {
                    "image": task.docker_image,
                    "digest": task_image_digests.get(task.id),
                }
                for task in tasks
                if task.docker_image is not None
            },
            **harness_configurations,
        },
        "state": "initializing",
        "started_at": previous_info.get("started_at", now),
    }
    if args.resume and existing_run:
        previous_count = previous_info.get("resume_count", 0)
        run_info["last_resumed_at"] = now
        run_info["resume_count"] = (
            previous_count + 1
            if isinstance(previous_count, int) and not isinstance(previous_count, bool)
            else 1
        )

    results: list[dict] | None = None
    gateway_ports: GatewayPortReservation | None = None
    with tempfile.TemporaryDirectory(
        prefix=".roller-bundle-", dir=output_root
    ) as temporary:
        bundle_store = TaskBundleStore(Path(temporary), bundle_policy)
        try:
            if openclaw_config is not None and args.network == "host":
                # Host-network gateways share the host's ports with any other
                # run here; hold this run's range until collection ends.
                gateway_ports = reserve_gateway_ports(
                    args.gateway_port,
                    args.gateway_port_step,
                    args.parallel + (simulator_gateway_slot_offset or 0),
                )
                ports = gateway_ports.ports
                openclaw_config = dataclasses.replace(
                    openclaw_config, gateway_port_start=ports.start
                )
                # The same dict is in run_info, so info.json records it too.
                harness_configurations["openclaw"]["gateway_port_start"] = ports.start
                logger.info(
                    "OpenClaw gateway ports %d-%d (every %d)",
                    ports.start,
                    ports[-1],
                    ports.step,
                )
            bundles = bundle_store.prepare_all(tasks)
            requests = bundle_store.attach(requests, bundles)
            run_dir.mkdir(parents=True, exist_ok=True)
            if tasks_snapshot is not None and not tasks_snapshot.exists():
                tasks_snapshot.write_bytes(bench_manifest or b"")
            execution = {
                "harness": run_label,
                "agent_harnesses": sorted(agent_harnesses),
                "simulator_harnesses": sorted(simulator_harnesses),
                "task_schema": "roller.task/v1",
                "output_collection": "all",
                "thinking_capabilities_sha256": thinking_capabilities_digest,
                "task_images": {
                    task.id: {
                        "image": task.docker_image,
                        "digest": task_image_digests.get(task.id),
                    }
                    for task in tasks
                    if task.docker_image is not None
                },
                **{
                    name: {
                        **payload,
                        "docker_image": images[name],
                        "docker_image_digest": docker_image_digests[name],
                    }
                    for name, payload in harness_configurations.items()
                },
            }
            lock_model_config = (
                model_config if "openclaw" in selected_harnesses else None
            )
            run_lock = build_run_lock(
                requests,
                roller_version=__version__,
                execution=execution,
                model_config=lock_model_config,
            )
            compatible_run_locks = []
            claude_execution = execution.get("claude-code")
            if args.resume and existing_run and isinstance(claude_execution, dict):
                # Previous v1 locks used a short BLAKE2b settings fingerprint.
                legacy_fingerprint = (
                    hashlib.blake2b(claude_settings_content, digest_size=16).hexdigest()
                    if claude_settings_content is not None
                    else None
                )
                legacy_claude_execution = dict(claude_execution)
                legacy_claude_execution.pop("config_sha256", None)
                legacy_claude_execution["settings_fingerprint"] = legacy_fingerprint
                compatible_run_locks.append(
                    build_run_lock(
                        requests,
                        roller_version=__version__,
                        execution={
                            **execution,
                            "claude-code": legacy_claude_execution,
                        },
                        model_config=lock_model_config,
                    )
                )
            lock_decision = ensure_run_lock(
                run_dir / "lock.json",
                run_lock,
                resume=args.resume,
                existing_run=existing_run,
                compatible_expected=compatible_run_locks,
            )
            bundle_store.persist_full_context(bundles.values(), run_dir)
            if lock_decision.legacy_run:
                logger.warning(
                    "legacy run has not completed trusted lock adoption; rerunning "
                    "all samples before enabling resume"
                )
            configuration = run_info["configuration"]
            if isinstance(configuration, dict):
                configuration["resume_reuse_enabled"] = (
                    lock_decision.reuse_existing_samples
                )
                for name, digest in docker_image_digests.items():
                    harness_info = configuration.get(name)
                    if isinstance(harness_info, dict):
                        harness_info["docker_image_digest"] = digest
            run_info["lock"] = {
                "path": "lock.json",
                "schema_version": RUN_LOCK_SCHEMA,
                "legacy_adopted": lock_decision.legacy_run,
            }
            atomic_write_json(run_dir / "info.json", run_info)
            keep_aux_traces = args.wire_trace == "full"
            grading_semaphore = threading.BoundedSemaphore(
                args.grader_parallel or args.parallel
            )

            harness_types = {
                "openclaw": (OpenClawHarness, openclaw_config),
                "codex": (CodexHarness, codex_config),
                "claude-code": (ClaudeCodeHarness, claude_code_config),
                "hermes": (HermesHarness, hermes_config),
                "opencode": (OpenCodeHarness, opencode_config),
                "deepseek": (DeepSeekHarness, deepseek_config),
                "pi": (PiHarness, pi_config),
            }

            def make_harness(
                name: str,
                slot: int,
                task_image: str | None = None,
                task_workspace: str | None = None,
                task_image_digest: str | None = None,
            ):
                if name not in harness_types:
                    raise ValueError(f"unknown Harness: {name}")
                harness_type, base_config = harness_types[name]
                assert base_config is not None
                replacements = {
                    "preflight_image_digest": (
                        task_image_digest
                        if task_image is not None
                        else docker_image_digests[name]
                    )
                }
                if task_image:
                    replacements["docker_image"] = task_image
                if task_workspace:
                    replacements["workspace_container_path"] = task_workspace
                return harness_type(
                    dataclasses.replace(base_config, **replacements), slot
                )

            def harness_factory(slot: int, request):
                agent = make_harness(
                    cli_harness or request.task.harness,
                    slot,
                    request.task.docker_image,
                    request.task.workspace_container_path,
                    (
                        task_image_digests.get(request.task.id)
                        if request.task.docker_image is not None
                        else None
                    ),
                )
                binding = binding_from_task_row(
                    getattr(request.task, "raw", None) or {}
                )
                if binding is not None and binding.grade is not None:
                    # Grading has to read the container the agent just wrote to,
                    # so it runs inside the session rather than after teardown.
                    try:
                        adapter = get_bench(binding.name)
                    except KeyError:
                        return GradedHarness(
                            agent,
                            binding,
                            grading_semaphore=grading_semaphore,
                        )
                    return adapter.wrap_harness(
                        agent,
                        binding,
                        grading_semaphore=grading_semaphore,
                    )
                spec = request.task.user_simulator
                if spec is None:
                    return agent
                simulator_slot = (
                    slot + simulator_gateway_slot_offset
                    if spec.harness == "openclaw"
                    else slot
                )
                simulator = make_harness(spec.harness, simulator_slot)
                return ConversationManager(
                    agent,
                    simulator,
                    keep_aux_traces=keep_aux_traces,
                )

            collector = Collector(
                CollectionConfig(
                    run_dir=run_dir,
                    parallel=args.parallel,
                    launch_interval=args.launch_interval,
                    retries=args.retries,
                    retry_backoff=args.retry_backoff,
                    resume=lock_decision.reuse_existing_samples,
                    verify_outputs=args.resume_verify_outputs,
                    keep_aux_traces=keep_aux_traces,
                    capture_cache_trace=(
                        args.cache_trace if "openclaw" in selected_harnesses else False
                    ),
                    keep_thinking=args.keep_thinking,
                ),
                harness_factory=harness_factory,
            )
            results = collector.collect(requests)
            if lock_decision.legacy_run:
                complete_legacy_adoption(run_dir / "lock.json")
        finally:
            bundle_store.cleanup_runtime()
            if gateway_ports is not None:
                gateway_ports.release()
    assert results is not None
    failed = sum(result.get("status") in {"error", "timeout"} for result in results)
    print(f"Collected {len(results)} trajectories ({failed} failed).")
    training = training_summary(results)
    print(
        f"SFT: {training['sft_included']} included, "
        f"{training['sft_segmented']} segmented, "
        f"{training['sft_excluded']} excluded. "
        f"Thinking audit: {training['thinking_mismatch']} mismatch; "
        f"wire {training['wire_unverified']} unverified, "
        f"{training['wire_not_observable']} not observable."
    )
    print(f"Run directory: {run_dir}")
    print(f"Info: {run_dir / 'info.json'}")
    if bench_context is not None:
        report_path = run_dir / "bench-report.json"
        try:
            _bench_report(
                argparse.Namespace(run_dir=str(run_dir), output=str(report_path))
            )
        except TaskFormatError:
            if not failed:
                raise
            logger.warning(
                "benchmark report was not generated because no graded samples were produced"
            )
    return 1 if failed else 0


def _bench(args: argparse.Namespace) -> int:
    if args.bench_command == "list":
        for adapter in list_benches():
            variants = (
                f" [variants: {', '.join(adapter.variants)}]"
                if adapter.variants
                else ""
            )
            print(f"{adapter.name:<16} {adapter.description}{variants}")
        return 0
    if args.bench_command == "run":
        from roller.benches.run_config import (
            load_bench_run_plan,
            temporary_environment,
        )

        with tempfile.TemporaryDirectory(prefix="roller-bench-run-") as temporary:
            plan = load_bench_run_plan(
                args.config,
                args.extra_config,
                temporary_root=Path(temporary),
            )
            collect_argv = list(plan.collect_argv)
            if args.dry_run:
                collect_argv.append("--dry-run")
            collect_args = _build_parser().parse_args(collect_argv)
            collect_args._bench_run_config = plan.provenance
            with temporary_environment(plan.environment):
                return _collect(collect_args)
    if args.bench_command == "export":
        adapter = get_bench(args.name)
        root = Path(args.root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"bench root is not a directory: {root}")
        rows = adapter.load(
            root,
            categories=args.category or None,
            task_ids=args.task_id or None,
            variant=args.variant,
            timeout_multiplier=args.timeout_multiplier,
        )
        if not rows:
            raise TaskFormatError(f"{adapter.name}: selection matched no tasks")
        rows = _repeat_bench_rows(rows, args.runs_per_task, adapter.name)
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        categories = sorted({row["bench"].get("category") or "" for row in rows} - {""})
        print(f"Exported {len(rows)} {adapter.name} tasks to {output}")
        if categories:
            print(f"Categories: {', '.join(categories)}")
        print(
            f"Run with: roller collect --tasks {output} --model <model> --task-timeouts"
        )
        return 0
    return _bench_report(args)


def _bench_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    samples_root = run_dir / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(f"run directory has no samples/: {run_dir}")
    rows: list[dict[str, Any]] = []
    bench_names: set[str] = set()
    for info_path in sorted(samples_root.glob("*/*/info.json")):
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("skipping unreadable sample info: %s", info_path)
            continue
        if not isinstance(info, dict):
            continue
        runtime = info.get("runtime")
        payload = runtime.get("bench") if isinstance(runtime, dict) else None
        if not isinstance(payload, dict):
            continue
        name = payload.get("bench")
        if isinstance(name, str) and name:
            bench_names.add(name)
        rows.append(
            {
                "task_id": info.get("task_id"),
                "model": info.get("model"),
                "status": info.get("status"),
                "error": info.get("error"),
                "bench_scores": payload.get("scores"),
                "bench_score_key": payload.get("score_key"),
                "bench_category": payload.get("category"),
                "bench_metadata": payload.get("metadata"),
                "bench_error": payload.get("error"),
            }
        )
    if not rows:
        raise TaskFormatError(f"no graded bench samples found under {samples_root}")
    label = ", ".join(sorted(bench_names)) or "unknown"
    try:
        report = (
            get_bench(sorted(bench_names)[0]).aggregate(rows)
            if len(bench_names) == 1
            else mean_aggregate(rows)
        )
    except KeyError:
        report = mean_aggregate(rows)
    print(format_report(label, report))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, report)
        print(f"Report written to {output}")
    return 0


def _init(args: argparse.Namespace) -> int:
    """Copy the starter examples to ``<directory>/examples``, keeping existing files."""

    package = Path(__file__).resolve().parent
    # Wheels ship the examples inside the package; a source checkout keeps them at the repository root.
    source = (
        package / "examples"
        if (package / "examples").is_dir()
        else package.parents[1] / "examples"
    )
    target = Path(args.directory) / "examples"
    written = kept = 0
    for name in ("tasks", "config", "benchmarks", "env"):
        for path in sorted((source / name).rglob("*")):
            if not path.is_file() or path.suffix == ".env":
                continue
            destination = target / path.relative_to(source)
            if destination.exists():
                kept += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            written += 1
    print(
        f"Wrote {written} files to {target}"
        + (f" ({kept} existing files kept)" if kept else "")
    )
    print(
        f"Next: cp {target / 'env' / 'api.env.example'} {target / 'env' / 'api.env'} and set the model service URL and API key"
    )
    return 0


def _view(args: argparse.Namespace) -> int:
    return serve_viewer(Path(args.run_dir), host=args.host, port=args.port)


def _export(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not (run_dir / "samples").is_dir():
        raise FileNotFoundError(f"{run_dir} has no samples directory")
    output = Path(args.output) if args.output else run_dir / args.format
    if args.format == "hf":
        rules = (
            hf_dataset.load_rules(Path(args.redact_rules))
            if args.redact_rules
            else None
        )
        rows, redaction = hf_dataset.export_run(
            run_dir, output, rules, redact=args.redact, runs=args.include_runs
        )
        print(f"Exported {len(rows)} samples as a Hugging Face dataset to {output}")
        for line in redaction.lines():
            print(line)
        return 0
    if args.redact_rules or args.include_runs:
        raise ValueError("--redact-rules and --include-runs apply to --format hf")
    written, skipped = export_run(run_dir, output)
    print(f"Exported {len(written)} ATIF trajectories to {output}")
    if skipped:
        print(f"Skipped {len(skipped)} samples:")
        for sample, reason in skipped.items():
            print(f"  {sample}: {reason}")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        if args.command == "collect":
            exit_code = _collect(args)
        elif args.command == "bench":
            exit_code = _bench(args)
        elif args.command == "view":
            exit_code = _view(args)
        elif args.command == "export":
            exit_code = _export(args)
        elif args.command == "init":
            exit_code = _init(args)
        else:
            parser.error(f"unsupported command: {args.command}")
            return
    except (
        FileNotFoundError,
        FileExistsError,
        TaskFormatError,
        ValueError,
        RuntimeError,
    ) as exc:
        logger.error("%s", exc)
        exit_code = 2
    raise SystemExit(exit_code)
