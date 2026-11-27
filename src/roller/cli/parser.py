"""Argument parser for the Roller CLI."""

from __future__ import annotations

import argparse

from roller import importer as importer_module
from roller.benches import list_benches
from roller.harness_registry import HARNESS_NAMES
from roller.harnesses import HARNESS_CLI_OPTIONS
from roller.option_specs import (
    BENCH_SELECTION_OPTIONS,
    EXECUTION_CORE_OPTIONS,
    EXECUTION_RUNTIME_OPTIONS,
    register_options,
)
from roller.version import __version__
from roller.viewer import DEFAULT_PORT as VIEWER_DEFAULT_PORT

_HARNESS_GROUP_LABELS = (
    ("openclaw", "OpenClaw"),
    ("hermes", "Hermes"),
    ("opencode", "OpenCode"),
    ("deepseek", "DeepSeek"),
    ("pi", "Pi"),
    ("stirrup", "Stirrup"),
    ("codex", "Codex"),
    ("claude-code", "Claude Code"),
)


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
        choices=("atif", "hf", "sft", "tokens", "xtuner"),
        default="atif",
        help=(
            "target format: atif, the trajectory format Harbor reads (default), "
            "hf, a Hugging Face dataset folder, sft, the OpenAI-format SFT "
            "records of the run with an index of the task, model and bench "
            "score of each, tokens, the token ids and logprobs --token-ids "
            "recorded, joined into one sequence per conversation, or xtuner, "
            "merged XTuner multimodal SFT JSONL with image_wh, parsed tool "
            "arguments and per-turn loss masks"
        ),
    )
    export.add_argument(
        "--output", "-o", help="directory to write (default: RUN_DIR/<format>)"
    )
    export.add_argument(
        "--min-score",
        type=float,
        metavar="X",
        help=(
            "with --format sft or xtuner, keep only the samples whose bench "
            "score is at least X and leave out the samples without a score. "
            "The score is the one `roller bench report` lists for the sample: "
            "from 0 to 1 for WildClawBench, PinchBench, QwenClawBench, "
            "SkillsBench and VIALS, and 0 or 1 for SWE-Bench Pro"
        ),
    )
    export.add_argument(
        "--inline-images",
        action="store_true",
        help=(
            "with --format xtuner, keep base64 image data URIs inline instead "
            "of writing image files under media/"
        ),
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
    export.add_argument(
        "--preserve-task-keys",
        action="store_true",
        help="with --format hf, preserve deliberate benchmark key fixtures while still redacting known collection credentials and paths",
    )

    importer = subparsers.add_parser(
        "import",
        help="import the local sessions of Claude Code and Codex as a Roller run",
    )
    importer.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help=(
            "session files or directories to import (default: the session "
            "directories of Claude Code and Codex)"
        ),
    )
    importer.add_argument(
        "--harness",
        choices=importer_module.HARNESSES,
        help="import only the sessions of one harness (default: both)",
    )
    importer.add_argument(
        "--output", default="runs", help="output root directory (default: runs)"
    )
    importer.add_argument(
        "--run-name", default="imported", help="run directory name (default: imported)"
    )
    importer.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="import only the sessions that started on or after this date",
    )
    importer.add_argument(
        "--project",
        action="append",
        metavar="GLOB",
        help="import only the sessions of matching projects; repeatable",
    )
    importer.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="sessions converted at once (default: 4)",
    )
    importer.add_argument(
        "--dry-run",
        action="store_true",
        help="list the sessions by project without writing anything",
    )

    cleaner = subparsers.add_parser(
        "clean",
        help="remove containers and bundle directories that dead collectors left behind",
    )
    cleaner.add_argument(
        "path",
        nargs="?",
        help=(
            "limit the cleanup to one run directory or output root (default: "
            "every Roller container, and the bundle directories under runs)"
        ),
    )
    cleaner.add_argument(
        "--dry-run", action="store_true", help="print what would be removed"
    )
    cleaner.add_argument(
        "--force",
        action="store_true",
        help=(
            "also remove what belongs to a running collector, to a collector on "
            "another host, or to no recorded collector"
        ),
    )
    cleaner.add_argument("--docker-binary", default="docker")

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
    bench_report_format = bench_report.add_mutually_exclusive_group()
    bench_report_format.add_argument(
        "--json",
        action="store_true",
        help="print the report JSON to stdout without summary formatting",
    )
    bench_report_format.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="stdout output format (default: text)",
    )

    collect = subparsers.add_parser("collect", help="collect trajectories")
    task_source = collect.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--tasks", help="JSON or JSONL task list")
    task_source.add_argument(
        "--bench",
        choices=tuple(adapter.name for adapter in list_benches()),
        help="load, run, grade, and report a checked-out benchmark in one command",
    )
    register_options(collect, BENCH_SELECTION_OPTIONS)
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
    register_options(collect, EXECUTION_CORE_OPTIONS)
    collect.add_argument(
        "--dry-run", action="store_true", help="validate and print the execution plan"
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
    for flag, label in (
        ("--openclaw-image", "OpenClaw image override (for mixed Harness runs)"),
        ("--codex-image", "Codex image override (for mixed Harness runs)"),
        ("--hermes-image", "Hermes image override (for mixed Harness runs)"),
        ("--opencode-image", "OpenCode image override (for mixed Harness runs)"),
        ("--deepseek-image", "DeepSeek image override (for mixed Harness runs)"),
        ("--pi-image", "Pi image override (for mixed Harness runs)"),
        (
            "--stirrup-image",
            "Stirrup code execution image override (for mixed Harness runs)",
        ),
        ("--claude-code-image", "Claude Code image override (for mixed Harness runs)"),
    ):
        runtime.add_argument(flag, help=label)
    register_options(runtime, EXECUTION_RUNTIME_OPTIONS)

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
    for flag, help_text in (
        (
            "--hermes-api-base",
            "Hermes-only OpenAI-compatible Chat Completions API base",
        ),
        ("--hermes-api-key-env", "Hermes-only API key environment variable"),
        ("--opencode-api-base", "OpenCode-only Chat Completions API base"),
        ("--deepseek-api-base", "DeepSeek-only Chat Completions API base"),
        ("--pi-api-base", "Pi-only Chat Completions API base"),
        ("--opencode-api-key-env", "OpenCode-only API key environment variable"),
        ("--deepseek-api-key-env", "DeepSeek-only API key environment variable"),
        ("--pi-api-key-env", "Pi-only API key environment variable"),
        ("--stirrup-api-base", "Stirrup-only Chat Completions API base"),
        ("--stirrup-api-key-env", "Stirrup-only API key environment variable"),
        ("--codex-api-base", "Codex-only API base override (for mixed Harness runs)"),
        (
            "--claude-api-base",
            "Claude Code-only API base override (for mixed Harness runs)",
        ),
    ):
        api.add_argument(flag, help=help_text)
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

    for harness_id, group_title in _HARNESS_GROUP_LABELS:
        group = collect.add_argument_group(group_title)
        register_options(group, HARNESS_CLI_OPTIONS[harness_id])

    return parser
