"""Validation, preflight checks, and Harness configuration assembly for ``collect``."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import math
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roller.benches import get_bench
from roller.bundles import BundlePolicy
from roller.cleanup import remove_leftover_containers
from roller.docker_runtime import DockerRuntime
from roller.harness_registry import HARNESS_NAMES, HARNESSES
from roller.harnesses.container import host_api_key, host_environment
from roller.tasks import TaskFormatError, build_requests, load_tasks
from roller.thinking import resolve_thinking
from roller.thinking_capabilities import (
    ThinkingCapabilities,
)
from roller.token_ids import logprobs_mode

logger = logging.getLogger(__name__)


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _stirrup_installed() -> bool:
    return importlib.util.find_spec("stirrup") is not None


def _remove_leftover_containers(binary: str, run_dir: Path) -> None:
    """Before a Resume, remove the containers dead collectors left for the run."""

    try:
        removed = remove_leftover_containers(DockerRuntime(binary), run_dir)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.warning("could not look for leftover containers of this run: %s", exc)
        return
    if removed:
        logger.info(
            "removed %d container(s) that an earlier collector of this run left behind",
            removed,
        )


def _docker_preflight(binary: str, image: str) -> str:
    if shutil.which(binary) is None:
        raise RuntimeError(f"Docker executable not found: {binary}")
    runtime = DockerRuntime(binary)
    try:
        result = runtime.version(timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Docker preflight failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Docker daemon is unavailable: {detail}")
    try:
        inspected = runtime.image_field(image, "Id", timeout=15)
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


def _with_task_timeout(request: Any) -> Any:
    """Apply a task's own ``timeout_seconds`` when the row declares one."""

    raw = getattr(request.task, "raw", None)
    declared = raw.get("timeout_seconds") if isinstance(raw, dict) else None
    if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
        return request
    if declared == request.timeout_seconds:
        return request
    return dataclasses.replace(request, timeout_seconds=declared)


def repeat_bench_rows(
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


def _preflight_environment(
    requests: Sequence[Any], cli_harness: str | None, configs: Mapping[str, Any]
) -> None:
    """Fail before the first rollout when a host variable a rollout reads is missing."""

    for request in requests:
        agent = cli_harness or request.task.harness
        host_environment(request.task, configs[agent].pass_env)
        spec = request.task.user_simulator
        for name in (agent, *([spec.harness] if spec is not None else [])):
            if name != "openclaw":
                host_api_key(HARNESSES[name].label, configs[name].api_key_env)


def _token_logprobs(
    requests: Sequence[Any], cli_harness: str | None, configs: Mapping[str, Any]
) -> dict[str, str]:
    """How the endpoint of each model reports logprobs: "processed" or "raw"."""

    modes: dict[str, str] = {}
    for request in requests:
        name = cli_harness or request.task.harness
        if request.model in modes:
            continue
        if name == "openclaw":
            provider, model = request.model.split("/", 1)
            entry = configs[name].model_config["providers"][provider]
            api_base, api_key = entry["baseUrl"], entry["apiKey"]
        else:
            model, api_base = request.model, configs[name].api_base
            api_key = host_api_key(HARNESSES[name].label, configs[name].api_key_env)
        modes[request.model] = logprobs_mode(
            api_base.rstrip("/") + HARNESSES[name].chat_completions, model, api_key
        )
    return modes


def validate_collect_args(args: argparse.Namespace) -> BundlePolicy:
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
    if args.grader_retries < 0:
        raise ValueError("--grader-retries cannot be negative")
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    if args.resume_verify_outputs and not args.resume:
        raise ValueError("--resume-verify-outputs requires --resume")
    if args.resume_unscored and not args.resume:
        raise ValueError("--resume-unscored requires --resume")
    if not math.isfinite(args.retry_backoff) or args.retry_backoff < 0:
        raise ValueError("--retry-backoff must be a finite number of seconds >= 0")
    if args.max_consecutive_errors < 0:
        raise ValueError("--max-consecutive-errors cannot be negative")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if args.bench_runs_per_task < 1:
        raise ValueError("--bench-runs-per-task must be at least 1")
    return BundlePolicy(
        max_file_bytes=args.max_input_file_bytes,
        max_task_input_bytes=args.max_task_input_bytes,
    )


@dataclass
class TaskResolution:
    workspace: Path | None
    tasks_path: Path
    tasks: list[Any]
    requests: list[Any]
    bench_context: dict[str, Any] | None
    bench_manifest: bytes | None
    bench_temporary: tempfile.TemporaryDirectory[str] | None
    agent_harnesses: set[str]
    simulator_harnesses: set[str]
    selected_harnesses: set[str]
    cli_harness: str | None
    remote: bool


def resolve_tasks_and_requests(args: argparse.Namespace) -> TaskResolution:
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
        rows = repeat_bench_rows(rows, args.bench_runs_per_task, adapter.name)
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
    if any(
        request.task.output_dir is None
        and (args.harness or request.task.harness) != "stirrup"
        for request in requests
    ):
        raise ValueError("files.output_dir null is supported by Stirrup only")
    remote = args.sandbox_provider != "docker"
    if remote:
        if not args.sandbox_template:
            raise ValueError("a remote sandbox requires --sandbox-template")
        if agent_harnesses not in ({"stirrup"}, {"claude-code"}) and (
            args.bench != "wildclawbench" or agent_harnesses != {"deepseek"}
        ):
            raise ValueError(
                "remote sandboxes currently support Stirrup and Claude Code, "
                "and WildClawBench with DeepSeek"
            )
        if agent_harnesses == {"claude-code"} and not args.claude_api_base:
            raise ValueError(
                "Claude Code in a remote sandbox requires --claude-api-base"
            )
        if simulator_harnesses:
            raise ValueError(
                "remote sandboxes do not support a separate user simulator"
            )
        if (
            args.docker_image
            or args.deepseek_image
            or args.stirrup_image
            or args.rebuild_task_images
            or any(task.docker_image for task in tasks)
        ):
            raise ValueError(
                "use a prebuilt sandbox template instead of a Docker or task image"
            )
        if args.keep_container_on_error:
            raise ValueError("--keep-container-on-error applies to Docker only")
        if any(
            task.inputs and task.input_mode == "mount" and not task.legacy_workspace
            for task in tasks
        ):
            raise ValueError(
                "remote sandbox tasks with explicit files must use copy input mode"
            )
    elif args.sandbox_template:
        raise ValueError("--sandbox-template requires a remote --sandbox-provider")
    if args.claude_capture_calls and "claude-code" not in selected_harnesses:
        raise ValueError("--claude-capture-calls requires a Claude Code participant")
    cli_harness = args.harness
    args._cli_harness = cli_harness
    args.harness = next(name for name in HARNESS_NAMES if name in agent_harnesses)
    return TaskResolution(
        workspace=workspace,
        tasks_path=tasks_path,
        tasks=tasks,
        requests=requests,
        bench_context=bench_context,
        bench_manifest=bench_manifest,
        bench_temporary=bench_temporary,
        agent_harnesses=agent_harnesses,
        simulator_harnesses=simulator_harnesses,
        selected_harnesses=selected_harnesses,
        cli_harness=cli_harness,
        remote=remote,
    )
