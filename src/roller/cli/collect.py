"""Execution driver for ``roller collect``."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roller.benches import binding_from_task_row, get_bench, grading_failure
from roller.benches.grading import GradedHarness
from roller.benches.report import brief_reason
from roller.bundles import TaskBundleStore
from roller.cleanup import BUNDLE_PREFIX, record_owner
from roller.cli.harness_setup import (
    build_harness_configurations,
    build_harness_setup,
)
from roller.cli.preflight import (
    _docker_preflight,
    _remove_leftover_containers,
    _token_logprobs,
    resolve_tasks_and_requests,
    validate_collect_args,
)
from roller.collector import (
    CollectionConfig,
    CollectionStopped,
    Collector,
    check_retired_cot_samples,
    failure_groups,
)
from roller.conversation import ConversationManager
from roller.harnesses import (
    ClaudeCodeHarness,
    CodexHarness,
    DeepSeekHarness,
    HermesHarness,
    OpenClawHarness,
    OpenCodeHarness,
    PiHarness,
    StirrupHarness,
)
from roller.harnesses.gateway_ports import GatewayPortReservation, reserve_gateway_ports
from roller.image_build import prepare_task_images
from roller.models import TaskBundle
from roller.run_lock import (
    RUN_LOCK_SCHEMA,
    build_run_lock,
    complete_legacy_adoption,
    ensure_run_lock,
)
from roller.sandbox import SandboxRun, SandboxTarget, load_sandbox_provider
from roller.storage import atomic_write_json, safe_component
from roller.tasks import TaskFormatError
from roller.trajectory_audit import training_summary
from roller.version import __version__

logger = logging.getLogger(__name__)


def _jsonl_equivalent(left: bytes, right: bytes) -> bool:
    """Compare JSONL snapshots by value, ignoring object key order."""

    try:
        return [json.loads(line) for line in left.splitlines()] == [
            json.loads(line) for line in right.splitlines()
        ]
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def _run_name(harness: str, value: str | None) -> str:
    if value:
        return safe_component(value, max_length=100)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{harness}_{timestamp}"


def _dry_run_payload(
    args: argparse.Namespace, requests: list[Any], bundles: Mapping[str, TaskBundle]
) -> dict[str, Any]:
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
        "docker_image": args.docker_image
        if getattr(args, "sandbox_provider", "docker") == "docker"
        else None,
        "tasks": len({request.task.id for request in requests}),
        "models": list(dict.fromkeys(request.model for request in requests)),
        "model_labels": list(
            dict.fromkeys(request.display_model for request in requests)
        ),
        "trajectories": len(requests),
        "parallel": args.parallel,
        "launch_interval": args.launch_interval,
        "retry_backoff": args.retry_backoff,
        "max_consecutive_errors": args.max_consecutive_errors,
        "grader_parallel": args.grader_parallel or args.parallel,
        "grader_retries": args.grader_retries,
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
                "task_bundle_digest": bundles[request.task.id].manifest_digest,
                "input_tree_digest": bundles[request.task.id].input_tree_digest,
                "context_manifest_digest": bundles[
                    request.task.id
                ].context_manifest_digest,
            }
            for request in requests
        ],
    }
    if getattr(args, "sandbox_provider", "docker") != "docker":
        payload.update(
            sandbox_provider=args.sandbox_provider,
            sandbox_template=args.sandbox_template,
        )
    bench_run_config = getattr(args, "_bench_run_config", None)
    if isinstance(bench_run_config, dict):
        payload["bench_run_config"] = bench_run_config
    return payload


@contextlib.contextmanager
def _run_log(path: Path) -> Iterator[None]:
    """Append this invocation's log, INFO and above, to ``path``."""

    root = logging.getLogger()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    level = root.level
    console = [(item, item.level) for item in root.handlers if level > logging.INFO]
    for item, own_level in console:
        item.setLevel(max(own_level, level))
    root.setLevel(min(level, logging.INFO))
    root.addHandler(handler)
    try:
        yield
    except Exception as exc:
        handler.handle(
            logging.LogRecord(__name__, logging.ERROR, __file__, 0, "%s", (exc,), None)
        )
        raise
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(level)
        for item, own_level in console:
            item.setLevel(own_level)


def _report_stop(stop: BaseException, run_dir: Path, args: argparse.Namespace) -> int:
    """Log why a run stopped early and how to continue it; return the exit code."""

    if getattr(args, "_bench_run_config", None) is not None:
        execution = {"resume": True}
        if not args.run_name:
            execution["run_name"] = run_dir.name
        resume = f"--extra-config '{json.dumps({'execution': execution})}'"
    else:
        resume = "--resume" if args.run_name else f"--resume --run-name {run_dir.name}"
    if isinstance(stop, CollectionStopped) and stop.exit_code < 128:
        logger.error(
            "stopped: %s. Fix the cause and rerun the same command with %s",
            stop,
            resume,
        )
    else:
        logger.error(
            "%s. Rerun the same command with %s to continue",
            f"stopped: {stop}" if str(stop) else "interrupted",
            resume,
        )
    logger.error("Run directory: %s", run_dir)
    return getattr(stop, "exit_code", 130)


def _print_grading_failures(results: Sequence[Mapping[str, Any]]) -> int:
    """Say how many samples finished but have no score, and why; return the count."""

    reasons = Counter(
        brief_reason(reason)
        for result in results
        if (reason := grading_failure(result)) is not None
    )
    count = sum(reasons.values())
    if not count:
        return 0
    print(
        f"Grading failed for {count} of {len(results)} samples; they have no score. "
        "Most common reasons:"
    )
    for reason, times in reasons.most_common(3):
        print(f"  {times}x {reason}")
    print(
        "Rerun them with --resume --resume-unscored once the cause is fixed; "
        "that runs the agent again."
    )
    return count


def _collect(args: argparse.Namespace) -> int:
    from roller.cli.bench import _bench_report

    bundle_policy = validate_collect_args(args)
    resolution = resolve_tasks_and_requests(args)
    setup = build_harness_setup(args, resolution)

    tasks = resolution.tasks
    requests = resolution.requests
    selected_harnesses = resolution.selected_harnesses
    agent_harnesses = resolution.agent_harnesses
    simulator_harnesses = resolution.simulator_harnesses
    cli_harness = resolution.cli_harness

    if args.dry_run:
        bundles = TaskBundleStore(
            Path(args.output), bundle_policy, materialize=False
        ).prepare_all(tasks)
        payload = _dry_run_payload(args, requests, bundles)
        if resolution.bench_context is not None:
            payload["bench"] = resolution.bench_context
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    run_label = next(iter(agent_harnesses)) if len(agent_harnesses) == 1 else "mixed"
    output_root = Path(args.output).expanduser().resolve()
    run_dir = output_root / _run_name(run_label, args.run_name)
    sandbox_run = None
    if resolution.remote:
        provider = load_sandbox_provider(args.sandbox_provider)
        version = provider.resolve(args.sandbox_template)
        if not version:
            raise ValueError("sandbox provider did not resolve an environment version")
        target = SandboxTarget(args.sandbox_provider, args.sandbox_template, version)
        sandbox_run = SandboxRun(run_dir, target, provider)
        for name in ("deepseek", "claude-code", "stirrup"):
            if setup.configs[name] is not None:
                setup.configs[name] = dataclasses.replace(
                    setup.configs[name], sandbox_run=sandbox_run
                )
        docker_image_digests: dict[str, str] = {}
        task_image_digests: dict[str, str] = {}
    else:
        docker_image_digests = {
            harness: _docker_preflight(args.docker_binary, image)
            for harness, image in setup.images.items()
        }
        task_image_digests = prepare_task_images(
            args.docker_binary,
            requests,
            setup.images,
            cli_harness=cli_harness,
            rebuild=args.rebuild_task_images,
        )

    existing_run = run_dir.exists() and any(run_dir.iterdir())
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"run directory already exists; use --resume or another --run-name: {run_dir}"
        )
    if existing_run:
        check_retired_cot_samples(run_dir, requests)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks_snapshot: Path | None = None
    if resolution.bench_manifest is not None:
        tasks_snapshot = run_dir / "tasks.jsonl"
        if tasks_snapshot.is_file():
            recorded_manifest = tasks_snapshot.read_bytes()
            if recorded_manifest != resolution.bench_manifest and not _jsonl_equivalent(
                recorded_manifest, resolution.bench_manifest
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
    harness_configurations = build_harness_configurations(args, setup, sandbox_run)
    simulator_gateway_slot_offset = (
        args.parallel if "openclaw" in simulator_harnesses else None
    )
    run_info: dict[str, Any] = {
        "schema_version": "roller.run/v1",
        "roller_version": __version__,
        "configuration": {
            "harness": run_label,
            "agent_harnesses": sorted(agent_harnesses),
            "simulator_harnesses": sorted(simulator_harnesses),
            "tasks": str(tasks_snapshot or resolution.tasks_path),
            "bench": (
                {**resolution.bench_context, "tasks_snapshot": "tasks.jsonl"}
                if resolution.bench_context is not None
                else None
            ),
            "bench_run_config": getattr(args, "_bench_run_config", None),
            "models": args.model,
            **(
                {
                    "token_ids": {
                        "logprobs": _token_logprobs(
                            requests, cli_harness, setup.configs
                        )
                    }
                }
                if args.token_ids
                else {}
            ),
            "trajectory_count": len(requests),
            "workspace": str(resolution.workspace) if resolution.workspace else None,
            "timeout_seconds": args.timeout,
            "parallel": args.parallel,
            "launch_interval": args.launch_interval,
            "grader_parallel": args.grader_parallel or args.parallel,
            "grader_retries": args.grader_retries,
            "samples": args.samples,
            "retries": args.retries,
            "retry_backoff": args.retry_backoff,
            "max_consecutive_errors": args.max_consecutive_errors,
            "start_index": args.start_index,
            "limit": args.limit,
            "resume": args.resume,
            "resume_unscored": args.resume_unscored,
            "training_files_only": args.training_files_only,
            "task_schema": "roller.task/v1",
            "thinking_capabilities_path": (
                str(setup.thinking_capabilities_path)
                if setup.thinking_capabilities_path
                else None
            ),
            "thinking_capabilities_sha256": setup.thinking_capabilities_digest,
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

    results: list[dict[str, Any]] | None = None
    gateway_ports: GatewayPortReservation | None = None
    with (
        tempfile.TemporaryDirectory(prefix=BUNDLE_PREFIX, dir=output_root) as temporary,
        contextlib.ExitStack() as run_log,
    ):
        record_owner(Path(temporary), run_dir)
        bundle_store = TaskBundleStore(Path(temporary), bundle_policy)
        try:
            if setup.configs["openclaw"] is not None and args.network == "host":
                gateway_ports = reserve_gateway_ports(
                    args.gateway_port,
                    args.gateway_port_step,
                    args.parallel + (simulator_gateway_slot_offset or 0),
                )
                ports = gateway_ports.ports
                setup.configs["openclaw"] = dataclasses.replace(
                    setup.configs["openclaw"], gateway_port_start=ports.start
                )
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
            run_log.enter_context(_run_log(run_dir / "collect.log"))
            if tasks_snapshot is not None and not tasks_snapshot.exists():
                tasks_snapshot.write_bytes(resolution.bench_manifest or b"")
            execution = {
                "harness": run_label,
                "agent_harnesses": sorted(agent_harnesses),
                "simulator_harnesses": sorted(simulator_harnesses),
                "task_schema": "roller.task/v1",
                "output_collection": "all",
                "thinking_capabilities_sha256": setup.thinking_capabilities_digest,
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
                        **(
                            {
                                "docker_image": setup.images[name],
                                "docker_image_digest": docker_image_digests[name],
                            }
                            if sandbox_run is None
                            else {}
                        ),
                    }
                    for name, payload in harness_configurations.items()
                },
            }
            lock_model_config = (
                setup.model_config if "openclaw" in selected_harnesses else None
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
                legacy_fingerprint = (
                    hashlib.blake2b(
                        setup.claude_settings_content, digest_size=16
                    ).hexdigest()
                    if setup.claude_settings_content is not None
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
                "openclaw": OpenClawHarness,
                "codex": CodexHarness,
                "claude-code": ClaudeCodeHarness,
                "hermes": HermesHarness,
                "opencode": OpenCodeHarness,
                "deepseek": DeepSeekHarness,
                "pi": PiHarness,
                "stirrup": StirrupHarness,
            }

            def make_harness(
                name: str,
                slot: int,
                task_image: str | None = None,
                task_workspace: str | None = None,
                task_image_digest: str | None = None,
            ) -> Any:
                if name not in harness_types:
                    raise ValueError(f"unknown Harness: {name}")
                base_config = setup.configs[name]
                assert base_config is not None
                replacements = {
                    "preflight_image_digest": (
                        task_image_digest
                        if task_image is not None
                        else docker_image_digests.get(name)
                    )
                }
                if task_image:
                    replacements["docker_image"] = task_image
                if task_workspace:
                    replacements["workspace_container_path"] = task_workspace
                return harness_types[name](
                    dataclasses.replace(base_config, **replacements), slot
                )

            def harness_factory(slot: int, request: Any) -> Any:
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
                    try:
                        adapter = get_bench(binding.name)
                    except KeyError:
                        return GradedHarness(
                            agent,
                            binding,
                            grading_semaphore=grading_semaphore,
                            grader_retries=args.grader_retries,
                        )
                    return adapter.wrap_harness(
                        agent,
                        binding,
                        grading_semaphore=grading_semaphore,
                        grader_retries=args.grader_retries,
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
                    max_consecutive_errors=args.max_consecutive_errors,
                    resume=lock_decision.reuse_existing_samples,
                    verify_outputs=args.resume_verify_outputs,
                    resume_unscored=args.resume_unscored,
                    training_files_only=args.training_files_only,
                    keep_aux_traces=keep_aux_traces,
                    capture_cache_trace=(
                        args.cache_trace if "openclaw" in selected_harnesses else False
                    ),
                    keep_thinking=args.keep_thinking,
                    sandbox_run=sandbox_run,
                ),
                harness_factory=harness_factory,
            )
            if args.resume and existing_run:
                if sandbox_run is None:
                    _remove_leftover_containers(args.docker_binary, run_dir)
                else:
                    sandbox_run.cleanup()
            try:
                results = collector.collect(requests)
            except (KeyboardInterrupt, CollectionStopped) as stop:
                if resolution.bench_temporary is not None:
                    resolution.bench_temporary.cleanup()
                return _report_stop(stop, run_dir, args)
            if lock_decision.legacy_run:
                complete_legacy_adoption(run_dir / "lock.json")
            failed = sum(
                result.get("status") in {"error", "timeout"} for result in results
            )
            if failed:
                logger.warning(
                    "%d rollout(s) failed; the full error of each is in its "
                    "sample's info.json. Most frequent reasons:",
                    failed,
                )
                for count, reason, task_id in failure_groups(results):
                    logger.warning("  %dx %s (for example %s)", count, reason, task_id)
        finally:
            bundle_store.cleanup_runtime()
            if gateway_ports is not None:
                gateway_ports.release()
    assert results is not None
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
    if resolution.bench_context is not None:
        report_path = run_dir / "bench-report.json"
        try:
            _bench_report(
                argparse.Namespace(run_dir=str(run_dir), output=str(report_path))
            )
        except TaskFormatError:
            if not failed:
                raise
            logger.warning(
                "benchmark report was not generated because the run has no bench samples"
            )
    grading_failed = _print_grading_failures(results)
    return 1 if failed or grading_failed else 0
