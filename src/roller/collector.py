from __future__ import annotations

import fnmatch
import inspect
import json
import logging
import math
import queue
import re
import shutil
import signal
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roller.benches.report import brief_reason, grading_failure
from roller.capture import (
    read_jsonl,
    recover_capture_retention,
    validate_capture,
)
from roller.cleanup import remove_own_containers
from roller.compaction_sft import (
    SFT_SEGMENTS_DIR,
    export_compaction_segments,
    has_main_compaction,
)
from roller.harness_registry import allows_raw_only_actf
from roller.harnesses.base import Harness
from roller.harnesses.docker import begin_collection
from roller.integrity import (
    has_legacy_claude_sidechain,
    keeps_thinking,
    trajectory_conversion_unavailable,
    validate_resumable_sample,
)
from roller.models import HarnessResult, RunRequest
from roller.sft import SFTConversionError, write_sft
from roller.sft_dataset import write_sft_dataset
from roller.storage import atomic_write_json, safe_component, write_jsonl
from roller.trajectory_audit import record_training_audits, training_summary

logger = logging.getLogger(__name__)
# New factories may inspect the request to select a task-level harness.  The
# one-argument form remains supported for downstream callers and older tests.
HarnessFactory = Callable[..., Harness]
RUN_CHECKPOINT_RESULT_INTERVAL = 25
RUN_CHECKPOINT_TIME_INTERVAL_SECONDS = 30.0
# Silence after which a run reports the rollouts it is still waiting for.
HEARTBEAT_SECONDS = 60.0


def _clear_sft_segments(sample_dir: Path) -> None:
    path = sample_dir / SFT_SEGMENTS_DIR
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def error_line(error: object, limit: int = 300) -> str:
    """The first line of an error, short enough for one log line."""

    text = str(error or "").strip()
    line = text.splitlines()[0].strip() if text else "no error message"
    return line if len(line) <= limit else line[: limit - 3] + "..."


def failure_groups(
    results: list[dict[str, Any]], limit: int = 5
) -> list[tuple[int, str, str]]:
    """Failed results by reason, most frequent first: (count, reason, example task)."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") in {"error", "timeout"}:
            # Ids, ports and counts differ between rollouts that fail alike.
            key = re.sub(r"[0-9a-f]{8,}|\d+", "#", error_line(result.get("error")))
            groups.setdefault(key, []).append(result)
    return [
        (len(group), error_line(group[0].get("error")), str(group[0].get("task_id")))
        for group in sorted(groups.values(), key=len, reverse=True)[:limit]
    ]


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    if not minutes:
        return f"{seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def _remove_path(path: Path) -> None:
    # Never follow a symlink out of the sample directory.
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _sample_directory(run_dir: Path, request: RunRequest) -> Path:
    model_dir = safe_component(request.display_model)
    task_dir = f"{request.task.source_index:08d}_{safe_component(request.task.id)}"
    if request.sample_count > 1:
        task_dir += f"__sample_{request.sample_index + 1:03d}"
    return run_dir / "samples" / model_dir / task_dir


def check_retired_cot_samples(run_dir: Path, requests: list[RunRequest]) -> None:
    """Recover capture publications and reject incompatible old samples before collection."""

    for request in requests:
        sample_dir = _sample_directory(run_dir, request)
        for base in (sample_dir, sample_dir / "user_simulator"):
            info_path = base / "info.json"
            capture_root = base / "capture"
            if capture_root.is_dir():
                interrupted = {
                    match.group(1)
                    for entry in capture_root.iterdir()
                    if (
                        match := re.fullmatch(
                            r"\.(attempt-\d+)\.retention-(?:stage|backup)", entry.name
                        )
                    )
                    is not None
                }
                for attempt in interrupted:
                    recover_capture_retention(
                        capture_root / attempt, prefer_raw=not info_path.is_file()
                    )
                if interrupted and not info_path.is_file():
                    raise ValueError(
                        f"{base}: capture retention was interrupted; a valid capture attempt was recovered, "
                        "but the sample has no finalized info.json. Resume stopped before rerunning the model."
                    )
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                isinstance(info, dict)
                and info.get("harness") == "claude-code"
                and info.get("status") == "ok"
            ):
                actf_path = base / "trace_ACTF.json"
                try:
                    actf = json.loads(actf_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    actf = None
                if isinstance(actf, dict) and has_legacy_claude_sidechain(actf):
                    raise ValueError(
                        f"{actf_path}: legacy Claude Code sidechain steps are mixed into root ACTF/SFT; "
                        "resume stopped before changing the run or calling the model. "
                        "Reproject the existing raw trace offline or use a new run directory."
                    )
            runtime = info.get("runtime") if isinstance(info, dict) else None
            if not isinstance(runtime, dict):
                continue
            for key in ("external_thinking", "signature_recovery"):
                value = runtime.get(key)
                mode = value.get("mode") if isinstance(value, dict) else value
                if mode in ("external", "external_always", "fill", "prefer"):
                    raise ValueError(
                        f"{info_path}: {key}={mode} is retired; use a new run "
                        "directory. Existing trajectory files were not changed."
                    )


@dataclass(frozen=True)
class CollectionConfig:
    run_dir: Path
    parallel: int = 1
    retries: int = 0
    resume: bool = False
    keep_aux_traces: bool = False
    capture_cache_trace: bool = True
    keep_thinking: bool = True
    # Minimum seconds between two rollout launches across all workers.
    launch_interval: float = 0.0
    # Seconds to wait before the first retry of a sample; doubles per retry.
    retry_backoff: float = 0.0
    # On Resume, reread recorded output files and compare their SHA-256.
    verify_outputs: bool = False
    # Stop the run after this many rollouts in a row end in error; 0 never stops.
    max_consecutive_errors: int = 0
    # On Resume, rerun bench samples whose agent finished but whose grading failed.
    resume_unscored: bool = False

    def __post_init__(self) -> None:
        if self.parallel < 1:
            raise ValueError("parallel must be at least 1")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")
        if not math.isfinite(self.launch_interval) or self.launch_interval < 0:
            raise ValueError("launch_interval must be a finite number of seconds >= 0")
        if not math.isfinite(self.retry_backoff) or self.retry_backoff < 0:
            raise ValueError("retry_backoff must be a finite number of seconds >= 0")
        if self.max_consecutive_errors < 0:
            raise ValueError("max_consecutive_errors cannot be negative")


class CollectionStopped(Exception):
    """The run stopped before every rollout ran; it is checkpointed for Resume."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class LaunchCancelled(Exception):
    """A rollout was waiting for its launch turn when the run was interrupted."""


class _LaunchGate:
    """Space rollout launches ``interval`` seconds apart, first come first served.

    Starting every worker at once stacks their start-up peaks: an OpenClaw
    gateway needs about twice its steady memory while it boots.
    """

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._next_launch = 0.0
        self._cancelled = threading.Event()

    def wait_turn(self) -> bool:
        """Block until this launch may start; False once the run is interrupted."""

        with self._lock:
            launch = max(time.monotonic(), self._next_launch)
            self._next_launch = launch + self._interval
        return not self._cancelled.wait(max(0.0, launch - time.monotonic()))

    def pause(self, seconds: float) -> bool:
        """Sleep without holding a launch turn; False once the run is interrupted."""

        return not self._cancelled.wait(seconds)

    def cancel(self) -> None:
        self._cancelled.set()


class Collector:
    def __init__(
        self, config: CollectionConfig, harness_factory: HarnessFactory
    ) -> None:
        self.config = config
        self.harness_factory = harness_factory
        self.run_dir = config.run_dir.expanduser().resolve()
        try:
            parameters = list(inspect.signature(harness_factory).parameters.values())
            positional = [
                parameter
                for parameter in parameters
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]
            self._factory_accepts_request = (
                any(
                    parameter.kind == inspect.Parameter.VAR_POSITIONAL
                    for parameter in parameters
                )
                or len(positional) >= 2
            )
            self._factory_request_keyword = not self._factory_accepts_request and any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            self._factory_accepts_request = True
            self._factory_request_keyword = False

    def _make_harness(self, worker_slot: int, request: RunRequest) -> Harness:
        if self._factory_accepts_request:
            return self.harness_factory(worker_slot, request)
        if self._factory_request_keyword:
            return self.harness_factory(worker_slot, request=request)
        return self.harness_factory(worker_slot)

    def _sample_dir(self, request: RunRequest) -> Path:
        return _sample_directory(self.run_dir, request)

    def _load_resumable(
        self, sample_dir: Path, request: RunRequest
    ) -> dict[str, Any] | None:
        info_path = sample_dir / "info.json"
        if not self.config.resume or not info_path.is_file():
            return None
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if request.bundle is not None:
            if not request.bundle.resume_safe:
                logger.info(
                    "[%s | %s] context capture=none disables resume",
                    request.task.id,
                    request.model,
                )
                return None
            provenance = payload.get("provenance")
            if (
                not isinstance(provenance, dict)
                or provenance.get("task_bundle_digest")
                != request.bundle.manifest_digest
            ):
                logger.info(
                    "[%s | %s] task bundle changed; rerunning",
                    request.task.id,
                    request.model,
                )
                return None
        try:
            rebuilt = validate_resumable_sample(
                sample_dir,
                request,
                payload,
                require_wire_trace=self.config.keep_aux_traces,
                require_cache_trace=(
                    self.config.keep_aux_traces and self.config.capture_cache_trace
                ),
                verify_output_sha256=self.config.verify_outputs,
                rebuild_sft=True,
            )
        except ValueError as exc:
            logger.info(
                "[%s | %s] existing sample failed integrity validation; rerunning: %s",
                request.task.id,
                request.model,
                exc,
            )
            return None
        if self.config.resume_unscored and (reason := grading_failure(payload)):
            logger.info(
                "[%s | %s] grading failed (%s); rerunning",
                request.task.id,
                request.model,
                brief_reason(reason),
            )
            return None
        if rebuilt:
            logger.info(
                "[%s | %s] rebuilt %s from the trace; it differed from the SFT "
                "this Roller version derives",
                request.task.id,
                request.model,
                ", ".join(path.relative_to(sample_dir).as_posix() for path in rebuilt),
            )
        if payload.get("request_index") != request.index:
            # The run grew: the sample takes its position in the current list.
            payload["request_index"] = request.index
            atomic_write_json(info_path, payload)
        resumed = dict(payload)
        resumed["status"] = "skipped"
        resumed["resumed_from_status"] = "ok"
        return resumed

    @staticmethod
    def _trace_source(sample_dir: Path, value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = sample_dir / path
        return path if path.is_file() else None

    @staticmethod
    def _prune_sample(
        sample_dir: Path,
        keep_aux_traces: bool = False,
        keep_rollout_logs: bool = False,
    ) -> None:
        keep = {
            "trace.jsonl",
            "trace_ACTF.json",
            "trace_SFT.json",
            SFT_SEGMENTS_DIR,
            "tools_capture.jsonl",  # Exact Codex Responses tool snapshots.
            "signature_recovery.jsonl",  # Preserve historical artifacts; never generated now.
            "tool_id_mappings.jsonl",  # Native IDs behind --claude-unique-tool-ids repairs.
            "info.json",
            "results",
            "conversation.json",
            "user_simulator",
            "capture",  # Formal per-attempt calls/native evidence, not diagnostics.
            # Written by bench grading, which runs before the container dies.
            "score.json",
        }
        if keep_aux_traces:
            # Full mode: retain the diagnostic layers alongside the ACTF trajectory.
            keep |= {"cache_trace.jsonl", "wire_trace.jsonl"}
        for path in sample_dir.iterdir():
            if path.name in keep:
                continue
            if (
                keep_rollout_logs
                and path.name == "rollout"
                and path.is_dir()
                and not path.is_symlink()
            ):
                # Failed samples keep every attempt's process logs for diagnosis.
                for item in path.iterdir():
                    if not (
                        item.is_file()
                        and not item.is_symlink()
                        and fnmatch.fnmatch(item.name, "*.attempt-*.log")
                    ):
                        _remove_path(item)
                if not any(path.iterdir()):
                    path.rmdir()
                continue
            _remove_path(path)

    def _finalize_sample(
        self,
        request: RunRequest,
        sample_dir: Path,
        harness_name: str,
        final: HarnessResult,
        attempts: list[dict[str, Any]],
        started_at: str,
    ) -> dict[str, Any]:
        metadata = final.metadata if isinstance(final.metadata, dict) else {}
        trace_source = self._trace_source(sample_dir, final.trace_path)
        trajectory_source = self._trace_source(sample_dir, final.trajectory_path)
        trace_path = sample_dir / "trace.jsonl"
        actf_path = sample_dir / "trace_ACTF.json"
        sft_path = sample_dir / "trace_SFT.json"
        sft_segments: list[dict[str, Any]] = []
        _clear_sft_segments(sample_dir)
        if trace_source is None and final.status == "ok":
            final.status = "error"
            final.error = "harness finished but no raw trace was found"
        elif (
            trajectory_source is None
            and final.status == "ok"
            and not (
                allows_raw_only_actf(harness_name)
                and trajectory_conversion_unavailable(metadata) == "actf"
            )
        ):
            final.status = "error"
            final.error = "harness finished but no canonical ACTF trajectory was found"

        if trace_source is not None:
            if trace_source.resolve() != trace_path.resolve():
                shutil.copy2(trace_source, trace_path)

        trajectory = None
        if trajectory_source is not None:
            if trajectory_source.resolve() != actf_path.resolve():
                shutil.copy2(trajectory_source, actf_path)
            try:
                trajectory = json.loads(actf_path.read_text(encoding="utf-8"))
                if not isinstance(trajectory, dict):
                    raise ValueError("canonical ACTF trajectory must be a JSON object")
                if final.status == "ok":
                    if trajectory_conversion_unavailable(metadata) == "sft":
                        raise SFTConversionError(
                            metadata["trajectory_conversion"]["error"]
                        )
                    capture = metadata.get("capture") or {}
                    manifest = (
                        capture.get("manifest") if isinstance(capture, dict) else None
                    )
                    if isinstance(manifest, str):
                        sft_segments = export_compaction_segments(
                            sample_dir,
                            harness_name,
                            manifest,
                            keep_thinking=keeps_thinking(metadata),
                        )
                        if sft_segments:
                            sft_path.unlink(missing_ok=True)
                        else:
                            write_sft(sft_path, trajectory)
                    elif has_main_compaction(trace_path, harness_name):
                        raise SFTConversionError(
                            "context reset SFT requires captured Calls"
                        )
                    else:
                        write_sft(sft_path, trajectory)
            except SFTConversionError as exc:
                logger.warning(
                    "[%s | %s] SFT export unavailable: %s",
                    request.task.id,
                    request.model,
                    exc,
                )
                metadata["trajectory_conversion"] = {
                    "status": "unavailable",
                    "target": "sft",
                    "error": str(exc),
                }
                sft_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.exception(
                    "[%s | %s] trajectory conversion failed",
                    request.task.id,
                    request.model,
                )
                final.status = "error"
                final.error = f"trajectory conversion failed: {exc}"
        # Failed/partial rollouts are diagnostic evidence, not training exports.
        # Also remove stale exports left by an earlier attempt or failed write.
        if final.status != "ok":
            sft_path.unlink(missing_ok=True)
            _clear_sft_segments(sample_dir)
            sft_segments = []
        record_training_audits(
            metadata,
            trajectory if isinstance(trajectory, dict) else None,
            sample_dir / "wire_trace.jsonl",
        )
        if attempts and attempts[-1].get("status") != final.status:
            attempts[-1]["status"] = final.status
            attempts[-1]["error"] = final.error

        provenance = metadata.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        if request.bundle is not None:
            provenance = {
                **provenance,
                "task_bundle_digest": request.bundle.manifest_digest,
                "input_tree_digest": request.bundle.input_tree_digest,
                "input_manifest_digest": request.bundle.input_manifest_digest,
                "context_manifest_digest": request.bundle.context_manifest_digest,
                "resume_safe": request.bundle.resume_safe,
            }

        result = {
            "schema_version": "roller.sample/v1",
            "request_index": request.index,
            "sample_index": request.sample_index,
            "sample_count": request.sample_count,
            "task_id": request.task.id,
            "source_index": request.task.source_index,
            "model": request.model,
            "model_label": request.display_model,
            "harness": harness_name,
            "status": final.status,
            "error": final.error,
            "usage": final.usage,
            "attempts": attempts,
            "timing": {
                "elapsed_seconds": round(final.elapsed_seconds, 3),
                "started_at": started_at,
                "finished_at": _now(),
            },
            "trajectory": {
                "raw": "trace.jsonl" if trace_path.is_file() else None,
                "raw_schema": metadata.get("raw_trace_schema"),
                "actf": "trace_ACTF.json" if actf_path.is_file() else None,
                "actf_schema": "ACTF_v1.0" if actf_path.is_file() else None,
                "sft": "trace_SFT.json" if sft_path.is_file() else None,
                **({"sft_segments": sft_segments} if sft_segments else {}),
            },
            "conversation": metadata.get("conversation"),
            "output_collection": metadata.get("output_collection"),
            "runtime": {
                key: value
                for key, value in metadata.items()
                if key not in {"conversation", "output_collection", "provenance"}
            },
            "provenance": provenance,
        }
        self._prune_sample(
            sample_dir,
            keep_aux_traces=self.config.keep_aux_traces,
            keep_rollout_logs=final.status != "ok",
        )
        atomic_write_json(sample_dir / "info.json", result)
        return result

    def _resumed(self, request: RunRequest) -> dict[str, Any] | None:
        """The stored result of a rollout that an earlier invocation completed."""

        resumed = self._load_resumable(self._sample_dir(request), request)
        if resumed is not None:
            resumed.update(
                {
                    "request_index": request.index,
                    "sample_index": request.sample_index,
                    "sample_count": request.sample_count,
                    "task_id": request.task.id,
                    "source_index": request.task.source_index,
                    "model": request.model,
                    "model_label": request.display_model,
                }
            )
            logger.debug(
                "[%s | %s] already complete; skipping", request.task.id, request.model
            )
        return resumed

    def _run_one(self, request: RunRequest, worker_slot: int) -> dict[str, Any]:
        sample_dir = self._sample_dir(request)
        # Wait before touching the sample directory, so a rollout cancelled
        # here leaves nothing behind, like a request that never left the queue.
        if not self._launch_gate.wait_turn():
            raise LaunchCancelled(f"{request.task.id}: run interrupted before launch")
        with self._running_lock:
            self._running[request.index] = (request, time.monotonic())
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        started_at = _now()
        attempt_rows: list[dict[str, Any]] = []
        final = HarnessResult.error_result("harness did not run")
        harness_name = "unknown"
        for attempt in range(1, self.config.retries + 2):
            if attempt > 1:
                backoff = self.config.retry_backoff * 2 ** (attempt - 2)
                if backoff:
                    logger.warning(
                        "[%s | %s] attempt %d failed: %s; waiting %gs before retrying",
                        request.task.id,
                        request.model,
                        attempt - 1,
                        error_line(final.error),
                        backoff,
                    )
                # The backoff runs first, so the launch turn is taken when the
                # retry is ready to start and no other launch waits behind it.
                if not (
                    self._launch_gate.pause(backoff) and self._launch_gate.wait_turn()
                ):
                    # Interrupted between attempts: keep the last failure.
                    logger.warning(
                        "[%s | %s] interrupted; not retrying after attempt %d",
                        request.task.id,
                        request.model,
                        attempt - 1,
                    )
                    break
                if not backoff:
                    logger.warning(
                        "[%s | %s] attempt %d failed: %s; retrying",
                        request.task.id,
                        request.model,
                        attempt - 1,
                        error_line(final.error),
                    )
            harness = self._make_harness(worker_slot, request)
            harness_name = getattr(harness, "name", "unknown")
            attempt_start = time.perf_counter()
            try:
                final = harness.run(request, sample_dir, attempt)
            except Exception as exc:
                logger.exception(
                    "[%s | %s] harness exception", request.task.id, request.model
                )
                final = HarnessResult.error_result(
                    str(exc), time.perf_counter() - attempt_start
                )
            attempt_rows.append(
                {
                    "attempt": attempt,
                    "status": final.status,
                    "error": final.error,
                    "elapsed_seconds": round(final.elapsed_seconds, 3),
                }
            )
            if final.status == "ok":
                break

        result = self._finalize_sample(
            request, sample_dir, harness_name, final, attempt_rows, started_at
        )
        if result["status"] != "ok":
            logger.warning(
                "[%s | %s] %s after %d attempt(s): %s",
                request.task.id,
                request.model,
                result["status"],
                len(attempt_rows),
                error_line(result["error"]),
            )
        return result

    @staticmethod
    def _usage_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
        totals: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }
        for result in results:
            usage = result.get("usage")
            if not isinstance(usage, dict):
                continue
            for key in totals:
                value = usage.get(key, 0)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] += value
        totals["cost_usd"] = round(float(totals["cost_usd"]), 6)
        return totals

    def _sample_dir_from_result(self, result: dict[str, Any]) -> Path:
        model = safe_component(
            str(result.get("model_label", result.get("model", "unknown")))
        )
        source_index = int(result.get("source_index", result.get("request_index", 0)))
        task = safe_component(str(result.get("task_id", "unknown")))
        task_dir = f"{source_index:08d}_{task}"
        sample_count = int(result.get("sample_count", 1))
        if sample_count > 1:
            sample_index = int(result.get("sample_index", 0))
            task_dir += f"__sample_{sample_index + 1:03d}"
        return self.run_dir / "samples" / model / task_dir

    def _write_run_info(
        self,
        results: list[dict[str, Any]],
        *,
        total_requests: int,
        state: str,
    ) -> None:
        info_path = self.run_dir / "info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            info = {"schema_version": "roller.run/v1"}
        summary = {
            "total": total_requests,
            "processed": len(results),
            "pending": max(0, total_requests - len(results)),
            "ok": sum(result.get("status") == "ok" for result in results),
            "error": sum(result.get("status") == "error" for result in results),
            "timeout": sum(result.get("status") == "timeout" for result in results),
            "skipped": sum(result.get("status") == "skipped" for result in results),
            "usage": self._usage_summary(results),
            "training": training_summary(results),
        }
        summary["completed"] = summary["ok"] + summary["skipped"]
        summary["failed"] = summary["error"] + summary["timeout"]
        info.update(
            {
                "state": state,
                "summary": summary,
                "samples": [
                    {
                        "request_index": result.get("request_index"),
                        "sample_index": result.get("sample_index", 0),
                        "sample_count": result.get("sample_count", 1),
                        "task_id": result.get("task_id"),
                        "model": result.get("model"),
                        "model_label": result.get("model_label", result.get("model")),
                        "status": result.get("status"),
                        "error": result.get("error"),
                        "path": self._sample_dir_from_result(result)
                        .relative_to(self.run_dir)
                        .as_posix(),
                        "usage": result.get("usage", {}),
                    }
                    for result in results
                ],
                "updated_at": _now(),
            }
        )
        if state == "finished":
            info["finished_at"] = info["updated_at"]
        else:
            info.pop("finished_at", None)
        atomic_write_json(info_path, info)

    def _write_sft_dataset(self, results: list[dict[str, Any]]) -> None:
        write_sft_dataset(
            self.run_dir,
            self.run_dir,
            [(self._sample_dir_from_result(result), result) for result in results],
        )

        def calls() -> Iterator[dict[str, Any]]:
            for result in results:
                if result.get("status") not in {"ok", "skipped"}:
                    continue
                capture = result.get("runtime", {}).get("capture")
                if not isinstance(capture, dict) or not capture.get("manifest"):
                    continue
                sample = self._sample_dir_from_result(result)
                validate_capture(sample, capture)
                manifest_path = sample / capture["manifest"]
                for call in read_jsonl(manifest_path.parent / "calls.jsonl"):
                    if call.get("export_status") == "available":
                        yield {
                            **call,
                            "sample": sample.relative_to(self.run_dir).as_posix(),
                            "capture_manifest": capture["manifest"],
                            "task_id": result["task_id"],
                            "model_label": result.get("model_label"),
                            "sample_index": result.get("sample_index", 0),
                        }

        # Separate schema; historical assistant content is input, output is the
        # current response. Never append these records to legacy sft.jsonl.
        if any(isinstance(r.get("runtime", {}).get("capture"), dict) for r in results):
            write_jsonl(self.run_dir / "calls.jsonl", calls())

    def collect(self, requests: list[RunRequest]) -> list[dict[str, Any]]:
        check_retired_cot_samples(self.run_dir, requests)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        total_requests = len(requests)
        self._write_run_info([], total_requests=total_requests, state="running")
        # Rebuilt per collect(): an interrupt cancels the gate for good.
        self._launch_gate = _LaunchGate(self.config.launch_interval)
        # Launched rollouts and when they started, by request index.
        self._running: dict[int, tuple[RunRequest, float]] = {}
        self._running_lock = threading.Lock()
        slots: queue.Queue[int] = queue.Queue()
        for slot in range(self.config.parallel):
            slots.put(slot)

        def with_slot(request: RunRequest) -> dict[str, Any]:
            slot = slots.get()
            try:
                return self._run_one(request, slot)
            finally:
                slots.put(slot)

        def collector_error(request: RunRequest, exc: Exception) -> dict[str, Any]:
            logger.exception(
                "[%s | %s] collector exception", request.task.id, request.model
            )
            sample_dir = self._sample_dir(request)
            sample_dir.mkdir(parents=True, exist_ok=True)
            self._prune_sample(
                sample_dir,
                keep_aux_traces=self.config.keep_aux_traces,
                keep_rollout_logs=True,
            )
            result = {
                "schema_version": "roller.sample/v1",
                "request_index": request.index,
                "sample_index": request.sample_index,
                "sample_count": request.sample_count,
                "source_index": request.task.source_index,
                "task_id": request.task.id,
                "model": request.model,
                "model_label": request.display_model,
                "harness": "unknown",
                "status": "error",
                "error": str(exc),
                "usage": {},
                "attempts": [],
            }
            atomic_write_json(sample_dir / "info.json", result)
            return result

        results: list[dict[str, Any]] = []
        checkpoint_count = 0
        checkpoint_time = time.monotonic()

        def checkpoint(state: str, *, force: bool = False) -> list[dict[str, Any]]:
            nonlocal checkpoint_count, checkpoint_time
            now = time.monotonic()
            if (
                state == "running"
                and not force
                and len(results) != 1
                and len(results) - checkpoint_count < RUN_CHECKPOINT_RESULT_INTERVAL
                and now - checkpoint_time < RUN_CHECKPOINT_TIME_INTERVAL_SECONDS
            ):
                return []
            ordered = sorted(results, key=lambda row: int(row.get("request_index", -1)))
            self._write_run_info(
                ordered,
                total_requests=total_requests,
                state=state,
            )
            checkpoint_count = len(results)
            checkpoint_time = now
            return ordered

        # Workers run even at parallel=1 so an interrupt lands here, never
        # inside a Harness that would then skip its container cleanup.
        pool = ThreadPoolExecutor(
            max_workers=self.config.parallel, thread_name_prefix="roller"
        )
        futures: dict[Future[dict[str, Any]], RunRequest] = {}
        # Finished futures and received signal numbers, in order of arrival.
        events: queue.SimpleQueue[Future[dict[str, Any]] | int] = queue.SimpleQueue()
        handlers: dict[int, Any] = {}
        # Raised once the running rollouts are dealt with; set by an interrupt,
        # a termination signal or too many consecutive errors.
        stop: BaseException | None = None
        abandon = False  # running rollouts lost their containers; do not wait

        def stop_launching() -> int:
            # Queued requests never start; running ones still finalize.
            # Workers still waiting for a launch turn give it up.
            self._launch_gate.cancel()
            return sum(future.cancel() for future in futures)

        def stored(request: RunRequest) -> dict[str, Any] | None:
            try:
                return self._resumed(request)
            except Exception as exc:
                return collector_error(request, exc)

        begin_collection(self.run_dir)
        try:
            try:
                pending = requests
                if self.config.resume:
                    found = list(pool.map(stored, requests))
                    results.extend(row for row in found if row is not None)
                    pending = [
                        request for request, row in zip(requests, found) if row is None
                    ]
                    logger.info(
                        "resume: %d of %d rollout(s) already complete, %d to run",
                        sum(row.get("status") == "skipped" for row in results),
                        total_requests,
                        len(pending),
                    )
                    checkpoint("running", force=True)
                if threading.current_thread() is threading.main_thread():
                    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                        # A signal the caller ignores, as under nohup, stays ignored.
                        if signal.getsignal(signum) is not signal.SIG_IGN:
                            handlers[signum] = signal.signal(
                                signum, lambda number, _frame: events.put(number)
                            )
                started = quiet_since = time.monotonic()
                for request in pending:
                    future = pool.submit(with_slot, request)
                    futures[future] = request
                    future.add_done_callback(events.put)
                outstanding = len(futures)
                interrupts = 0
                resumed = len(results)
                failed = sum(row.get("status") != "skipped" for row in results)
                streak: list[dict[str, Any]] = []  # rollouts in a row ending in error
                while outstanding:
                    wait = quiet_since + HEARTBEAT_SECONDS - time.monotonic()
                    try:
                        event = events.get(timeout=max(0.0, wait))
                    except queue.Empty:
                        quiet_since = time.monotonic()
                        with self._running_lock:
                            running = sorted(
                                self._running.values(), key=lambda item: item[1]
                            )
                        logger.info(
                            "no rollout finished in the last %s: %d running, "
                            "%d waiting to start; longest running: %s",
                            _duration(HEARTBEAT_SECONDS),
                            len(running),
                            outstanding - len(running),
                            ", ".join(
                                f"{request.task.id} | {request.model} "
                                f"{_duration(quiet_since - launched)}"
                                for request, launched in running[:3]
                            )
                            or "none",
                        )
                        continue
                    if isinstance(event, int):
                        interrupts += event == signal.SIGINT
                        if event == signal.SIGINT and interrupts == 1:
                            cancelled = stop_launching()
                            if stop is None:
                                stop = KeyboardInterrupt()
                            logger.warning(
                                "interrupted: cancelled %d queued request(s) and any "
                                "rollouts waiting for a launch turn; waiting for %d "
                                "running rollout(s) to finish. Press Ctrl-C again "
                                "to stop them now",
                                cancelled,
                                len(self._running),
                            )
                            continue
                        # A second Ctrl-C, or a signal that leaves no time to wait.
                        stop_launching()
                        abandon = True
                        name = signal.Signals(event).name
                        stop = (
                            KeyboardInterrupt()
                            if event == signal.SIGINT
                            else CollectionStopped(f"received {name}", 128 + event)
                        )
                        logger.warning(
                            "%s: stopped %d running rollout(s) and removed %d "
                            "container(s) of this run",
                            name,
                            len(self._running),
                            remove_own_containers(),
                        )
                        break
                    outstanding -= 1
                    request = futures[event]
                    with self._running_lock:
                        launched = self._running.pop(
                            request.index, (request, time.monotonic())
                        )[1]
                    if event.cancelled():
                        continue
                    try:
                        result = event.result()
                    except LaunchCancelled:
                        continue
                    except Exception as exc:
                        result = collector_error(request, exc)
                    except BaseException as exc:
                        # A Harness that lets an interrupt through ends the run
                        # the way one Ctrl-C does.
                        if stop is None:
                            stop = exc
                            logger.warning(
                                "interrupted: cancelled %d queued request(s) and any "
                                "rollouts waiting for a launch turn; "
                                "waiting for running samples to finish",
                                stop_launching(),
                            )
                        continue
                    results.append(result)
                    quiet_since = time.monotonic()
                    status = result.get("status")
                    failed += status != "ok"
                    remaining = total_requests - len(results)
                    pace = (quiet_since - started) / (len(results) - resumed)
                    logger.info(
                        "[%d/%d] %s %s | %s in %s (ok %d, failed %d)%s",
                        len(results),
                        total_requests,
                        status,
                        request.task.id,
                        request.model,
                        _duration(quiet_since - launched),
                        len(results) - failed,
                        failed,
                        f", ETA {_duration(remaining * pace)}"
                        if remaining and stop is None
                        else "",
                    )
                    if status == "ok":
                        streak.clear()
                    elif status == "error":
                        streak.append(result)
                    limit = self.config.max_consecutive_errors
                    if stop is None and limit and len(streak) >= limit:
                        count, reason, _ = failure_groups(streak, 1)[0]
                        stop = CollectionStopped(
                            f"{len(streak)} rollouts in a row ended in error, "
                            f"{count} of them with: {reason}",
                            1,
                        )
                        logger.error(
                            "%s; cancelled %d queued request(s), waiting for %d "
                            "running rollout(s) to finish",
                            stop,
                            stop_launching(),
                            len(self._running),
                        )
                    checkpoint("running")
            except BaseException:
                stop_launching()
                raise
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
                pool.shutdown(wait=not abandon, cancel_futures=True)
                if not abandon:
                    begin_collection(None)
            if stop is not None:
                raise stop
        except BaseException:
            ordered = checkpoint("interrupted", force=True)
            try:
                self._write_sft_dataset(ordered)
            except Exception:
                logger.exception("failed to aggregate partial SFT dataset")
            raise

        ordered = checkpoint("finalizing", force=True)
        try:
            self._write_sft_dataset(ordered)
        except BaseException:
            checkpoint("finalization_error", force=True)
            raise
        checkpoint("finished", force=True)
        return ordered
