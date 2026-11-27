"""Interactive OpenClaw session running inside a disposable Docker container."""

from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roller.artifacts import ArtifactCollectionResult
from roller.docker_runtime import remove_container
from roller.harness_registry import convert_native_trace
from roller.harnesses.base import InteractiveSession, TurnResult
from roller.harnesses.container import (
    build_provenance,
    output_collection_metadata,
)
from roller.harnesses.openclaw_config import (
    OPENCLAW_STATE_CONTAINER_PATH,
    TASK_ENVIRONMENT_CONTAINER_PATH,
    _model_base_urls,
    _redact_url,
)
from roller.models import HarnessResult, RunRequest
from roller.native_images import openclaw_message
from roller.storage import safe_component
from roller.usage import ZERO_USAGE, extract_openclaw_usage
from roller.wire_transforms import extract_wire_tools

if TYPE_CHECKING:
    from roller.harnesses.openclaw import OpenClawHarness

logger = logging.getLogger(__name__)


class OpenClawSession(InteractiveSession):
    """One live OpenClaw container used turn-by-turn by a conversation manager."""

    def __init__(
        self,
        harness: OpenClawHarness,
        request: RunRequest,
        output_dir: Path,
        attempt: int,
        *,
        collect_outputs: bool,
    ) -> None:
        self.harness = harness
        self.request = request
        self.output_dir = output_dir
        self.attempt = attempt
        self.collect_outputs = collect_outputs
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.container = (
            f"roller-{request.index}-{safe_component(request.display_model, 24)}-"
            f"{uuid.uuid4().hex[:8]}"
        ).lower()
        self.gateway_token = uuid.uuid4().hex
        self.gateway_process: subprocess.Popen[str] | None = None
        self.container_started = False
        self.version = ""
        self.state_dir = OPENCLAW_STATE_CONTAINER_PATH
        self.return_codes: list[int] = []
        self.messages: list[str] = []
        self.session_id = harness.config.session_id
        self.session_ids = [self.session_id]
        self.environment: dict[str, str] = {}
        self.image_digest = harness._image_digest()
        self._assistant_count = 0
        self._finished: HarnessResult | None = None

    def start(self) -> None:
        if self.container_started:
            raise RuntimeError("OpenClaw session has already started")
        # Retries share one sample directory. Every trace layer must represent
        # the current attempt rather than a previous partial session.
        for relative in (
            "rollout/trace.jsonl",
            "cache_trace.jsonl",
            "wire_trace.jsonl",
            "rollout/wire_strip.jsonl",
        ):
            (self.output_dir / relative).unlink(missing_ok=True)
        self.harness.deadline = self.started + self.request.timeout_seconds
        self.harness._start_wire_proxy(self.request, self.output_dir, self.attempt)
        # Mark before run: a timed-out Docker client may already have created it.
        self.container_started = True
        self.environment = self.harness._start_container(
            self.container, self.request, self.image_digest
        )
        self.harness._prepare_workspace(self.container, self.request)
        self.harness._prepare_context_and_setup(self.container, self.request)
        self.version = self.harness._configure(
            self.container, self.request, self.environment
        )
        self.state_dir = self.harness._state_dir_for_version(self.version)
        if self.harness.config.capture_calls:
            patch = (
                Path(__file__).resolve().parents[1] / "assets/openclaw_capture.py"
            ).read_text()
            self.harness._exec_checked(
                self.container,
                ["python3", "-c", patch],
                "OpenClaw native session capture hook",
            )
        # The legacy 2026.3.11 image keeps its gateway config under
        # /root/.openclaw while the temporary OPENCLAW_STATE_DIR is used by
        # the CLI. Set both token locations explicitly so gateway and agent
        # clients authenticate with the same per-rollout token.
        for setting in ("gateway.auth.token", "gateway.remote.token"):
            self.harness._exec_checked(
                self.container,
                [
                    "openclaw",
                    "config",
                    "set",
                    setting,
                    self.gateway_token,
                ],
                f"gateway token setup ({setting})",
            )
        self.harness._ensure_gateway_port_free(self.container)
        self.gateway_process = self.harness.runtime.exec_logged(
            self.container,
            [
                "/bin/bash",
                "-lc",
                "set -a; "
                f"if [ -f {TASK_ENVIRONMENT_CONTAINER_PATH} ]; then "
                f". {TASK_ENVIRONMENT_CONTAINER_PATH}; fi; "
                "set +a; "
                f"exec openclaw gateway --port {self.harness.gateway_port}",
            ],
            self.output_dir / "rollout" / f"gateway.attempt-{self.attempt}.log",
            env={
                "OPENCLAW_GATEWAY_TOKEN": self.gateway_token,
                "OPENCLAW_STATE_DIR": self.state_dir,
            },
            workdir=self.harness.config.workspace_container_path,
        )
        self.harness._wait_for_gateway(self.container, self.gateway_process)

    def send(self, message: str, timeout_seconds: float) -> TurnResult:
        if not self.container_started or self.gateway_process is None:
            raise RuntimeError("OpenClaw session is not running")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("interactive message must be non-empty")
        if timeout_seconds <= 0:
            raise TimeoutError("interactive turn has no time remaining")
        turn_started = time.monotonic()
        command = [
            "openclaw",
            "agent",
            "--session-id",
            self.session_id,
            "--timeout",
            str(max(1, int(timeout_seconds))),
        ]
        thinking = self.harness._effective_thinking(self.request)
        if thinking:
            # The CLI flag reaches this live session; changing the default after
            # gateway startup does not reliably affect an existing agent turn.
            command += ["--thinking", thinking]
        command += [
            "--message",
            openclaw_message(
                self.request,
                message,
                self.harness.config.workspace_container_path,
                first_message=not self.messages,
            ),
        ]
        agent_log_path = (
            self.output_dir / "rollout" / f"agent.attempt-{self.attempt}.log"
        )
        process = self.harness.runtime.exec_logged(
            self.container,
            command,
            agent_log_path,
            env={
                "OPENCLAW_GATEWAY_TOKEN": self.gateway_token,
                "OPENCLAW_STATE_DIR": self.state_dir,
            },
            workdir=self.harness.config.workspace_container_path,
            append=bool(self.messages),
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            self.harness.runtime.exec(self.container, ["pkill", "-f", "openclaw agent"])
            raise TimeoutError("OpenClaw interactive turn timed out") from exc
        finally:
            self.harness.runtime.close_log(process)
        self.return_codes.append(return_code)
        if return_code != 0:
            detail = ""
            try:
                detail = agent_log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()[-2000:]
            except OSError:
                pass
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"OpenClaw agent turn exited with code {return_code}{suffix}"
            )
        if self.harness.config.capture_calls:
            remaining = self.request.timeout_seconds - (time.monotonic() - self.started)
            try:
                if remaining <= 0:
                    raise TimeoutError(
                        "OpenClaw capture deadline exhausted before subagent drain"
                    )
                script = (
                    Path(__file__).resolve().parents[1] / "assets/openclaw_drain.mjs"
                ).read_text()
                self.harness._exec_checked(
                    self.container,
                    [
                        "node",
                        "--input-type=module",
                        "-e",
                        script,
                    ],
                    "OpenClaw subagent completion drain",
                    env={
                        "OPENCLAW_STATE_DIR": self.state_dir,
                        "ROLLER_CAPTURE_ROOT_SESSION": self.session_id,
                        "ROLLER_CAPTURE_TIMEOUT_MS": str(max(1, int(remaining * 1000))),
                    },
                    timeout=remaining,
                )
            except Exception as exc:
                # The root turn already succeeded. Incomplete collection must
                # remain a capture failure, not fabricate a failed root turn.
                self.harness.capture.errors.append(
                    f"OpenClaw subagent drain failed: {exc}"
                )
        content, assistant_count = self.harness._latest_assistant_content(
            self.container,
            state_dir=self.state_dir,
            session_id=self.session_id,
        )
        if assistant_count <= self._assistant_count:
            raise RuntimeError("OpenClaw turn produced no new assistant response")
        self._assistant_count = assistant_count
        self.messages.append(message)
        return TurnResult(content, time.monotonic() - turn_started, return_code)

    def reset(self) -> None:
        """Select a fresh OpenClaw session while preserving this container."""

        if not self.container_started or self.gateway_process is None:
            raise RuntimeError("OpenClaw session is not running")
        self.session_id = (
            f"{self.harness.config.session_id}-{len(self.session_ids) + 1}"
        )
        self.session_ids.append(self.session_id)
        self._assistant_count = 0

    def prepare_transcript_for_grading(self) -> str | None:
        if not self.container_started:
            return None
        return self.harness._export_combined_transcript(
            self.container,
            self.session_ids,
            state_dir=self.state_dir,
        )

    def finish(self, status: str, error: str | None = None) -> HarnessResult:
        if self._finished is not None:
            return self._finished
        trace_path: Path | None = None
        trajectory_path: Path | None = None
        legacy_records: list[dict[str, Any]] = []
        artifact_result: ArtifactCollectionResult | None = None
        output_copy_error: str | None = None
        warnings: list[str] = []
        trajectory_conversion = None
        if self.container_started:
            try:
                trace_path = self.harness._collect_trace(
                    self.container,
                    self.output_dir,
                    self.attempt,
                    state_dir=self.state_dir,
                    session_ids=self.session_ids,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] trace collection failed: %s", self.request.task.id, exc
                )
            try:
                self.harness._collect_cache_trace(self.container, self.output_dir)
            except Exception as exc:
                logger.warning(
                    "[%s] cache trace collection failed: %s",
                    self.request.task.id,
                    exc,
                )
            if self.collect_outputs:
                try:
                    artifact_result, output_copy_error = self.harness._collect_outputs(
                        self.container, self.request, self.output_dir
                    )
                except Exception as exc:
                    output_copy_error = str(exc)
                    logger.warning(
                        "[%s] output collection failed: %s", self.request.task.id, exc
                    )
                try:
                    legacy_records = self.harness._collect_legacy_artifacts(
                        self.container, self.output_dir
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] legacy artifact collection failed: %s",
                        self.request.task.id,
                        exc,
                    )
        try:
            try:
                self.harness._stop_process(self.gateway_process)
            finally:
                self.harness.runtime.close_log(self.gateway_process)
        finally:
            self.harness._stop_wire_proxy()

        capture_metadata = {}
        if self.harness.capture is not None:
            capture = self.harness.capture
            cache = self.output_dir / "cache_trace.jsonl"
            if cache.is_file():
                capture.add_native(cache, "cache_trace.jsonl")
            capture.copy_native(
                self.harness.runtime,
                self.container,
                self.state_dir + "/agents/main/sessions/.",
                "sessions",
            )
            capture.copy_native(
                self.harness.runtime,
                self.container,
                "/tmp/roller-subagent-runs.json",
                "subagent-runs.json",
            )
            capture.copy_native(
                self.harness.runtime,
                self.container,
                "/tmp/roller-capture-stream.jsonl",
                "stream-contexts.jsonl",
            )
            if (
                self.harness.runtime.exec(
                    self.container,
                    ["test", "-f", "/tmp/roller-tool-id-aliases.jsonl"],
                ).returncode
                == 0
            ):
                capture.copy_native(
                    self.harness.runtime,
                    self.container,
                    "/tmp/roller-tool-id-aliases.jsonl",
                    "tool-id-aliases.jsonl",
                )
            capture_metadata = capture.finish(status)
            # Main ACTF consumes only root cache contexts. Full native evidence
            # remains immutable in capture/native/cache_trace.jsonl.
            if cache.is_file():
                rows = [
                    json.loads(line)
                    for line in cache.read_text().split("\n")
                    if line.strip()
                ]
                rows = [r for r in rows if ":subagent:" not in r.get("sessionKey", "")]
                cache.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                )

        if trace_path is not None:
            try:
                cache_trace_path = self.output_dir / "cache_trace.jsonl"
                wire_trace_path = self.output_dir / "wire_trace.jsonl"
                trajectory_path = self.output_dir / "rollout" / "trace_ACTF.json"
                convert_native_trace(
                    "openclaw",
                    trace_path,
                    trajectory_path,
                    user_messages=self.messages,
                    cache_trace=(
                        cache_trace_path if cache_trace_path.is_file() else None
                    ),
                    wire_trace=(wire_trace_path if wire_trace_path.is_file() else None),
                    tools=(
                        extract_wire_tools(wire_trace_path)
                        if wire_trace_path.is_file()
                        else None
                    ),
                    keep_thinking=self.harness.config.keep_thinking,
                )
            except Exception as exc:
                trajectory_path = None
                conversion_error = str(exc)
                warnings.append(
                    "OpenClaw trajectory conversion failed: " + conversion_error
                )
                trajectory_conversion = {
                    "status": "unavailable",
                    "target": "actf",
                    "error": conversion_error,
                }

        if status == "ok" and trace_path is None:
            status = "error"
            error = "OpenClaw finished but no session trace was found"
        if status == "ok" and output_copy_error is not None:
            status = "error"
            error = f"output collection failed: {output_copy_error}"
        if (
            status == "ok"
            and artifact_result is not None
            and not artifact_result.passed
        ):
            status = "error"
            error = "unsafe output files were rejected"
        retained = (
            self.container_started
            and self.harness.config.keep_container_on_error
            and status != "ok"
        )
        if self.container_started:
            try:
                if retained:
                    # Preserve the filesystem for debugging without leaving
                    # the host-network gateway alive to occupy a future
                    # worker's deterministic port.
                    stop_result = self.harness.runtime.stop(
                        self.container, 1, timeout=30
                    )
                    if stop_result.returncode != 0:
                        logger.warning(
                            "[%s] failed to stop retained container %s",
                            self.request.task.id,
                            self.container,
                        )
                else:
                    remove_result = remove_container(
                        self.harness.runtime, self.container
                    )
                    if remove_result.returncode != 0:
                        logger.warning(
                            "[%s] failed to remove container %s",
                            self.request.task.id,
                            self.container,
                        )
            except Exception as exc:
                action = "stop" if retained else "remove"
                logger.warning(
                    "[%s] failed to %s container %s: %s",
                    self.request.task.id,
                    action,
                    self.container,
                    exc,
                )

        elapsed = time.monotonic() - self.started
        usage: dict[str, Any] = (
            extract_openclaw_usage(trace_path)
            if trace_path is not None
            else dict(ZERO_USAGE)
        )
        usage["elapsed_seconds"] = round(elapsed, 3)
        api_base_urls = _model_base_urls(self.harness.config.model_config or {})
        if self.environment.get("OPENROUTER_BASE_URL"):
            api_base_urls.append(_redact_url(self.environment["OPENROUTER_BASE_URL"]))
        provenance = build_provenance(
            self.request,
            harness=self.harness.name,
            openclaw_version=self.version,
            docker_image=self.harness.config.docker_image,
            docker_image_digest=self.image_digest,
            api_base_urls=list(dict.fromkeys(api_base_urls)),
            environment=self.environment,
            attempt=self.attempt,
            started_at=self.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._finished = HarnessResult(
            status=status,
            elapsed_seconds=elapsed,
            error=error,
            usage=usage,
            trace_path=(
                trace_path.relative_to(self.output_dir).as_posix()
                if trace_path is not None
                else None
            ),
            trajectory_path=(
                trajectory_path.relative_to(self.output_dir).as_posix()
                if trajectory_path is not None
                else None
            ),
            metadata={
                **capture_metadata,
                "raw_trace_schema": "openclaw-session-jsonl",
                "openclaw_version": self.version,
                "gateway_port": self.harness.gateway_port,
                "container": (self.container if retained else None),
                "agent_return_codes": self.return_codes,
                "turns": len(self.messages),
                "output_collection": output_collection_metadata(
                    artifact_result, self.request, self.output_dir
                ),
                "output_copy_error": output_copy_error,
                "legacy_artifacts": legacy_records,
                "warnings": warnings,
                "trajectory_conversion": trajectory_conversion,
                "attempt": self.attempt,
                "settings": {
                    "reasoning_effort": self.harness._effective_thinking(self.request),
                    "keep_thinking": self.harness.config.keep_thinking,
                },
                "provenance": provenance,
            },
        )
        return self._finished
