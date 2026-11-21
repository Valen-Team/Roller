"""Shared lifecycle for Harnesses driven by a pinned in-container bridge runner.

Hermes, OpenCode, DeepSeek and Pi each ship a ``runner.py`` under
``roller/assets`` that executes one turn from ``turn.json``, writes
``result.json`` and appends to a native journal. This module owns the Docker
side of that contract; subclasses declare what differs per Harness. Stirrup
reuses the container lifecycle with its agent in the collecting process.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

from roller.artifacts import collect_output_directory
from roller.capture_runtime import CaptureRuntime
from roller.harnesses.base import InteractiveHarness, InteractiveSession, TurnResult
from roller.harnesses.container import (
    CONTEXT_CONTAINER_PATH,
    ENV_NAME,
    INPUTS_CONTAINER_PATH,
    OUTPUTS_CONTAINER_PATH,
    WORKSPACE_INPUT_CONTAINER_PATH,
    build_provenance,
    collect_outputs,
    docker_run_args,
    host_api_key,
    host_environment,
    output_collection_metadata,
)
from roller.harnesses.docker import (
    DockerRuntime,
    remove_container,
    warn_if_image_changed,
)
from roller.models import HarnessResult, RunRequest
from roller.redaction import REDACTED, redact_file_literals
from roller.sandbox import (
    Sandbox,
    SandboxProcess,
    SandboxRun,
    route_judge_environment,
    route_sandbox_http,
)
from roller.storage import atomic_write_json
from roller.thinking import resolve_thinking
from roller.token_ids import TOKEN_IDS_FILE
from roller.usage import ZERO_USAGE
from roller.wire_trace import WireTraceProxy, redact_jsonl_file

logger = logging.getLogger(__name__)


def validate_bridge_config(config: Any, label: str) -> None:
    """Checks shared by every bridge Harness config; limits stay per Harness."""

    if not config.docker_image.strip():
        raise ValueError(f"{label} docker image must be non-empty")
    if config.container_user is not None and not config.container_user.strip():
        raise ValueError(f"{label} container user must be non-empty or None")
    if not config.workspace_container_path.startswith("/"):
        raise ValueError(f"{label} workspace container path must be absolute")
    if not all(
        ENV_NAME.fullmatch(name) for name in (config.api_key_env, *config.pass_env)
    ):
        raise ValueError(f"invalid {label} environment variable name")
    if config.api_base:
        try:
            parsed = urlsplit(config.api_base)
            parsed.port  # noqa: B018 - reading it raises for an invalid port
            valid = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and not (
                    parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                )
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                f"{label} API base must be an absolute http(s) URL without credentials, query or fragment"
            )
    if config.wire_trace_mode not in {"off", "actf", "full"}:
        raise ValueError(f"{label} wire trace mode must be off|actf|full")
    # A model client in the collecting process reaches the local proxy itself.
    remote = getattr(config, "remote_sandbox", False) or not getattr(
        config, "agent_in_container", True
    )
    if config.wire_trace_mode == "full" and (
        not config.api_base or (not remote and config.network != "host")
    ):
        raise ValueError(
            f"{label} --wire-trace full requires --api-base and --network host"
        )
    if config.capture_calls and (
        not config.api_base
        or (not remote and config.network != "host")
        or config.wire_trace_mode == "off"
    ):
        raise ValueError(
            f"{label} call capture requires --api-base, --network host and wire tracing"
        )
    if config.token_ids and not config.capture_calls:
        raise ValueError(f"{label} --token-ids requires --capture-calls")


def positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def validate_step_limits(config: Any, label: str) -> None:
    """``max_steps``/``context_length``/``output_tokens`` used by the step-based runners."""

    if not positive_int(config.max_steps):
        raise ValueError(f"{label} max steps must be a positive integer")
    if not positive_int(config.context_length) or config.context_length < 1024:
        raise ValueError(f"{label} context length must be at least 1024 tokens")
    if (
        not positive_int(config.output_tokens)
        or config.output_tokens >= config.context_length
    ):
        raise ValueError(
            f"{label} output tokens must be positive and smaller than context length"
        )


class BridgeSession(InteractiveSession):
    """One disposable container in which the bridge runner keeps native state."""

    label: ClassVar[str]
    state_dir: ClassVar[str]
    runner: ClassVar[str]
    runner_source: ClassVar[Path]
    raw_trace_schema: ClassVar[str]
    python: ClassVar[str] = "python3"
    api_key_name: ClassVar[str] = "OPENAI_API_KEY"
    default_api_base: ClassVar[str] = "https://api.openai.com/v1"
    version_check: ClassVar[str] = "version check"
    # (file next to runner_source, container destination, description)
    assets: ClassVar[tuple[tuple[str, str, str], ...]] = ()
    native_trace_name: ClassVar[str] = "trace.jsonl"
    redact_trace: ClassVar[bool] = False
    conversion_failure_is_error: ClassVar[bool] = True
    redact_capture_errors: ClassVar[bool] = False
    supports_reset: ClassVar[bool] = False
    # The runner can put ``files.attach_images`` on the first user message.
    attaches_images: ClassVar[bool] = False
    # False when the agent runs in the collecting process: the container then
    # only executes commands, gets no model key, and model calls stay local.
    agent_in_container: ClassVar[bool] = True

    def __init__(
        self,
        harness: BridgeHarness,
        request: RunRequest,
        output_dir: Path,
        attempt: int,
        *,
        collect_outputs: bool = True,
    ) -> None:
        self.harness, self.request = harness, request
        self.output_dir, self.attempt = Path(output_dir), attempt
        self.collect_outputs = collect_outputs
        self.container = (
            f"roller-{harness.name}-{request.index}-{uuid.uuid4().hex[:12]}"
        )
        self.container_started = False
        self.transcript_container_path = self.state_dir + "/trace.jsonl"
        self.home = f"/roller/{harness.name}-home"
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.environment: dict[str, str] = {}
        self.image_digest: str | None = None
        self.version: str | None = None
        self.proxy: WireTraceProxy | None = None
        self.sandbox_run: SandboxRun | None = getattr(
            harness.config, "sandbox_run", None
        )
        self.sandbox: Sandbox | None = None
        self.remote_proxy: SandboxProcess | None = None
        self.remote_native: SandboxProcess | None = None
        self.remote_wire_dir = f"{self.state_dir}/wire"
        config = harness.config
        self.capture = (
            CaptureRuntime(self.output_dir, attempt, harness.name, config.capture_debug)
            if config.capture_calls
            else None
        )
        self.capture_metadata: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.trajectory_conversion: dict[str, Any] | None = None
        self._result: HarnessResult | None = None
        self.turns = 0

    # Per-Harness declarations -------------------------------------------------

    def effort(self) -> str | None:
        """The validated native effort for this task."""

        raise NotImplementedError

    def harness_environment(self, api_base: str) -> dict[str, str]:
        raise NotImplementedError

    def turn_options(self, timeout: float) -> dict[str, Any]:
        """Harness-specific ``turn.json`` fields."""

        raise NotImplementedError

    def convert_trace(self, trace: Path, destination: Path) -> dict[str, Any]:
        raise NotImplementedError

    def extract_usage(self, trace: Path) -> dict[str, int | float]:
        raise NotImplementedError

    def version_fields(self) -> dict[str, Any]:
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        raise NotImplementedError

    def setup_script(self) -> str:
        """Extra shell steps run after the state directories exist."""

        return ""

    def inherited_environment(self) -> dict[str, str]:
        return host_environment(self.request.task, self.harness.config.pass_env)

    def system_prompt(self) -> str:
        config, task = self.harness.config, self.request.task
        context = ", ".join(item.target for item in task.context_files)
        return (
            f"You are running a Roller task. Work in {config.workspace_container_path}. "
            f"Task inputs are in the workspace. Save deliverables under {task.output_dir}. "
            f"Task context files (read when relevant): {context or '(none)'}. "
            "Complete the requested task and return a concise final answer."
        )

    def collect_diagnostics(self) -> None:
        """Copy extra runner logs before the journal; failures are non-fatal."""

    def collect_native_evidence(self, usage: dict[str, int | float]) -> None:
        """Add Harness-specific evidence to ``self.capture`` before it is sealed."""

    def result_metadata(self) -> dict[str, Any]:
        return {}

    # Shared lifecycle -----------------------------------------------------------

    def remaining(self) -> float:
        remaining = self.request.timeout_seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            raise TimeoutError(f"{self.label} task timed out")
        return remaining

    def redact(self, text: str) -> str:
        key = os.environ.get(self.harness.config.api_key_env)
        return text.replace(key, REDACTED) if key else text

    def redact_file(self, path: Path, *, errors: str = "strict") -> None:
        """Redact the API key in a journal or log without loading it whole."""

        redact_file_literals(
            path, [os.environ.get(self.harness.config.api_key_env, "")], errors=errors
        )

    def checked(
        self,
        args: Sequence[str],
        description: str,
        *,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        limit = min(timeout, self.remaining())
        try:
            result = self.harness.runtime.run(args, timeout=limit)
        except subprocess.TimeoutExpired:
            # Only a limit cut down to the task budget is a task timeout; the
            # exception's command text can carry container credentials.
            if limit < timeout:
                raise TimeoutError(f"{self.label} task timed out") from None
            raise RuntimeError(f"{description} timed out after {limit:g}s") from None
        if result.returncode:
            detail = self.redact(
                (result.stderr or result.stdout or "unknown error").strip()
            )[-2000:]
            raise RuntimeError(f"{description} failed: {detail}")
        return result

    def command(
        self,
        argv: Sequence[str],
        description: str,
        *,
        cwd: str | None = None,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        if self.sandbox is None:
            return self.checked(
                self.harness.runtime.exec_args(self.container, argv, workdir=cwd),
                description,
                timeout=timeout,
            )
        result = self.sandbox.run(
            argv, cwd=cwd, env=self.environment, timeout=min(timeout, self.remaining())
        )
        if result.returncode:
            detail = self.redact((result.stderr or result.stdout).strip())[-2000:]
            raise RuntimeError(f"{description} failed: {detail}")
        return result

    def upload(self, source: Path, target: str, description: str) -> None:
        if self.sandbox is None:
            self.checked(
                self.harness.runtime.copy_in_args(source, self.container, target),
                description,
            )
        else:
            self.sandbox.upload(source, target, timeout=min(60, self.remaining()))

    def _start_sandbox(self, workspace: str, api_base: str) -> str:
        assert self.sandbox_run is not None
        box = self.sandbox_run.create(
            self.container, self.request.timeout_seconds + 900, self.remaining()
        )
        self.sandbox = box
        self.container_started = True
        if box.version != self.sandbox_run.target.version:
            raise RuntimeError("sandbox environment version changed since preflight")
        timeout = self.request.timeout_seconds
        route_judge_environment(box, self.environment, timeout)
        if self.agent_in_container:
            model_key = self.environment[self.api_key_name]
            api_base, self.environment[self.api_key_name] = route_sandbox_http(
                box, api_base, model_key, timeout
            )
        self.command(
            ["mkdir", "-p", workspace, self.state_dir, self.home, "/roller/task"],
            "sandbox workspace setup",
        )
        task, bundle = self.request.task, self.request.bundle
        if bundle:
            if bundle.legacy_workspace or task.input_mode == "copy":
                box.upload(
                    bundle.inputs_root,
                    workspace,
                    contents=True,
                    timeout=self.remaining(),
                )
                self.command(
                    ["chmod", "-R", "u+w", workspace], "sandbox workspace permissions"
                )
            box.upload(
                bundle.context_root, workspace, contents=True, timeout=self.remaining()
            )
        elif task.workspace:
            box.upload(
                task.workspace, workspace, contents=True, timeout=self.remaining()
            )
        if not self.agent_in_container or (
            self.harness.config.wire_trace_mode != "full" and self.capture is None
        ):
            return api_base
        directory = self.remote_wire_dir
        self.command(["mkdir", "-p", directory], "wire recorder setup")
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "config.json"
            atomic_write_json(
                settings,
                {
                    "api_base": api_base,
                    "calls": self.capture is not None,
                    "token_ids": self.harness.config.token_ids,
                },
            )
            self.upload(
                settings, f"{directory}/config.json", "wire recorder configuration"
            )
        self.remote_proxy = box.spawn(
            [self.python, "-m", "roller.sandbox_wire", directory],
            env=self.environment,
            log_path=self.output_dir / "rollout" / "wire-recorder.log",
        )
        deadline = time.monotonic() + min(30, self.remaining())
        while time.monotonic() < deadline:
            ready = box.run(
                ["cat", f"{directory}/ready.json"], timeout=min(5, self.remaining())
            )
            if ready.returncode == 0:
                return json.loads(ready.stdout)["url"]
            time.sleep(0.1)
        raise RuntimeError("sandbox wire recorder did not become ready")

    def start(self) -> None:
        if self.container_started or self._result is not None:
            raise RuntimeError(f"{self.label} session has already started or finished")
        config, task, bundle = (
            self.harness.config,
            self.request.task,
            self.request.bundle,
        )
        if task.attach_images and not self.attaches_images:
            raise ValueError(f"{self.label} does not support files.attach_images")
        self.effort()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        environment = self.inherited_environment()
        if self.agent_in_container:
            environment[self.api_key_name] = host_api_key(
                self.label, config.api_key_env
            )
        api_base = config.api_base or self.default_api_base
        if (self.sandbox_run is None or not self.agent_in_container) and (
            config.wire_trace_mode == "full" or config.capture_calls
        ):
            token_ids_path = self.output_dir / TOKEN_IDS_FILE
            token_ids_path.unlink(missing_ok=True)
            self.proxy = WireTraceProxy(
                api_base,
                self.output_dir / "wire_trace.jsonl",
                mode="full",
                capture_recorder=self.capture.recorder if self.capture else None,
                capture_identity_header="x-roller-native-session-id"
                if self.capture
                else None,
                # A model client here names each call, so its journal and the
                # captured Calls can be matched.
                capture_run_header="x-roller-native-run-id"
                if self.capture and not self.agent_in_container
                else None,
                token_ids_path=token_ids_path if config.token_ids else None,
            )
            self.proxy.start()
            api_base = self.proxy.proxy_base_url
        self.environment = environment
        if self.sandbox_run is not None:
            api_base = self._start_sandbox(config.workspace_container_path, api_base)
            environment.update(self.harness_environment(api_base))
        else:
            environment.update(self.harness_environment(api_base))
            self.image_digest = self.checked(
                self.harness.runtime.image_field_args(config.docker_image, "Id"),
                f"{self.label} image inspection",
            ).stdout.strip()
            warn_if_image_changed(
                logger,
                config.docker_image,
                config.preflight_image_digest,
                self.image_digest,
            )
            args = docker_run_args(
                self.harness.name, self.container, config, environment, self.request
            )
            # Use the inspected image ID so the recorded version is the version in use.
            args += [
                "--entrypoint",
                "/bin/bash",
                self.image_digest,
                "-c",
                "tail -f /dev/null",
            ]
            # A timed-out Docker client may already have created the container.
            self.container_started = True
            self.checked(args, f"{self.label} container startup")
        workspace = shlex.quote(config.workspace_container_path)
        output = shlex.quote(f"{config.workspace_container_path}/{task.output_dir}")
        script = (
            f"mkdir -p {workspace} {self.state_dir} {self.home} /roller/task"
            + self.setup_script()
        )
        if self.sandbox is None:
            if bundle and (bundle.legacy_workspace or task.input_mode == "copy"):
                script += f" && cp -a {INPUTS_CONTAINER_PATH}/. {workspace}/ && chmod -R u+w {workspace}"
            elif task.workspace and not bundle:
                script += f" && cp -a {WORKSPACE_INPUT_CONTAINER_PATH}/. {workspace}/ && chmod -R u+w {workspace}"
            if bundle:
                script += f" && cp -a {CONTEXT_CONTAINER_PATH}/. {workspace}/"
        script += f" && rm -rf {output} && mkdir -p {output} && ln -s {output} {OUTPUTS_CONTAINER_PATH}"
        self.command(["/bin/bash", "-c", script], f"{self.label} workspace setup")
        if self.agent_in_container:
            self.upload(self.runner_source, self.runner, f"{self.label} runner copy")
        for name, destination, description in self.assets:
            self.upload(
                self.runner_source.with_name(name),
                destination,
                f"{self.label} {description}",
            )
        for command in task.setup_commands:
            self.command(
                ["/bin/bash", "-c", command],
                "task setup",
                cwd=config.workspace_container_path,
                timeout=self.remaining(),
            )
        if self.agent_in_container:
            self.version = self.command(
                [self.python, self.runner, "--check"],
                f"{self.label} {self.version_check}",
            ).stdout.strip()

    def send(self, message: str, timeout_seconds: float) -> TurnResult:
        if not self.container_started or self._result is not None:
            raise RuntimeError(f"{self.label} session is not active")
        started = time.monotonic()
        timeout = min(timeout_seconds, self.remaining())
        if timeout <= 0:
            raise TimeoutError(f"{self.label} turn timed out")
        config = self.harness.config
        self.turns += 1
        payload = {
            "capture_calls": config.capture_calls,
            "session_id": self.container,
            "model": self.request.model,
            "message": message,
            "thinking": self.effort(),
            "timeout_seconds": timeout,
            "system_prompt": self.system_prompt(),
            **self.turn_options(timeout),
        }
        config_path = self.output_dir / f".{self.harness.name}-turn.json"
        atomic_write_json(config_path, payload)
        log_path = (
            self.output_dir
            / "rollout"
            / f"{self.harness.name}.attempt-{self.attempt}.log"
        )
        process = None
        try:
            remote_config = self.state_dir + "/turn.json"
            self.upload(config_path, remote_config, f"{self.label} turn configuration")
            argv = [self.python, self.runner, "--config", remote_config]
            if self.sandbox is None:
                process = self.harness.runtime.exec_logged(
                    self.container,
                    argv,
                    log_path,
                    workdir=config.workspace_container_path,
                    append=True,
                )
            else:
                self._start_remote_native(remote_config)
                process = self.sandbox.spawn(
                    argv,
                    cwd=config.workspace_container_path,
                    env=self.environment,
                    log_path=log_path,
                )
            try:
                return_code = process.wait(
                    timeout=max(0.001, timeout - (time.monotonic() - started))
                )
            except subprocess.TimeoutExpired:
                process.kill()
                if self.sandbox is None:
                    process.wait(timeout=10)
                self._stop_turn()
                raise TimeoutError(f"{self.label} turn timed out") from None
            response = self.command(
                ["cat", self.state_dir + "/result.json"],
                f"{self.label} turn result",
            )
            result = json.loads(response.stdout)
            if result.get("timeout") is True:
                self._stop_turn()
                raise TimeoutError(f"{self.label} turn timed out")
            if (
                return_code
                or result.get("ok") is not True
                or not isinstance(result.get("content"), str)
                or not result["content"].strip()
            ):
                raise RuntimeError(
                    self.redact(
                        str(
                            result.get("error")
                            or f"{self.label} runner exited with code {return_code}"
                        )
                    )
                )
            return TurnResult(
                result["content"], time.monotonic() - started, return_code
            )
        finally:
            config_path.unlink(missing_ok=True)
            if self.sandbox is None:
                self.harness.runtime.close_log(process)
            if log_path.exists():
                self.redact_file(log_path, errors="replace")

    def _start_remote_native(self, config_path: str) -> None:
        if self.remote_native is not None:
            return
        assert self.sandbox is not None
        self.remote_native = self.sandbox.spawn(
            [self.python, self.runner, "--serve", config_path],
            env=self.environment,
            cwd=self.harness.config.workspace_container_path,
            log_path=self.output_dir / "rollout" / "native-runtime.log",
        )
        self.command(
            [
                "/bin/sh",
                "-c",
                f"until test -S {self.state_dir}/runner.sock; do sleep 0.1; done",
            ],
            "native session startup",
            timeout=30,
        )

    def _stop_turn(self) -> None:
        if self.sandbox is None:
            self.harness.runtime.kill(self.container, timeout=30)
        else:
            self._stop_remote_native()

    def _stop_remote_native(self) -> None:
        if self.remote_native is not None:
            self.remote_native.kill()
            self.remote_native = None

    def fetch_trace(self, trace: Path) -> bool:
        """Copy the native journal out of the container; False when it fails."""

        if self.sandbox is None:
            return not self.harness.runtime.copy_out(
                self.container, self.transcript_container_path, trace, timeout=30
            ).returncode
        self.sandbox.download(self.transcript_container_path, trace, timeout=60)
        return True

    def _collect_trace(
        self, problems: list[str]
    ) -> tuple[Path | None, Path | None, dict[str, int | float]]:
        usage: dict[str, int | float] = dict(ZERO_USAGE)
        trace = self.output_dir / "rollout" / "trace.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        if (
            not self.fetch_trace(trace)
            or not trace.is_file()
            or not trace.stat().st_size
        ):
            problems.append(f"no {self.label} journal was collected")
            return None, None, usage
        try:
            if self.redact_trace:
                self.redact_file(trace)
            usage = self.extract_usage(trace)
            trajectory = trace.with_name("trace_ACTF.json")
            if not self.convert_trace(trace, trajectory).get("steps"):
                raise ValueError(f"{self.label} journal produced no assistant steps")
            return trace, trajectory, usage
        except Exception as exc:
            message = f"{self.label} trace conversion failed: " + self.redact(str(exc))
            if self.conversion_failure_is_error:
                problems.append(message)
            else:
                # The native journal is retained and bench grading has already
                # observed the workspace; a missing text-only trajectory must
                # not turn a valid evaluation sample into an agent error.
                self.warnings.append(message)
                self.trajectory_conversion = {
                    "status": "unavailable",
                    "target": "actf",
                    "error": self.redact(str(exc)),
                }
            return trace, None, usage

    def _collect_remote_wire(self) -> None:
        if self.remote_proxy is None:
            return
        assert self.sandbox is not None
        box, directory = self.sandbox, self.remote_wire_dir
        result = box.run(["touch", f"{directory}/stop"], timeout=15)
        if result.returncode != 0:
            self.remote_proxy.kill()
            raise RuntimeError("could not stop the sandbox wire recorder")
        if self.remote_proxy.wait(timeout=30):
            raise RuntimeError("sandbox wire recorder exited with an error")
        if self.capture is not None:
            box.download(
                f"{directory}/capture",
                self.capture.recorder.directory,
                contents=True,
                timeout=120,
            )
            finished = box.run(["cat", f"{directory}/finished.json"], timeout=15)
            finished.check_returncode()
            self.capture.recorder.error = json.loads(finished.stdout)["error"]
        source = f"{directory}/wire_trace.jsonl"
        if box.run(["test", "-f", source], timeout=15).returncode == 0:
            wire = self.output_dir / "wire_trace.jsonl"
            box.download(source, wire, timeout=120)
            redact_jsonl_file(wire)
        source = f"{directory}/{TOKEN_IDS_FILE}"
        if box.run(["test", "-f", source], timeout=15).returncode == 0:
            box.download(source, self.output_dir / TOKEN_IDS_FILE, timeout=120)

    def _collect_remote_outputs(self):
        assert self.sandbox is not None
        workspace = self.harness.config.workspace_container_path
        with tempfile.TemporaryDirectory(dir=self.output_dir) as temporary:
            stage = Path(temporary)
            self.sandbox.download(
                f"{workspace}/{self.request.task.output_dir}",
                stage,
                contents=True,
                timeout=120,
            )
            return collect_output_directory(stage, self.output_dir), None

    def finish(self, status: str, error: str | None = None) -> HarnessResult:
        if self._result is not None:
            return self._result
        config, runtime = self.harness.config, self.harness.runtime
        trace = trajectory = None
        output_metadata = None
        usage: dict[str, int | float] = dict(ZERO_USAGE)
        problems: list[str] = []
        try:
            if self.container_started:
                self.collect_diagnostics()
                if self.sandbox is not None:
                    try:
                        self._stop_remote_native()
                    except Exception as exc:
                        problems.append(self.redact(str(exc)))
                try:
                    trace, trajectory, usage = self._collect_trace(problems)
                except Exception as exc:
                    problems.append(
                        f"{self.label} trace conversion failed: "
                        + self.redact(str(exc))
                    )
                if self.collect_outputs:
                    if self.sandbox is None:
                        collected, copy_error = collect_outputs(
                            runtime,
                            self.container,
                            self.request,
                            self.output_dir,
                            config.workspace_container_path,
                            timeout=30,
                        )
                    else:
                        collected, copy_error = self._collect_remote_outputs()
                    if copy_error is not None:
                        problems.append("output collection failed")
                    output_metadata = output_collection_metadata(
                        collected, self.request, self.output_dir
                    )
                    if not collected.passed:
                        problems.append("unsafe output files were rejected")
            else:
                problems.append(f"{self.label} container did not start")
        except Exception as exc:
            problems.append(self.redact(str(exc)))
        finally:
            if self.sandbox is not None:
                try:
                    self._collect_remote_wire()
                except Exception as exc:
                    message = self.redact(str(exc))
                    problems.append(message)
                    if self.capture is not None:
                        self.capture.errors.append(message)
            if self.proxy is not None:
                try:
                    self.proxy.stop()
                    if self.proxy.log_path.exists():
                        redact_jsonl_file(self.proxy.log_path)
                except Exception as exc:
                    problems.append(self.redact(str(exc)))
            if self.capture is not None:
                if trace is not None:
                    try:
                        self.capture.add_native(trace, self.native_trace_name)
                    except OSError as exc:
                        self.capture.errors.append(
                            f"native evidence collection failed: {self.native_trace_name}: {exc}"
                        )
                self.collect_native_evidence(usage)
                if self.redact_capture_errors:
                    self.capture_metadata = self.capture.finish(
                        status, error_formatter=lambda exc: self.redact(str(exc))
                    )
                else:
                    self.capture_metadata = self.capture.finish(status)
            if problems and status == "ok":
                status, error = "error", "; ".join(problems)
            retained = (
                self.container_started
                and config.keep_container_on_error
                and status != "ok"
            )
            if self.sandbox is not None and self.sandbox_run is not None:
                try:
                    self.sandbox_run.close(self.container, self.sandbox)
                except Exception:
                    logger.warning(
                        "sandbox removal failed: %s", self.container, exc_info=True
                    )
            elif self.container_started and not retained:
                try:
                    removed = remove_container(runtime, self.container)
                    if removed.returncode:
                        logger.warning(
                            "%s container removal failed: %s",
                            self.label,
                            self.container,
                        )
                except Exception:
                    logger.warning(
                        "%s container removal failed: %s",
                        self.label,
                        self.container,
                        exc_info=True,
                    )
        elapsed = time.monotonic() - self.started
        usage["elapsed_seconds"] = round(elapsed, 3)
        provenance = build_provenance(
            self.request,
            harness=self.harness.name,
            docker_image=config.docker_image if self.sandbox_run is None else None,
            docker_image_digest=self.image_digest,
            api_base_urls=[config.api_base or self.default_api_base],
            environment=self.environment,
            attempt=self.attempt,
            started_at=self.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            **(self.sandbox_run.target.provenance() if self.sandbox_run else {}),
            **self.version_fields(),
        )
        self._result = HarnessResult(
            status=status,
            error=self.redact(error) if error else None,
            elapsed_seconds=elapsed,
            usage=usage,
            trace_path=trace.relative_to(self.output_dir).as_posix() if trace else None,
            trajectory_path=trajectory.relative_to(self.output_dir).as_posix()
            if trajectory
            else None,
            metadata={
                **self.capture_metadata,
                "raw_trace_schema": self.raw_trace_schema,
                **self.version_fields(),
                "provenance": provenance,
                "output_collection": output_metadata,
                "turn_count": self.turns,
                **self.result_metadata(),
                "retained_container": self.container if retained else None,
                "settings": {
                    "reasoning_effort": resolve_thinking(
                        self.request.task.thinking, config.reasoning_effort
                    ),
                    **self.settings(),
                    "keep_thinking": config.keep_thinking,
                },
            },
        )
        return self._result


class BridgeHarness(InteractiveHarness):
    session_class: ClassVar[type[BridgeSession]]

    def __init__(
        self, config: Any, worker_slot: int, runtime: DockerRuntime | None = None
    ):
        self.config, self.worker_slot = config, worker_slot
        self.runtime = runtime or DockerRuntime(config.docker_binary)

    def create_session(
        self,
        request: RunRequest,
        output_dir: Path,
        attempt: int,
        *,
        collect_outputs: bool = True,
    ) -> BridgeSession:
        return self.session_class(
            self, request, output_dir, attempt, collect_outputs=collect_outputs
        )

    def run(self, request: RunRequest, output_dir: Path, attempt: int) -> HarnessResult:
        session = self.create_session(request, output_dir, attempt)
        status, error = "ok", None
        try:
            session.start()
            for index, message in enumerate(request.task.prompts):
                if (
                    session.supports_reset
                    and index < len(request.task.new_session)
                    and request.task.new_session[index]
                ):
                    session.reset()
                session.send(message, session.remaining())
        except (TimeoutError, subprocess.TimeoutExpired):
            status, error = "timeout", f"task exceeded {request.timeout_seconds}s"
        except Exception as exc:
            status, error = "error", session.redact(str(exc))
        return session.finish(status, error)
