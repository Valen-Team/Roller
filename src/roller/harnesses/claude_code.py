from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import roller.messages_bridge as messages_bridge
from roller.artifacts import ArtifactCollectionResult
from roller.capture_runtime import CaptureRuntime
from roller.docker_runtime import DockerRuntime, remove_container
from roller.harness_registry import convert_native_trace
from roller.harnesses.base import InteractiveHarness, InteractiveSession, TurnResult
from roller.harnesses.claude_code_config import (
    CLAUDE_HOME_CONTAINER_PATH,
    CLAUDE_INPUT_CONTAINER_PATH,
    GRADING_TRANSCRIPT_CONTAINER_PATH,
    TOOL_ID_MAPPINGS_NAME,
    ClaudeCodeConfig,
    actf_assistant_messages,
    claude_effort,
)
from roller.harnesses.claude_code_sandbox import SandboxAgent
from roller.harnesses.container import (
    CONTEXT_CONTAINER_PATH,
    INPUTS_CONTAINER_PATH,
    OUTPUTS_CONTAINER_PATH,
    ContainerCommands,
    build_provenance,
    docker_run_args,
    host_api_key,
    host_environment,
    output_collection_metadata,
)
from roller.models import HarnessResult, RunRequest
from roller.native_images import claude_user_message
from roller.redaction import (
    redact_anthropic_base_url,
    redact_jsonl_file,
)
from roller.sandbox import Sandbox
from roller.storage import safe_component
from roller.thinking import resolve_thinking
from roller.token_ids import TOKEN_IDS_FILE
from roller.usage import ZERO_USAGE, extract_claude_code_usage
from roller.wire_trace import WireTraceProxy

logger = logging.getLogger(__name__)


class ClaudeCodeHarness(ContainerCommands, InteractiveHarness):
    """Run Claude Code in an isolated container and retain native session JSONL."""

    name = "claude-code"
    error_limit = 4000

    def __init__(
        self,
        config: ClaudeCodeConfig,
        worker_slot: int,
        runtime: DockerRuntime | None = None,
    ) -> None:
        # Set when the CLI runs in a remote sandbox; the Docker runtime is then
        # unused and every container step goes through ``SandboxAgent``.
        self.sandbox_run = config.sandbox_run
        self.config = config
        self.worker_slot = worker_slot
        self.runtime = runtime or DockerRuntime(config.docker_binary)
        self._wire_proxy: WireTraceProxy | None = None
        self.capture: CaptureRuntime | None = None

    def create_session(
        self,
        request: RunRequest,
        output_dir: Path,
        attempt: int,
        *,
        collect_outputs: bool = True,
    ) -> InteractiveSession:
        return ClaudeCodeSession(
            self, request, output_dir, attempt, collect_outputs=collect_outputs
        )

    def _start_wire_proxy(self, output_dir: Path, attempt: int = 1) -> None:
        self._wire_proxy = None
        self.capture = None
        if self.config.wire_trace_mode == "off" or not self.config.api_base:
            return
        log_path = output_dir / "wire_trace.jsonl"
        log_path.unlink(missing_ok=True)
        mappings_path = output_dir / TOOL_ID_MAPPINGS_NAME
        mappings_path.unlink(missing_ok=True)
        token_ids_path = output_dir / TOKEN_IDS_FILE
        token_ids_path.unlink(missing_ok=True)
        if self.config.capture_calls:
            self.capture = CaptureRuntime(
                output_dir, attempt, "claude-code", self.config.capture_debug
            )
        self._wire_proxy = WireTraceProxy(
            self.config.api_base,
            log_path,
            mode="full"
            if self.config.wire_trace_mode == "full" or self.config.capture_calls
            else "tools",
            capture_recorder=self.capture.recorder if self.capture else None,
            # Claude Code 2.1.222 uses an SDK-internal ``?beta=true`` route.
            # Anthropic-compatible gateways commonly implement the stable
            # Messages path but reject this transport-only query flag.
            drop_query_parameters=("beta",),
            drop_request_headers=("anthropic-beta",),
            tool_id_mappings_path=mappings_path
            if self.config.unique_tool_ids
            else None,
            anthropic_pdf_as_images=self.config.pdf_as_images,
            chat_bridge=messages_bridge if self.config.chat_completions else None,
            chat_bridge_options={"tool_images_inline": True}
            if self.config.tool_images_inline
            else {},
            token_ids_path=token_ids_path if self.config.token_ids else None,
        )
        self._wire_proxy.start()

    def _stop_wire_proxy(self) -> None:
        if self._wire_proxy is None:
            return
        proxy = self._wire_proxy
        try:
            proxy.stop()
            capture = getattr(self, "capture", None)
            recorder = capture.recorder if capture else None
            if recorder is not None:
                recorder.seal()
            if proxy.log_path.is_file():
                redact_jsonl_file(proxy.log_path)
        finally:
            self._wire_proxy = None

    def _effective_effort(self, request: RunRequest) -> str | None:
        selected = resolve_thinking(request.task.thinking, self.config.reasoning_effort)
        return claude_effort(selected)

    def _model_name(self, request: RunRequest) -> str:
        if self.config.api_base:
            return request.model
        return request.model.rsplit("/", 1)[-1]

    def _environment(self, request: RunRequest) -> dict[str, str]:
        environment = host_environment(request.task, self.config.pass_env)
        environment["ANTHROPIC_API_KEY"] = host_api_key(
            "Claude Code", self.config.api_key_env
        )
        api_base = (
            self._wire_proxy.proxy_base_url
            if self._wire_proxy is not None
            else self.config.api_base
        )
        if api_base:
            environment["ANTHROPIC_BASE_URL"] = api_base
        model_name = self._model_name(request)
        environment.update(
            {
                "ANTHROPIC_MODEL": model_name,
                "CLAUDE_CONFIG_DIR": CLAUDE_HOME_CONTAINER_PATH,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                "IS_SANDBOX": "1",
            }
        )
        if self.config.api_base:
            environment.update(
                {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": model_name,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": model_name,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_name,
                    "CLAUDE_CODE_SUBAGENT_MODEL": model_name,
                }
            )
        if self.config.max_thinking_tokens is not None:
            environment["MAX_THINKING_TOKENS"] = str(self.config.max_thinking_tokens)
        return environment

    def _start_container(
        self, container: str, request: RunRequest, image: str | None = None
    ) -> dict[str, str]:
        environment = self._environment(request)
        args = docker_run_args(self.name, container, self.config, environment, request)
        args.extend(
            [image or self.config.docker_image, "/bin/bash", "-c", "tail -f /dev/null"]
        )
        self._checked(args, "Claude Code container startup", timeout=60)
        return environment

    def _prepare_workspace(self, container: str, request: RunRequest) -> None:
        workspace = self.config.workspace_container_path
        output_root = f"{workspace}/{request.task.output_dir}"
        if request.bundle is not None:
            if request.bundle.legacy_workspace or request.task.input_mode == "copy":
                setup = (
                    f"mkdir -p {shlex.quote(workspace)} && "
                    f"cp -a {INPUTS_CONTAINER_PATH}/. {shlex.quote(workspace)}/ && "
                    f"chmod -R u+w {shlex.quote(workspace)} && "
                    f"rm -rf {shlex.quote(workspace)}/context && "
                    f"ln -s {CONTEXT_CONTAINER_PATH} {shlex.quote(workspace)}/context"
                )
            else:
                setup = f"mkdir -p {shlex.quote(workspace)}"
        elif request.task.workspace is not None:
            setup = (
                f"mkdir -p {shlex.quote(workspace)} && "
                f"cp -a /roller-input/. {shlex.quote(workspace)}/ && "
                f"chmod -R u+w {shlex.quote(workspace)}"
            )
        else:
            setup = f"mkdir -p {shlex.quote(workspace)}"
        script = (
            f"{setup} && rm -rf {shlex.quote(output_root)} && "
            f"mkdir -p {shlex.quote(output_root)} {CLAUDE_HOME_CONTAINER_PATH} && "
            f"mkdir -p /roller/task && rm -rf {OUTPUTS_CONTAINER_PATH} && "
            f"ln -s {shlex.quote(output_root)} {OUTPUTS_CONTAINER_PATH}"
        )
        self._checked(
            self.runtime.exec_args(container, ["/bin/bash", "-c", script]),
            "Claude Code workspace setup",
        )

    def _prepare_context_and_setup(self, container: str, request: RunRequest) -> None:
        if request.bundle is not None:
            workspace = shlex.quote(self.config.workspace_container_path)
            script = (
                f"cp -a {CONTEXT_CONTAINER_PATH}/. {workspace}/ && "
                f"mkdir -p {CLAUDE_HOME_CONTAINER_PATH}/skills && "
                f"if [ -d {CONTEXT_CONTAINER_PATH}/skills ]; then "
                f"cp -a {CONTEXT_CONTAINER_PATH}/skills/. "
                f"{CLAUDE_HOME_CONTAINER_PATH}/skills/; fi"
            )
            self._checked(
                self.runtime.exec_args(container, ["/bin/bash", "-c", script]),
                "Claude Code context injection",
            )
        for command in request.task.setup_commands:
            self._checked(
                self.runtime.exec_args(
                    container,
                    ["/bin/bash", "-lc", command],
                    workdir=self.config.workspace_container_path,
                ),
                f"task setup command {command!r}",
            )

    def _prepare_settings(self, container: str) -> None:
        self._checked(
            self.runtime.exec_args(
                container, ["mkdir", "-p", CLAUDE_HOME_CONTAINER_PATH]
            ),
            "Claude Code config directory setup",
        )
        if self.config.settings_path is None:
            return
        self._checked(
            self.runtime.copy_in_args(
                self.config.settings_path,
                container,
                f"{CLAUDE_HOME_CONTAINER_PATH}/settings.json",
            ),
            "Claude Code settings copy",
        )

    def claude_argv(
        self,
        request: RunRequest,
        prompt: str,
        *,
        resume: bool,
        input_path: str | None = None,
    ) -> list[str]:
        """One ``claude --print`` turn as the argv to run in the agent's container.

        ``input_path`` is a stream-json user message inside the container; it
        replaces the prompt argument when the message carries images. The
        remote-sandbox backend runs this argv directly, so the turn's flags stay
        the same whichever backend executes it.
        """

        command = [
            "claude",
            "--verbose",
            "--output-format=stream-json",
            f"--permission-mode={self.config.permission_mode}",
            "--model",
            self._model_name(request),
        ]
        if self.config.settings_path is not None:
            command.extend(
                ["--settings", f"{CLAUDE_HOME_CONTAINER_PATH}/settings.json"]
            )
        if self.config.max_turns is not None:
            command.extend(["--max-turns", str(self.config.max_turns)])
        effort = self._effective_effort(request)
        if effort:
            command.extend(["--effort", effort])
        if resume:
            command.append("--continue")
        if input_path is None:
            command.extend(["--print", prompt])
        else:
            command.extend(["--print", "--input-format=stream-json"])
            # Neither ``docker exec`` nor the sandbox API forwards stdin; read
            # the message from the file instead.
            index = command.index("claude")
            command[index:] = [
                "/bin/sh",
                "-c",
                'exec "$@" < "$0"',
                input_path,
                *command[index:],
            ]
        return command

    def _claude_command(
        self,
        container: str,
        request: RunRequest,
        prompt: str,
        *,
        resume: bool,
        input_path: str | None = None,
    ) -> list[str]:
        """``claude_argv`` wrapped in the ``docker exec`` that runs it."""

        return self.runtime.exec_args(
            container,
            self.claude_argv(request, prompt, resume=resume, input_path=input_path),
            env={"CLAUDE_CONFIG_DIR": CLAUDE_HOME_CONTAINER_PATH},
            workdir=self.config.workspace_container_path,
        )

    def _copy_image_input(
        self, container: str, request: RunRequest, prompt: str, staging: Path
    ) -> str:
        payload_path = staging / ".claude-input.jsonl"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            json.dumps(claude_user_message(request, prompt)) + "\n",
            encoding="utf-8",
        )
        try:
            self._checked(
                self.runtime.copy_in_args(
                    payload_path, container, CLAUDE_INPUT_CONTAINER_PATH
                ),
                "Claude Code image input copy",
            )
        finally:
            payload_path.unlink(missing_ok=True)
        return CLAUDE_INPUT_CONTAINER_PATH

    def _run_turn(
        self,
        container: str,
        request: RunRequest,
        prompt: str,
        *,
        resume: bool,
        timeout_seconds: float,
        log_path: Path,
    ) -> int:
        input_path = None
        if request.task.attach_images and not resume:
            input_path = self._copy_image_input(
                container, request, prompt, log_path.parent
            )
        process = self.runtime.popen(
            self._claude_command(
                container, request, prompt, resume=resume, input_path=input_path
            ),
            log_path,
            append=resume,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            self.runtime.exec(container, ["pkill", "-f", "claude"])
            raise TimeoutError("Claude Code turn timed out") from exc
        finally:
            self.runtime.close_log(process)
        if return_code != 0:
            detail = ""
            try:
                detail = log_path.read_text(encoding="utf-8", errors="replace").strip()[
                    -4000:
                ]
            except OSError:
                pass
            raise RuntimeError(
                f"Claude Code exited with code {return_code}"
                + (f": {detail}" if detail else "")
            )
        return return_code

    @staticmethod
    def _combine_session_files(files: list[Path], destination: Path) -> None:
        rows: list[tuple[str, int, dict[str, Any]]] = []
        seen_uuids: set[str] = set()
        sequence = 0
        for path in files:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    event_uuid = event.get("uuid")
                    if isinstance(event_uuid, str) and event_uuid:
                        if event_uuid in seen_uuids:
                            continue
                        seen_uuids.add(event_uuid)
                    rows.append((str(event.get("timestamp", "")), sequence, event))
                    sequence += 1
        rows.sort(key=lambda item: (item[0], item[1]))
        with destination.open("w", encoding="utf-8") as stream:
            for _, _, event in rows:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def trace_staging_root(self, output_dir: Path) -> Path:
        """An empty directory for a downloaded copy of the CLI's session files."""

        staging_root = output_dir / ".claude-session-copy"
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        return staging_root

    def trace_from_staging(
        self, staging_root: Path, output_dir: Path, attempt: int
    ) -> Path | None:
        """Combine the session files under ``staging_root`` into the run's trace.

        Both backends reach this with the same directory contents, copied out of
        a Docker container or downloaded from a remote sandbox.
        """

        rollout_dir = output_dir / "rollout"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        primary_files = [
            path
            for path in staging_root.rglob("*.jsonl")
            if "subagents" not in path.relative_to(staging_root).parts
            and path.is_file()
        ]
        if not primary_files:
            return None
        primary = max(primary_files, key=lambda path: path.stat().st_mtime_ns)
        sidechain_dir = primary.parent / primary.stem / "subagents"
        files = [primary]
        if sidechain_dir.is_dir():
            files.extend(
                sorted(
                    path
                    for path in sidechain_dir.rglob("*.jsonl")
                    if not path.is_symlink()
                )
            )
        if self.config.capture_calls and self.capture is not None:
            from roller.storage import atomic_write_json

            native_dir = self.capture.recorder.directory / "native"
            native_dir.mkdir(parents=True, exist_ok=True)
            sources = []
            for index, path in enumerate(files):
                if path.is_symlink():
                    continue
                name = f"source-{index:04d}.jsonl"
                self.capture.add_native(path, name)
                sources.append(
                    {
                        "path": f"native/{name}",
                        "native_path": path.relative_to(staging_root).as_posix(),
                    }
                )
            atomic_write_json(native_dir.parent / "native_index.json", sources)
        destination = rollout_dir / f"trace.attempt-{attempt}.jsonl"
        self._combine_session_files(files, destination)
        if not destination.is_file() or destination.stat().st_size == 0:
            return None
        trace_path = rollout_dir / "trace.jsonl"
        shutil.copyfile(destination, trace_path)
        return trace_path

    def _collect_trace(
        self, container: str, output_dir: Path, attempt: int
    ) -> Path | None:
        staging_root = self.trace_staging_root(output_dir)
        try:
            copied = self.runtime.copy_out(
                container,
                f"{CLAUDE_HOME_CONTAINER_PATH}/projects/.",
                staging_root,
                timeout=60,
            )
            if copied.returncode != 0:
                return None
            return self.trace_from_staging(staging_root, output_dir, attempt)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def _capture_metadata(
        self, output_dir: Path, attempt: int, status: str
    ) -> dict[str, Any]:
        if not self.config.capture_calls:
            return {}
        if self.capture is None:
            return {
                "capture": {
                    "manifest": None,
                    "capture_status": "partial",
                    "error": "capture recorder unavailable",
                }
            }
        metadata = self.capture.finish(status, error_formatter=str)
        if metadata["capture"]["manifest"] is None:
            logger.warning(
                "Claude call capture finalization failed: %s",
                metadata["capture"]["error"],
            )
        return metadata

    def run(self, request: RunRequest, output_dir: Path, attempt: int) -> HarnessResult:
        session = ClaudeCodeSession(self, request, output_dir, attempt)
        status, error = "error", None
        try:
            session.start()
            for index, prompt in enumerate(request.task.prompts):
                remaining = request.timeout_seconds - (
                    time.monotonic() - session.started_monotonic
                )
                if remaining <= 0:
                    raise TimeoutError("Claude Code task timed out")
                session.return_codes.append(
                    session.run_turn(
                        prompt, resume=index > 0, timeout_seconds=remaining
                    )
                )
            status = "ok"
        except (TimeoutError, subprocess.TimeoutExpired):
            status = "timeout"
            error = f"task exceeded {request.timeout_seconds}s"
        except Exception as exc:
            status = "error"
            error = str(exc)
        return session.finish(status, error, user_messages=request.task.prompts)


class ClaudeCodeSession(InteractiveSession):
    """Persistent Claude Code session for model-driven conversations."""

    def __init__(
        self,
        harness: ClaudeCodeHarness,
        request: RunRequest,
        output_dir: Path,
        attempt: int,
        *,
        collect_outputs: bool = True,
    ) -> None:
        self.harness = harness
        self.request = request
        self.output_dir = output_dir
        self.attempt = attempt
        self.collect_outputs = collect_outputs
        self.started = False
        self.container_started = False
        self.container = (
            f"roller-claude-{request.index}-{safe_component(request.display_model, 20)}-"
            f"{uuid.uuid4().hex[:8]}"
        ).lower()
        self.messages: list[str] = []
        self.return_codes: list[int] = []
        self._pending_message = ""
        self.environment: dict[str, str] = {}
        self.claude_version = ""
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.started_monotonic = time.monotonic()
        self.image_digest = None if harness.sandbox_run else harness._image_digest()
        self.stream_path = (
            output_dir / "rollout" / f"claude-code.attempt-{attempt}.jsonl"
        )
        self.agent: SandboxAgent | None = (
            SandboxAgent(
                harness, request, output_dir, self.container, harness.sandbox_run
            )
            if harness.sandbox_run is not None
            else None
        )
        self._finished: HarnessResult | None = None

    def run_turn(self, prompt: str, *, resume: bool, timeout_seconds: float) -> int:
        """One turn, run by whichever backend holds this sample's container."""

        if self.agent is not None:
            return self.agent.run_turn(
                prompt,
                resume=resume,
                timeout_seconds=timeout_seconds,
                log_path=self.stream_path,
            )
        return self.harness._run_turn(
            self.container,
            self.request,
            prompt,
            resume=resume,
            timeout_seconds=timeout_seconds,
            log_path=self.stream_path,
        )

    def collect_trace(self) -> Path | None:
        if self.agent is not None:
            return self.agent.collect_trace(self.output_dir, self.attempt)
        return self.harness._collect_trace(
            self.container, self.output_dir, self.attempt
        )

    @property
    def sandbox(self) -> Sandbox | None:
        """The remote sandbox this sample runs in, which bench grading reads.

        ``None`` on the Docker path, where a grader reaches the agent's work
        through ``docker exec`` instead.
        """

        return self.agent.box if self.agent is not None else None

    def upload_for_grading(self, source: Path, target: str) -> None:
        """Place one host file inside this sample's container or sandbox."""

        if self.agent is not None:
            self.agent.upload(source, target)
            return
        copied = self.harness.runtime.copy_in(
            source, self.container, target, timeout=60
        )
        if copied.returncode != 0:
            detail = (copied.stderr or copied.stdout or "").strip()[-300:]
            raise RuntimeError(
                f"could not copy {source} into {self.container}:{target}"
                + (f": {detail}" if detail else "")
            )

    def prepare_transcript_for_grading(self) -> str | None:
        """Place the run's assistant messages where a bench grader reads them.

        The CLI names its session files after the project directory and the
        session id, and a run's subagents write their own, so no fixed path
        inside the container holds the conversation. Roller already combines
        those files on this host and converts them to ACTF, so grading reads
        that conversion's visible assistant turns: a bench then scores exactly
        the text ``rollout/trace_ACTF.json`` records as the run's last reply,
        sidechains left out. Both backends are served, ``docker cp`` for a
        local container and the sandbox API's upload for a remote one.
        """

        if not self.container_started:
            return None
        trace = self.collect_trace()
        if trace is None:
            # The harness treats a missing native trace as a failed run, so
            # grading says so too rather than scoring an empty transcript zero.
            raise RuntimeError("Claude Code produced no native session trace to grade")
        trajectory = convert_native_trace(
            "claude-code",
            trace,
            self.output_dir / "rollout" / ".grading.ACTF.json",
            user_messages=list(self.messages),
            wire_trace=self.output_dir / "wire_trace.jsonl",
            keep_thinking=self.harness.config.keep_thinking,
        )
        staging = self.output_dir / "rollout" / ".grading-transcript.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(
            json.dumps(actf_assistant_messages(trajectory), ensure_ascii=False),
            encoding="utf-8",
        )
        self.upload_for_grading(staging, GRADING_TRANSCRIPT_CONTAINER_PATH)
        return GRADING_TRANSCRIPT_CONTAINER_PATH

    def start(self) -> None:
        if self.started:
            raise RuntimeError("Claude Code session has already started")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stream_path.parent.mkdir(parents=True, exist_ok=True)
        for relative in (
            "rollout/trace.jsonl",
            f"rollout/trace.attempt-{self.attempt}.jsonl",
            "wire_trace.jsonl",
        ):
            (self.output_dir / relative).unlink(missing_ok=True)
        self.harness.deadline = self.started_monotonic + self.request.timeout_seconds
        self.harness._start_wire_proxy(self.output_dir, self.attempt)
        # Mark before run: a timed-out client may already have created it.
        self.container_started = True
        if self.agent is not None:
            self.environment = self.agent.start(
                self.harness._environment(self.request),
                self.harness._wire_proxy.proxy_base_url
                if self.harness._wire_proxy is not None
                else self.harness.config.api_base,
            )
            self.claude_version = self.agent.version()
            self.started = True
            return
        self.environment = self.harness._start_container(
            self.container, self.request, self.image_digest
        )
        self.harness._prepare_workspace(self.container, self.request)
        self.harness._prepare_context_and_setup(self.container, self.request)
        self.harness._prepare_settings(self.container)
        self.claude_version = self.harness._checked(
            self.harness.runtime.exec_args(self.container, ["claude", "--version"]),
            "Claude Code version check",
        ).stdout.strip()
        self.started = True

    def _latest_reply(self) -> str:
        trace = self.collect_trace()
        if trace is None:
            raise RuntimeError("Claude Code turn produced no native session trace")
        destination = (
            self.output_dir / "rollout" / f".turn-{len(self.messages) + 1}.ACTF.json"
        )
        trajectory = convert_native_trace(
            "claude-code",
            trace,
            destination,
            user_messages=[*self.messages, self._pending_message],
            wire_trace=self.output_dir / "wire_trace.jsonl",
            keep_thinking=self.harness.config.keep_thinking,
        )
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            raise RuntimeError("Claude Code turn produced an invalid ACTF trajectory")
        for step in reversed(steps):
            if not isinstance(step, Mapping):
                continue
            assistant = step.get("assistant_content")
            if not isinstance(assistant, Mapping):
                continue
            content = assistant.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        raise RuntimeError("Claude Code turn produced no visible assistant reply")

    def send(self, message: str, timeout_seconds: float) -> TurnResult:
        if not self.started or not self.container_started:
            raise RuntimeError("Claude Code session has not started")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Claude Code session messages must be non-empty strings")
        prompt = message.strip()
        turn_started = time.monotonic()
        self._pending_message = prompt
        self.return_codes.append(
            self.run_turn(
                prompt, resume=bool(self.messages), timeout_seconds=timeout_seconds
            )
        )
        reply = self._latest_reply()
        self.messages.append(prompt)
        return TurnResult(reply, time.monotonic() - turn_started, self.return_codes[-1])

    def finish(
        self,
        status: str,
        error: str | None = None,
        *,
        user_messages: Sequence[str] | None = None,
    ) -> HarnessResult:
        """Collect traces and outputs, release the container and build the result.

        ``user_messages`` defaults to the messages delivered through :meth:`send`;
        :meth:`ClaudeCodeHarness.run` passes the task prompts it ran directly.
        """

        if self._finished is not None:
            return self._finished
        harness, config = self.harness, self.harness.config
        task_id = self.request.task.id
        messages = list(self.messages if user_messages is None else user_messages)
        trace_path: Path | None = None
        trajectory_path: Path | None = None
        artifact_result: ArtifactCollectionResult | None = None
        output_copy_error: str | None = None
        try:
            harness._stop_wire_proxy()
        except Exception as exc:
            logger.warning(
                "[%s] Claude Code wire proxy shutdown failed: %s", task_id, exc
            )
        if self.container_started:
            try:
                trace_path = self.collect_trace()
            except Exception as exc:
                logger.warning(
                    "[%s] Claude Code trace collection failed: %s", task_id, exc
                )
            if self.collect_outputs:
                try:
                    artifact_result, output_copy_error = (
                        self.agent.collect_outputs(self.output_dir)
                        if self.agent is not None
                        else harness._collect_outputs(
                            self.container, self.request, self.output_dir
                        )
                    )
                except Exception as exc:
                    output_copy_error = str(exc)
                    logger.warning(
                        "[%s] Claude Code output collection failed: %s", task_id, exc
                    )
        if trace_path is not None:
            try:
                trajectory_path = self.output_dir / "rollout" / "trace_ACTF.json"
                trajectory = convert_native_trace(
                    "claude-code",
                    trace_path,
                    trajectory_path,
                    user_messages=messages,
                    wire_trace=self.output_dir / "wire_trace.jsonl",
                    keep_thinking=config.keep_thinking,
                )
                if not trajectory.get("steps"):
                    raise ValueError(
                        "Claude Code native trace produced no assistant steps"
                    )
            except Exception as exc:
                trajectory_path = None
                if status == "ok":
                    status, error = (
                        "error",
                        f"Claude Code trajectory conversion failed: {exc}",
                    )
        if status == "ok" and trace_path is None:
            status, error = (
                "error",
                "Claude Code finished but no native session trace was found",
            )
        if status == "ok" and output_copy_error is not None:
            status, error = "error", f"output collection failed: {output_copy_error}"
        if (
            status == "ok"
            and artifact_result is not None
            and not artifact_result.passed
        ):
            status, error = "error", "unsafe output files were rejected"
        retained_container: str | None = None
        if self.container_started and self.agent is not None:
            # A remote sandbox is the run's to release, whatever the outcome:
            # --keep-container-on-error applies to Docker only.
            try:
                self.agent.close()
            except Exception as exc:
                logger.warning(
                    "[%s] failed to release sandbox %s: %s",
                    task_id,
                    self.container,
                    exc,
                )
        elif self.container_started:
            if config.keep_container_on_error and status != "ok":
                retained_container = self.container
            else:
                removed = remove_container(harness.runtime, self.container)
                if removed.returncode != 0:
                    logger.warning(
                        "[%s] failed to remove container %s", task_id, self.container
                    )
        elapsed = time.monotonic() - self.started_monotonic
        usage = (
            extract_claude_code_usage(trace_path, self.stream_path)
            if trace_path is not None
            else dict(ZERO_USAGE)
        )
        usage["elapsed_seconds"] = round(elapsed, 3)
        provenance = build_provenance(
            self.request,
            harness=harness.name,
            claude_code_version=self.claude_version,
            docker_image=config.docker_image,
            docker_image_digest=self.image_digest,
            api_base_urls=[redact_anthropic_base_url(config.api_base)]
            if config.api_base
            else [],
            environment=self.environment,
            attempt=self.attempt,
            started_at=self.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        try:
            reasoning_effort = resolve_thinking(
                self.request.task.thinking, config.reasoning_effort
            )
            transport_effort = harness._effective_effort(self.request)
        except Exception:
            reasoning_effort = None
            transport_effort = None
        self._finished = HarnessResult(
            status=status,
            elapsed_seconds=elapsed,
            error=error,
            usage=usage,
            trace_path=trace_path.relative_to(self.output_dir).as_posix()
            if trace_path
            else None,
            trajectory_path=trajectory_path.relative_to(self.output_dir).as_posix()
            if trajectory_path
            else None,
            metadata={
                **harness._capture_metadata(self.output_dir, self.attempt, status),
                "raw_trace_schema": "claude-code-session-jsonl",
                "claude_code_version": self.claude_version,
                "container": retained_container,
                **(
                    harness.sandbox_run.target.provenance()
                    if harness.sandbox_run is not None
                    else {}
                ),
                "agent_return_codes": self.return_codes,
                "turns": len(messages),
                "reasoning_effort": reasoning_effort,
                "transport_effort": transport_effort,
                "max_thinking_tokens": config.max_thinking_tokens,
                "max_turns": config.max_turns,
                "permission_mode": config.permission_mode,
                "wire_trace": config.wire_trace_mode,
                **({"unique_tool_ids": True} if config.unique_tool_ids else {}),
                **({"tool_images_inline": True} if config.tool_images_inline else {}),
                **({"pdf_as_images": True} if config.pdf_as_images else {}),
                "keep_thinking": config.keep_thinking,
                "config_sha256": config.config_sha256,
                "output_collection": output_collection_metadata(
                    artifact_result, self.request, self.output_dir
                ),
                "output_copy_error": output_copy_error,
                "attempt": self.attempt,
                "provenance": provenance,
            },
        )
        return self._finished
