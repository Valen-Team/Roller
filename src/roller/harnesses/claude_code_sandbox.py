"""Run Claude Code in a remote sandbox instead of a local Docker container.

The CLI is a Linux binary the sandbox template mounts read-only, and a sandbox
created without egress cannot reach the recording proxy on the collecting host.
The provider's ``route_http`` carries the CLI's requests out through the sandbox
API instead, so the host keeps the same :class:`~roller.wire_trace.WireTraceProxy`
the Docker path uses and the run's wire trace, call capture, token ids and blobs
are written by the same code either way. Nothing of Roller runs in the sandbox:
it holds the task's files and the CLI, and the agent's own requests are the only
traffic that leaves it.

The paths the CLI sees are the Docker path's, so a collected sample's workspace,
config directory and task directory read the same whichever backend ran it.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roller.artifacts import ArtifactCollectionResult
from roller.execution import SandboxWorkspace, stage_sandbox_inputs
from roller.harnesses.container import host_api_key
from roller.models import RunRequest
from roller.sandbox import Sandbox, SandboxRun

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime only
    from roller.harnesses.claude_code import ClaudeCodeHarness

logger = logging.getLogger(__name__)

# Where the CLI's own files live, matching the Docker path's container paths.
CLAUDE_HOME = "/roller/claude-home"
TASK_DIR = "/roller/task"
# Default location of the read-only volume the sandbox template mounts the
# pinned CLI release at; ``--claude-sandbox-binary`` overrides it.
DEFAULT_BINARY = "/opt/claude-code/bin/claude"
# A sandbox outlives its sample's budget by this much, so collecting the trace
# after a turn that used all of its time still has a live instance to read.
LIFETIME_MARGIN = 900
# How long the CLI waits for one reply. The relay hands it a whole response at
# once, so a long reply plus the request layer's retries on a busy replica have
# to fit inside this; the sample's own timeout is what bounds the run.
REPLY_TIMEOUT_MS = 2 * 3600 * 1000
# CLI settings this backend fixes, so what a sample sends is the benchmark's
# protocol rather than the collecting machine's defaults.
PROTOCOL_ENVIRONMENT = {
    # The CLI's attribution block varies per conversation, which would defeat
    # the served model's prefix cache.
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "API_TIMEOUT_MS": str(REPLY_TIMEOUT_MS),
}
SETUP_TIMEOUT = 300
VERSION_TIMEOUT = 120
DOWNLOAD_TIMEOUT = 900


def ending(log_path: Path) -> str:
    """How the CLI said its turn ended, from the ``result`` event of its stream.

    A stream-json log carries the images the CLI viewed, so its last bytes are
    usually the middle of a base64 blob: the event the CLI writes about itself
    says far more than a tail of the file does.
    """

    result: dict[str, Any] | None = None
    try:
        with log_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                # Only the result event is wanted, and it is short; skipping the
                # rest keeps a multi-megabyte log out of memory.
                if '"type":"result"' not in line.replace(" ", ""):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:  # a line still being written
                    continue
                if isinstance(event, dict) and event.get("type") == "result":
                    result = event
    except OSError:
        return "its stream could not be read"
    if result is None:
        return "its stream holds no result event"
    subtype = str(result.get("subtype") or "unknown")
    said = str(result.get("result") or "").strip()[:500]
    turns = result.get("num_turns")
    counted = f", {turns} turns" if isinstance(turns, int) else ""
    return f"{subtype}{counted}" + (f": {said}" if said else "")


class SandboxAgent:
    """One sample's sandbox: start it, run the CLI's turns, read what it wrote."""

    def __init__(
        self,
        harness: ClaudeCodeHarness,
        request: RunRequest,
        output_dir: Path,
        name: str,
        sandbox_run: SandboxRun,
    ) -> None:
        self.harness = harness
        self.request = request
        self.output_dir = output_dir
        self.name = name
        self.sandbox_run = sandbox_run
        self.box: Sandbox | None = None
        self.backend: SandboxWorkspace | None = None
        self.environment: dict[str, str] = {}

    @property
    def binary(self) -> str:
        return self.harness.config.sandbox_binary or DEFAULT_BINARY

    def _checked(self, argv: list[str], description: str, timeout: float) -> Any:
        assert self.box is not None
        try:
            result = self.box.run(argv, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            # A setup step hitting its own limit is not the task running out of
            # time, which is what a TimeoutError would be reported as.
            raise RuntimeError(
                f"{description} timed out after {exc.timeout:g}s"
            ) from None
        if result.returncode != 0:
            raise RuntimeError(
                f"{description} failed: {self.harness._error_text(result)}"
            )
        return result

    def start(
        self, environment: dict[str, str], api_base: str | None
    ) -> dict[str, str]:
        """Create the sandbox, route the CLI's API base through it, place the task.

        ``api_base`` is the recording proxy's URL on this host when one runs.
        The returned environment is the CLI's, with the base URL and token that
        reach that proxy from inside the sandbox.
        """

        timeout = self.request.timeout_seconds
        # The provider clips this to its template's own startup limit, which is
        # how long that environment takes to come up; a shorter one here only
        # abandons instances that were still starting.
        try:
            box = self.sandbox_run.create(self.name, timeout + LIFETIME_MARGIN, timeout)
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            # An instance that never came up is the cluster's doing, not the
            # task running out of its own time, which is what a TimeoutError
            # reaching the collector is reported as.
            raise RuntimeError(f"sandbox did not start: {exc}") from None
        self.box = box
        self.backend = SandboxWorkspace(
            box, name=self.name, sandbox_run=self.sandbox_run
        )
        if box.version != self.sandbox_run.target.version:
            raise RuntimeError("sandbox environment version changed since preflight")
        workspace = self.harness.config.workspace_container_path
        directories = [workspace, CLAUDE_HOME, TASK_DIR]
        if self.request.task.output_dir:
            # The Harness collects this directory before teardown, so it has to
            # exist even when the task's protocol writes nothing into it.
            directories.append(f"{workspace}/{self.request.task.output_dir}")
        self._checked(
            ["mkdir", "-p", *directories], "sandbox workspace setup", SETUP_TIMEOUT
        )
        self._place_task(box, workspace)
        self._place_settings(box)
        if api_base:
            key_name = self.harness.config.api_key_env
            key = environment.get(key_name) or host_api_key("Claude Code", key_name)
            # The provider hands back a token for the relay; the real key stays
            # on this host, where the relay's forwarder adds it again.
            api_base, environment[key_name] = self.backend.route_http(
                api_base, key, timeout
            )
            environment["ANTHROPIC_BASE_URL"] = api_base
            environment["ANTHROPIC_AUTH_TOKEN"] = environment[key_name]
        environment["CLAUDE_CONFIG_DIR"] = CLAUDE_HOME
        # Only loopback is reachable from the sandbox; a proxy variable the
        # collecting host passed through would send the CLI's calls nowhere.
        environment["NO_PROXY"] = "127.0.0.1,localhost"
        environment.update(PROTOCOL_ENVIRONMENT)
        window = self.harness.config.context_window
        if window is not None:
            # The served window, so the CLI compacts its history before the
            # server refuses it rather than ending the sample.
            environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(window)
        self.environment = environment
        return environment

    def _place_task(self, box: Sandbox, workspace: str) -> None:
        """Copy the task's files in via the shared sandbox staging helper."""

        timeout = min(self.request.timeout_seconds, DOWNLOAD_TIMEOUT)
        stage_sandbox_inputs(
            box,
            self.request,
            workspace,
            upload_timeout=timeout,
            run_checked=lambda argv, desc: self._checked(argv, desc, SETUP_TIMEOUT),
            chmod_legacy_workspace=True,
        )
        for command in self.request.task.setup_commands:
            self._checked(
                ["/bin/bash", "-lc", command],
                f"task setup command {command!r}",
                SETUP_TIMEOUT,
            )

    def _place_settings(self, box: Sandbox) -> None:
        settings = self.harness.config.settings_path
        if settings is None:
            return
        box.upload(settings, f"{CLAUDE_HOME}/settings.json", timeout=SETUP_TIMEOUT)

    def version(self) -> str:
        result = self._checked(
            [self.binary, "--version"], "Claude Code version check", VERSION_TIMEOUT
        )
        return str(result.stdout).strip()

    def _image_input(self, prompt: str, staging: Path) -> str:
        """Place the turn's stream-json user message, which carries its images."""

        from roller.native_images import claude_user_message

        assert self.box is not None
        payload = staging / ".claude-input.jsonl"
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text(
            json.dumps(claude_user_message(self.request, prompt)) + "\n",
            encoding="utf-8",
        )
        target = "/roller/claude-input.jsonl"
        try:
            self.box.upload(payload, target, timeout=SETUP_TIMEOUT)
        finally:
            payload.unlink(missing_ok=True)
        return target

    def run_turn(
        self, prompt: str, *, resume: bool, timeout_seconds: float, log_path: Path
    ) -> int:
        """Run one ``claude --print`` turn, streaming its output to ``log_path``."""

        assert self.box is not None
        input_path = None
        if self.request.task.attach_images and not resume:
            input_path = self._image_input(prompt, log_path.parent)
        argv = self.harness.claude_argv(
            self.request, prompt, resume=resume, input_path=input_path
        )
        argv = [self.binary if item == "claude" else item for item in argv]
        process = self.box.spawn(
            argv,
            cwd=self.harness.config.workspace_container_path,
            env=self.environment,
            log_path=log_path,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            process.kill()
            raise TimeoutError("Claude Code turn timed out") from exc
        except Exception:
            process.kill()
            raise
        if return_code != 0:
            raise RuntimeError(
                f"Claude Code exited with code {return_code}: {ending(log_path)}"
            )
        return return_code

    def collect_trace(self, output_dir: Path, attempt: int) -> Path | None:
        """Download the CLI's native session files and combine them as usual."""

        if self.box is None:
            return None
        staging_root = self.harness.trace_staging_root(output_dir)
        try:
            try:
                self.box.download(
                    f"{CLAUDE_HOME}/projects",
                    staging_root,
                    contents=True,
                    timeout=DOWNLOAD_TIMEOUT,
                )
            except Exception as exc:
                logger.warning("Claude Code session download failed: %s", exc)
                return None
            return self.harness.trace_from_staging(staging_root, output_dir, attempt)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def upload(self, source: Path, target: str) -> None:
        """Place one host file in the sandbox, as ``docker cp`` does locally."""

        if self.box is None:
            raise RuntimeError("the sandbox has been released")
        self.box.upload(source, target, timeout=SETUP_TIMEOUT)

    def collect_outputs(
        self, output_dir: Path
    ) -> tuple[ArtifactCollectionResult, str | None]:
        """Download the task's output directory and vet it via the unified backend."""

        assert self.backend is not None
        workspace = self.harness.config.workspace_container_path
        source = f"{workspace}/{self.request.task.output_dir}"
        return self.backend.collect_outputs(
            source, output_dir, timeout=DOWNLOAD_TIMEOUT, catch_errors=True
        )

    def close(self) -> None:
        if self.backend is None:
            return
        backend, self.backend, self.box = self.backend, None, None
        backend.close()
