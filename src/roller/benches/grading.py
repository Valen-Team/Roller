from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from roller.benches.base import BenchBinding, GradeSpec
from roller.benches.report import brief_reason
from roller.execution import WorkspaceBackend, workspace_backend
from roller.harnesses.base import Harness, InteractiveHarness, InteractiveSession
from roller.models import HarnessResult, RunRequest
from roller.option_specs import GRADER_RETRY_MAX_SECONDS
from roller.sandbox import Sandbox
from roller.storage import atomic_write_json

logger = logging.getLogger(__name__)

ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
GRADER_CONTAINER_DIR = "/tmp/roller-bench"
RUNNER_CONTAINER_PATH = f"{GRADER_CONTAINER_DIR}/_grade_runner.py"
TRANSCRIPT_CONTAINER_PATH = f"{GRADER_CONTAINER_DIR}/transcript.jsonl"
# Wait before the first grader retry; doubles before each further one.
GRADER_RETRY_BACKOFF_SECONDS = 5.0

# The grader is a plain function in the bench's own source. It is invoked with
# the transcript and workspace the agent actually produced, which is why it has
# to run in the container rather than on the host.
_RUNNER_TEMPLATE = """\
import json
import sys
from itertools import chain

def _load_jsonl_rows(lines):
    for line in lines:
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except (ValueError, RecursionError):
                continue

def _load_transcript(path):
    import pathlib
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        return []
    try:
        with file_path.open(encoding="utf-8", errors="ignore") as source:
            first = next((line for line in source if line.strip()), None)
            if first is None:
                return []
            try:
                parsed = json.loads(first.strip())
            except (ValueError, RecursionError):
                # A pretty-printed JSON document can span several lines.
                source.seek(0)
                try:
                    parsed = json.load(source)
                except (ValueError, RecursionError):
                    source.seek(0)
                    return list(_load_jsonl_rows(source))
            else:
                second = next((line for line in source if line.strip()), None)
                if second is not None:
                    # Native JSONL journals can embed large images in their rows.
                    return [parsed, *_load_jsonl_rows(chain((second,), source))]
    except OSError:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("transcript", "messages", "chat", "events"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
    return [parsed]


__ROLLER_GRADER_SOURCE__


def _main():
    kwargs = json.loads(__ROLLER_KWARGS__)
    if __ROLLER_NEEDS_TRANSCRIPT__:
        kwargs["transcript"] = _load_transcript(__ROLLER_TRANSCRIPT_PATH__)
    kwargs["workspace_path"] = __ROLLER_WORKSPACE__
    result = __ROLLER_ENTRYPOINT__(**kwargs)
    if not isinstance(result, dict):
        raise TypeError("grader must return a dict, got %r" % type(result).__name__)
    sys.stdout.write("__ROLLER_SCORE__" + json.dumps(result, ensure_ascii=False))


_main()
"""


@dataclass(frozen=True)
class GradeOutcome:
    scores: dict[str, Any]
    elapsed_seconds: float
    error: str | None = None
    # False when there was no container to run the grader in.
    retryable: bool = True
    attempts: int = 1

    @property
    def failure(self) -> str | None:
        """Why grading failed; ``None`` for a score, zero included."""

        return self.error or self.scores.get("error") or None


def _error_scores(message: str) -> dict[str, Any]:
    return {"overall_score": 0.0, "score_eligible": False, "error": message}


def build_runner_source(spec: GradeSpec, workspace_path: str) -> str:
    """Wrap the bench's grader source in a self-contained runner script."""

    return (
        _RUNNER_TEMPLATE.replace("__ROLLER_GRADER_SOURCE__", spec.code)
        .replace("__ROLLER_KWARGS__", repr(json.dumps(spec.kwargs)))
        .replace("__ROLLER_NEEDS_TRANSCRIPT__", repr(bool(spec.needs_transcript)))
        .replace("__ROLLER_TRANSCRIPT_PATH__", repr(TRANSCRIPT_CONTAINER_PATH))
        .replace("__ROLLER_WORKSPACE__", repr(workspace_path))
        .replace("__ROLLER_ENTRYPOINT__", spec.entrypoint)
    )


def parse_grader_stdout(stdout: str) -> dict[str, Any] | None:
    """Recover the score payload from grader stdout.

    Graders routinely print progress and judge traffic, so the payload is
    tagged rather than assumed to be the whole of stdout.
    """

    marker = "__ROLLER_SCORE__"
    index = stdout.rfind(marker)
    if index >= 0:
        candidate = stdout[index + len(marker) :].strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


class ContainerGrader:
    """Run a bench's grader inside a live sample container."""

    def __init__(self, spec: GradeSpec) -> None:
        self.spec = spec

    def _env_args(self) -> list[str]:
        """Pass the grader the variables that are set.

        A missing required one stops the run before any container starts, see
        ``roller.benches.missing_grader_env``.
        """

        args: list[str] = []
        for name in (*self.spec.env_names, *self.spec.optional_env_names):
            if not ENV_NAME_RE.fullmatch(name):
                raise ValueError(f"invalid grading environment variable name: {name!r}")
            if not os.environ.get(name):
                continue
            # Docker/Podman copies NAME from this process's environment. Do not
            # place secret values in argv, where process listings can expose them.
            args.extend(["-e", name])
        # HTTPX does not interpret NO_PROXY CIDRs. Expand configured ranges for
        # the grader's actual endpoints without bypassing other proxy traffic.
        bypass = os.environ.get("NO_PROXY") or os.environ.get("no_proxy", "")
        entries = [value.strip() for value in bypass.split(",") if value.strip()]
        networks = []
        for value in entries:
            if "/" in value:
                try:
                    networks.append(ipaddress.ip_network(value, strict=False))
                except ValueError:
                    pass
        hosts = []
        for name in (*self.spec.env_names, *self.spec.optional_env_names):
            if name.endswith("BASE_URL"):
                host = urlsplit(os.environ.get(name, "")).hostname
                try:
                    address = ipaddress.ip_address(host or "")
                except ValueError:
                    continue
                if (
                    any(address in network for network in networks)
                    and host not in entries
                ):
                    hosts.append(host)
        if hosts:
            expanded = ",".join(dict.fromkeys([*entries, *hosts]))
            args.extend(["-e", "NO_PROXY=" + expanded, "-e", "no_proxy=" + expanded])
        return args

    def grade(
        self,
        runtime: Any,
        container: str,
        *,
        task_id: str,
        workspace_path: str,
        transcript_container_path: str | None,
        sandbox: Sandbox | None = None,
    ) -> GradeOutcome:
        started = time.monotonic()
        workspace = self.spec.workspace_path or workspace_path
        execution = workspace_backend(runtime, container, sandbox)
        if self.spec.collector_files:
            return self._grade_in_collector(execution, workspace, started)
        prepare = execution.run(["mkdir", "-p", GRADER_CONTAINER_DIR], timeout=30)
        if prepare.returncode != 0:
            detail = (prepare.stderr or prepare.stdout or "").strip()[-500:]
            return GradeOutcome(
                _error_scores(f"grader staging failed: {detail}"),
                time.monotonic() - started,
                error=detail or "grader staging failed",
            )

        if self.spec.needs_transcript:
            self._stage_transcript(execution, transcript_container_path, task_id)

        staging_error = self._stage_files(execution, workspace, task_id)
        if staging_error is not None:
            # Without its answer key the grader would score a correct run zero,
            # which reads as a model failure. Surface it as a grading error.
            return GradeOutcome(
                _error_scores(f"grader staging failed: {staging_error}"),
                time.monotonic() - started,
                error=staging_error,
            )

        runner_source = build_runner_source(self.spec, workspace)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(runner_source)
            host_runner = handle.name
        try:
            copied = execution.upload(
                Path(host_runner), RUNNER_CONTAINER_PATH, timeout=60
            )
            if copied.returncode != 0:
                detail = (copied.stderr or copied.stdout or "").strip()[-500:]
                return GradeOutcome(
                    _error_scores(f"grader copy failed: {detail}"),
                    time.monotonic() - started,
                    error=detail or "grader copy failed",
                )
            try:
                result = execution.run(
                    ["python3", RUNNER_CONTAINER_PATH],
                    cwd=workspace,
                    options=self._env_args(),
                    timeout=self.spec.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                message = f"grading exceeded {self.spec.timeout_seconds}s"
                return GradeOutcome(
                    _error_scores(message), time.monotonic() - started, error=message
                )
        finally:
            Path(host_runner).unlink(missing_ok=True)

        return self._outcome(result, time.monotonic() - started)

    def _grade_in_collector(
        self, execution: WorkspaceBackend, workspace: str, started: float
    ) -> GradeOutcome:
        """Copy ``collector_files`` out, then run the grader here on the copy."""

        with tempfile.TemporaryDirectory(prefix="roller-grade-") as temporary:
            root = Path(temporary)
            for relative in self.spec.collector_files:
                path = shlex.quote(f"{workspace}/{relative}")
                read = execution.run(
                    [
                        "/bin/sh",
                        "-c",
                        f"if [ -f {path} ]; then base64 {path}; else exit 3; fi",
                    ],
                    timeout=60,
                )
                if read.returncode == 3:
                    continue  # the agent wrote no such file
                if read.returncode != 0:
                    detail = (read.stderr or read.stdout or "").strip()[-500:]
                    message = f"could not read {relative}: {detail}"
                    return GradeOutcome(
                        _error_scores(message),
                        time.monotonic() - started,
                        error=message,
                    )
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(read.stdout))
            for source, target in self.spec.stage:
                destination = root / target
                destination.parent.mkdir(parents=True, exist_ok=True)
                if Path(source).is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copyfile(source, destination)
            runner = root / "_grade_runner.py"
            runner.write_text(
                build_runner_source(self.spec, str(root)), encoding="utf-8"
            )
            try:
                result = subprocess.run(
                    [sys.executable, str(runner)],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.spec.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                message = f"grading exceeded {self.spec.timeout_seconds}s"
                return GradeOutcome(
                    _error_scores(message), time.monotonic() - started, error=message
                )
        return self._outcome(result, time.monotonic() - started)

    @staticmethod
    def _outcome(result: subprocess.CompletedProcess, elapsed: float) -> GradeOutcome:
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()[-2000:]
            return GradeOutcome(
                _error_scores(f"grader failed: {detail}"), elapsed, error=detail
            )
        scores = parse_grader_stdout(result.stdout or "")
        if scores is None:
            detail = (result.stdout or "").strip()[-500:]
            message = f"grader produced no JSON payload: {detail}"
            return GradeOutcome(_error_scores(message), elapsed, error=message)
        return GradeOutcome(scores, elapsed)

    def _stage_files(
        self, execution: WorkspaceBackend, workspace: str, task_id: str
    ) -> str | None:
        """Copy the spec's ``stage`` sources into the container's workspace.

        This runs after the agent's last turn, so files placed here — answer
        keys, reference outputs — were never visible to the model. Returns a
        failure description, or ``None`` when everything landed.
        """

        for source, target in self.spec.stage:
            host = Path(source)
            if not host.exists():
                return f"staged source is missing on the host: {source}"
            destination = f"{workspace}/{target}"
            parent = str(PurePosixPath(destination).parent)
            # ``docker cp`` needs the destination's parent to exist, and for a
            # directory source the destination itself.
            container_dir = destination if host.is_dir() else parent
            made = execution.run(["mkdir", "-p", container_dir], timeout=30)
            if made.returncode != 0:
                detail = (made.stderr or made.stdout or "").strip()[-300:]
                return f"could not create {container_dir}: {detail}"
            copied = execution.upload(
                host, destination, contents=host.is_dir(), timeout=120
            )
            if copied.returncode != 0:
                detail = (copied.stderr or copied.stdout or "").strip()[-300:]
                return f"could not copy {source} to {destination}: {detail}"
            logger.debug(
                "[%s] staged %s -> %s for grading", task_id, source, destination
            )
        return None

    @staticmethod
    def _stage_transcript(
        execution: WorkspaceBackend,
        transcript_container_path: str | None,
        task_id: str,
    ) -> None:
        """Copy the live session transcript to a fixed path for the grader.

        Each harness stores its transcript somewhere different; normalizing the
        location here keeps bench graders harness-agnostic.
        """

        if not transcript_container_path:
            return
        script = (
            f"if [ -f {shlex.quote(transcript_container_path)} ]; then "
            f"cp {shlex.quote(transcript_container_path)} "
            f"{shlex.quote(TRANSCRIPT_CONTAINER_PATH)}; fi"
        )
        result = execution.run(["/bin/sh", "-c", script], timeout=60)
        if result.returncode != 0:
            logger.warning(
                "[%s] transcript staging failed: %s",
                task_id,
                (result.stderr or result.stdout or "").strip()[-300:],
            )


def transcript_container_path(
    session: InteractiveSession, spec: GradeSpec | None = None
) -> str | None:
    """Best-effort location of a session's transcript inside its container.

    Only OpenClaw exposes a derivable session file; for other harnesses a bench
    can pin the path via ``GradeSpec.transcript_path``.
    """

    if spec is not None and spec.transcript_path:
        return spec.transcript_path
    prepare = getattr(session, "prepare_transcript_for_grading", None)
    if callable(prepare):
        prepared = prepare()
        if isinstance(prepared, str) and prepared:
            return prepared
    explicit = getattr(session, "transcript_container_path", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    harness = getattr(session, "harness", None)
    config = getattr(harness, "config", None)
    state_dir = getattr(session, "state_dir", None)
    session_id = getattr(config, "session_id", None)
    if isinstance(state_dir, str) and isinstance(session_id, str):
        return f"{state_dir}/agents/main/sessions/{session_id}.jsonl"
    return None


def workspace_container_path(session: InteractiveSession) -> str:
    harness = getattr(session, "harness", None)
    config = getattr(harness, "config", None)
    value = getattr(config, "workspace_container_path", None)
    return value if isinstance(value, str) and value else "/tmp_workspace"


def prepare_benchmark_workspace(
    session: InteractiveSession, spec: GradeSpec | None
) -> None:
    """Expose a bench's canonical workspace path before the agent runs."""

    if spec is None or spec.workspace_path is None:
        return
    actual = workspace_container_path(session)
    expected = spec.workspace_path
    # Some benchmark graders intentionally run from the container root so
    # staged files keep upstream absolute paths such as /verifier. The root
    # already exists and cannot be replaced with a workspace symlink.
    if actual == expected or expected == "/":
        return
    container = getattr(session, "container", None)
    runtime = getattr(getattr(session, "harness", None), "runtime", None)
    if (
        not getattr(session, "container_started", False)
        or not container
        or runtime is None
    ):
        raise RuntimeError(
            "cannot prepare benchmark workspace alias without a live container"
        )
    expected_path = PurePosixPath(expected)
    script = (
        f"mkdir -p {shlex.quote(str(expected_path.parent))}; "
        f"if [ -L {shlex.quote(expected)} ]; then "
        f'test "$(readlink -f {shlex.quote(expected)})" = '
        f'"$(readlink -f {shlex.quote(actual)})"; '
        f"elif [ -e {shlex.quote(expected)} ]; then "
        f"echo 'benchmark workspace alias target already exists' >&2; exit 1; "
        f"else ln -s {shlex.quote(actual)} {shlex.quote(expected)}; fi"
    )
    execution = workspace_backend(runtime, container, getattr(session, "sandbox", None))
    result = execution.run(["/bin/sh", "-c", script], timeout=30)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-500:]
        raise RuntimeError(
            f"benchmark workspace alias {expected} -> {actual} failed"
            + (f": {detail}" if detail else "")
        )


class GradedHarness(Harness):
    """Drive an interactive harness, then grade before the container dies.

    Grading has to observe the container's filesystem, so it cannot run after
    ``run()`` returns — by then the harness has already torn the container
    down. This wrapper reproduces the plain run loop and slips the grader in
    between the agent's last turn and teardown, leaving the wrapped harness
    untouched.
    """

    def __init__(
        self,
        inner: InteractiveHarness,
        binding: BenchBinding,
        *,
        write_score_file: bool = True,
        grading_semaphore: threading.Semaphore | None = None,
        grader_retries: int = 0,
    ) -> None:
        if not isinstance(inner, InteractiveHarness):
            raise TypeError(
                f"bench grading requires an interactive harness; {type(inner).__name__} "
                "cannot expose a live container"
            )
        self.inner = inner
        self.binding = binding
        self.write_score_file = write_score_file
        self.grading_semaphore = grading_semaphore
        # How often a failed grader is run again while the container is up.
        self.grader_retries = grader_retries
        self.name = inner.name

    def run(self, request: RunRequest, output_dir: Path, attempt: int) -> HarnessResult:
        session = self.inner.create_session(request, output_dir, attempt)
        status = "error"
        error: str | None = None
        started = time.monotonic()
        try:
            session.start()
            self._prepare_session(session)
            for index, prompt in enumerate(self._prompts(request)):
                remaining = request.timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"task exceeded {request.timeout_seconds}s")
                if (
                    index < len(request.task.new_session)
                    and request.task.new_session[index]
                ):
                    session.reset()
                session.send(prompt, remaining)
            status = "ok"
        except (TimeoutError, subprocess.TimeoutExpired):
            status = "timeout"
            error = f"task exceeded {request.timeout_seconds}s"
        except Exception as exc:
            status = "error"
            error = str(exc)

        try:
            outcome = self._grade_with_retries(session, request, status)
        finally:
            # Grader failures must not skip saving the agent's work and teardown.
            result = session.finish(status, error)
        if outcome is not None:
            self._attach(result, request, output_dir, outcome)
        return result

    def _grade_with_retries(
        self, session: InteractiveSession, request: RunRequest, status: str
    ) -> GradeOutcome | None:
        """Grade, running a failed grader again while the container is still up.

        A judge request error or a grader crash leaves the agent's work intact
        in the container, so grading again costs far less than a new rollout.
        A score is never retried, a zero included.
        """

        spec = self.binding.grade
        # A failed agent is graded only where the bench asks for it.
        graded = status == "ok" or (spec is not None and spec.grade_on_agent_failure)
        attempt = 1
        while True:
            started = time.monotonic()
            try:
                outcome = self._grade(session, request, status)
            except Exception as exc:
                logger.exception("[%s] grading raised", request.task.id)
                outcome = GradeOutcome(
                    _error_scores(str(exc)), time.monotonic() - started, error=str(exc)
                )
            if (
                outcome is None
                or outcome.failure is None
                or not (graded and outcome.retryable)
                or outcome.elapsed_seconds > GRADER_RETRY_MAX_SECONDS
                or attempt > self.grader_retries
            ):
                break
            delay = GRADER_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
            logger.warning(
                "[%s] grading failed (%s); retry %d of %d in %gs",
                request.task.id,
                brief_reason(outcome.failure),
                attempt,
                self.grader_retries,
                delay,
            )
            time.sleep(delay)
            attempt += 1
        if outcome is not None and attempt > 1:
            outcome = replace(outcome, attempts=attempt)
        return outcome

    def _prepare_session(self, session: InteractiveSession) -> None:
        prepare_benchmark_workspace(session, self.binding.grade)

    def _prompts(self, request: RunRequest) -> tuple[str, ...]:
        """Defer to the wrapped harness's own prompt expansion when it has one."""

        getter = getattr(self.inner, "_prompts", None)
        if callable(getter):
            return tuple(getter(request))
        return request.task.prompts

    def _grade(
        self, session: InteractiveSession, request: RunRequest, status: str
    ) -> GradeOutcome | None:
        spec = self.binding.grade
        if spec is None:
            return None
        if status != "ok" and not spec.grade_on_agent_failure:
            # Agent timeouts/errors are excluded by benchmark protocol. A timed
            # out harness may already have stopped its container, so attempting
            # an exec here only obscures the real failure with a runtime error.
            message = f"agent {status}; benchmark grading skipped"
            return GradeOutcome(_error_scores(message), 0.0, error=message)
        if status != "ok":
            logger.info(
                "[%s] grading workspace after agent %s per benchmark protocol",
                request.task.id,
                status,
            )
        container = getattr(session, "container", None)
        if not getattr(session, "container_started", False) or not container:
            # The container never came up, so there is nothing to inspect. A
            # zero here is the honest score: the agent produced no work.
            return GradeOutcome(
                _error_scores(f"container unavailable for grading (status={status})"),
                0.0,
                error="container unavailable",
                retryable=False,
            )
        runtime = getattr(getattr(session, "harness", None), "runtime", None)
        if runtime is None:
            return GradeOutcome(
                _error_scores("harness exposes no docker runtime"),
                0.0,
                error="no docker runtime",
                retryable=False,
            )
        grader = ContainerGrader(spec)
        grade_started = time.monotonic()
        try:
            if self.grading_semaphore is None:
                logger.info("[%s] benchmark grading started", request.task.id)
                outcome = grader.grade(
                    runtime,
                    container,
                    task_id=request.task.id,
                    workspace_path=workspace_container_path(session),
                    transcript_container_path=transcript_container_path(session, spec),
                    sandbox=getattr(session, "sandbox", None),
                )
            else:
                logger.info("[%s] waiting for benchmark grader slot", request.task.id)
                with self.grading_semaphore:
                    logger.info("[%s] benchmark grading started", request.task.id)
                    outcome = grader.grade(
                        runtime,
                        container,
                        task_id=request.task.id,
                        workspace_path=workspace_container_path(session),
                        transcript_container_path=transcript_container_path(
                            session, spec
                        ),
                        sandbox=getattr(session, "sandbox", None),
                    )
            logger.info(
                "[%s] benchmark grading finished in %.1fs%s",
                request.task.id,
                time.monotonic() - grade_started,
                f": {outcome.error}" if outcome.error else "",
            )
            return outcome
        except Exception as exc:
            logger.exception("[%s] grading raised", request.task.id)
            return GradeOutcome(_error_scores(str(exc)), 0.0, error=str(exc))

    def _attach(
        self,
        result: HarnessResult,
        request: RunRequest,
        output_dir: Path,
        outcome: GradeOutcome,
    ) -> None:
        spec = self.binding.grade
        payload = {
            "bench": self.binding.name,
            "category": self.binding.category,
            "metadata": dict(self.binding.metadata),
            "task_id": request.task.id,
            "model": request.model,
            "score_key": spec.score_key if spec is not None else "overall_score",
            "elapsed_seconds": round(outcome.elapsed_seconds, 3),
            "error": outcome.error,
            "scores": outcome.scores,
        }
        if outcome.attempts > 1:
            payload["grade_attempts"] = outcome.attempts
        if isinstance(result.metadata, dict):
            result.metadata["bench"] = payload
        if self.write_score_file:
            try:
                atomic_write_json(output_dir / "score.json", payload)
            except OSError as exc:
                logger.warning(
                    "[%s] could not write score.json: %s", request.task.id, exc
                )
