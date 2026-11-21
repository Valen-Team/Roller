"""Stirrup's agent loop in the collecting process, with the task container as its code backend.

The agent, its model client and its journal run here; the container or remote
sandbox only executes commands and holds files. Needs the ``stirrup`` extra
(Python 3.12 or later).
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import json
import posixpath
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError
from stirrup import Agent
from stirrup.clients.utils import content_to_openai, to_openai_messages, to_openai_tools
from stirrup.core.exceptions import (
    CompactionRequiredError,
    ContextOverflowError,
    OutputTokenLimitError,
)
from stirrup.core.models import (
    AssistantBlock,
    AssistantMessage,
    ChatMessage,
    ImageContentBlock,
    ReasoningBlock,
    TextBlock,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolUseCountMetadata,
    UserMessage,
)
from stirrup.prompts import MESSAGE_SUMMARIZER
from stirrup.tools import ViewImageToolProvider
from stirrup.tools.code_backends.base import (
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
)
from stirrup.utils.logging import AgentLoggerBase

STIRRUP_VERSION = version("stirrup")
RETRYABLE_HTTP = (408, 429, 500, 502, 503, 504)
# A truncated turn whose prompt and output come this close to the context
# length ran out of context rather than out of output tokens.
CONTEXT_SLACK = 64
# Raw bytes kept per output stream; Stirrup then shortens what the model sees.
OUTPUT_CAP = 1 << 20
_ENVELOPE = (
    'd=/tmp/.roller-exec; mkdir -p "$d"; cd {workdir} || exit 97; '
    'timeout -k 5 {timeout}s bash -c {command} >"$d/out" 2>"$d/err" </dev/null; rc=$?; '
    'emit() {{ if [ "$(stat -c %s "$1")" -gt {cap} ]; then head -c {half} "$1"; tail -c {half} "$1"; '
    'else cat "$1"; fi | base64; }}; '
    'printf "%s:" "$rc"; emit "$d/out"; printf ":"; emit "$d/err"'
)

Run = Callable[[list[str], float], subprocess.CompletedProcess[str]]
Upload = Callable[[Path, str], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Journal:
    """The native trace: what each model call added, its outcome, and every tool result.

    A request records only the messages after the prefix it shares with the
    previous request and its reply, so history and images are written once.
    Tool results are written when the agent next acts, after Stirrup has moved
    their images into user messages.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.pending: list[ToolMessage] = []
        self.calls = 0
        self.summaries = 0

    def _append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**row, "timestamp": _now()}, ensure_ascii=False))
            stream.write("\n")

    def flush(self) -> None:
        for message in self.pending:
            self._append(
                {
                    "type": "tool",
                    "tool_call_id": message.tool_call_id,
                    "name": message.name,
                    "success": message.success,
                    "content": content_to_openai(message.content),
                }
            )
        self.pending.clear()

    def write(self, kind: str, **fields: Any) -> None:
        self.flush()
        self._append({"type": kind, **fields})


class JournalLogger(AgentLoggerBase):
    """Stirrup's logger interface, keeping only what the journal needs."""

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.name, self.model, self.max_turns, self.depth = "", None, None, 0
        self.finish_params, self.run_metadata, self.output_dir = None, None, None

    def __enter__(self) -> JournalLogger:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def on_step(self, *_: object, **__: object) -> None:
        return None

    def assistant_message(self, *_: object) -> None:
        return None  # the model client records each reply

    def user_message(self, *_: object) -> None:
        return None  # each request records the messages it adds

    def task_message(self, *_: object) -> None:
        return None

    def tool_result(self, tool_message: ToolMessage) -> None:
        self.journal.pending.append(tool_message)

    def context_summarization_start(self, *_: object) -> None:
        return None

    def context_summarization_complete(self, summary: str, bridge: str) -> None:
        self.journal.summaries += 1
        self.journal.write(
            "summary", id=f"summary-{self.journal.summaries:04d}", summary=summary
        )

    def debug(self, *_: object) -> None:
        return None

    info = warning = error = debug


def _summary_request(messages: list[ChatMessage]) -> bool:
    last = messages[-1] if messages else None
    return (
        isinstance(last, UserMessage)
        and isinstance(last.content, str)
        and last.content.startswith(MESSAGE_SUMMARIZER)
    )


def _common_prefix(left: list[Any], right: list[Any]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


class ChatClient:
    """Stirrup's LLM client: one Chat Completions request per turn, journaled."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        body: dict[str, Any],
        context_window: int,
        timeout: float,
        attempts: int,
        journal: Journal,
        session_id: str | None,
        remaining: Callable[[], float],
    ) -> None:
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.api_key, self.body = api_key, body
        self._context_window = context_window
        self.timeout, self.attempts = timeout, attempts
        self.journal, self.session_id, self.remaining = journal, session_id, remaining
        # A local recording proxy is reached directly, never through the host's proxy.
        local = urlsplit(self.url).hostname in {"127.0.0.1", "localhost"}
        self.opener = urllib.request.build_opener(
            *([urllib.request.ProxyHandler({})] if local else [])
        )
        self.previous: list[dict[str, Any]] = []
        self.previous_tools: list[dict[str, Any]] | None = None

    @property
    def model_slug(self) -> str:
        return str(self.body["model"])

    @property
    def context_window_tokens(self) -> int:
        return self._context_window

    def _post(self, payload: dict[str, Any], call_id: str) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        if self.session_id:
            headers["x-roller-native-session-id"] = self.session_id
            headers["x-roller-native-run-id"] = call_id
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers
        )
        with self.opener.open(
            request, timeout=min(self.timeout, self.remaining())
        ) as response:
            return json.load(response)

    def _record_request(
        self, call_id: str, purpose: str, payload: dict[str, Any]
    ) -> None:
        messages, tools = payload["messages"], payload.get("tools", [])
        base = _common_prefix(self.previous, messages)
        self.journal.write(
            "request",
            id=call_id,
            purpose=purpose,
            base=base,
            messages=messages[base:],
            **({"tools": tools} if tools != self.previous_tools else {}),
        )
        self.previous, self.previous_tools = messages, tools

    def _complete(self, payload: dict[str, Any], purpose: str) -> tuple[str, dict]:
        """Send one turn, retrying transport faults; returns the call ID and response."""
        for attempt in range(self.attempts):
            self.journal.calls += 1
            call_id = f"call-{self.journal.calls:04d}"
            self._record_request(call_id, purpose, payload)
            fields: dict[str, Any]
            try:
                raw = self._post(payload, call_id)
                choice = raw["choices"][0]
                message = choice["message"]
                if not isinstance(message.get("content") or "", str):
                    raise ValueError("Expected text in message.content")
                if choice["finish_reason"] not in ("stop", "length", "tool_calls"):
                    raise ValueError(
                        f"Unexpected finish_reason: {choice['finish_reason']}"
                    )
                return call_id, raw
            except urllib.error.HTTPError as exc:
                body = exc.read(2000).decode("utf-8", "replace")
                fields = {"http_status": exc.code, "error": body}
                if exc.code == 400 and "maximum context length" in body:
                    self.journal.write(
                        "response", id=call_id, outcome="context_overflow", **fields
                    )
                    raise ContextOverflowError(
                        "the prompt exceeds the served context"
                    ) from None
                retry = exc.code in RETRYABLE_HTTP
            except (
                OSError,
                http.client.HTTPException,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                if isinstance(exc, TimeoutError) and self.remaining() <= 1:
                    raise
                fields, retry = {"error": f"{type(exc).__name__}: {exc}"[:500]}, True
            last = not retry or attempt + 1 == self.attempts
            self.journal.write(
                "response", id=call_id, outcome="error" if last else "retry", **fields
            )
            if last:
                raise RuntimeError(f"model request failed: {fields}"[:1000])
            time.sleep(min(2**attempt, self.remaining()))
        raise AssertionError("unreachable")

    async def generate(
        self, messages: list[ChatMessage], tools: dict[str, Tool]
    ) -> AssistantMessage:
        payload = dict(self.body, messages=to_openai_messages(messages))
        if tools:
            payload.update(tools=to_openai_tools(tools), tool_choice="auto")
        purpose = "summary" if _summary_request(messages) else "main"
        started = time.perf_counter()
        call_id, raw = self._complete(payload, purpose)
        choice = raw["choices"][0]
        message, finish = choice["message"], choice["finish_reason"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        calls = message.get("tool_calls") or []
        usage = raw.get("usage") or {}
        record = {"id": call_id, "finish_reason": finish, "usage": usage}
        if finish == "length" and not calls:
            # The server caps a turn's output at the context left, so a turn
            # cut there overflowed it.
            used = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            if used >= self._context_window - CONTEXT_SLACK:
                self.journal.write("response", outcome="context_overflow", **record)
                raise ContextOverflowError("the response filled the served context")
            self.journal.write("response", outcome="output_limit", **record)
            raise OutputTokenLimitError(
                model_slug=self.model_slug,
                max_tokens=payload.get("max_tokens", 0),
                provider_reason="length",
            )
        blocks: list[AssistantBlock] = []
        if reasoning:
            blocks.append(ReasoningBlock(content=reasoning))
        if content:
            blocks.append(TextBlock(text=content))
        for call in calls:
            blocks.append(
                ToolCall.from_provider(
                    provider_id=call.get("id"),
                    name=call["function"]["name"],
                    arguments=call["function"].get("arguments") or "",
                )
            )
        thinking = (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ) or 0
        reply = AssistantMessage(
            blocks=blocks,
            request_start_time=started,
            request_end_time=time.perf_counter(),
            token_usage=TokenUsage(
                input=usage.get("prompt_tokens", 0),
                reasoning=thinking,
                answer=usage.get("completion_tokens", 0) - thinking,
            ),
        )
        replayed = to_openai_messages([reply])[0]
        outcome = "step" if purpose == "main" else "summary" if content else "discarded"
        self.journal.write("response", outcome=outcome, message=replayed, **record)
        self.previous = [*self.previous, replayed]
        return reply


class ContainerCodeExec(CodeExecToolProvider):
    """Stirrup's code execution backend on the sample's container or sandbox.

    ``run`` executes an argv there and ``upload`` copies a host file in; the
    session owns the container's lifecycle.
    """

    def __init__(
        self, run: Run, upload: Upload, *, workdir: str, shell_timeout: int
    ) -> None:
        super().__init__(shell_timeout=shell_timeout)
        self.run, self.upload, self.workdir = run, upload, workdir

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        return self.get_code_exec_tool()

    async def __aexit__(self, *_: object) -> None:
        return None

    def _resolve(self, path: str) -> str:
        return posixpath.normpath(posixpath.join(self.workdir, path))

    def shell(self, command: str, timeout: float) -> tuple[int, str, str]:
        """Run in the working directory with the exact exit code and output."""
        script = _ENVELOPE.format(
            workdir=shlex.quote(self.workdir),
            timeout=timeout,
            command=shlex.quote(command),
            cap=OUTPUT_CAP,
            half=OUTPUT_CAP // 2,
        )
        result = self.run(["/bin/bash", "-c", script], timeout + 60)
        code, out, err = result.stdout.split(":")
        return (
            int(code),
            base64.b64decode(out).decode("utf-8", "replace"),
            base64.b64decode(err).decode("utf-8", "replace"),
        )

    async def run_command(
        self, cmd: str, *, timeout: int | None = None
    ) -> CommandResult:
        timeout = self._shell_timeout if timeout is None else timeout
        code, out, err = self.shell(cmd, timeout)
        if code == 124:
            return CommandResult(
                exit_code=1,
                stdout=out,
                stderr=f"Command timed out after {timeout} seconds",
                error_kind="timeout",
            )
        return CommandResult(exit_code=code, stdout=out, stderr=err)

    def exists(self, path: str) -> bool:
        return self.shell("test -f " + shlex.quote(self._resolve(path)), 30)[0] == 0

    def read(self, path: str) -> bytes:
        if not self.exists(path):
            raise FileNotFoundError(path)
        result = self.run(["base64", self._resolve(path)], 120)
        if result.returncode:
            raise RuntimeError(f"could not read {path}: {result.stderr[-500:]}")
        return base64.b64decode(result.stdout)

    async def file_exists(self, path: str) -> bool:
        return self.exists(path)

    async def is_directory(self, path: str) -> bool:
        return self.shell("test -d " + shlex.quote(self._resolve(path)), 30)[0] == 0

    async def list_files(self, path: str) -> list[str]:
        code, out, _ = self.shell(
            f"cd {shlex.quote(self._resolve(path))} && find . -type f -printf '%P\\n'",
            60,
        )
        return out.splitlines() if code == 0 else []

    async def read_file_bytes(self, path: str) -> bytes:
        return self.read(path)

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        target = self._resolve(path)
        self.run(["mkdir", "-p", posixpath.dirname(target)], 30).check_returncode()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "content"
            source.write_bytes(content)
            self.upload(source, target)

    async def view_image(self, path: str) -> ImageContentBlock:
        if await self.is_directory(path):
            raise ValueError(f"Path `{path}` is a directory, not an image.")
        try:
            return ImageContentBlock(data=self.read(path))
        except ValidationError as exc:
            raise ValueError("You submitted a corrupt/unsupported image file.") from exc


async def _agent_run(
    client: ChatClient,
    backend: ContainerCodeExec,
    prompt: str,
    max_turns: int,
    logger: JournalLogger,
) -> dict[str, Any]:
    """Run the loop; the stop reason says how it ended."""
    agent = Agent(
        client=client,
        name="roller",
        max_turns=max_turns,
        logger=logger,
        tools=[backend, ViewImageToolProvider(backend)],
    )
    outcome: dict[str, Any] = {"stop": "max_turns"}
    # State caching writes to the home directory and installs a signal
    # handler; neither fits a collector worker thread.
    async with agent.session(cache_on_interrupt=False) as session:
        try:
            finish, _, _ = await session.run(prompt)
            if finish is not None:
                outcome.update(stop="finish", finish=finish.model_dump())
        except OutputTokenLimitError:
            outcome["stop"] = "output_limit"
        except (ContextOverflowError, CompactionRequiredError) as exc:
            outcome.update(stop="context_overflow", detail=str(exc)[:300])
    return outcome


def run_episode(
    *,
    journal_path: Path,
    session: dict[str, Any],
    prompt: str,
    run: Run,
    upload: Upload,
    workdir: str,
    command_timeout: int,
    max_turns: int,
    client_options: dict[str, Any],
) -> dict[str, Any]:
    """One Stirrup episode; the journal keeps the turns completed until it ended."""
    journal = Journal(journal_path)
    journal.write("session", stirrup_version=STIRRUP_VERSION, **session)
    client = ChatClient(journal=journal, **client_options)
    backend = ContainerCodeExec(
        run, upload, workdir=workdir, shell_timeout=command_timeout
    )
    outcome: dict[str, Any] = {"stop": "error"}
    try:
        outcome = asyncio.run(
            _agent_run(client, backend, prompt, max_turns, JournalLogger(journal))
        )
        return outcome
    finally:
        journal.write("end", **outcome)
