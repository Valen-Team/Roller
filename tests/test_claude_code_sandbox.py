from __future__ import annotations

import dataclasses
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from roller.harnesses.claude_code import ClaudeCodeConfig, ClaudeCodeHarness
from roller.harnesses.claude_code_sandbox import (
    CLAUDE_HOME,
    DEFAULT_BINARY,
    TASK_DIR,
    SandboxAgent,
)
from roller.models import RunRequest, Task
from roller.sandbox import SandboxTarget

RELAY_URL = "http://127.0.0.1:8766"


class FakeProcess:
    def __init__(self, code: int | None, log_path: Path, output: str) -> None:
        self.code = code
        self.log_path = log_path
        self.output = output
        self.killed = 0

    def wait(self, timeout: float | None = None) -> int:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(self.output, encoding="utf-8")
        if self.code is None:
            raise subprocess.TimeoutExpired("claude", timeout or 0)
        return self.code

    def kill(self) -> None:
        self.killed += 1


class FakeBox:
    """Enough of Roller's ``Sandbox`` protocol to drive the backend."""

    def __init__(self, version: str = "sha256:spec", exit_code: int | None = 0) -> None:
        self.version = version
        self.exit_code = exit_code
        self.stream = '{"type":"result"}\n'
        self.commands: list[tuple[list[str], str | None]] = []
        self.uploads: list[tuple[str, str, bool]] = []
        self.downloads: list[tuple[str, bool]] = []
        self.spawned: list[dict[str, Any]] = []
        self.routed: list[tuple[str, str]] = []
        self.closed = 0
        self.version_output = "2.1.311 (Claude Code)"

    def run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: Any = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append((list(argv), cwd))
        stdout = self.version_output if argv[-1] == "--version" else ""
        return subprocess.CompletedProcess(list(argv), 0, stdout, "")

    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: Any = None,
        log_path: Path,
    ) -> FakeProcess:
        self.spawned.append(
            {"argv": list(argv), "cwd": cwd, "env": dict(env or {}), "log": log_path}
        )
        return FakeProcess(self.exit_code, log_path, self.stream)

    def upload(
        self, source: Path, target: str, *, contents: bool = False, timeout: float
    ) -> None:
        self.uploads.append((str(source), target, contents))

    def download(
        self, source: str, target: Path, *, contents: bool = False, timeout: float
    ) -> None:
        self.downloads.append((source, contents))
        target.mkdir(parents=True, exist_ok=True)
        if source.endswith("/projects"):
            session = target / "project" / "session.jsonl"
            session.parent.mkdir(parents=True, exist_ok=True)
            session.write_text(
                '{"uuid":"a","timestamp":"1","type":"assistant"}\n', encoding="utf-8"
            )

    def route_http(self, base: str, key: str, timeout: float) -> str:
        self.routed.append((base, key))
        return RELAY_URL

    def close(self) -> None:
        self.closed += 1


class FakeRun:
    def __init__(self, box: FakeBox) -> None:
        self.box = box
        self.target = SandboxTarget("h-cluster", "zerobench-claude-code", "sha256:spec")
        self.created: list[tuple[str, float]] = []
        self.released: list[str] = []

    def create(self, name: str, lifetime: float, timeout: float) -> FakeBox:
        self.created.append((name, lifetime))
        return self.box

    def close(self, name: str, box: FakeBox) -> None:
        self.released.append(name)
        box.close()


def _request(workspace: Path | None = None) -> RunRequest:
    return RunRequest(
        index=0,
        task=Task(
            id="task",
            prompts=("solve it",),
            source_index=0,
            workspace=workspace,
            input_mode="copy",
            output_dir="output",
        ),
        model="glm-5-3-flash-bf16",
        timeout_seconds=120,
    )


def _agent(
    output_dir: Path,
    box: FakeBox,
    *,
    config: ClaudeCodeConfig | None = None,
    request: RunRequest | None = None,
) -> tuple[SandboxAgent, FakeRun, ClaudeCodeHarness]:
    run = FakeRun(box)
    settings = config or ClaudeCodeConfig(
        api_base="http://127.0.0.1:1",
        wire_trace_mode="full",
        chat_completions=True,
        tool_images_inline=True,
    )
    settings = dataclasses.replace(settings, sandbox_run=run)
    harness = ClaudeCodeHarness(settings, 0)
    task = request or _request()
    return SandboxAgent(harness, task, output_dir, "sample", run), run, harness


class SandboxAgentTests(unittest.TestCase):
    def test_start_routes_the_proxy_and_hides_the_real_key(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            box = FakeBox()
            agent, run, _ = _agent(Path(folder), box)

            environment = agent.start(
                {"ANTHROPIC_API_KEY": "real-secret"}, "http://127.0.0.1:9999"
            )

            self.assertEqual(box.routed, [("http://127.0.0.1:9999", "real-secret")])
            self.assertEqual(environment["ANTHROPIC_BASE_URL"], RELAY_URL)
            # The provider's relay token is what the CLI gets; the real key is
            # the forwarder's on the collecting host.
            self.assertNotIn("real-secret", "".join(environment.values()))
            self.assertEqual(
                environment["ANTHROPIC_API_KEY"], environment["ANTHROPIC_AUTH_TOKEN"]
            )
            self.assertEqual(environment["CLAUDE_CONFIG_DIR"], CLAUDE_HOME)
            self.assertEqual(environment["NO_PROXY"], "127.0.0.1,localhost")
            self.assertEqual(run.created, [("sample", 120 + 900)])
            directories = next(
                argv for argv, _ in box.commands if argv[:2] == ["mkdir", "-p"]
            )
            # The output directory is collected before teardown, so it has to
            # exist even for a protocol that writes nothing into it.
            self.assertEqual(
                directories[2:],
                [
                    "/roller/workspace",
                    CLAUDE_HOME,
                    TASK_DIR,
                    "/roller/workspace/output",
                ],
            )

    def test_start_fixes_the_protocol_settings_of_the_cli(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent, _, _ = _agent(
                Path(folder),
                FakeBox(),
                config=ClaudeCodeConfig(
                    api_base="http://127.0.0.1:1",
                    wire_trace_mode="full",
                    chat_completions=True,
                    context_window=98304,
                ),
            )

            environment = agent.start({}, None)

            # The attribution block varies per conversation, so leaving it on
            # would defeat the served model's prefix cache.
            self.assertEqual(environment["CLAUDE_CODE_ATTRIBUTION_HEADER"], "0")
            self.assertEqual(environment["DISABLE_TELEMETRY"], "1")
            self.assertEqual(environment["DISABLE_ERROR_REPORTING"], "1")
            self.assertEqual(environment["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"], "1")
            self.assertEqual(environment["API_TIMEOUT_MS"], str(2 * 3600 * 1000))
            self.assertEqual(environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "98304")

    def test_an_unset_context_window_is_left_to_the_cli(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent, _, _ = _agent(Path(folder), FakeBox())

            environment = agent.start({}, None)

            self.assertNotIn("CLAUDE_CODE_MAX_CONTEXT_TOKENS", environment)

    def test_a_sandbox_that_never_starts_is_not_a_task_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent, run, _ = _agent(Path(folder), FakeBox())

            def refuse(name: str, lifetime: float, timeout: float) -> FakeBox:
                raise subprocess.TimeoutExpired("create", timeout)

            run.create = refuse  # type: ignore[assignment]
            with self.assertRaisesRegex(RuntimeError, "sandbox did not start"):
                agent.start({}, None)

    def test_start_refuses_an_environment_version_that_moved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent, _, _ = _agent(Path(folder), FakeBox(version="sha256:other"))
            with self.assertRaisesRegex(RuntimeError, "version changed"):
                agent.start({}, None)

    def test_the_task_workspace_is_copied_in(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / "task-files"
            workspace.mkdir()
            (workspace / "image.png").write_bytes(b"\x89PNG")
            box = FakeBox()
            agent, _, _ = _agent(root, box, request=_request(workspace))

            agent.start({}, None)

            self.assertEqual(box.uploads, [(str(workspace), "/roller/workspace", True)])

    def test_version_comes_from_the_mounted_binary(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            box = FakeBox()
            agent, _, _ = _agent(Path(folder), box)
            agent.start({}, None)

            self.assertEqual(agent.version(), "2.1.311 (Claude Code)")
            self.assertIn(
                [DEFAULT_BINARY, "--version"], [argv for argv, _ in box.commands]
            )

    def test_a_turn_runs_the_configured_binary_in_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            box = FakeBox()
            agent, _, harness = _agent(
                root,
                box,
                config=ClaudeCodeConfig(
                    api_base="http://127.0.0.1:1",
                    wire_trace_mode="full",
                    chat_completions=True,
                    sandbox_binary="/opt/cc/bin/claude",
                    max_turns=100,
                ),
            )
            agent.start({}, None)

            code = agent.run_turn(
                "solve it",
                resume=False,
                timeout_seconds=30,
                log_path=root / "rollout" / "stream.jsonl",
            )

            self.assertEqual(code, 0)
            spawned = box.spawned[0]
            self.assertEqual(spawned["argv"][0], "/opt/cc/bin/claude")
            self.assertNotIn("claude", spawned["argv"])
            self.assertIn("--max-turns", spawned["argv"])
            self.assertEqual(spawned["cwd"], "/roller/workspace")
            self.assertEqual(spawned["log"], root / "rollout" / "stream.jsonl")
            self.assertEqual(
                (root / "rollout" / "stream.jsonl").read_text(encoding="utf-8"),
                '{"type":"result"}\n',
            )

    def test_a_turn_that_exits_non_zero_reports_the_clis_own_ending(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            box = FakeBox(exit_code=1)
            # What the CLI writes when it runs out of turns, after a view of an
            # image whose base64 is the tail of the log.
            box.stream = (
                '{"type":"assistant","message":{"content":[{"type":"text",'
                '"text":"/9j/4AAQSkZJRgABAQAAAQ"}]}}\n'
                '{"type":"result","subtype":"error_max_turns","num_turns":12,'
                '"is_error":true,"result":"Reached the maximum number of turns"}\n'
            )
            agent, _, _ = _agent(root, box)
            agent.start({}, None)

            with self.assertRaises(RuntimeError) as caught:
                agent.run_turn(
                    "solve it",
                    resume=False,
                    timeout_seconds=30,
                    log_path=root / "rollout" / "stream.jsonl",
                )

            message = str(caught.exception)
            self.assertIn("error_max_turns, 12 turns", message)
            self.assertIn("Reached the maximum number of turns", message)
            # The log's own bytes are not what the failure reports.
            self.assertNotIn("/9j/4AAQ", message)

    def test_a_stream_without_a_result_event_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            box = FakeBox(exit_code=1)
            box.stream = '{"type":"assistant","message":{"content":[]}}\n'
            agent, _, _ = _agent(root, box)
            agent.start({}, None)

            with self.assertRaisesRegex(RuntimeError, "no result event"):
                agent.run_turn(
                    "solve it",
                    resume=False,
                    timeout_seconds=30,
                    log_path=root / "rollout" / "stream.jsonl",
                )

    def test_a_turn_that_runs_out_of_time_kills_the_process(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            box = FakeBox(exit_code=None)
            agent, _, _ = _agent(root, box)
            agent.start({}, None)

            with self.assertRaisesRegex(TimeoutError, "timed out"):
                agent.run_turn(
                    "solve it",
                    resume=False,
                    timeout_seconds=1,
                    log_path=root / "rollout" / "stream.jsonl",
                )

    def test_the_trace_is_downloaded_and_combined(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            box = FakeBox()
            agent, _, _ = _agent(root, box)
            agent.start({}, None)

            trace = agent.collect_trace(root, 1)

            self.assertEqual(box.downloads, [(f"{CLAUDE_HOME}/projects", True)])
            assert trace is not None
            self.assertEqual(trace, root / "rollout" / "trace.jsonl")
            self.assertIn("assistant", trace.read_text(encoding="utf-8"))
            self.assertFalse((root / ".claude-session-copy").exists())

    def test_outputs_are_downloaded_from_the_task_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            box = FakeBox()
            agent, _, _ = _agent(root, box)
            agent.start({}, None)

            collected, copy_error = agent.collect_outputs(root)

            self.assertIsNone(copy_error)
            self.assertTrue(collected.passed)
            self.assertIn(("/roller/workspace/output", True), box.downloads)

    def test_close_releases_the_instance_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            box = FakeBox()
            agent, run, _ = _agent(Path(folder), box)
            agent.start({}, None)

            agent.close()
            agent.close()

            self.assertEqual(run.released, ["sample"])
            self.assertEqual(box.closed, 1)


class SandboxConfigTests(unittest.TestCase):
    def test_wire_tracing_in_a_sandbox_does_not_need_host_networking(self) -> None:
        run = FakeRun(FakeBox())
        config = ClaudeCodeConfig(
            api_base="http://127.0.0.1:1",
            wire_trace_mode="full",
            capture_calls=True,
            network="none",
            sandbox_run=run,
        )

        self.assertTrue(config.sandbox)

    def test_host_networking_is_still_required_without_a_sandbox(self) -> None:
        with self.assertRaisesRegex(ValueError, "--network host"):
            ClaudeCodeConfig(
                api_base="http://127.0.0.1:1",
                wire_trace_mode="full",
                capture_calls=True,
                network="none",
            )

    def test_inline_tool_images_need_the_chat_completions_translation(self) -> None:
        with self.assertRaisesRegex(ValueError, "--claude-chat-completions"):
            ClaudeCodeConfig(
                api_base="http://127.0.0.1:1",
                wire_trace_mode="full",
                tool_images_inline=True,
            )

    def test_a_relative_sandbox_binary_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            ClaudeCodeConfig(sandbox_binary="bin/claude")


if __name__ == "__main__":
    unittest.main()
