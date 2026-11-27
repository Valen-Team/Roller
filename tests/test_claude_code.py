from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from roller.docker_runtime import DockerRuntime
from roller.harnesses.claude_code import (
    CLAUDE_INPUT_CONTAINER_PATH,
    ClaudeCodeConfig,
    ClaudeCodeHarness,
    claude_effort,
)
from roller.models import InputFile, RunRequest, Task
from roller.redaction import normalize_anthropic_base_url, redact_anthropic_base_url


class ClaudeCodeHarnessTests(unittest.TestCase):
    def _request(self, thinking: str | None = None) -> RunRequest:
        return RunRequest(
            index=0,
            task=Task(
                id="task",
                prompts=("hello", "follow up"),
                source_index=0,
                thinking=thinking,
            ),
            model="claude-opus-5",
            timeout_seconds=30,
        )

    def test_normalizes_full_messages_endpoint_and_redacts_credentials(self) -> None:
        self.assertEqual(
            normalize_anthropic_base_url(
                "http://example.test:3888/v1/messages?api_key=secret"
            ),
            "http://example.test:3888",
        )
        self.assertEqual(
            normalize_anthropic_base_url("http://example.test:3888/v1"),
            "http://example.test:3888",
        )
        self.assertEqual(
            redact_anthropic_base_url(
                "https://user:secret@example.test:8443/v1/messages?api_key=x"
            ),
            "https://example.test:8443/v1/messages",
        )

    def test_effort_mapping_is_claude_specific(self) -> None:
        self.assertEqual(claude_effort("high"), "high")
        self.assertEqual(claude_effort("ultra"), "ultracode")
        with self.assertRaisesRegex(ValueError, "no exact Claude Code mapping"):
            claude_effort("off")
        with self.assertRaisesRegex(ValueError, "no exact Claude Code mapping"):
            claude_effort("adaptive")

    def test_maps_named_secret_and_custom_endpoint(self) -> None:
        harness = ClaudeCodeHarness(
            ClaudeCodeConfig(
                api_base="http://example.test/v1/messages",
                api_key_env="ROLLER_TEST_ANTHROPIC_KEY",
                wire_trace_mode="off",
            ),
            0,
        )
        with patch.dict(
            os.environ,
            {"ROLLER_TEST_ANTHROPIC_KEY": "secret", "HTTPS_PROXY": "proxy"},
            clear=True,
        ):
            environment = harness._environment(self._request())
        self.assertEqual(environment["ANTHROPIC_API_KEY"], "secret")
        self.assertEqual(environment["ANTHROPIC_BASE_URL"], "http://example.test")
        self.assertEqual(environment["ANTHROPIC_MODEL"], "claude-opus-5")
        self.assertEqual(environment["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "claude-opus-5")
        self.assertEqual(environment["HTTPS_PROXY"], "proxy")
        self.assertNotIn("ROLLER_TEST_ANTHROPIC_KEY", environment)

    def test_official_api_strips_optional_provider_prefix(self) -> None:
        harness = ClaudeCodeHarness(
            ClaudeCodeConfig(
                api_key_env="ROLLER_TEST_ANTHROPIC_KEY",
                wire_trace_mode="off",
            ),
            0,
        )
        request = RunRequest(
            index=0,
            task=Task(id="task", prompts=("hello",), source_index=0),
            model="anthropic/claude-opus-5",
            timeout_seconds=30,
        )
        with patch.dict(
            os.environ,
            {"ROLLER_TEST_ANTHROPIC_KEY": "secret"},
            clear=True,
        ):
            environment = harness._environment(request)
        command = harness._claude_command("container", request, "hello", resume=False)
        self.assertEqual(environment["ANTHROPIC_MODEL"], "claude-opus-5")
        self.assertNotIn("ANTHROPIC_DEFAULT_SONNET_MODEL", environment)
        self.assertEqual(command[command.index("--model") + 1], "claude-opus-5")

    def test_attached_images_are_sent_as_a_stream_json_first_message(self) -> None:
        harness = ClaudeCodeHarness(ClaudeCodeConfig(wire_trace_mode="off"), 0)
        harness.runtime.run = MagicMock()
        harness.runtime.popen = MagicMock()
        harness.runtime.popen.return_value.wait.return_value = 0
        copied: list[dict] = []

        def run(args, timeout=None):
            if args[0] == "cp":
                copied.append(json.loads(Path(args[1]).read_text(encoding="utf-8")))
            return subprocess.CompletedProcess(args, 0, "", "")

        harness.runtime.run.side_effect = run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "chart.png").write_bytes(b"png-bytes")
            request = RunRequest(
                0,
                Task(
                    "task",
                    ("hello", "next"),
                    0,
                    inputs=(InputFile(root / "chart.png", "inputs/chart.png"),),
                    attach_images=("inputs/chart.png",),
                ),
                "anthropic/claude-opus-5",
                30,
            )
            log_path = root / "rollout" / "claude.log"
            harness._run_turn(
                "container",
                request,
                "hello",
                resume=False,
                timeout_seconds=30,
                log_path=log_path,
            )
            harness._run_turn(
                "container",
                request,
                "next",
                resume=True,
                timeout_seconds=30,
                log_path=log_path,
            )
            self.assertEqual(list(log_path.parent.iterdir()), [])

        first, resumed = (call.args[0] for call in harness.runtime.popen.call_args_list)
        shell = first.index("/bin/sh")
        self.assertEqual(
            first[shell : shell + 5],
            [
                "/bin/sh",
                "-c",
                'exec "$@" < "$0"',
                CLAUDE_INPUT_CONTAINER_PATH,
                "claude",
            ],
        )
        self.assertEqual(first[-2:], ["--print", "--input-format=stream-json"])
        self.assertIn("--output-format=stream-json", first)
        self.assertNotIn("hello", first)
        self.assertEqual(resumed[-3:], ["--continue", "--print", "next"])
        self.assertNotIn("/bin/sh", resumed)
        self.assertEqual(len(copied), 1)
        self.assertEqual(
            copied[0]["message"]["content"],
            [
                {"type": "text", "text": "hello"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(b"png-bytes").decode(),
                    },
                },
            ],
        )

    def test_builds_print_and_resume_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "settings.json"
            settings.write_text("{}", encoding="utf-8")
            harness = ClaudeCodeHarness(
                ClaudeCodeConfig(
                    settings_path=settings,
                    reasoning_effort="high",
                    max_turns=12,
                    permission_mode="bypassPermissions",
                    wire_trace_mode="off",
                ),
                0,
            )
            first = harness._claude_command(
                "container", self._request(), "hello", resume=False
            )
            resumed = harness._claude_command(
                "container", self._request(), "follow up", resume=True
            )
        self.assertIn("--verbose", first)
        self.assertIn("--output-format=stream-json", first)
        self.assertIn("--permission-mode=bypassPermissions", first)
        self.assertEqual(first[first.index("--effort") + 1], "high")
        self.assertEqual(first[first.index("--max-turns") + 1], "12")
        self.assertNotIn("--continue", first)
        self.assertIn("--continue", resumed)
        self.assertEqual(resumed[-2:], ["--print", "follow up"])

    def test_wire_trace_requires_compatible_network_and_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "full requires --api-base"):
            ClaudeCodeConfig(wire_trace_mode="full")
        with self.assertRaisesRegex(ValueError, "network host"):
            ClaudeCodeConfig(
                api_base="http://example.test/v1/messages",
                wire_trace_mode="actf",
                network="bridge",
            )

    @patch("roller.harnesses.claude_code.WireTraceProxy")
    def test_actf_wire_proxy_captures_tools_without_retaining_full_body(
        self, proxy_class
    ) -> None:
        harness = ClaudeCodeHarness(
            ClaudeCodeConfig(
                api_base="http://example.test/v1/messages",
                wire_trace_mode="actf",
            ),
            0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            harness._start_wire_proxy(Path(temporary))
        self.assertEqual(proxy_class.call_args.args[0], "http://example.test")
        self.assertEqual(proxy_class.call_args.kwargs["mode"], "tools")
        self.assertEqual(
            proxy_class.call_args.kwargs["drop_query_parameters"], ("beta",)
        )
        self.assertEqual(
            proxy_class.call_args.kwargs["drop_request_headers"],
            ("anthropic-beta",),
        )
        proxy_class.return_value.start.assert_called_once_with()

    def test_combines_main_and_sidechain_events_chronologically_and_dedupes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main = root / "main.jsonl"
            side = root / "side.jsonl"
            main.write_text(
                json.dumps({"uuid": "u2", "timestamp": "2026-01-01T00:00:02Z"})
                + "\n"
                + json.dumps({"uuid": "dupe", "timestamp": "2026-01-01T00:00:03Z"})
                + "\n",
                encoding="utf-8",
            )
            side.write_text(
                json.dumps({"uuid": "u1", "timestamp": "2026-01-01T00:00:01Z"})
                + "\n"
                + json.dumps({"uuid": "dupe", "timestamp": "2026-01-01T00:00:03Z"})
                + "\n",
                encoding="utf-8",
            )
            destination = root / "combined.jsonl"
            ClaudeCodeHarness._combine_session_files([main, side], destination)
            rows = [json.loads(line) for line in destination.read_text().splitlines()]
        self.assertEqual([row["uuid"] for row in rows], ["u1", "u2", "dupe"])

    def test_collects_main_trace_under_subagents_output_directory(self) -> None:
        runtime = DockerRuntime()
        runtime.run = MagicMock()

        def copy_session(args, timeout):
            staging = Path(args[-1]) / "-roller-workspace"
            staging.mkdir(parents=True)
            (staging / "session.jsonl").write_text(
                json.dumps({"uuid": "root", "timestamp": "2026-01-01T00:00:01Z"}) + "\n"
            )
            child = staging / "session" / "subagents"
            child.mkdir(parents=True)
            (child / "agent.jsonl").write_text(
                json.dumps({"uuid": "child", "timestamp": "2026-01-01T00:00:02Z"})
                + "\n"
            )
            return MagicMock(returncode=0)

        runtime.run.side_effect = copy_session
        harness = ClaudeCodeHarness(ClaudeCodeConfig(wire_trace_mode="off"), 0, runtime)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "subagents" / "case"
            trace = harness._collect_trace("container", output, 1)
            self.assertIsNotNone(trace)
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual([row["uuid"] for row in rows], ["root", "child"])

    def test_wire_proxy_shutdown_redacts_body_credentials(self) -> None:
        harness = ClaudeCodeHarness(ClaudeCodeConfig(wire_trace_mode="off"), 0)
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "wire.jsonl"
            log_path.write_text(
                json.dumps({"body": {"api_key": "secret", "messages": []}}) + "\n",
                encoding="utf-8",
            )
            proxy = MagicMock(log_path=log_path)
            harness._wire_proxy = proxy
            harness._stop_wire_proxy()
            entry = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(entry["body"]["api_key"], "<redacted>")
        self.assertIsNone(harness._wire_proxy)


if __name__ == "__main__":
    unittest.main()
