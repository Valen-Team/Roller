from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from roller.actf.deepseek import (
    convert_deepseek_trace,
    deepseek_events_to_actf,
    extract_deepseek_usage,
)
from roller.cli import _build_parser, _collect
from roller.docker_runtime import DockerRuntime
from roller.harnesses.deepseek import RUNNER_PATH, DeepSeekConfig, DeepSeekHarness
from roller.models import RunRequest, Task
from roller.sft import actf_to_sft

TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run shell",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def native(kind, data, seq):
    event = {"type": kind, "data": data, "seq": seq, "time": 1000 + seq}
    if kind in {"user/message", "assistant/message", "tool/result"}:
        event["surfaceOp"] = "append"
    return {
        "type": "notification",
        "method": "session.event",
        "payload": {"sessionId": "s1", "event": event},
    }


def sample_events():
    request = {
        "type": "request",
        "body": {
            "messages": [
                {"role": "system", "content": "Actual DeepSeek system prompt"}
            ],
            "tools": [deepcopy(TOOL)],
        },
    }
    return [
        request,
        native("turn/start", {"turn": 1}, 0),
        native(
            "user/message",
            {"role": "user", "content": [{"type": "text", "text": "Read token"}]},
            1,
        ),
        native(
            "assistant/message",
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "Read the file."},
                        {
                            "type": "tool-call",
                            "id": "call-1",
                            "name": "bash",
                            "arguments": '{"command":"cat token.txt"}',
                        },
                    ],
                },
                "usage": {"inputTokens": 10, "outputTokens": 3, "cacheReadTokens": 4},
            },
            2,
        ),
        native(
            "tool/result",
            {
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "call-1",
                            "content": [{"type": "text", "text": "TOKEN"}],
                        }
                    ],
                }
            },
            3,
        ),
        deepcopy(request),
        native(
            "assistant/message",
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "TOKEN"}],
                },
                "usage": {"inputTokens": 20, "outputTokens": 2, "totalTokens": 22},
            },
            4,
        ),
        native("turn/end", {"turn": 1, "reason": {"kind": "completed"}}, 5),
    ]


def write_events(path, events=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (sample_events() if events is None else events)
        )
    )


def locked_configuration(*flags):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tasks = root / "tasks.json"
        tasks.write_text(
            json.dumps([{"harness": "deepseek", "model": "model", "prompt": "OK"}])
        )
        command = [
            "collect",
            "--tasks",
            str(tasks),
            "--output",
            str(root / "runs"),
            "--run-name",
            "case",
            "--api-base",
            "https://deepseek.example/v1",
            "--api-key-env",
            "LOCAL_TEST_KEY",
            *flags,
        ]
        with (
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
            patch("roller.cli.collect._docker_preflight", return_value="sha256:test"),
            patch("roller.cli.collect.Collector") as collector,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            collector.return_value.collect.return_value = []
            assert _collect(_build_parser().parse_args(command)) == 0
        return json.loads((root / "runs/case/lock.json").read_text())["execution"][
            "deepseek"
        ]


class DeepSeekProfileTests(unittest.TestCase):
    def session(self, **kwargs):
        harness = DeepSeekHarness(DeepSeekConfig(**kwargs), 0, Mock())
        session = harness.create_session(
            RunRequest(0, Task("test", ("one",), 0), "model", 60), Path("unused"), 1
        )
        session.version = "0.2.2rc2"
        return session

    def test_default_lock_settings_and_environment_are_unchanged(self):
        self.assertEqual(
            locked_configuration(),
            {
                "docker_image": "roller-deepseek:0.2.2rc2",
                "container_user": "0",
                "network": "host",
                "api_base": "https://deepseek.example/v1",
                "api_key_env": "LOCAL_TEST_KEY",
                "reasoning_effort": None,
                "max_steps": 12,
                "deepseek_version": "0.2.2rc2",
                "context_length": 128000,
                "output_tokens": 8192,
                "tools": ["bash"],
                "api_mode": "chat_completions",
                "wire_trace": "actf",
                "keep_thinking": True,
                "pass_env_names": [
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "NO_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "no_proxy",
                ],
                "profile": "sdk-minimal",
                "retry_count": 0,
                "docker_image_digest": "sha256:test",
            },
        )
        session = self.session()
        self.assertEqual(
            session.settings(),
            {
                "max_steps": 12,
                "context_length": 128000,
                "output_tokens": 8192,
                "tools": ["bash"],
            },
        )
        self.assertEqual(
            session.version_fields(),
            {"deepseek_version": "0.2.2rc2", "profile": "sdk-minimal"},
        )
        self.assertEqual(session.turn_options(60)["profile"], "sdk-minimal")
        self.assertEqual(
            session.harness_environment("https://deepseek.example/v1"),
            {
                "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
                "HOME": "/roller/deepseek-home",
                "DSH_HOME": "/roller/deepseek-home",
                "DSH_RUNTIME_MODE": "exe",
                "ROLLER_DEEPSEEK_JOURNAL": "/roller/deepseek-state/trace.jsonl",
            },
        )

    def test_selected_profile_and_permission_mode_are_threaded(self):
        locked = locked_configuration(
            "--deepseek-profile",
            "sdk",
            "--deepseek-permission-mode",
            "danger-full-access",
        )
        self.assertIsNone(locked["tools"])
        self.assertEqual(
            (locked["profile"], locked["permission_mode"]),
            ("sdk", "danger-full-access"),
        )
        session = self.session(profile="sdk", permission_mode="danger-full-access")
        self.assertIsNone(session.settings()["tools"])
        self.assertEqual(
            session.version_fields(),
            {
                "deepseek_version": "0.2.2rc2",
                "profile": "sdk",
                "permission_mode": "danger-full-access",
            },
        )
        self.assertEqual(session.turn_options(60)["profile"], "sdk")
        self.assertEqual(
            session.harness_environment("https://deepseek.example/v1")[
                "DSH_PERMISSION_MODE"
            ],
            "danger-full-access",
        )

    def test_other_profiles_run_commands_unconfined_unless_a_mode_is_chosen(self):
        # Their default file sandbox has no backend in the task container.
        locked = locked_configuration("--deepseek-profile", "sdk")
        self.assertEqual(locked["permission_mode"], "danger-full-access")
        session = self.session(profile="sdk")
        self.assertEqual(
            session.harness_environment("https://deepseek.example/v1")[
                "DSH_PERMISSION_MODE"
            ],
            "danger-full-access",
        )
        self.assertEqual(
            session.version_fields()["permission_mode"], "danger-full-access"
        )
        chosen = self.session(profile="sdk", permission_mode="read-only")
        self.assertEqual(
            chosen.harness_environment("https://deepseek.example/v1")[
                "DSH_PERMISSION_MODE"
            ],
            "read-only",
        )

    def test_input_modalities_are_declared_to_the_sdk(self):
        self.assertNotIn("input_modalities", locked_configuration())
        locked = locked_configuration(
            "--deepseek-profile", "sdk", "--deepseek-input-modalities", "text,image"
        )
        self.assertEqual(locked["input_modalities"], ["text", "image"])
        session = self.session(profile="sdk", input_modalities=("text", "image"))
        self.assertEqual(
            session.turn_options(60)["input_modalities"], ["text", "image"]
        )
        self.assertEqual(
            session.version_fields()["input_modalities"], ["text", "image"]
        )
        self.assertEqual(self.session().turn_options(60)["input_modalities"], ["text"])
        assets = RUNNER_PATH.parent
        self.assertIn(
            "inputModalities: process.env.ROLLER_DEEPSEEK_INPUT_MODALITIES.split(',')",
            (assets / "roller.patch.yml").read_text(),
        )
        self.assertIn(
            'os.environ["ROLLER_DEEPSEEK_INPUT_MODALITIES"]',
            (assets / "runner.py").read_text(),
        )
        for value in ((), ("image",), ("text", "audio"), "text"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "modalities"),
            ):
                DeepSeekConfig(input_modalities=value)

    def test_delegation_tools_are_withheld_from_the_model(self):
        source = (RUNNER_PATH.parent / "journal.mjs").read_text()
        for name in (
            "subagent",
            "subagent_fork",
            "list_agents",
            "interrupt_agent",
            "send_message",
            "workflow",
            "ralph",
        ):
            self.assertIn(f"'{name}'", source)
        self.assertIn(
            "body.tools.filter(tool => !DELEGATION_TOOLS.has(tool.function?.name))",
            source,
        )

    def test_invalid_profile_and_permission_mode(self):
        for settings in (
            {"profile": ""},
            {"profile": "../sdk"},
            {"profile": "sdk\n"},
            {"profile": None},
            {"permission_mode": "full"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                DeepSeekConfig(**settings)


IMAGE_URLS = ["data:image/png;base64,AA==", "data:image/png;base64,AQ=="]


def image_events(count=1, sent=None):
    """A read_image call whose result names ``count`` attachments."""
    tool = {
        "type": "function",
        "function": {"name": "read_image", "parameters": {"type": "object"}},
    }
    events = sample_events()
    for row in events:
        if row.get("type") == "request":
            row["body"]["tools"].append(deepcopy(tool))
    events[3]["payload"]["event"]["data"]["message"]["content"][1].update(
        name="read_image", arguments='{"file_path":"a.png"}'
    )
    events[4]["payload"]["event"]["data"]["message"]["content"][0]["content"] = [
        {"type": "text", "text": "image/png image"},
        *(
            {
                "type": "image",
                "attachment": {"attachmentId": f"sha256:{n}", "mediaType": "image/png"},
            }
            for n in range(count)
        ),
    ]
    urls = [IMAGE_URLS[n] for n in range(count)] if sent is None else sent
    if urls:
        events[5]["body"]["messages"] += [
            {"role": "tool", "content": "image/png image"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Attached image(s) from tool result:"},
                    *({"type": "image_url", "image_url": {"url": url}} for url in urls),
                ],
            },
        ]
    return events


class DeepSeekTraceTests(unittest.TestCase):
    def test_tool_result_images_survive_actf_and_sft_in_order(self):
        actf = deepseek_events_to_actf(image_events(2))
        blocks = actf["steps"][0]["observation"][0]["content"]
        self.assertEqual(
            [block["type"] for block in blocks], ["text", "image", "image"]
        )
        sft = actf_to_sft(actf)
        self.assertEqual(
            sft["messages"][3]["content"],
            [
                {"type": "text", "text": "image/png image"},
                *(
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in IMAGE_URLS
                ),
            ],
        )
        self.assertNotIn("images", sft)

    def test_tool_images_must_match_what_the_model_received(self):
        for sent in ([], IMAGE_URLS[:1], ["https://example.test/a.png", IMAGE_URLS[1]]):
            with (
                self.subTest(sent=sent),
                self.assertRaisesRegex(ValueError, "inline data"),
            ):
                deepseek_events_to_actf(image_events(2, sent=sent))
        with self.assertRaisesRegex(ValueError, "no tool result declares"):
            deepseek_events_to_actf(image_events(1, sent=IMAGE_URLS))

    def test_native_tools_reasoning_and_usage(self):
        events = sample_events()
        actf = deepseek_events_to_actf(events)
        self.assertEqual(actf["tools"], [TOOL])
        self.assertEqual(actf["system_prompt"], "Actual DeepSeek system prompt")
        metric = actf["steps"][0]["metric"]
        self.assertEqual(
            (
                metric["prompt_tokens_len"],
                metric["cache_read_tokens"],
                metric["cache_write_tokens"],
            ),
            (14, 4, 0),
        )
        sft = actf_to_sft(actf)
        self.assertEqual(
            [m["role"] for m in sft["messages"]],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(sft["messages"][3]["tool_call_id"], "call-1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            write_events(path, events)
            usage = extract_deepseek_usage(path)
            self.assertEqual(
                (
                    usage["request_count"],
                    usage["input_tokens"],
                    usage["output_tokens"],
                    usage["cache_read_tokens"],
                    usage["total_tokens"],
                ),
                (2, 30, 5, 4, 39),
            )

    def test_export_reasoning_preserves_native_history(self):
        events = sample_events()
        original = deepcopy(events)
        actf = deepseek_events_to_actf(events, keep_thinking=False)
        self.assertEqual(events, original)
        self.assertTrue(
            all(not s["assistant_content"]["reasoning_content"] for s in actf["steps"])
        )

    def test_two_turns_preserve_identical_replies(self):
        events = sample_events()
        events += [
            native("turn/start", {"turn": 2}, 6),
            native(
                "user/message",
                {"role": "user", "content": [{"type": "text", "text": "Again"}]},
                7,
            ),
            native(
                "assistant/message",
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "TOKEN"}],
                    }
                },
                8,
            ),
            native("turn/end", {"turn": 2, "reason": {"kind": "completed"}}, 9),
        ]
        sft = actf_to_sft(deepseek_events_to_actf(events))
        self.assertEqual(sum(m["role"] == "user" for m in sft["messages"]), 2)
        self.assertEqual(
            sum(
                m["role"] == "assistant" and m.get("content") == "TOKEN"
                for m in sft["messages"]
            ),
            2,
        )

    def test_incomplete_or_lossy_data_rejected(self):
        for case in (
            "schema",
            "arguments",
            "duplicate",
            "unpaired",
            "image",
            "interrupted",
            "runner",
            "max-tokens",
            "error",
            "no-end",
            "open-next-turn",
            "replace",
            "system",
        ):
            with self.subTest(case=case):
                events = sample_events()
                assistant = events[3]["payload"]["event"]["data"]
                if case == "schema":
                    for e in events:
                        if e["type"] == "request":
                            e["body"]["tools"] = []
                elif case == "arguments":
                    assistant["message"]["content"][1]["arguments"] = "[]"
                elif case == "duplicate":
                    assistant["message"]["content"].append(
                        deepcopy(assistant["message"]["content"][1])
                    )
                elif case == "unpaired":
                    events.pop(4)
                elif case == "image":
                    assistant["message"]["content"].append({"type": "image"})
                elif case == "interrupted":
                    assistant["interrupted"] = True
                elif case == "runner":
                    events.append({"type": "runner_error", "error": "failure"})
                elif case in {"max-tokens", "error"}:
                    events[-1]["payload"]["event"]["data"]["reason"]["kind"] = case
                elif case == "no-end":
                    events.pop()
                elif case == "open-next-turn":
                    events.append(native("turn/start", {"turn": 2}, 6))
                elif case == "replace":
                    events[3]["payload"]["event"]["surfaceOp"] = {"op": "replace"}
                else:
                    for e in events:
                        if e["type"] == "request":
                            e["body"]["messages"] = []
                with self.assertRaises(ValueError):
                    deepseek_events_to_actf(events)

    def test_corrupt_journal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "trace.jsonl"
            source.write_text('{"type":')
            with self.assertRaises(ValueError):
                convert_deepseek_trace(source, Path(tmp) / "trace_ACTF.json")

    def test_unicode_line_separator_inside_json_string_is_preserved(self):
        events = sample_events()
        events[-2]["payload"]["event"]["data"]["message"]["content"][0]["text"] = (
            "TOKEN\u2028next"
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "trace.jsonl"
            source.write_text(
                "".join(
                    json.dumps(event, ensure_ascii=False) + "\n" for event in events
                ),
                encoding="utf-8",
            )
            destination = Path(tmp) / "trace_ACTF.json"
            actf = convert_deepseek_trace(source, destination)
            self.assertEqual(
                actf_to_sft(actf)["messages"][-1]["content"], "TOKEN\u2028next"
            )


class FakeRuntime(DockerRuntime):
    def __init__(self):
        self.calls = []
        self.result = {"ok": True, "content": "TOKEN"}
        self.process = Mock()
        self.process.wait.return_value = 0

    def run(self, args, timeout=None):
        self.calls.append(list(args))
        stdout = ""
        if args[:2] == ["image", "inspect"]:
            stdout = "sha256:test"
        elif args[-1] == "--check":
            stdout = "0.2.2rc2"
        elif args[-1].endswith("/result.json"):
            stdout = json.dumps(self.result)
        elif args[0] == "cp" and args[1].endswith(
            ":/roller/deepseek-state/trace.jsonl"
        ):
            write_events(Path(args[2]))
        elif args[0] == "cp" and args[1].endswith("/outputs/."):
            (Path(args[2]) / "answer.txt").write_text("TOKEN")
        return subprocess.CompletedProcess(args, 0, stdout, "")

    def popen(self, args, log_path, append=False):
        self.calls.append(list(args))
        return self.process

    def close_log(self, process):
        pass


class DeepSeekHarnessTests(unittest.TestCase):
    def test_attached_images_are_rejected_before_the_container_starts(self):
        runtime = FakeRuntime()
        harness = DeepSeekHarness(
            DeepSeekConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
            0,
            runtime,
        )
        task = Task("test", ("one",), 0, attach_images=("inputs/chart.png",))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
        ):
            result = harness.run(RunRequest(0, task, "model", 60), Path(tmp), 1)
        self.assertEqual(result.status, "error")
        self.assertIn("DeepSeek does not support files.attach_images", result.error)
        self.assertFalse(any(call[0] == "run" for call in runtime.calls))

    def test_session_outputs_cleanup_and_redaction(self):
        runtime = FakeRuntime()
        harness = DeepSeekHarness(
            DeepSeekConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
            0,
            runtime,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
        ):
            result = harness.run(
                RunRequest(0, Task("test", ("one", "two"), 0), "model", 60),
                Path(tmp),
                1,
            )
            self.assertEqual(result.status, "ok", result.error)
            self.assertEqual(result.metadata["turn_count"], 2)
            self.assertEqual((Path(tmp) / "results/answer.txt").read_text(), "TOKEN")
            self.assertFalse((Path(tmp) / ".deepseek-turn.json").exists())
            self.assertTrue(any(a[:2] == ["rm", "-f"] for a in runtime.calls))
            runtime.result = {"ok": False, "error": "failed test-secret"}
            result = harness.run(
                RunRequest(1, Task("error", ("one",), 0), "model", 60),
                Path(tmp) / "error",
                1,
            )
            self.assertEqual(result.status, "error")
            self.assertNotIn("test-secret", result.error)

    def test_timeout_stops_container_and_recovers_outputs(self):
        runtime = FakeRuntime()
        runtime.process.wait.side_effect = [subprocess.TimeoutExpired("run", 1), 0]
        harness = DeepSeekHarness(
            DeepSeekConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
            0,
            runtime,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
        ):
            result = harness.run(
                RunRequest(0, Task("test", ("one",), 0), "model", 60), Path(tmp), 1
            )
            self.assertEqual(result.status, "timeout")
            self.assertTrue(any(a[0] == "kill" for a in runtime.calls))
            self.assertTrue((Path(tmp) / "results/answer.txt").is_file())

    def test_conversion_failure_keeps_raw_and_recovers_outputs(self):
        runtime = FakeRuntime()
        harness = DeepSeekHarness(
            DeepSeekConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
            0,
            runtime,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
            patch(
                "roller.actf.deepseek.convert_deepseek_trace",
                side_effect=ValueError("invalid journal"),
            ),
        ):
            result = harness.run(
                RunRequest(0, Task("test", ("one",), 0), "model", 60), Path(tmp), 1
            )
            self.assertEqual(result.status, "error")
            self.assertIn("invalid journal", result.error)
            self.assertIsNone(result.trajectory_path)
            self.assertTrue((Path(tmp) / result.trace_path).is_file())
            self.assertEqual((Path(tmp) / "results/answer.txt").read_text(), "TOKEN")
            self.assertTrue(any(a[:2] == ["rm", "-f"] for a in runtime.calls))

    def test_output_transfer_failure_keeps_the_converted_trajectory(self):
        harness = DeepSeekHarness(
            DeepSeekConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
            0,
            FakeRuntime(),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
            patch(
                "roller.harnesses.bridge.collect_outputs",
                side_effect=EOFError("truncated output archive"),
            ),
        ):
            result = harness.run(
                RunRequest(0, Task("test", ("one",), 0), "model", 60), Path(tmp), 1
            )
            self.assertEqual(result.status, "error")
            self.assertIn("truncated output archive", result.error)
            self.assertTrue((Path(tmp) / result.trace_path).is_file())
            self.assertTrue((Path(tmp) / result.trajectory_path).is_file())

    def test_invalid_settings_and_mixed_dry_run(self):
        for settings in (
            {"max_steps": 0},
            {"max_steps": True},
            {"context_length": 100},
            {"output_tokens": 128000},
            {"api_base": "https://user:secret@example.org"},
            {"reasoning_effort": "adaptive"},
            {"wire_trace_mode": "full"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                DeepSeekConfig(**settings)
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "tasks.json"
            tasks.write_text(
                json.dumps(
                    [
                        {
                            "id": "one",
                            "harness": "deepseek",
                            "prompt": "hi",
                            "model": "model",
                        },
                        {
                            "id": "two",
                            "harness": "hermes",
                            "user_simulator": {
                                "harness": "deepseek",
                                "instruction": "finish",
                                "max_turns": 1,
                            },
                            "model": "model",
                        },
                    ]
                )
            )
            args = ["collect", "--tasks", str(tasks), "--dry-run"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_collect(_build_parser().parse_args(args)), 0)


if __name__ == "__main__":
    unittest.main()
