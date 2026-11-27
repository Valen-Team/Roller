from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from roller.actf.opencode import (
    convert_opencode_trace,
    extract_opencode_usage,
    native_messages,
    opencode_events_to_actf,
)
from roller.cli import _build_parser, _collect
from roller.docker_runtime import DockerRuntime
from roller.harnesses.container import PROXY_ENV_NAMES
from roller.harnesses.opencode import (
    RUNNER_PATH,
    OpenCodeConfig,
    OpenCodeHarness,
    _ensure_local_proxy_bypass,
    _system_prompt,
)
from roller.models import RunRequest, Task
from roller.sft import SFTConversionError, actf_to_sft

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


def sample_events():
    messages = [
        {
            "info": {
                "id": "m1",
                "role": "user",
                "sessionID": "s1",
                "time": {"created": 1000},
            },
            "parts": [
                {"id": "p1", "messageID": "m1", "type": "text", "text": "Read token"}
            ],
        },
        {
            "info": {
                "id": "m2",
                "role": "assistant",
                "sessionID": "s1",
                "parentID": "m1",
                "time": {"created": 1001, "completed": 1002},
                "finish": "tool-calls",
                "tokens": {"input": 10, "output": 3, "cache": {"read": 4, "write": 0}},
            },
            "parts": [
                {
                    "id": "p2",
                    "messageID": "m2",
                    "type": "reasoning",
                    "text": "Read the file.",
                },
                {
                    "id": "p3",
                    "messageID": "m2",
                    "type": "tool",
                    "callID": "call-1",
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        "input": {"command": "cat token.txt"},
                        "output": "TOKEN",
                        "time": {"start": 1001, "end": 1002},
                    },
                },
            ],
        },
        {
            "info": {
                "id": "m3",
                "role": "assistant",
                "sessionID": "s1",
                "parentID": "m1",
                "time": {"created": 1003, "completed": 1004},
                "finish": "stop",
                "tokens": {"input": 20, "output": 2, "total": 22},
            },
            "parts": [{"id": "p4", "messageID": "m3", "type": "text", "text": "TOKEN"}],
        },
    ]
    return [
        {
            "type": "request",
            "body": {
                "messages": [
                    {"role": "system", "content": "Actual OpenCode system prompt"}
                ],
                "tools": [TOOL],
            },
        },
        {
            "type": "session_export",
            "data": {"info": {"id": "s1"}, "messages": messages},
        },
    ]


def write_events(path, events=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in (events or sample_events())))


class OpenCodeTraceTests(unittest.TestCase):
    def test_user_image_file_parts_reach_actf_and_sft(self):
        events = sample_events()
        parts = events[-1]["data"]["messages"][0]["parts"]
        parts.append(
            {
                "id": "p1a",
                "messageID": "m1",
                "type": "file",
                "mime": "image/PNG",
                "filename": "chart.png",
                "url": "data:image/png;base64,AA==",
            }
        )
        actf = opencode_events_to_actf(events)
        self.assertEqual(
            actf["steps"][0]["user_content"],
            [
                {"type": "text", "text": "Read token"},
                {
                    "type": "image",
                    "data": "data:image/png;base64,AA==",
                    "mimeType": "image/png",
                },
            ],
        )
        record = actf_to_sft(actf)
        self.assertNotIn("images", record)
        self.assertEqual(
            record["messages"][1]["content"],
            [
                {"type": "text", "text": "Read token"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        )

        # A path-only reference stays in ACTF; SFT export never reads files.
        parts[-1]["url"] = "file:///roller/workspace/chart.png"
        actf = opencode_events_to_actf(events)
        self.assertEqual(
            actf["steps"][0]["user_content"][1]["data"],
            "file:///roller/workspace/chart.png",
        )
        with self.assertRaises(SFTConversionError):
            actf_to_sft(actf)

        parts[-1].update(mime="application/pdf", url="data:application/pdf;base64,AA==")
        with self.assertRaisesRegex(ValueError, "lose data"):
            opencode_events_to_actf(events)

    def test_native_tools_reasoning_and_usage(self):
        events = sample_events()
        actf = opencode_events_to_actf(events)
        self.assertEqual(actf["tools"], [TOOL])
        self.assertEqual(actf["system_prompt"], "Actual OpenCode system prompt")
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
            write_events(path, events + [events[-1]])
            usage = extract_opencode_usage(path)
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

    def test_export_reasoning_does_not_change_native_history(self):
        events = sample_events()
        original = deepcopy(events)
        actf = opencode_events_to_actf(events, keep_thinking=False)
        self.assertEqual(events, original)
        self.assertTrue(
            all(not s["assistant_content"]["reasoning_content"] for s in actf["steps"])
        )

    def test_native_updates_and_exports_do_not_duplicate_messages(self):
        events = sample_events()
        messages = events[-1]["data"]["messages"]
        updates = []
        for m in messages:
            updates.append(
                {
                    "type": "event",
                    "event": {
                        "type": "message.updated",
                        "properties": {"info": m["info"]},
                    },
                }
            )
            updates.extend(
                {
                    "type": "event",
                    "event": {
                        "type": "message.part.updated",
                        "properties": {"part": p},
                    },
                }
                for p in m["parts"]
            )
        self.assertEqual(
            opencode_events_to_actf([events[0], *updates, events[-1], events[-1]]),
            opencode_events_to_actf(events),
        )
        self.assertEqual(
            opencode_events_to_actf([events[0], *updates]),
            opencode_events_to_actf(events),
        )

    def test_message_update_without_session_id_uses_native_part_identity(self):
        events = sample_events()
        updates = [
            {
                "type": "event",
                "event": {
                    "type": "session.created",
                    "properties": {"info": {"id": "s1"}},
                },
            }
        ]
        for message in events[-1]["data"]["messages"]:
            info = {
                key: deepcopy(value)
                for key, value in message["info"].items()
                if key != "sessionID"
            }
            updates.append(
                {
                    "type": "event",
                    "event": {"type": "message.updated", "properties": {"info": info}},
                }
            )
            updates.extend(
                {
                    "type": "event",
                    "event": {
                        "type": "message.part.updated",
                        "properties": {
                            "part": {**part, "sessionID": message["info"]["sessionID"]}
                        },
                    },
                }
                for part in message["parts"]
            )
        journal = [events[0], *updates]
        self.assertEqual(
            {row["info"]["sessionID"] for row in native_messages(journal)}, {"s1"}
        )
        self.assertEqual(
            opencode_events_to_actf(journal), opencode_events_to_actf(events)
        )

    def test_partial_message_update_keeps_exported_session_identity(self):
        events = sample_events()
        update = {
            "type": "event",
            "event": {
                "type": "message.updated",
                "properties": {"info": {"id": "m3", "finish": "stop"}},
            },
        }
        self.assertEqual(
            opencode_events_to_actf([*events, update]), opencode_events_to_actf(events)
        )

    def test_null_session_id_update_keeps_previous_native_identity(self):
        events = sample_events()
        for message in events[-1]["data"]["messages"]:
            if message["info"]["id"] == "m3":
                for part in message["parts"]:
                    part.pop("sessionID", None)
        update = {
            "type": "event",
            "event": {
                "type": "message.updated",
                "properties": {
                    "info": {"id": "m3", "sessionID": None, "finish": "stop"}
                },
            },
        }
        rows = native_messages([*events, update], root_only=True)
        self.assertIn("m3", [row["info"]["id"] for row in rows])

    def test_missing_message_identity_is_explicitly_rejected(self):
        events = sample_events()
        for message in events[-1]["data"]["messages"]:
            message["info"].pop("sessionID")
            for part in message["parts"]:
                part.pop("sessionID", None)
        with self.assertRaisesRegex(ValueError, "session identity"):
            opencode_events_to_actf(events)

    def test_missing_child_message_session_id_does_not_enter_root(self):
        events = sample_events()
        events.insert(
            1,
            {
                "type": "event",
                "event": {
                    "type": "session.created",
                    "properties": {"info": {"id": "child", "parentID": "s1"}},
                },
            },
        )
        child = {
            "info": {
                "id": "child-message",
                "role": "assistant",
                "time": {"created": 1005},
                "finish": "stop",
            },
            "parts": [
                {
                    "id": "child-part",
                    "messageID": "child-message",
                    "sessionID": "child",
                    "type": "text",
                    "text": "child only",
                }
            ],
        }
        events[-1]["data"]["messages"].append(child)
        self.assertNotIn(
            "child-message",
            [row["info"]["id"] for row in native_messages(events, root_only=True)],
        )
        self.assertEqual(
            opencode_events_to_actf(events), opencode_events_to_actf(sample_events())
        )

    def test_empty_unfinished_timeout_placeholder_is_not_training_data(self):
        events = sample_events()
        events[-1]["data"]["messages"].append(
            {
                "info": {
                    "id": "m4",
                    "role": "assistant",
                    "sessionID": "s1",
                    "parentID": "m1",
                    "time": {"created": 1005},
                    "tokens": {
                        "input": 0,
                        "output": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
                "parts": [],
            }
        )

        actf = opencode_events_to_actf(events)

        self.assertEqual(len(actf["steps"]), 2)
        self.assertEqual(actf_to_sft(actf)["messages"][-1]["content"], "TOKEN")

    def test_completed_reasoning_only_turn_is_skipped_when_thinking_is_removed(self):
        events = sample_events()
        messages = events[-1]["data"]["messages"]
        messages.insert(
            -1,
            {
                "info": {
                    "id": "m-reasoning",
                    "role": "assistant",
                    "sessionID": "s1",
                    "parentID": "m1",
                    "time": {"created": 1002.5, "completed": 1002.7},
                    "finish": "stop",
                },
                "parts": [
                    {
                        "id": "p-reasoning",
                        "messageID": "m-reasoning",
                        "type": "reasoning",
                        "text": "private only",
                    }
                ],
            },
        )

        stripped = opencode_events_to_actf(events, keep_thinking=False)
        retained = opencode_events_to_actf(events, keep_thinking=True)

        self.assertEqual(len(stripped["steps"]), 2)
        self.assertEqual(len(retained["steps"]), 3)
        actf_to_sft(stripped)

    def test_two_user_turns_keep_native_ids_and_identical_replies(self):
        events = sample_events()
        second = deepcopy(events[-1])
        rows = second["data"]["messages"]
        user, assistant = deepcopy(rows[0]), deepcopy(rows[-1])
        for row, mid, created in ((user, "m4", 1005), (assistant, "m5", 1006)):
            row["info"].update(
                id=mid, time={"created": created, "completed": created + 1}
            )
            for part in row["parts"]:
                part.update(id="p-" + mid, messageID=mid)
        assistant["info"]["parentID"] = "m4"
        rows.extend([user, assistant])
        sft = actf_to_sft(opencode_events_to_actf(events + [second]))
        self.assertEqual(sum(m["role"] == "user" for m in sft["messages"]), 2)
        self.assertEqual(
            sum(
                m["role"] == "assistant" and m.get("content") == "TOKEN"
                for m in sft["messages"]
            ),
            2,
        )

    def test_unsafe_or_incomplete_native_data_is_rejected(self):
        for kind in (
            "schema",
            "changed_schema",
            "arguments",
            "duplicate",
            "pending",
            "attachment_shape",
            "media",
            "error",
            "system",
        ):
            with self.subTest(kind=kind):
                events = sample_events()
                rows = events[-1]["data"]["messages"]
                tool = rows[1]["parts"][1]
                if kind == "schema":
                    events[0]["body"]["tools"] = []
                elif kind == "changed_schema":
                    changed = deepcopy(events[0])
                    changed["body"]["tools"][0]["function"]["description"] = "changed"
                    events.insert(1, changed)
                elif kind == "arguments":
                    tool["state"]["input"] = []
                elif kind == "duplicate":
                    other = deepcopy(tool)
                    other["id"] = "p31"
                    rows[1]["parts"].append(other)
                elif kind == "pending":
                    tool["state"]["status"] = "pending"
                elif kind == "attachment_shape":
                    tool["state"]["attachments"] = {"type": "file", "mime": "image/png"}
                elif kind == "media":
                    rows[0]["parts"][0]["type"] = "file"
                elif kind == "error":
                    rows[1]["info"]["error"] = {"name": "APIError"}
                else:
                    events[0]["body"]["messages"] = []
                with self.assertRaises(ValueError):
                    opencode_events_to_actf(events)

    def test_context_overflow_retains_the_observed_conversation(self):
        events = sample_events()
        overflow = events[-1]["data"]["messages"][-1]
        overflow["info"]["error"] = {"name": "ContextOverflowError"}
        overflow["info"].pop("finish")
        overflow["parts"] = []

        actf = opencode_events_to_actf(events, keep_thinking=True)

        (step,) = actf["steps"]
        self.assertEqual(step["user_content"], "Read token")
        (result,) = step["observation"]
        self.assertEqual(result["content"], [{"type": "text", "text": "TOKEN"}])
        with self.assertRaisesRegex(SFTConversionError, "context overflow"):
            actf_to_sft(actf)

    def test_running_tool_keeps_its_arguments_without_a_result(self):
        events = sample_events()
        messages = events[-1]["data"]["messages"]
        messages.pop()
        state = messages[-1]["parts"][-1]["state"]
        state["status"] = "running"
        state.pop("output")

        actf = opencode_events_to_actf(events, keep_thinking=True)

        (step,) = actf["steps"]
        (call,) = step["assistant_content"]["tool_calls"]
        self.assertEqual(call["arguments"], {"command": "cat token.txt"})
        self.assertEqual(step["observation"], [])
        with self.assertRaisesRegex(SFTConversionError, "running tool call"):
            actf_to_sft(actf)

    def test_tool_image_attachments_are_bounded_and_exported_to_sft(self):
        events = sample_events()
        data_url = "data:image/png;base64," + "A" * 5000
        tool = events[-1]["data"]["messages"][1]["parts"][1]
        tool["state"]["attachments"] = [
            {"type": "file", "mime": "image/png", "url": data_url}
        ]

        actf = opencode_events_to_actf(events)
        observation = actf["steps"][0]["observation"][0]
        attachment = observation["details"]["attachments"][0]
        self.assertEqual(attachment["type"], "file")
        self.assertEqual(attachment["mime"], "image/png")
        self.assertTrue(attachment["url"]["omitted"])
        self.assertEqual(attachment["url"]["scheme"], "data")
        self.assertEqual(attachment["url"]["payload_characters"], 5000)
        self.assertEqual(
            observation["content"],
            [
                {"type": "text", "text": "TOKEN"},
                {
                    "type": "image",
                    "data": data_url,
                    "mimeType": "image/png",
                },
            ],
        )
        self.assertEqual(json.dumps(actf).count(data_url), 1)

        sft = actf_to_sft(actf)
        tool_result = next(m for m in sft["messages"] if m["role"] == "tool")
        self.assertEqual(
            tool_result["content"],
            [
                {"type": "text", "text": "TOKEN"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        )
        self.assertNotIn("images", sft)

    def test_tool_image_attachments_keep_occurrence_order(self):
        events = sample_events()
        first = "data:image/png;base64,AA=="
        second = "data:image/jpeg;base64,AQ=="
        tool = events[-1]["data"]["messages"][1]["parts"][1]
        tool["state"]["attachments"] = [
            {"type": "file", "mime": "image/png", "url": first},
            {"type": "file", "mime": "image/jpeg", "url": second},
        ]

        sft = actf_to_sft(opencode_events_to_actf(events))

        tool_result = next(m for m in sft["messages"] if m["role"] == "tool")
        self.assertEqual(
            tool_result["content"],
            [
                {"type": "text", "text": "TOKEN"},
                {"type": "image_url", "image_url": {"url": first}},
                {"type": "image_url", "image_url": {"url": second}},
            ],
        )
        self.assertNotIn("images", sft)

    def test_non_image_attachment_keeps_bounded_audit_summary(self):
        events = sample_events()
        tool = events[-1]["data"]["messages"][1]["parts"][1]
        tool["state"]["attachments"] = [
            {"type": "file", "mime": "application/pdf", "url": "file:///tmp/a.pdf"}
        ]

        actf = opencode_events_to_actf(events)
        sft = actf_to_sft(actf)

        observation = actf["steps"][0]["observation"][0]
        self.assertEqual(observation["content"][0], {"type": "text", "text": "TOKEN"})
        self.assertIn("OpenCode attachments omitted", observation["content"][1]["text"])
        tool_result = next(m for m in sft["messages"] if m["role"] == "tool")
        self.assertIn("OpenCode attachments omitted", tool_result["content"])

    def test_image_attachment_without_embedded_bytes_fails_sft_export(self):
        events = sample_events()
        tool = events[-1]["data"]["messages"][1]["parts"][1]
        tool["state"]["attachments"] = [
            {"type": "file", "mime": "image/png", "url": "https://example.test/a.png"}
        ]

        actf = opencode_events_to_actf(events)

        self.assertEqual(
            actf["steps"][0]["observation"][0]["content"][1],
            {
                "type": "image",
                "data": "https://example.test/a.png",
                "mimeType": "image/png",
            },
        )
        with self.assertRaisesRegex(SFTConversionError, "valid base64"):
            actf_to_sft(actf)

    def test_corrupt_journal_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "trace.jsonl"
            source.write_text('{"type":')
            with self.assertRaises(ValueError):
                convert_opencode_trace(source, Path(tmp) / "trace_ACTF.json")

    def test_unicode_line_separator_inside_json_string_is_preserved(self):
        events = sample_events()
        events[-1]["data"]["messages"][-1]["parts"][0]["text"] = "TOKEN\u2028next"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "trace.jsonl"
            source.write_text(
                "".join(
                    json.dumps(event, ensure_ascii=False) + "\n" for event in events
                ),
                encoding="utf-8",
            )
            destination = Path(tmp) / "trace_ACTF.json"
            actf = convert_opencode_trace(source, destination)
            self.assertEqual(
                actf_to_sft(actf)["messages"][-1]["content"], "TOKEN\u2028next"
            )

    def test_final_reply_checks_completion_and_current_user(self):
        spec = importlib.util.spec_from_file_location(
            "opencode_runner_test", RUNNER_PATH
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        export = sample_events()[-1]["data"]
        self.assertEqual(runner.native_reply(export), "TOKEN")
        for field, value in (
            ("finish", "length"),
            ("parentID", "unknown"),
            ("error", {"name": "APIError"}),
            ("time", {"created": 1}),
        ):
            changed = deepcopy(export)
            changed["messages"][-1]["info"][field] = value
            with self.assertRaises(ValueError):
                runner.native_reply(changed)

    def test_session_export_streams_large_json_to_file(self):
        spec = importlib.util.spec_from_file_location(
            "opencode_runner_export_test", RUNNER_PATH
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        payload = {"info": {"id": "s1"}, "content": "x" * 200_000}

        def fake_run(args, *, stdout, stderr, text, timeout):
            self.assertEqual(args, ["opencode", "export", "s1"])
            self.assertEqual(stderr, subprocess.PIPE)
            self.assertTrue(text)
            self.assertEqual(timeout, 30)
            json.dump(payload, stdout)
            return subprocess.CompletedProcess(args, 0, "", "")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner.subprocess, "run", side_effect=fake_run),
        ):
            state = Path(tmp)
            self.assertEqual(runner.export_session(state, "s1", 30), payload)
            self.assertGreater(
                (state / "session-export.json").stat().st_size, 64 * 1024
            )


class FakeRuntime(DockerRuntime):
    def __init__(self):
        self.calls = []
        self.turn_configs = []
        self.result = {"ok": True, "content": "TOKEN"}
        self.process = Mock()
        self.process.wait.return_value = 0

    def run(self, args, timeout=None):
        self.calls.append(list(args))
        stdout = ""
        if args[:2] == ["image", "inspect"]:
            stdout = "sha256:test"
        elif args[-1] == "--check":
            stdout = "1.18.30"
        elif args[-1].endswith("/result.json"):
            stdout = json.dumps(self.result)
        elif args[0] == "cp" and args[1].endswith(".opencode-turn.json"):
            self.turn_configs.append(json.loads(Path(args[1]).read_text()))
        elif args[0] == "cp" and args[1].endswith(
            ":/roller/opencode-state/trace.jsonl"
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


class OpenCodeHarnessTests(unittest.TestCase):
    def test_local_services_bypass_inherited_proxy(self):
        environment = {
            "NO_PROXY": "models.example.test,localhost",
            "no_proxy": "example.test",
        }

        _ensure_local_proxy_bypass(environment)

        expected = "localhost,127.0.0.1,::1,models.example.test,example.test"
        self.assertEqual(environment["NO_PROXY"], expected)
        self.assertEqual(environment["no_proxy"], expected)

    def test_attached_images_become_file_arguments_of_the_first_turn(self):
        task = Task("test", ("one", "two"), 0, attach_images=("inputs/chart.png",))
        runtime = FakeRuntime()
        harness = OpenCodeHarness(
            OpenCodeConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
            0,
            runtime,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
        ):
            harness.run(RunRequest(0, task, "model", 60), Path(tmp), 1)
            self.assertEqual(
                [turn.get("files") for turn in runtime.turn_configs],
                [["/roller/workspace/inputs/chart.png"], None],
            )
            blind = OpenCodeHarness(
                OpenCodeConfig(
                    api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off", vision=False
                ),
                0,
                FakeRuntime(),
            )
            result = blind.run(RunRequest(0, task, "model", 60), Path(tmp) / "blind", 1)
            self.assertEqual(result.status, "error")
            self.assertIn("--opencode-vision", result.error)

        runner = self.runner()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"OPENAI_BASE_URL": "http://example.test/v1"}),
            patch.object(
                runner.subprocess, "Popen", side_effect=RuntimeError("stop")
            ) as popen,
        ):
            config = {
                "timeout_seconds": 30,
                "model": "m",
                "context_length": 2048,
                "output_tokens": 512,
                "max_steps": 2,
                "message": "one",
                "files": ["/roller/workspace/inputs/chart.png"],
            }
            with self.assertRaisesRegex(RuntimeError, "stop"):
                runner.run_turn(config, Path(tmp))
        args = popen.call_args.args[0]
        self.assertEqual(
            args[args.index("--file") + 1], "/roller/workspace/inputs/chart.png"
        )

    def runner(self):
        spec = importlib.util.spec_from_file_location(
            "opencode_runner_tools", RUNNER_PATH
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        return runner

    def session(self, **kwargs):
        harness = OpenCodeHarness(OpenCodeConfig(**kwargs), 0, Mock())
        return harness.create_session(
            RunRequest(0, Task("test", ("one",), 0), "model", 60), Path("unused"), 1
        )

    def test_default_lock_settings_and_permissions_are_unchanged(self):
        tools = [
            "bash",
            "read",
            "write",
            "edit",
            "apply_patch",
            "glob",
            "grep",
            "skill",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.json"
            tasks.write_text(
                json.dumps([{"harness": "opencode", "model": "model", "prompt": "OK"}])
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
                "https://opencode.example/v1",
                "--api-key-env",
                "LOCAL_TEST_KEY",
            ]
            with (
                patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
                patch(
                    "roller.cli.collect._docker_preflight", return_value="sha256:test"
                ),
                patch("roller.cli.collect.Collector") as collector,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                collector.return_value.collect.return_value = []
                self.assertEqual(_collect(_build_parser().parse_args(command)), 0)
                with self.assertRaisesRegex(Exception, "lock|configuration|mismatch"):
                    _collect(
                        _build_parser().parse_args(
                            command + ["--resume", "--opencode-native-tools"]
                        )
                    )
            locked = json.loads((root / "runs/case/lock.json").read_text())[
                "execution"
            ]["opencode"]
        self.assertEqual(
            locked,
            {
                "docker_image": "roller-opencode:1.18.30",
                "container_user": "0",
                "network": "host",
                "api_base": "https://opencode.example/v1",
                "api_key_env": "LOCAL_TEST_KEY",
                "reasoning_effort": None,
                "max_steps": 12,
                "opencode_version": "1.18.30",
                "context_length": 128000,
                "output_tokens": 8192,
                "vision": True,
                "tools": tools,
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
                "compaction": {"auto": False, "prune": False},
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
                "vision": True,
                "tools": tools,
            },
        )
        self.assertEqual(
            session.turn_options(60),
            {
                "max_steps": 12,
                "context_length": 128000,
                "output_tokens": 8192,
                "vision": True,
            },
        )
        captured = self.session(
            capture_calls=True, api_base="http://localhost:1234/v1"
        ).settings()
        self.assertEqual(captured["tools"], [*tools, "task"])
        self.assertEqual(
            captured["subagent_profile_policy"], "explore-tool-restrictions/v1"
        )
        runner = self.runner()
        allowed = {name: "allow" for name in tools}
        self.assertEqual(
            runner.tool_permissions({}),
            {"*": "deny", **allowed, "external_directory": "allow"},
        )
        self.assertEqual(
            runner.tool_permissions({"capture_calls": True}),
            {"*": "deny", **allowed, "external_directory": "allow", "task": "allow"},
        )
        self.assertEqual(
            runner.agent_profiles({"max_steps": 12}),
            {
                "build": {"steps": 12},
                "title": {"disable": True},
                "summary": {"disable": True},
            },
        )
        self.assertEqual(
            runner.agent_profiles({"max_steps": 12, "capture_calls": True})["explore"],
            {
                "steps": 12,
                "permission": {
                    "write": "deny",
                    "edit": "deny",
                    "apply_patch": "deny",
                    "task": "deny",
                    "skill": "deny",
                },
            },
        )

    def test_native_tools_keep_subagents_behind_call_capture(self):
        runner = self.runner()
        for capture in (False, True):
            with self.subTest(capture_calls=capture):
                session = self.session(
                    native_tools=True,
                    deny_tools=("webfetch",),
                    capture_calls=capture,
                    api_base="http://localhost:1234/v1",
                )
                self.assertEqual(
                    {
                        key: session.settings()[key]
                        for key in ("tools", "tool_policy", "deny_tools")
                    },
                    {
                        "tools": None,
                        "tool_policy": "native",
                        "deny_tools": ["webfetch"],
                    },
                )
                turn = {"capture_calls": capture, **session.turn_options(60)}
                self.assertEqual(
                    (turn["native_tools"], turn["deny_tools"]), (True, ["webfetch"])
                )
                policy = runner.tool_permissions(turn)
                # OpenCode applies the last matching rule.
                self.assertEqual(list(policy)[0], "*")
                self.assertEqual(policy["*"], "allow")
                self.assertEqual(policy["webfetch"], "deny")
                self.assertEqual(
                    policy.get("task", policy["*"]), "allow" if capture else "deny"
                )
                agents = runner.agent_profiles(turn)
                if capture:
                    self.assertEqual(
                        agents["explore"]["permission"],
                        {
                            "*": "deny",
                            "bash": "allow",
                            "read": "allow",
                            "glob": "allow",
                            "grep": "allow",
                            "external_directory": "allow",
                            "webfetch": "deny",
                        },
                    )
                else:
                    self.assertNotIn("explore", agents)
        self.assertEqual(
            runner.tool_permissions(
                {"capture_calls": True, "native_tools": True, "deny_tools": ["task"]}
            )["task"],
            "deny",
        )
        with self.assertRaises(ValueError):
            OpenCodeConfig(deny_tools=("webfetch",))
        with self.assertRaises(ValueError):
            OpenCodeConfig(native_tools=True, deny_tools=("",))

    def test_skillsbench_prompt_preserves_upstream_output_paths(self):
        task = Task(
            "test",
            ("one",),
            0,
            raw={"bench": {"name": "skillsbench"}},
            output_dir=".roller_outputs",
        )
        prompt = _system_prompt(RunRequest(0, task, "model", 60), "/root")
        self.assertIn("exact requested output paths", prompt)
        self.assertNotIn(".roller_outputs", prompt)

    def test_task_image_cache_location_is_not_overridden(self):
        runtime = FakeRuntime()
        harness = OpenCodeHarness(
            OpenCodeConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
            0,
            runtime,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
        ):
            # A proxy configured on the host would replace the defaults checked below.
            for name in PROXY_ENV_NAMES:
                os.environ.pop(name, None)
            harness.run(
                RunRequest(0, Task("test", ("one",), 0), "model", 60),
                Path(tmp),
                1,
            )
        docker_run = next(call for call in runtime.calls if call[0] == "run")
        self.assertFalse(
            any(value.startswith("XDG_CACHE_HOME=") for value in docker_run)
        )
        self.assertIn("NO_PROXY=localhost,127.0.0.1,::1", docker_run)
        self.assertIn("no_proxy=localhost,127.0.0.1,::1", docker_run)

    def test_output_budget_reaches_the_native_cli(self):
        session = self.session(output_tokens=32768)
        environment = session.harness_environment("http://localhost:8000/v1")
        self.assertEqual(environment["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"], "32768")

    def test_session_outputs_cleanup_and_redaction(self):
        runtime = FakeRuntime()
        harness = OpenCodeHarness(
            OpenCodeConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
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
            self.assertTrue(runtime.turn_configs[0]["vision"])
            self.assertTrue(result.metadata["settings"]["vision"])
            self.assertEqual((Path(tmp) / "results/answer.txt").read_text(), "TOKEN")
            self.assertFalse((Path(tmp) / ".opencode-turn.json").exists())
            self.assertTrue(any(a[:2] == ["rm", "-f"] for a in runtime.calls))
            runtime.result = {"ok": False, "error": "failed test-secret"}
            result = harness.run(
                RunRequest(1, Task("error", ("one",), 0), "model", 60),
                Path(tmp) / "error",
                1,
            )
            self.assertEqual(result.status, "error")
            self.assertNotIn("test-secret", result.error)

    def test_vision_capability_can_be_disabled(self):
        runtime = FakeRuntime()
        harness = OpenCodeHarness(
            OpenCodeConfig(
                api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off", vision=False
            ),
            0,
            runtime,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
        ):
            result = harness.run(
                RunRequest(0, Task("test", ("one",), 0), "model", 60),
                Path(tmp),
                1,
            )
        self.assertEqual(result.status, "ok", result.error)
        self.assertFalse(runtime.turn_configs[0]["vision"])
        self.assertFalse(result.metadata["settings"]["vision"])

    def test_new_session_boundary_clears_only_conversation_state(self):
        runtime = FakeRuntime()
        harness = OpenCodeHarness(
            OpenCodeConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
            0,
            runtime,
        )
        task = Task(
            "test",
            ("one", "two"),
            0,
            new_session=(False, True),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
        ):
            result = harness.run(RunRequest(0, task, "model", 60), Path(tmp), 1)

        self.assertEqual(result.status, "ok", result.error)
        reset = [
            call
            for call in runtime.calls
            if call[-3:] == ["rm", "-f", "/roller/opencode-state/session.json"]
        ]
        self.assertEqual(len(reset), 1)

    def test_timeout_stops_container_and_recovers_outputs(self):
        runtime = FakeRuntime()
        runtime.process.wait.side_effect = [subprocess.TimeoutExpired("run", 1), 0]
        harness = OpenCodeHarness(
            OpenCodeConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
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
        harness = OpenCodeHarness(
            OpenCodeConfig(api_key_env="LOCAL_TEST_KEY", wire_trace_mode="off"),
            0,
            runtime,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "test-secret"}),
            patch(
                "roller.actf.opencode.convert_opencode_trace",
                side_effect=ValueError("invalid journal"),
            ),
        ):
            result = harness.run(
                RunRequest(0, Task("test", ("one",), 0), "model", 60), Path(tmp), 1
            )
            self.assertEqual(result.status, "ok")
            self.assertIsNone(result.error)
            self.assertTrue(
                any("invalid journal" in item for item in result.metadata["warnings"])
            )
            self.assertEqual(
                result.metadata["trajectory_conversion"],
                {
                    "status": "unavailable",
                    "target": "actf",
                    "error": "invalid journal",
                },
            )
            self.assertIsNone(result.trajectory_path)
            self.assertTrue((Path(tmp) / result.trace_path).is_file())
            self.assertEqual((Path(tmp) / "results/answer.txt").read_text(), "TOKEN")
            self.assertTrue(any(a[:2] == ["rm", "-f"] for a in runtime.calls))

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
                OpenCodeConfig(**settings)
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "tasks.json"
            tasks.write_text(
                json.dumps(
                    [
                        {
                            "id": "one",
                            "harness": "opencode",
                            "prompt": "hi",
                            "model": "model",
                        },
                        {
                            "id": "two",
                            "harness": "hermes",
                            "user_simulator": {
                                "harness": "opencode",
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
