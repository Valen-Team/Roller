"""Stirrup: journal projection, windows around summaries, and the in-process loop."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from roller.cli import _build_parser, _collect, _request_field
from roller.compaction_sft import build_compaction_segments, has_main_compaction
from roller.harnesses.docker import DockerRuntime
from roller.harnesses.stirrup import StirrupConfig, StirrupHarness, sample_seed
from roller.models import RunRequest, Task
from roller.sandbox import SandboxRun, SandboxTarget
from roller.sft import actf_to_sft, validate_sft_record
from roller.stirrup_actf import (
    STIRRUP_COMPACTION_SFT_REASON,
    extract_stirrup_usage,
    read_stirrup_native,
    stirrup_events_to_actf,
)

PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
    "AAAABJRU5ErkJggg=="
)
URL = "data:image/png;base64," + PNG
TOOLS = [
    {"type": "function", "function": {"name": name, "description": name}}
    for name in ("code_exec", "view_image", "finish")
]
SYSTEM = {"role": "system", "content": [{"type": "text", "text": "Stirrup base"}]}
TASK = {"role": "user", "content": [{"type": "text", "text": "Read a.png"}]}


def text(value: str) -> list[dict]:
    return [{"type": "text", "text": value}]


def assistant(call_id: str, name: str, arguments: dict, reasoning: str = "") -> dict:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }
    if reasoning:
        message["reasoning_content"] = reasoning
    return message


def tool(call_id: str, name: str, content: list[dict]) -> dict:
    return {"role": "tool", "content": content, "tool_call_id": call_id, "name": name}


VIEW = assistant("t1", "view_image", {"path": "a.png"}, "look first")
VIEWED = tool(
    "t1",
    "view_image",
    text("Viewing image at path: a.png")
    + text("Done! The User will provide the image for tool call t1"),
)
IMAGE = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Here is the image for tool call t1"},
        {"type": "image_url", "image_url": {"url": URL}},
    ],
}
FINISH = assistant("t2", "finish", {"reason": "done", "paths": []})


def journal(calls: list[dict]) -> list[dict]:
    """Journal rows as the agent writes them: each request adds what is new."""
    rows = [{"type": "session", "session_id": "s1", "timestamp": "t0"}]
    previous: list[dict] = []
    for index, call in enumerate(calls, 1):
        messages = call["messages"]
        base = 0
        while base < min(len(previous), len(messages)) and (
            previous[base] == messages[base]
        ):
            base += 1
        rows.append(
            {
                "type": "request",
                "id": call["id"],
                "purpose": call.get("purpose", "main"),
                "base": base,
                "messages": messages[base:],
                **({"tools": TOOLS} if index == 1 else {}),
                "timestamp": f"2026-11-21T00:00:{index:02d}+00:00",
            }
        )
        reply = call.get("reply")
        rows.append(
            {
                "type": "response",
                "id": call["id"],
                "outcome": call["outcome"],
                **({"message": reply, "finish_reason": "stop"} if reply else {}),
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "timestamp": f"2026-11-21T00:00:{index:02d}.400000+00:00",
            }
        )
        previous = [*messages, reply] if reply else messages
        for row in call.get("after", []):
            rows.append(
                {**row, "timestamp": f"2026-11-21T00:00:{index:02d}.500000+00:00"}
            )
    return rows


def tool_row(message: dict, success: bool = True) -> dict:
    return {
        "type": "tool",
        "tool_call_id": message["tool_call_id"],
        "name": message["name"],
        "success": success,
        "content": message["content"],
    }


LINEAR = [
    {
        "id": "call-0001",
        "messages": [SYSTEM, TASK],
        "reply": VIEW,
        "outcome": "step",
        "after": [tool_row(VIEWED)],
    },
    {
        "id": "call-0002",
        "messages": [SYSTEM, TASK, VIEW, VIEWED, IMAGE],
        "outcome": "retry",
    },
    {
        "id": "call-0003",
        "messages": [SYSTEM, TASK, VIEW, VIEWED, IMAGE],
        "reply": FINISH,
        "outcome": "step",
        "after": [tool_row(tool("t2", "finish", text("done")))],
    },
]


class StirrupJournalTests(unittest.TestCase):
    def test_actf_keeps_the_moved_image_and_skips_the_retried_call(self):
        rows = journal(LINEAR)
        actf = stirrup_events_to_actf(rows)
        self.assertEqual(actf["system_prompt"], "Stirrup base")
        self.assertEqual(actf["tools"], TOOLS)
        self.assertEqual(len(actf["steps"]), 2)
        self.assertEqual(actf["steps"][1]["extra"], {"call_id": "call-0003"})
        record = actf_to_sft(actf)
        validate_sft_record(record)
        self.assertEqual(
            [message["role"] for message in record["messages"]],
            ["system", "user", "assistant", "tool", "user", "assistant", "tool"],
        )
        self.assertEqual(record["messages"][2]["reasoning_content"], "look first")
        self.assertEqual(
            record["messages"][4]["content"],
            "Here is the image for tool call t1\n<image>",
        )
        self.assertEqual(record["images"], [URL])
        self.assertNotIn("sft_unavailable_reason", actf)
        without = stirrup_events_to_actf(rows, keep_thinking=False)
        self.assertEqual(
            without["steps"][0]["assistant_content"]["reasoning_content"], ""
        )

    def test_usage_and_capture_participant(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "native" / "trace.jsonl"
            trace.parent.mkdir()
            trace.write_text("".join(json.dumps(row) + "\n" for row in journal(LINEAR)))
            usage = extract_stirrup_usage(trace)
            self.assertEqual(usage["request_count"], 3)
            self.assertEqual(usage["input_tokens"], 30)
            native = read_stirrup_native(Path(tmp))
        self.assertEqual(list(native.participants), ["session:s1"])
        self.assertEqual(native.participants["session:s1"]["role"], "root")
        self.assertEqual(native.expected_call_counts["session:s1"], 2)


SUMMARIZE = {"role": "user", "content": text("The context window is approaching")}
BRIDGE = {"role": "user", "content": text("**Context Continuation** SUMMARY")}
ACK = {"role": "user", "content": text("Got it, thanks!")}
SUMMARY = {
    "role": "assistant",
    "content": text("SUMMARY"),
    "reasoning_content": "sum up",
}
COMPACTED = [
    {
        "id": "call-0001",
        "messages": [SYSTEM, TASK],
        "reply": VIEW,
        "outcome": "step",
        "after": [tool_row(VIEWED)],
    },
    {
        "id": "call-0002",
        "messages": [SYSTEM, TASK, VIEW, VIEWED, IMAGE],
        "outcome": "retry",
    },
    {
        "id": "call-0003",
        "messages": [SYSTEM, TASK, VIEW, VIEWED, IMAGE, SUMMARIZE],
        "purpose": "summary",
        "reply": SUMMARY,
        "outcome": "summary",
        "after": [{"type": "summary", "id": "summary-0001", "summary": "SUMMARY"}],
    },
    {
        "id": "call-0004",
        "messages": [SYSTEM, TASK, BRIDGE, ACK],
        "reply": FINISH,
        "outcome": "step",
        "after": [tool_row(tool("t2", "finish", text("done")))],
    },
]


def write_compacted_sample(directory: Path) -> str:
    rows = journal(COMPACTED)
    (directory / "trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    actf = stirrup_events_to_actf(rows)
    (directory / "trace_ACTF.json").write_text(json.dumps(actf))
    calls = []
    for index, call in enumerate(COMPACTED, 1):
        reply = call.get("reply")
        output = None
        if reply:
            message = dict(reply)
            # vLLM names the thinking ``reasoning``.
            if "reasoning_content" in message:
                message["reasoning"] = message.pop("reasoning_content")
            output = {"choices": [{"message": message, "finish_reason": "stop"}]}
        calls.append(
            {
                "call_id": f"c{index}",
                "request_seq": index,
                "participant_id": "session:s1",
                "protocol": "openai-chat-completions",
                "context": {"messages": call["messages"], "tools": TOOLS},
                "output": output,
                "native_identity": {"session_id": "s1", "run_id": call["id"]},
                "export_status": "available" if reply else "unavailable",
                "started_at": f"2026-11-21T00:00:{index:02d}Z",
            }
        )
    capture = directory / "capture" / "attempt-0001"
    capture.mkdir(parents=True)
    (capture / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (capture / "manifest.json").write_text(
        json.dumps({"participants": [{"participant_id": "session:s1", "role": "root"}]})
    )
    return "capture/attempt-0001/manifest.json"


class StirrupCompactionTests(unittest.TestCase):
    def test_summary_request_is_a_window_of_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp)
            manifest = write_compacted_sample(sample)
            actf = json.loads((sample / "trace_ACTF.json").read_text())
            self.assertEqual(
                actf["sft_unavailable_reason"], STIRRUP_COMPACTION_SFT_REASON
            )
            self.assertTrue(has_main_compaction(sample / "trace.jsonl", "stirrup"))
            segments = build_compaction_segments(sample, "stirrup", manifest)
            without = build_compaction_segments(
                sample, "stirrup", manifest, keep_thinking=False
            )
        self.assertEqual(
            [meta["boundary_source"] for meta, _ in segments],
            ["initial", "summary_request", "native_boundary"],
        )
        first, summary, after = (record for _, record in segments)
        self.assertEqual(first["target_start_index"], 2)
        self.assertNotIn("images", first)
        self.assertEqual(summary["target_start_index"], 6)
        self.assertEqual(summary["images"], [URL])
        self.assertEqual(
            summary["messages"][4]["content"],
            "Here is the image for tool call t1\n<image>",
        )
        self.assertEqual(summary["messages"][-1]["content"], "SUMMARY")
        self.assertEqual(summary["messages"][-1]["reasoning_content"], "sum up")
        self.assertEqual(after["target_start_index"], 4)
        self.assertEqual(segments[2][0]["boundary_event_id"], "summary-0001")
        self.assertEqual(segments[0][0]["first_call_id"], "c1")
        self.assertEqual(without[1][1]["messages"][-1]["reasoning_content"], "")


class StirrupConfigTests(unittest.TestCase):
    def test_request_body_order_and_sample_seed(self):
        config = StirrupConfig(
            temperature=1.0,
            top_p=0.95,
            output_tokens=65536,
            context_length=98304,
            reasoning_effort="max",
            seed=42,
            request_fields=(("n", 1), ("stream", False)),
        )
        harness = StirrupHarness(config, 0, DockerRuntime())
        request = RunRequest(0, Task("q1", ("p",), 0), "glm", 60, sample_index=2)
        body = harness.create_session(request, Path("unused"), 1).request_body()
        self.assertEqual(
            list(body),
            [
                "temperature",
                "top_p",
                "max_tokens",
                "chat_template_kwargs",
                "model",
                "seed",
                "n",
                "stream",
            ],
        )
        self.assertEqual(body["chat_template_kwargs"], {"reasoning_effort": "max"})
        self.assertEqual(body["seed"], sample_seed(42, "q1", 2))
        self.assertNotEqual(sample_seed(42, "q1", 1), sample_seed(42, "q1", 2))
        self.assertEqual(_request_field("stream=false"), ("stream", False))

    def test_cli_accepts_remote_sandboxes_and_names_the_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "tasks.jsonl"
            tasks.write_text(json.dumps({"id": "t", "prompt": "p"}) + "\n")
            argv = [
                "collect",
                "--tasks",
                str(tasks),
                "--harness",
                "stirrup",
                "--model",
                "m",
                "--output",
                tmp,
                "--sandbox-provider",
                "test",
                "--sandbox-template",
                "runtime",
                "--dry-run",
            ]
            parser = _build_parser()
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "k"}),
                patch("roller.cli._stirrup_installed", return_value=False),
                self.assertRaisesRegex(ValueError, r"roller-agent\[stirrup\]"),
            ):
                _collect(parser.parse_args(argv))
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "k"}),
                patch("roller.cli._stirrup_installed", return_value=True),
                patch("sys.stdout"),
            ):
                self.assertEqual(_collect(parser.parse_args(argv)), 0)


class ScriptedModel(BaseHTTPRequestHandler):
    """Answers each request with the next scripted (status, body) pair."""

    script: list[tuple[int, dict]] = []
    bodies: list[bytes] = []
    headers_seen: list[dict] = []

    def log_message(self, *args):
        return None

    def do_POST(self):
        self.bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).headers_seen.append(dict(self.headers))
        status, body = self.script.pop(0)
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def reply(message: dict, finish: str = "stop", usage: dict | None = None) -> dict:
    return {
        "id": "r",
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 10},
    }


def call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ModelServer:
    def __init__(self, script: list[tuple[int, dict]]):
        ScriptedModel.script, ScriptedModel.bodies = list(script), []
        ScriptedModel.headers_seen = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedModel)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def answer(argv: list[str]) -> str:
    """What bash prints for the backend's commands in the task container."""
    if argv[-1].startswith("d=/tmp/.roller-exec"):
        script = argv[-1]
        code, out = 0, ""
        if "test -d" in script:
            code = 1
        elif "sleep" in script:
            code = 124
        elif "python3" in script:
            out = "391\n"
        return f"{code}:{base64.b64encode(out.encode()).decode()}:"
    return PNG if argv[0] == "base64" else ""


class FakeContainer(DockerRuntime):
    """Docker calls of one session."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def run(self, args, timeout=None):
        args = list(args)
        self.calls.append(args)
        stdout = ""
        if args[:2] == ["image", "inspect"]:
            stdout = "sha256:test"
        elif args[0] == "exec":
            stdout = answer(args[2:])
        elif args[0] == "cp" and args[1].endswith("/outputs/."):
            (Path(args[2]) / "answer.txt").write_text("391")
        return subprocess.CompletedProcess(args, 0, stdout, "")


class FakeSandbox:
    """A remote sandbox: commands run there, outputs come back by download."""

    version = "runtime-1"

    def __init__(self):
        self.runs: list[list[str]] = []
        self.envs: list[dict] = []

    def run(self, argv, *, cwd=None, env=None, timeout):
        self.runs.append(list(argv))
        self.envs.append(dict(env or {}))
        return subprocess.CompletedProcess(argv, 0, answer(list(argv)), "")

    def spawn(self, argv, *, cwd=None, env=None, log_path):
        raise AssertionError("the agent runs in the collecting process")

    def upload(self, source, target, *, contents=False, timeout):
        self.runs.append(["upload", target])

    def download(self, source, target, *, contents=False, timeout):
        if source.endswith("/outputs"):
            (Path(target) / "answer.txt").write_text("391")

    def close(self):
        self.runs.append(["close"])


class FakeProvider:
    def __init__(self, box):
        self.box = box

    def create(self, template, version, *, name, lifetime, timeout, allocated):
        allocated("instance")
        return self.box

    def remove(self, instance_id):
        return None


def episode_script() -> list[tuple[int, dict]]:
    return [
        (
            200,
            reply(
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "view it",
                    "tool_calls": [call("a", "view_image", {"path": "a.png"})],
                },
                "tool_calls",
            ),
        ),
        (503, {"error": "busy"}),
        (
            200,
            reply(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        call("b", "code_exec", {"cmd": "python3 solve.py"}),
                        call("c", "code_exec", {"cmd": "sleep 999"}),
                    ],
                },
                "tool_calls",
            ),
        ),
        (
            200,
            reply(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [call("d", "finish", {"reason": "391", "paths": []})],
                },
                "tool_calls",
            ),
        ),
    ]


class StirrupLoopTests(unittest.TestCase):
    def setUp(self):
        pytest.importorskip("stirrup")

    def test_episode_in_process_with_the_container_as_code_backend(self):
        server = ModelServer(episode_script())
        self.addCleanup(server.close)
        runtime = FakeContainer()
        config = StirrupConfig(
            api_base=server.base,
            api_key_env="LOCAL_TEST_KEY",
            temperature=1.0,
            reasoning_effort="max",
            seed=7,
            command_timeout=5,
            request_fields=(("n", 1),),
            capture_calls=True,
            wire_trace_mode="full",
        )
        harness = StirrupHarness(config, 0, runtime)
        task = Task("boxes", ("Read a.png",), 0)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_TEST_KEY": "secret"}),
            patch("roller.stirrup_agent.time.sleep"),
        ):
            result = harness.run(RunRequest(0, task, "m", 120), Path(tmp), 1)
            self.assertEqual(result.status, "ok", result.error)
            self.assertEqual(result.metadata["stop_reason"], "finish")
            self.assertTrue(result.metadata["capture"]["manifest"])
            actf = json.loads((Path(tmp) / result.trajectory_path).read_text())
            record = actf_to_sft(actf)
            wire = (Path(tmp) / "wire_trace.jsonl").read_text()
            captured = [
                json.loads(line)
                for line in (Path(tmp) / "capture/attempt-0001/calls.jsonl").open()
            ]
            self.assertEqual((Path(tmp) / "results/answer.txt").read_text(), "391")
        # The model key stays in this process; the container never sees it.
        self.assertFalse(any("secret" in " ".join(args) for args in runtime.calls))
        self.assertIn('"seq": 1', wire)
        # Each captured Call carries the journal's call ID; the proxy drops it.
        self.assertEqual(
            [call["native_identity"]["run_id"] for call in captured],
            ["call-0001", "call-0002", "call-0003", "call-0004"],
        )
        self.assertNotIn("x-roller", json.dumps(ScriptedModel.headers_seen).lower())
        first = json.loads(ScriptedModel.bodies[0])
        self.assertEqual(
            list(first),
            [
                "temperature",
                "max_tokens",
                "chat_template_kwargs",
                "model",
                "seed",
                "n",
                "messages",
                "tools",
                "tool_choice",
            ],
        )
        self.assertEqual(ScriptedModel.bodies[0], json.dumps(first).encode())
        self.assertEqual(ScriptedModel.bodies[1], ScriptedModel.bodies[2])
        self.assertEqual(
            [tool["function"]["name"] for tool in first["tools"]],
            ["finish", "code_exec", "view_image"],
        )
        self.assertEqual(len(actf["steps"]), 3)
        self.assertEqual(
            actf["steps"][0]["assistant_content"]["reasoning_content"], "view it"
        )
        # Stirrup sends every image as PNG of at most one megapixel.
        self.assertEqual(len(record["images"]), 1)
        self.assertTrue(record["images"][0].startswith("data:image/png;base64,"))
        third = json.loads(ScriptedModel.bodies[3])["messages"]
        self.assertIn(
            "Command timed out after 5 seconds",
            third[-1]["content"][0]["text"],
        )
        self.assertIn("391", third[-2]["content"][0]["text"])

    def test_remote_sandbox_shares_the_backend_and_the_local_proxy(self):
        server = ModelServer(episode_script())
        self.addCleanup(server.close)
        box = FakeSandbox()
        with tempfile.TemporaryDirectory() as tmp:
            sandbox_run = SandboxRun(
                Path(tmp) / "run",
                SandboxTarget("test", "runtime", "runtime-1"),
                FakeProvider(box),
            )
            config = StirrupConfig(
                api_base=server.base,
                api_key_env="LOCAL_TEST_KEY",
                command_timeout=5,
                capture_calls=True,
                token_ids=False,
                wire_trace_mode="full",
                remote_sandbox=True,
                network=None,
                sandbox_run=sandbox_run,
            )
            harness = StirrupHarness(config, 0, FakeContainer())
            task = Task("boxes", ("Read a.png",), 0)
            sample = Path(tmp) / "sample"
            with (
                patch.dict(os.environ, {"LOCAL_TEST_KEY": "secret"}),
                patch("roller.stirrup_agent.time.sleep"),
            ):
                result = harness.run(RunRequest(0, task, "m", 120), sample, 1)
            self.assertEqual(result.status, "ok", result.error)
            self.assertEqual((sample / "results/answer.txt").read_text(), "391")
            self.assertTrue((sample / "wire_trace.jsonl").is_file())
            self.assertEqual(
                result.metadata["provenance"]["sandbox_version"], "runtime-1"
            )
        self.assertEqual(len(ScriptedModel.bodies), 4)
        self.assertEqual(box.runs[-1], ["close"])
        self.assertTrue(any(argv[0] == "base64" for argv in box.runs))
        # No recorder or model key in the sandbox: the model is called from here.
        self.assertFalse(any("sandbox_wire" in " ".join(argv) for argv in box.runs))
        self.assertFalse(any("secret" in json.dumps(env) for env in box.envs))

    def test_client_overflow_output_limit_and_reasoning(self):
        from stirrup.core.exceptions import (
            ContextOverflowError,
            OutputTokenLimitError,
        )
        from stirrup.core.models import SystemMessage, UserMessage

        from roller.stirrup_agent import ChatClient, Journal

        server = ModelServer(
            [
                (400, {"error": "This model's maximum context length is 98304 tokens"}),
                (
                    200,
                    reply(
                        {"role": "assistant", "content": "x"},
                        "length",
                        {"prompt_tokens": 900, "completion_tokens": 90},
                    ),
                ),
                (
                    200,
                    reply(
                        {"role": "assistant", "content": "x"},
                        "length",
                        {"prompt_tokens": 100, "completion_tokens": 90},
                    ),
                ),
            ]
        )
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(Path(tmp) / "trace.jsonl")
            client = ChatClient(
                api_base=server.base,
                api_key="",
                body={"max_tokens": 90, "model": "m"},
                context_window=1000,
                timeout=30,
                attempts=3,
                journal=journal,
                session_id=None,
                remaining=lambda: 60.0,
            )
            messages = [SystemMessage(content="s"), UserMessage(content="u")]
            for error in (
                ContextOverflowError,
                ContextOverflowError,
                OutputTokenLimitError,
            ):
                with self.assertRaises(error):
                    asyncio.run(client.generate(messages, {}))
            rows = [
                json.loads(line)
                for line in journal.path.read_text().splitlines()
                if '"response"' in line
            ]
        self.assertEqual(
            [row["outcome"] for row in rows],
            ["context_overflow", "context_overflow", "output_limit"],
        )
