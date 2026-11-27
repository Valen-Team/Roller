from __future__ import annotations

import gzip
import json
import tempfile
import threading
import unittest
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from roller.actf.claude_code import claude_code_events_to_actf
from roller.collector import Collector
from roller.harnesses.claude_code import (
    TOOL_ID_MAPPINGS_NAME,
    ClaudeCodeConfig,
    ClaudeCodeHarness,
)
from roller.sft import actf_to_sft
from roller.tool_id_identity import ToolIdRegistry, scope_claude_tool_ids
from roller.wire_trace import WireTraceProxy


def _call(message_id: str, call_id: str, second: int) -> dict:
    return {
        "type": "assistant",
        "uuid": f"assistant-{message_id}",
        "timestamp": f"2026-08-22T01:00:{second:02d}Z",
        "message": {
            "id": message_id,
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": "Read",
                    "input": {"file_path": message_id},
                },
            ],
            "stop_reason": "tool_use",
        },
    }


def _result(name: str, call_id: str, second: int) -> dict:
    return {
        "type": "user",
        "uuid": f"result-{name}",
        "timestamp": f"2026-08-22T01:00:{second:02d}Z",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"content of {name}",
                },
            ],
        },
    }


def _session() -> list[dict]:
    return [
        {
            "type": "user",
            "uuid": "user-1",
            "timestamp": "2026-08-22T01:00:00Z",
            "message": {"role": "user", "content": "read both files"},
        },
        _call("msg-1", "call_0", 1),
        _result("msg-1", "call_0", 2),
        _call("msg-2", "call_0", 3),
        _result("msg-2", "call_0", 4),
        {
            "type": "assistant",
            "uuid": "assistant-3",
            "timestamp": "2026-08-22T01:00:05Z",
            "message": {
                "id": "msg-3",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "Done."}],
                "stop_reason": "end_turn",
            },
        },
    ]


class ScopeClaudeToolIdsTests(unittest.TestCase):
    def test_sequential_reuse_keeps_both_calls_and_their_results(self) -> None:
        events = _session()
        original = deepcopy(events)
        trajectory = claude_code_events_to_actf(events)
        self.assertEqual(events, original)

        calls = [
            call
            for step in trajectory["steps"]
            for call in step["assistant_content"]["tool_calls"]
        ]
        results = [item for step in trajectory["steps"] for item in step["observation"]]
        self.assertEqual(
            [call["id"] for call in calls], ["call_0", "call_0__occurrence_2"]
        )
        self.assertEqual(
            [call["arguments"]["file_path"] for call in calls], ["msg-1", "msg-2"]
        )
        self.assertNotIn("original_tool_call_id", calls[0])
        self.assertEqual(calls[1]["original_tool_call_id"], "call_0")
        self.assertEqual(
            [item["toolCallId"] for item in results], ["call_0", "call_0__occurrence_2"]
        )
        self.assertEqual(
            [item["content"][0]["text"] for item in results],
            ["content of msg-1", "content of msg-2"],
        )
        record = actf_to_sft(trajectory)
        self.assertEqual(
            [
                message["tool_call_id"]
                for message in record["messages"]
                if message["role"] == "tool"
            ],
            ["call_0", "call_0__occurrence_2"],
        )

    def test_unique_ids_and_split_messages_are_returned_unchanged(self) -> None:
        events = _session()
        events[3] = _call("msg-2", "call_1", 3)
        events[4] = _result("msg-2", "call_1", 4)
        # Claude Code writes one row per content block of the same message.
        events.insert(
            2, {**_call("msg-1", "call_0", 1), "uuid": "assistant-msg-1-again"}
        )
        scoped = scope_claude_tool_ids(events)
        self.assertTrue(all(before is after for before, after in zip(events, scoped)))

    def test_generated_id_avoids_native_ids_and_sidechains_are_left_alone(self) -> None:
        events = _session()
        events.insert(5, _call("msg-2b", "call_0__occurrence_2", 4))
        events.insert(6, _result("msg-2b", "call_0__occurrence_2", 4))
        events.append({**_call("child", "call_0", 6), "isSidechain": True})
        scoped = scope_claude_tool_ids(events)
        self.assertEqual(
            scoped[3]["message"]["content"][0]["id"], "call_0__occurrence_3"
        )
        self.assertEqual(
            scoped[4]["message"]["content"][0]["tool_use_id"], "call_0__occurrence_3"
        )
        self.assertEqual(
            scoped[4]["message"]["content"][0]["roller_original_tool_id"], "call_0"
        )
        self.assertIs(scoped[-1], events[-1])

    def test_reuse_before_the_earlier_result_is_rejected(self) -> None:
        events = _session()
        events[2], events[3] = (
            _call("msg-2", "call_0", 2),
            _result("msg-1", "call_0", 3),
        )
        with self.assertRaisesRegex(ValueError, "reused before its result: call_0"):
            claude_code_events_to_actf(events)


class ToolIdRegistryTests(unittest.TestCase):
    def test_only_repeated_tool_use_ids_are_replaced(self) -> None:
        registry = ToolIdRegistry()
        start = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "call_0",
                "name": "Read",
                "input": {},
            },
        }
        first = deepcopy(start)
        self.assertEqual(registry.repair(first), [])
        self.assertEqual(first, start)

        message = {
            "type": "message",
            "content": [
                {"type": "text", "text": "call_0"},
                {"type": "tool_use", "id": "call_0", "name": "Read", "input": {}},
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {}},
            ],
        }
        mappings = registry.repair(message)
        delivered = message["content"][1]["id"]
        self.assertEqual(mappings, [{"native_id": "call_0", "delivered_id": delivered}])
        self.assertRegex(delivered, r"^toolu_roller_[0-9a-f]{32}$")
        self.assertEqual(message["content"][2]["id"], "call_1")
        for other in (
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "tool_use"},
            },
            {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            },
            [],
        ):
            self.assertEqual(registry.repair(other), [])


class ToolIdRepairProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.responses: list[tuple[str, bytes, str]] = []
        self.request_headers: list[dict[str, str]] = []
        owner = self

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                owner.request_headers.append(
                    {key.lower(): value for key, value in self.headers.items()}
                )
                content_type, payload, encoding = owner.responses.pop(0)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                if encoding:
                    self.send_header("Content-Encoding", encoding)
                self.end_headers()
                self.wfile.write(payload)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream.daemon_threads = True
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        self.upstream = f"http://127.0.0.1:{upstream.server_port}"

    def start(self, **config) -> WireTraceProxy:
        harness = ClaudeCodeHarness(
            ClaudeCodeConfig(api_base=self.upstream, **config), 0
        )
        harness._start_wire_proxy(self.root)
        self.addCleanup(harness._stop_wire_proxy)
        return harness._wire_proxy

    def post(
        self, proxy: WireTraceProxy, path: str = "/v1/messages?beta=true"
    ) -> bytes:
        request = Request(
            proxy.proxy_base_url + path,
            data=b'{"model":"m","messages":[]}',
            headers={"Content-Type": "application/json", "Accept-Encoding": "gzip"},
        )
        with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
            return response.read()

    @staticmethod
    def stream(call_id: str) -> bytes:
        events = [
            {"type": "message_start", "message": {"id": "msg", "content": []}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": "Read",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"tool_use": 1}',
                },
            },
            {"type": "message_stop"},
        ]
        return "".join(
            f"event: {event['type']}\r\ndata: {json.dumps(event)}\r\n\r\n"
            for event in events
        ).encode()

    def test_streamed_and_json_responses_get_unique_ids_and_a_mapping_row(self) -> None:
        proxy = self.start(unique_tool_ids=True)
        message = json.dumps(
            {
                "type": "message",
                "content": [
                    {"type": "tool_use", "id": "call_0", "name": "Read", "input": {}}
                ],
            }
        ).encode()
        self.responses = [
            ("text/event-stream", self.stream("call_0"), ""),
            ("text/event-stream", self.stream("call_0"), ""),
            ("application/json", message, ""),
        ]
        first, second, third = (self.post(proxy) for _ in range(3))

        self.assertEqual(first, self.stream("call_0"))
        lines = second.split(b"\r\n")
        repaired = json.loads(lines[4][len(b"data: ") :])
        delivered = repaired["content_block"]["id"]
        self.assertRegex(delivered, r"^toolu_roller_[0-9a-f]{32}$")
        untouched = self.stream("call_0").split(b"\r\n")
        self.assertEqual(lines[:4] + lines[5:], untouched[:4] + untouched[5:])
        third_id = json.loads(third)["content"][0]["id"]
        self.assertNotIn(third_id, {"call_0", delivered})

        rows = [
            json.loads(line)
            for line in (self.root / TOOL_ID_MAPPINGS_NAME).read_text().splitlines()
        ]
        self.assertEqual(
            rows,
            [
                {"native_id": "call_0", "delivered_id": delivered},
                {"native_id": "call_0", "delivered_id": third_id},
            ],
        )
        self.assertTrue(
            all(
                headers["accept-encoding"] == "identity"
                for headers in self.request_headers
            )
        )

    def test_other_routes_and_compressed_responses_pass_through(self) -> None:
        proxy = self.start(unique_tool_ids=True)
        body = self.stream("call_0")
        self.responses = [
            ("text/event-stream", body, ""),
            ("text/event-stream", body, ""),
            ("text/event-stream", gzip.compress(body), "gzip"),
        ]
        self.assertEqual(self.post(proxy), body)
        self.assertEqual(self.post(proxy, "/v1/messages/count_tokens"), body)
        self.assertEqual(gzip.decompress(self.post(proxy)), body)
        self.assertFalse((self.root / TOOL_ID_MAPPINGS_NAME).exists())

    def test_without_the_flag_responses_and_request_headers_are_untouched(self) -> None:
        proxy = self.start()
        self.assertIsNone(proxy.tool_id_mappings_path)
        body = self.stream("call_0")
        self.responses = [
            ("text/event-stream", body, ""),
            ("text/event-stream", body, ""),
        ]
        self.assertEqual([self.post(proxy), self.post(proxy)], [body, body])
        self.assertEqual(self.request_headers[0]["accept-encoding"], "gzip")
        self.assertFalse((self.root / TOOL_ID_MAPPINGS_NAME).exists())

    def test_mapping_file_survives_sample_pruning(self) -> None:
        sample = self.root / "sample"
        sample.mkdir()
        for name in (TOOL_ID_MAPPINGS_NAME, "trace.jsonl", "claude.log"):
            (sample / name).write_text("{}\n")
        Collector._prune_sample(sample)
        self.assertEqual(
            sorted(path.name for path in sample.iterdir()),
            [TOOL_ID_MAPPINGS_NAME, "trace.jsonl"],
        )

    def test_repair_needs_the_request_proxy(self) -> None:
        for options in (
            {},
            {"api_base": "http://example.test", "wire_trace_mode": "off"},
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "tool ID repair requires"),
            ):
                ClaudeCodeConfig(unique_tool_ids=True, **options)


if __name__ == "__main__":
    unittest.main()
