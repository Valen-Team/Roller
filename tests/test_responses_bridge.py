from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from roller import responses_bridge
from roller.capture import read_jsonl
from roller.responses_bridge import Stream, chat_request
from roller.wire_trace import WireTraceProxy

EXEC = {
    "type": "function",
    "name": "exec_command",
    "description": "Run a command",
    "strict": False,
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
}
SPAWN = {"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}}

RESPONSES_REQUEST = {
    "model": "glm",
    "instructions": "You are Codex.",
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "reasoning": {"effort": "high", "summary": "auto"},
    "store": False,
    "stream": True,
    "include": ["reasoning.encrypted_content"],
    "tools": [
        EXEC,
        {"type": "namespace", "name": "multi_agent_v1", "tools": [SPAWN]},
        {"type": "web_search"},
    ],
    "input": [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "<permissions>"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "# AGENTS.md"},
                {"type": "input_text", "text": "List the files."},
            ],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "I will run ls."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Looking."}],
        },
        {
            "type": "function_call",
            "name": "exec_command",
            "arguments": '{"cmd": "ls"}',
            "call_id": "call-1",
        },
        {
            "type": "function_call",
            "name": "spawn_agent",
            "namespace": "multi_agent_v1",
            "arguments": "{}",
            "call_id": "call-2",
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "a.png"},
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": [
                {"type": "input_text", "text": "an image"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "A square."}],
        },
    ],
}

# What vLLM streams for a response with reasoning, text and two tool calls.
CHUNKS = [
    {
        "id": "chat-1",
        "model": "glm",
        "prompt_token_ids": [1, 2, 3],
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {"reasoning": "Hm"},
                "token_ids": [10, 11],
                "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.25}]},
            }
        ]
    },
    {"choices": [{"index": 0, "delta": {"reasoning": "m."}, "token_ids": [12]}]},
    {"choices": [{"index": 0, "delta": {"content": "Looking."}, "token_ids": [13]}]},
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-3",
                            "type": "function",
                            "function": {"name": "exec_command", "arguments": ""},
                        }
                    ]
                },
                "token_ids": [14],
            }
        ]
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"cmd": "ls"}'}}
                    ]
                },
                "token_ids": [15],
            }
        ]
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call-4",
                            "type": "function",
                            "function": {
                                "name": "multi_agent_v1__spawn_agent",
                                "arguments": "{}",
                            },
                        }
                    ]
                },
                "finish_reason": "tool_calls",
                "token_ids": [16],
            }
        ]
    },
    {
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
    },
]


class ChatRequestTests(unittest.TestCase):
    def test_the_items_of_one_response_become_one_assistant_message(self):
        request = chat_request(RESPONSES_REQUEST)

        self.assertEqual(
            request["messages"],
            [
                {"role": "system", "content": "You are Codex."},
                {
                    "role": "developer",
                    "content": [{"type": "text", "text": "<permissions>"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "# AGENTS.md"},
                        {"type": "text", "text": "List the files."},
                    ],
                },
                {
                    "role": "assistant",
                    "reasoning": "I will run ls.",
                    "content": "Looking.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": '{"cmd": "ls"}',
                            },
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "multi_agent_v1__spawn_agent",
                                "arguments": "{}",
                            },
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "a.png"},
                {
                    "role": "tool",
                    "tool_call_id": "call-2",
                    "content": [
                        {"type": "text", "text": "an image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                },
                {"role": "assistant", "content": "A square."},
            ],
        )

    def test_function_tools_are_sent_and_namespaces_are_flattened(self):
        request = chat_request(RESPONSES_REQUEST)

        self.assertEqual(
            request["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "description": "Run a command",
                        "strict": False,
                        "parameters": EXEC["parameters"],
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "multi_agent_v1__spawn_agent",
                        "parameters": {"type": "object"},
                    },
                },
            ],
        )
        self.assertEqual(
            {
                key: value
                for key, value in request.items()
                if key not in ("messages", "tools")
            },
            {
                "model": "glm",
                "stream": True,
                "stream_options": {"include_usage": True},
                "reasoning_effort": "high",
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
        )


class StreamTests(unittest.TestCase):
    def test_each_output_item_is_opened_filled_and_closed(self):
        stream = Stream(RESPONSES_REQUEST)
        events = [event for chunk in CHUNKS for event in stream.feed(chunk)]
        events += stream.finish()

        self.assertEqual(
            [event["type"] for event in events],
            [
                "response.created",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                "response.output_item.done",
                "response.output_item.added",
                "response.output_text.delta",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(events[1]["item"]["content"][0]["text"], "")
        response = events[-1]["response"]
        reasoning, message, first, second = response["output"]
        self.assertEqual(
            reasoning["content"], [{"type": "reasoning_text", "text": "Hmm."}]
        )
        self.assertEqual(message["content"][0]["text"], "Looking.")
        self.assertEqual(
            (first["call_id"], first["name"], first["arguments"]),
            ("call-3", "exec_command", '{"cmd": "ls"}'),
        )
        self.assertEqual(
            (second["name"], second["namespace"]), ("spawn_agent", "multi_agent_v1")
        )
        self.assertEqual(
            [event["output_index"] for event in events[1:-1:1] if "item" in event],
            [0, 0, 1, 1, 2, 2, 3, 3],
        )
        self.assertEqual(
            (response["usage"]["input_tokens"], response["usage"]["output_tokens"]),
            (3, 7),
        )


class BridgedProxyTests(unittest.TestCase):
    """Codex served by a Chat Completions upstream."""

    def test_stream_is_translated_and_token_ids_are_saved(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        requests = []
        stream = b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in CHUNKS)

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                requests.append((self.path, json.loads(body)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(stream) + 14))
                self.end_headers()
                self.wfile.write(stream + b"data: [DONE]\n\n")

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream.daemon_threads = True
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        proxy = WireTraceProxy(
            f"http://127.0.0.1:{upstream.server_port}/v1",
            root / "wire.jsonl",
            mode="full",
            responses_tools_path=root / "tools_capture.jsonl",
            chat_bridge=responses_bridge,
            token_ids_path=root / "token_ids.jsonl",
        )
        proxy.start()
        self.addCleanup(proxy.stop)

        request = Request(
            proxy.proxy_base_url + "/responses",
            data=json.dumps(RESPONSES_REQUEST).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer k"},
        )
        with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
            payload = response.read().decode()

        events = [
            json.loads(line[5:])
            for line in payload.splitlines()
            if line.startswith("data:")
        ]
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(
            [item["type"] for item in events[-1]["response"]["output"]],
            ["reasoning", "message", "function_call", "function_call"],
        )
        self.assertNotIn("token_ids", payload)

        path, sent = requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual((sent["return_token_ids"], sent["logprobs"]), (True, True))
        row = read_jsonl(root / "token_ids.jsonl")[0]
        self.assertEqual(row["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(row["completion_token_ids"], [10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(row["finish_reason"], "tool_calls")
        wire = read_jsonl(root / "wire.jsonl")
        self.assertEqual(wire[0]["path"], "/v1/responses")
        self.assertEqual(wire[0]["upstream_path"], "/v1/chat/completions")
        # The tools Codex sent are still captured from its own request.
        tools = read_jsonl(root / "tools_capture.jsonl")
        self.assertEqual(tools[0]["kind"], "codex_tools")


if __name__ == "__main__":
    unittest.main()
