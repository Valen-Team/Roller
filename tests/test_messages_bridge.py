from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from roller.capture import CallRecorder, read_jsonl
from roller.messages_bridge import (
    MessagesStream,
    chat_request,
    messages_error,
    messages_response,
)
from roller.model_protocols import anthropic_response
from roller.wire_trace import WireTraceProxy

IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
}
IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

MESSAGES_REQUEST = {
    "model": "glm",
    "max_tokens": 32000,
    "stream": True,
    "thinking": {"type": "adaptive"},
    "output_config": {"effort": "high"},
    "metadata": {"user_id": "u"},
    "system": [
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=1;"},
        {"type": "text", "text": "You are an agent."},
        {"type": "text", "text": "\nBe brief.", "cache_control": {"type": "ephemeral"}},
    ],
    "tools": [
        {
            "name": "Bash",
            "description": "Run a command",
            "input_schema": {"type": "object", "properties": {"command": {}}},
        },
        {"type": "web_search_20250305", "name": "web_search"},
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                {"type": "text", "text": "List the files."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I will run ls.", "signature": "s1"},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "Bash",
                    "input": {"command": "ls"},
                },
                {
                    "type": "tool_use",
                    "id": "call-2",
                    "name": "Read",
                    "input": {"file_path": "a.png"},
                },
            ],
        },
        {"role": "system", "content": "Available agent types: none."},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-1", "content": "a.png"},
                {
                    "type": "tool_result",
                    "tool_use_id": "call-2",
                    "is_error": False,
                    "content": [{"type": "text", "text": "an image"}, IMAGE],
                },
                {"type": "text", "text": "Describe it."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "A square."}]},
        {"role": "user", "content": "Thanks."},
    ],
}


class ChatRequestTests(unittest.TestCase):
    def test_messages_keep_their_order_and_tool_results_follow_their_calls(self):
        request = chat_request(MESSAGES_REQUEST)

        self.assertEqual(
            request["messages"],
            [
                {"role": "system", "content": "You are an agent.\nBe brief."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>ctx</system-reminder>",
                        },
                        {"type": "text", "text": "List the files."},
                    ],
                },
                {
                    "role": "assistant",
                    "reasoning": "I will run ls.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": '{"command": "ls"}',
                            },
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_path": "a.png"}',
                            },
                        },
                    ],
                },
                {"role": "system", "content": "Available agent types: none."},
                {"role": "tool", "tool_call_id": "call-1", "content": "a.png"},
                {"role": "tool", "tool_call_id": "call-2", "content": "an image"},
                {"role": "user", "content": [IMAGE_PART]},
                {"role": "user", "content": "Describe it."},
                {"role": "assistant", "content": "A square."},
                {"role": "user", "content": "Thanks."},
            ],
        )

    def test_parameters_and_client_tools_are_carried_over(self):
        request = chat_request(
            {
                **MESSAGES_REQUEST,
                "temperature": 0.7,
                "stop_sequences": ["END"],
                "tool_choice": {"type": "tool", "name": "Bash"},
            }
        )

        self.assertEqual(
            {key: value for key, value in request.items() if key != "messages"},
            {
                "model": "glm",
                "max_tokens": 32000,
                "temperature": 0.7,
                "stream": True,
                "stream_options": {"include_usage": True},
                "stop": ["END"],
                "reasoning_effort": "high",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "description": "Run a command",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {}},
                            },
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "Bash"}},
            },
        )


CHUNKS = [
    {
        "id": "chatcmpl-1",
        "model": "glm",
        "prompt_token_ids": [1, 2, 3],
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
    },
    {
        "choices": [
            {
                "delta": {"reasoning": "Run"},
                "token_ids": [10],
                "logprobs": {"content": [{"token": "Run", "logprob": -0.5}]},
            }
        ]
    },
    {
        "choices": [
            {
                "delta": {"reasoning": " ls."},
                "token_ids": [11, 12],
                "logprobs": {
                    "content": [
                        {"token": " ls", "logprob": -0.25},
                        {"token": ".", "logprob": -0.125},
                    ]
                },
            }
        ]
    },
    {"choices": [{"delta": {"content": "Listing."}, "token_ids": [13]}]},
    {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": ""},
                        }
                    ]
                }
            }
        ]
    },
    {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"command":'}}
                    ]
                },
                "token_ids": [14],
            }
        ]
    },
    {
        "choices": [
            {
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": ' "ls"}'}}]
                },
                "token_ids": [15],
            }
        ]
    },
    {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call-2",
                            "function": {"name": "Read", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    },
    {"choices": [{"delta": {}, "finish_reason": "tool_calls", "token_ids": [16]}]},
    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 7}},
]


def stream_bytes(chunks) -> bytes:
    frames = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return "".join([*frames, "data: [DONE]\n\n"]).encode()


class MessagesStreamTests(unittest.TestCase):
    def test_chunks_become_thinking_text_and_tool_use_blocks(self):
        stream = MessagesStream()
        events = [event for chunk in CHUNKS for event in stream.feed(chunk)]
        events += stream.finish()

        self.assertEqual(
            [event["type"] for event in events],
            ["message_start"]
            + ["content_block_start", "content_block_delta", "content_block_delta"]
            + ["content_block_delta", "content_block_stop"]  # thinking + signature
            + ["content_block_start", "content_block_delta", "content_block_stop"]
            + ["content_block_start", "content_block_delta", "content_block_delta"]
            + ["content_block_stop"]
            + ["content_block_start", "content_block_delta", "content_block_stop"]
            + ["message_delta", "message_stop"],
        )
        self.assertEqual(events[0]["message"]["usage"]["input_tokens"], 3)
        starts = [e for e in events if e["type"] == "content_block_start"]
        self.assertEqual(
            [(e["index"], e["content_block"]["type"]) for e in starts],
            [(0, "thinking"), (1, "text"), (2, "tool_use"), (3, "tool_use")],
        )
        self.assertEqual(
            starts[2]["content_block"],
            {"type": "tool_use", "id": "call-1", "name": "Bash", "input": {}},
        )
        self.assertEqual(events[4]["delta"]["type"], "signature_delta")
        self.assertEqual(
            events[-2],
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 7,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        )


class UnstreamedTests(unittest.TestCase):
    def test_response_and_error_take_the_messages_shape(self):
        message = messages_response(
            {
                "id": "chatcmpl-1",
                "model": "glm",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "Half",
                            "reasoning": "Hmm",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": "Bash", "arguments": "{}"},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }
        )

        self.assertEqual(
            [block["type"] for block in message["content"]],
            ["thinking", "text", "tool_use"],
        )
        self.assertEqual(message["stop_reason"], "max_tokens")
        self.assertEqual(message["usage"]["input_tokens"], 5)
        self.assertEqual(
            messages_error(429, b'{"error": {"message": "slow down"}}'),
            {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "slow down"},
            },
        )


class BridgedProxyTests(unittest.TestCase):
    """A Messages client served by a Chat Completions upstream."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.requests = []
        self.response = (200, "text/event-stream", stream_bytes(CHUNKS))
        owner = self

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append((self.path, dict(self.headers), body))
                status, content_type, payload = owner.response
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream.daemon_threads = True
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        self.recorder = CallRecorder(self.root / "capture")
        self.proxy = WireTraceProxy(
            f"http://127.0.0.1:{upstream.server_port}",
            self.root / "wire.jsonl",
            mode="full",
            capture_recorder=self.recorder,
            drop_query_parameters=("beta",),
            messages_as_chat=True,
            token_ids_path=self.root / "token_ids.jsonl",
        )
        self.proxy.start()
        self.addCleanup(self.proxy.stop)

    def post(self, path="/v1/messages?beta=true"):
        request = Request(
            self.proxy.proxy_base_url + path,
            data=json.dumps(MESSAGES_REQUEST).encode(),
            headers={"Content-Type": "application/json", "x-api-key": "secret"},
        )
        try:
            response = build_opener(ProxyHandler({})).open(request, timeout=5)
        except HTTPError as exc:
            response = exc
        with response:
            return response.code, response.headers["Content-Type"], response.read()

    def test_stream_is_translated_and_token_ids_are_saved(self):
        status, content_type, payload = self.post()

        self.assertEqual(status, 200)
        message = anthropic_response(payload, content_type)
        self.assertEqual(
            [block["type"] for block in message["content"]],
            ["thinking", "text", "tool_use", "tool_use"],
        )
        self.assertEqual(message["content"][2]["input"], {"command": "ls"})
        self.assertEqual(message["stop_reason"], "tool_use")

        path, headers, body = self.requests[0]
        sent = json.loads(body)
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual((sent["return_token_ids"], sent["logprobs"]), (True, True))
        self.assertEqual(sent["messages"][0]["role"], "system")

        self.assertEqual(
            read_jsonl(self.root / "token_ids.jsonl"),
            [
                {
                    "request_seq": 1,
                    "continues": None,
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [10, 11, 12, 13, 14, 15, 16],
                    "logprobs": [-0.5, -0.25, -0.125],
                    "finish_reason": "tool_calls",
                    "complete": True,
                    "request": {
                        key: value
                        for key, value in sent.items()
                        if key not in ("messages", "tools")
                    },
                }
            ],
        )
        wire = read_jsonl(self.root / "wire.jsonl")
        self.assertEqual(wire[0]["path"], "/v1/messages?beta=true")
        self.assertEqual(wire[0]["upstream_path"], "/v1/chat/completions")
        self.assertEqual(wire[0]["body"]["system"], MESSAGES_REQUEST["system"])
        self.assertEqual(wire[1]["finish_reasons"], ["tool_use"])
        # The capture holds what Claude Code sent and received.
        self.recorder.seal()
        captured = (self.root / "capture").glob("**/*")
        stored = b"".join(p.read_bytes() for p in captured if p.is_file())
        self.assertIn(b"message_start", stored)
        self.assertNotIn(b"prompt_token_ids", stored)

    def test_upstream_error_takes_the_messages_shape(self):
        self.response = (400, "application/json", b'{"error": {"message": "too long"}}')

        status, _, payload = self.post()

        self.assertEqual(status, 400)
        self.assertEqual(
            json.loads(payload),
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "too long"},
            },
        )
        self.assertFalse((self.root / "token_ids.jsonl").exists())

    def test_other_routes_are_forwarded_as_they_are(self):
        self.response = (200, "application/json", b'{"input_tokens": 3}')

        _, _, payload = self.post("/v1/messages/count_tokens?beta=true")

        self.assertEqual(self.requests[0][0], "/v1/messages/count_tokens")
        self.assertEqual(json.loads(self.requests[0][2]), MESSAGES_REQUEST)
        self.assertEqual(payload, b'{"input_tokens": 3}')


if __name__ == "__main__":
    unittest.main()
