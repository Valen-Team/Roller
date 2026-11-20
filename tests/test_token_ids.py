from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from roller.capture import CallRecorder, read_jsonl
from roller.token_ids import TokenIds, TokenLog, logprobs_mode
from roller.wire_trace import WireTraceProxy

CHUNKS = [
    {
        "id": "chatcmpl-1",
        "prompt_token_ids": [1, 2, 3],
        "choices": [{"delta": {"role": "assistant", "content": ""}}],
    },
    {
        "id": "chatcmpl-1",
        "choices": [
            {
                "delta": {"content": "Red"},
                "token_ids": [10, 11],
                "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.25}]},
            }
        ],
    },
    {
        "id": "chatcmpl-1",
        "choices": [
            {
                "delta": {},
                "finish_reason": "stop",
                "token_ids": [12],
                "logprobs": {"content": [{"logprob": -0.125}]},
            }
        ],
    },
    {"id": "chatcmpl-1", "choices": [], "usage": {"completion_tokens": 3}},
]
# The same stream from an endpoint that was not asked for token ids.
PLAIN_CHUNKS = [
    {"id": "chatcmpl-1", "choices": [{"delta": {"role": "assistant", "content": ""}}]},
    {"id": "chatcmpl-1", "choices": [{"delta": {"content": "Red"}}]},
    {"id": "chatcmpl-1", "choices": [{"delta": {}, "finish_reason": "stop"}]},
    {"id": "chatcmpl-1", "choices": [], "usage": {"completion_tokens": 3}},
]
REQUEST = {
    "model": "glm",
    "stream": True,
    "temperature": 0.7,
    "messages": [{"role": "user", "content": "Name a colour."}],
}


def tokens(prompt, completion) -> TokenIds:
    taken = TokenIds({"model": "glm"})
    taken.add({"prompt_token_ids": prompt, "choices": [{"token_ids": completion}]})
    return taken


class TokenIdsTests(unittest.TestCase):
    def test_request_asks_for_them_and_chunks_give_them_up(self):
        request = dict(REQUEST)
        taken = TokenIds(request)
        chunks = json.loads(json.dumps(CHUNKS))
        for chunk in chunks:
            taken.take(chunk)

        self.assertEqual(
            (request["return_token_ids"], request["logprobs"]), (True, True)
        )
        self.assertNotIn("messages", taken.request)
        self.assertEqual(taken.request["temperature"], 0.7)
        self.assertEqual(taken.prompt, [1, 2, 3])
        self.assertEqual(taken.completion, [10, 11, 12])
        self.assertEqual(taken.logprobs, [-0.5, -0.25, -0.125])
        self.assertEqual(taken.finish_reason, "stop")
        self.assertEqual(chunks, PLAIN_CHUNKS)


class TokenLogTests(unittest.TestCase):
    def test_a_call_holds_only_the_prompt_after_the_call_it_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "token_ids.jsonl"
            log = TokenLog(path)
            log.write(1, tokens([1, 2], [3, 4]), True)
            # A subagent starts between two calls of the root conversation.
            log.write(2, tokens([7, 8], [9]), True)
            log.write(3, tokens([1, 2, 3, 4, 5], [6]), True)
            log.write(4, tokens([1, 2, 3, 4, 5, 6, 7], [8]), True)
            # A compacted context does not continue what came before.
            log.write(5, tokens([1, 2, 9], [9]), True)

            rows = read_jsonl(path)

        self.assertEqual(
            [(row["request_seq"], row["continues"]) for row in rows],
            [(1, None), (2, None), (3, 1), (4, 3), (5, None)],
        )
        self.assertEqual(
            [row["prompt_token_ids"] for row in rows],
            [[1, 2], [7, 8], [5], [7], [1, 2, 9]],
        )
        self.assertEqual(rows[2]["completion_token_ids"], [6])


class ChatCompletionsProxyTests(unittest.TestCase):
    """A Harness that calls Chat Completions itself."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.requests = []
        frames = [f"data: {json.dumps(chunk)}\n\n" for chunk in CHUNKS]
        stream = "".join([*frames, "data: [DONE]\n\n"]).encode()
        self.response = ("text/event-stream", stream)
        owner = self

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append((self.path, json.loads(body)))
                content_type, payload = owner.response
                self.send_response(200)
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
            f"http://127.0.0.1:{upstream.server_port}/v1",
            self.root / "wire.jsonl",
            mode="full",
            capture_recorder=self.recorder,
            token_ids_path=self.root / "token_ids.jsonl",
        )
        self.proxy.start()
        self.addCleanup(self.proxy.stop)

    def post(self, path="/chat/completions", body=REQUEST) -> bytes:
        request = Request(
            self.proxy.proxy_base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
            return response.read()

    def stored(self) -> bytes:
        self.recorder.seal()
        captured = (self.root / "capture").glob("**/*")
        return b"".join(path.read_bytes() for path in captured if path.is_file())

    def test_stream_reaches_the_harness_without_the_token_ids(self):
        payload = self.post()

        events = [
            json.loads(line[5:])
            for line in payload.split(b"\n")
            if line.startswith(b"data:") and line != b"data: [DONE]"
        ]
        self.assertEqual(events, PLAIN_CHUNKS)
        self.assertTrue(payload.endswith(b"data: [DONE]\n\n"))
        path, sent = self.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual((sent["return_token_ids"], sent["logprobs"]), (True, True))
        self.assertEqual(sent["messages"], REQUEST["messages"])
        self.assertEqual(
            read_jsonl(self.root / "token_ids.jsonl"),
            [
                {
                    "request_seq": 1,
                    "continues": None,
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [10, 11, 12],
                    "logprobs": [-0.5, -0.25, -0.125],
                    "finish_reason": "stop",
                    "complete": True,
                    "request": {
                        "model": "glm",
                        "stream": True,
                        "temperature": 0.7,
                        "return_token_ids": True,
                        "logprobs": True,
                    },
                }
            ],
        )
        # The records hold the request the Harness sent and what it received.
        self.assertEqual(read_jsonl(self.root / "wire.jsonl")[0]["body"], REQUEST)
        stored = self.stored()
        self.assertIn(b"Red", stored)
        self.assertNotIn(b"token_ids", stored)

    def test_unstreamed_response_and_the_next_call_of_the_conversation(self):
        def body(prompt, completion):
            choice = {
                "message": {"role": "assistant", "content": "Red"},
                "finish_reason": "stop",
                "token_ids": completion,
                "logprobs": {"content": [{"logprob": -0.5}] * len(completion)},
            }
            return json.dumps({"prompt_token_ids": prompt, "choices": [choice]})

        self.response = ("application/json", body([1, 2, 3], [10]).encode())
        first = json.loads(self.post(body={**REQUEST, "stream": False}))
        self.response = ("application/json", body([1, 2, 3, 10, 4], [11]).encode())
        self.post(body={**REQUEST, "stream": False})

        self.assertEqual(
            first,
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Red"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
        rows = read_jsonl(self.root / "token_ids.jsonl")
        self.assertEqual(
            [(row["continues"], row["prompt_token_ids"]) for row in rows],
            [(None, [1, 2, 3]), (1, [4])],
        )

    def test_other_routes_are_forwarded_as_they_are(self):
        self.response = ("application/json", b'{"data": []}')

        payload = self.post("/models")

        self.assertEqual(self.requests[0], ("/v1/models", REQUEST))
        self.assertEqual(payload, b'{"data": []}')
        self.assertFalse((self.root / "token_ids.jsonl").exists())


class LogprobsModeTests(unittest.TestCase):
    def mode(self, processed: bool, token_ids=(1, 2)) -> str:
        class Endpoint(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                gap = 0.8 / body["temperature"] if processed else 0.8
                top = [{"logprob": -0.2}, {"logprob": -0.2 - gap}]
                payload = json.dumps(
                    {
                        "prompt_token_ids": list(token_ids),
                        "choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        return logprobs_mode(url, "glm", "key")

    def test_gap_that_grows_as_temperature_drops_is_processed(self):
        self.assertEqual(self.mode(processed=True), "processed")
        self.assertEqual(self.mode(processed=False), "raw")

    def test_endpoint_without_token_ids_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "returned no token ids"):
            self.mode(processed=True, token_ids=())


if __name__ == "__main__":
    unittest.main()
