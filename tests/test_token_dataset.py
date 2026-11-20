from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from roller.messages_bridge import logprobs_mode
from roller.storage import write_jsonl
from roller.token_dataset import export_run, sequences


def call(seq, prompt, completion, finish="tool_calls"):
    return {
        "request_seq": seq,
        "prompt_token_ids": prompt,
        "completion_token_ids": completion,
        "logprobs": [-0.5] * len(completion),
        "finish_reason": finish,
        "complete": True,
        "request": {"model": "glm", "max_tokens": 64},
    }


CALLS = [
    call(1, [1, 2], [3, 4]),
    # A subagent starts between two calls of the root conversation.
    call(2, [7, 8], [9], "stop"),
    call(3, [1, 2, 3, 4, 5], [6], "stop"),
    # A compacted context does not continue what came before.
    call(4, [1, 9], [9], "stop"),
]
PARTICIPANTS = {1: "root:a", 2: "agent:b", 3: "root:a", 4: "root:a"}


class SequenceTests(unittest.TestCase):
    def test_calls_that_continue_each_other_share_one_sequence(self):
        self.assertEqual(
            sequences(CALLS, PARTICIPANTS),
            [
                {
                    "participant_id": "root:a",
                    "request_seqs": [1, 3],
                    "finish_reasons": ["tool_calls", "stop"],
                    "token_ids": [1, 2, 3, 4, 5, 6],
                    "generated": [0, 0, 1, 1, 0, 1],
                    "logprobs": [None, None, -0.5, -0.5, None, -0.5],
                },
                {
                    "participant_id": "agent:b",
                    "request_seqs": [2],
                    "finish_reasons": ["stop"],
                    "token_ids": [7, 8, 9],
                    "generated": [0, 0, 1],
                    "logprobs": [None, None, -0.5],
                },
                {
                    "participant_id": "root:a",
                    "request_seqs": [4],
                    "finish_reasons": ["stop"],
                    "token_ids": [1, 9, 9],
                    "generated": [0, 0, 1],
                    "logprobs": [None, None, -0.5],
                },
            ],
        )

    def test_export_writes_one_row_per_sample_with_its_sequences(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            sample = run / "samples" / "glm" / "00000000_task"
            (run / "samples" / "glm" / "00000001_untraced").mkdir(parents=True)
            write_jsonl(sample / "token_ids.jsonl", CALLS)
            write_jsonl(
                sample / "capture" / "attempt-0001" / "calls.jsonl",
                [
                    {"request_seq": seq, "participant_id": participant}
                    for seq, participant in PARTICIPANTS.items()
                ],
            )
            configuration = {"claude-code": {"token_logprobs": {"glm": "processed"}}}
            (run / "info.json").write_text(json.dumps({"configuration": configuration}))
            result = {"task_id": "task", "model": "glm", "status": "ok"}
            for directory in (sample, sample.with_name("00000001_untraced")):
                (directory / "info.json").write_text(json.dumps(result))

            counts = export_run(run, run / "tokens")

            rows = (run / "tokens" / "tokens.jsonl").read_text().splitlines()
        row = json.loads(rows[0])
        self.assertEqual(counts, (1, 3, 12))
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            (row["task_id"], row["sample"], row["status"], row["logprobs_mode"]),
            ("task", "samples/glm/00000000_task", "ok", "processed"),
        )
        self.assertEqual(row["request"], {"model": "glm", "max_tokens": 64})
        self.assertEqual(row["sequences"], sequences(CALLS, PARTICIPANTS))


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
        return logprobs_mode(f"http://127.0.0.1:{server.server_port}", "glm", "key")

    def test_gap_that_grows_as_temperature_drops_is_processed(self):
        self.assertEqual(self.mode(processed=True), "processed")
        self.assertEqual(self.mode(processed=False), "raw")

    def test_endpoint_without_token_ids_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "returned no token ids"):
            self.mode(processed=True, token_ids=())


if __name__ == "__main__":
    unittest.main()
