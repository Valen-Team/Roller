from __future__ import annotations

import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from roller.cli import main
from roller.collector import Collector
from roller.storage import write_jsonl
from roller.token_dataset import TOKENS_FILE, export_run, sequences


def call(seq, prompt, completion, finish="tool_calls", continues=None, logprobs=None):
    return {
        "request_seq": seq,
        "continues": continues,
        "prompt_token_ids": prompt,
        "completion_token_ids": completion,
        "logprobs": [-0.5] * len(completion) if logprobs is None else logprobs,
        "finish_reason": finish,
        "complete": True,
        "request": {"model": "glm", "max_tokens": 64},
    }


CALLS = [
    call(1, [1, 2], [3, 4]),
    # A subagent starts between two calls of the root conversation.
    call(2, [7, 8], [9], "stop"),
    call(3, [5], [6], "stop", continues=1),
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
            configuration = {"token_ids": {"logprobs": {"glm": "processed"}}}
            (run / "info.json").write_text(json.dumps({"configuration": configuration}))
            result = {"task_id": "task", "model": "glm", "status": "ok"}
            for directory in (sample, sample.with_name("00000001_untraced")):
                (directory / "info.json").write_text(json.dumps(result))

            counts = export_run(run, run / "tokens")

            rows = (run / "tokens" / "tokens.jsonl").read_text().splitlines()
            # --training-files-only removes the capture and keeps its map.
            shutil.rmtree(sample / "capture")
            pruned = {
                **result,
                "runtime": {
                    "capture": {
                        "participant_ids": {str(k): v for k, v in PARTICIPANTS.items()}
                    }
                },
            }
            (sample / "info.json").write_text(json.dumps(pruned))
            export_run(run, run / "pruned")
            self.assertEqual(
                (run / "pruned" / "tokens.jsonl").read_text().splitlines(), rows
            )
        row = json.loads(rows[0])
        self.assertEqual(counts, (1, 3, 12))
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            (row["task_id"], row["sample"], row["status"], row["logprobs_mode"]),
            ("task", "samples/glm/00000000_task", "ok", "processed"),
        )
        self.assertEqual(row["request"], {"model": "glm", "max_tokens": 64})
        self.assertEqual(row["sequences"], sequences(CALLS, PARTICIPANTS))

    def test_tool_image_turns_append_prompts_of_any_length(self):
        calls = [
            call(1, [10, 11, 12], [13, 14], logprobs=[-0.1, -0.2]),
            # A tool returned an image: the next prompt is far longer than
            # the text turns around it.
            call(2, list(range(100, 140)), [15], continues=1, logprobs=[-0.3]),
            # A turn whose prompt adds one token and whose completion is empty.
            call(3, [16], [], "length", continues=2, logprobs=[]),
            call(4, [17, 18], [19, 20, 21], "stop", 3, [-0.4, -0.5, -0.6]),
        ]

        (joined,) = sequences(calls, {seq: "root:a" for seq in range(1, 5)})

        self.assertEqual(joined["request_seqs"], [1, 2, 3, 4])
        self.assertEqual(
            joined["finish_reasons"], ["tool_calls", "tool_calls", "length", "stop"]
        )
        self.assertEqual(
            joined["token_ids"],
            [10, 11, 12, 13, 14, *range(100, 140), 15, 16, 17, 18, 19, 20, 21],
        )
        self.assertEqual(
            joined["generated"], [0, 0, 0, 1, 1, *[0] * 40, 1, 0, 0, 0, 1, 1, 1]
        )
        self.assertEqual(
            joined["logprobs"],
            [None] * 3
            + [-0.1, -0.2]
            + [None] * 40
            + [-0.3]
            # The empty completion adds its prompt token and no logprob.
            + [None] * 3
            + [-0.4, -0.5, -0.6],
        )
        self.assertEqual(
            len(joined["token_ids"]), len(joined["generated"]), len(joined["logprobs"])
        )

    def test_compaction_and_interleaved_subagents_reset_continues(self):
        calls = [
            call(1, [1], [2]),
            # Two subagents start while the root conversation waits.
            call(2, [30], [31]),
            call(3, [40], [41]),
            call(4, [32], [33], "stop", continues=2),
            call(5, [3], [4], continues=1),
            call(6, [42], [43], "stop", continues=3),
            # The root conversation is compacted: its summary starts afresh,
            # and so does the conversation that follows it.
            call(7, [50], [51], "stop"),
            call(8, [60], [61]),
            call(9, [62], [63], "stop", continues=8),
            # A subagent after the compaction continues none of the above.
            call(10, [70], [71], "stop"),
        ]
        participants = {
            1: "root:a",
            2: "agent:b",
            3: "agent:c",
            4: "agent:b",
            5: "root:a",
            6: "agent:c",
            7: "root:a",
            8: "root:a",
            9: "root:a",
            # The participant of call 10 is unknown.
        }

        joined = sequences(calls, participants)

        self.assertEqual(
            [
                (item["participant_id"], item["request_seqs"], item["token_ids"])
                for item in joined
            ],
            [
                ("root:a", [1, 5], [1, 2, 3, 4]),
                ("agent:b", [2, 4], [30, 31, 32, 33]),
                ("agent:c", [3, 6], [40, 41, 42, 43]),
                ("root:a", [7], [50, 51]),
                ("root:a", [8, 9], [60, 61, 62, 63]),
                (None, [10], [70, 71]),
            ],
        )
        self.assertEqual(sum(len(item["token_ids"]) for item in joined), 20)


def _capture_calls(participants):
    return [
        {"request_seq": seq, "participant_id": participant}
        for seq, participant in participants.items()
    ]


def _write_sample(run, name, result, calls=None, captures=()):
    """A sample directory with ``info.json``, and the files given."""

    sample = run / "samples" / result["model"] / name
    sample.mkdir(parents=True)
    (sample / "info.json").write_text(json.dumps(result))
    if calls is not None:
        write_jsonl(sample / "token_ids.jsonl", calls)
    for attempt, participants in captures:
        write_jsonl(
            sample / "capture" / attempt / "calls.jsonl", _capture_calls(participants)
        )
    return sample


def _export(run, output):
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            main(["export", str(run), "--format", "tokens", "-o", str(output)])
    except SystemExit as exit:
        lines = (output / TOKENS_FILE).read_text().splitlines()
        return exit.code, stdout.getvalue(), lines
    raise AssertionError("main() returned without exiting")


class MixedRunTests(unittest.TestCase):
    """A run of several samples, models, statuses and captures."""

    ROOT = [
        call(1, [1, 2], [3]),
        call(2, list(range(200, 220)), [4], continues=1),
        call(3, [5], [6], "stop", continues=2),
    ]
    WITH_SUBAGENT = [
        call(1, [1], [2]),
        call(2, [8], [9], "stop"),
        call(3, [3], [4], "stop", continues=1),
    ]

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name) / "run"
        self.run_dir.mkdir()
        configuration = {"token_ids": {"logprobs": {"glm": "processed", "qwen": "raw"}}}
        (self.run_dir / "info.json").write_text(
            json.dumps({"configuration": configuration})
        )
        self.samples = {}

        def sample(name, result, calls=None, captures=()):
            result = {"task_id": name.split("_", 1)[1], **result}
            self.samples[name] = _write_sample(
                self.run_dir, name, result, calls, captures
            )

        # Directory order differs from request order.
        sample(
            "00000000_late",
            {"model": "glm", "status": "ok", "request_index": 3},
            self.ROOT,
            # Only the capture of the last attempt names the participants.
            [
                ("attempt-0001", {1: "root:stale"}),
                ("attempt-0002", {1: "root:a", 2: "root:a", 3: "root:a"}),
            ],
        )
        sample(
            "00000001_subagent",
            {"model": "qwen", "status": "ok", "request_index": 0},
            self.WITH_SUBAGENT,
            [("attempt-0001", {1: "root:a", 2: "agent:b", 3: "root:a"})],
        )
        sample(
            "00000002_failed",
            {"model": "glm", "status": "error", "request_index": 1},
            [call(1, [1], [2], "length")],
            [("attempt-0001", {1: "root:a"})],
        )
        # No capture, and no participant_ids in runtime.capture to fall back on.
        sample(
            "00000003_uncaptured",
            {
                "model": "other",
                "status": "ok",
                "request_index": 2,
                "runtime": {"capture": None},
            },
            [call(1, [1], [2], "stop")],
        )
        # The endpoint was never called: an empty token_ids.jsonl.
        sample(
            "00000004_empty",
            {"model": "glm", "status": "timeout", "request_index": 4},
            [],
        )
        # Collected without --token-ids: no token_ids.jsonl.
        sample("00000005_untraced", {"model": "glm", "status": "ok"})

    def rows(self, output="tokens"):
        code, printed, lines = _export(self.run_dir, self.run_dir / output)
        self.assertEqual(code, 0)
        return printed, lines

    def test_rows_follow_request_order_and_keep_every_status(self):
        printed, lines = self.rows()
        rows = [json.loads(line) for line in lines]

        self.assertEqual(
            [
                (row["task_id"], row["model"], row["status"], row["logprobs_mode"])
                for row in rows
            ],
            [
                ("subagent", "qwen", "ok", "raw"),
                ("failed", "glm", "error", "processed"),
                # A model the run has no logprobs mode for.
                ("uncaptured", "other", "ok", None),
                ("late", "glm", "ok", "processed"),
                ("empty", "glm", "timeout", "processed"),
            ],
        )
        by_task = {row["task_id"]: row for row in rows}
        self.assertEqual(
            [item["participant_id"] for item in by_task["subagent"]["sequences"]],
            ["root:a", "agent:b"],
        )
        self.assertEqual(
            by_task["late"]["sequences"],
            sequences(self.ROOT, {1: "root:a", 2: "root:a", 3: "root:a"}),
        )
        self.assertEqual(
            [item["participant_id"] for item in by_task["uncaptured"]["sequences"]],
            [None],
        )
        self.assertEqual(
            (by_task["empty"]["request"], by_task["empty"]["sequences"]), (None, [])
        )
        # 2 + 1 + 1 + 1 sequences of 6 + 2 + 2 + 26 tokens.
        self.assertIn("Exported 5 samples as 5 sequences of 36 tokens", printed)

    def test_training_files_only_exports_the_same_rows(self):
        _, full = self.rows("full")
        for name in ("00000000_late", "00000001_subagent"):
            sample_dir = self.samples[name]
            result = json.loads((sample_dir / "info.json").read_text())
            runtime = copy.deepcopy(result.get("runtime") or {})
            runtime["capture"] = {"manifest": "capture/manifest.json"}
            (sample_dir / "trace.jsonl").write_text("")
            Collector._keep_training_files(sample_dir, runtime, self.run_dir / "blobs")
            # The collector writes info.json after pruning, from the metadata.
            result["runtime"] = runtime
            (sample_dir / "info.json").write_text(json.dumps(result))
            self.assertFalse((sample_dir / "capture").exists())
            self.assertFalse((sample_dir / "trace.jsonl").exists())
            self.assertIn("participant_ids", runtime["capture"])

        _, pruned = self.rows("pruned")

        self.assertEqual(pruned, full)

    def test_logprobs_mode_is_none_when_the_run_recorded_none(self):
        for configuration in ({}, {"token_ids": None}, {"token_ids": {}}):
            with self.subTest(configuration=configuration):
                (self.run_dir / "info.json").write_text(
                    json.dumps({"configuration": configuration})
                )
                _, lines = self.rows()
                self.assertEqual(
                    {json.loads(line)["logprobs_mode"] for line in lines}, {None}
                )


class EmptyRunTests(unittest.TestCase):
    def test_a_run_without_token_ids_exports_an_empty_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            (run / "info.json").write_text(json.dumps({"configuration": {}}))
            for samples in ("nothing", "untraced"):
                with self.subTest(samples=samples):
                    if samples == "untraced":
                        _write_sample(
                            run, "00000000_task", {"model": "glm", "status": "ok"}
                        )
                    else:
                        (run / "samples").mkdir()
                    self.assertEqual(export_run(run, run / "tokens"), (0, 0, 0))
                    code, printed, lines = _export(run, run / "cli")
                    self.assertEqual((code, lines), (0, []))
                    self.assertIn(
                        "Exported 0 samples as 0 sequences of 0 tokens", printed
                    )


if __name__ == "__main__":
    unittest.main()
