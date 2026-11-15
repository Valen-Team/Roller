from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from roller.atif import actf_to_atif, export_run, write_atif
from roller.cli import main

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
PNG_PATH = f"images/{hashlib.sha256(PNG).hexdigest()[:16]}.png"


def _trajectory() -> dict:
    return {
        "schema_version": "ACTF_v1.0",
        "system_prompt": "Use tools carefully.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ],
        "steps": [
            {
                "step_id": 1,
                "user_content": "Read input.txt",
                "assistant_content": {
                    "reasoning_content": "I need the file.",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "read",
                            "arguments": {"path": "input.txt"},
                        }
                    ],
                },
                "observation": [
                    {
                        "role": "toolResult",
                        "toolCallId": "call-1",
                        "toolName": "read",
                        "content": [
                            {"type": "text", "text": "line one"},
                            {"type": "text", "text": "line two"},
                        ],
                        "isError": False,
                    }
                ],
                "metric": {
                    "prompt_tokens_len": 120,
                    "completion_tokens_len": 30,
                    "llm_infer_ms": None,
                    "env_action_ms": None,
                    "stop_reason": "toolUse",
                    "cache_read_tokens": 100,
                },
                "started_at": "2026-11-16 03:00:00.000000",
                "finished_at": "2026-11-16 03:00:02.500000",
            },
            {
                "step_id": 2,
                "user_content": "",
                "assistant_content": {
                    "reasoning_content": "",
                    "content": "Summary",
                    "tool_calls": [],
                },
                "observation": [],
                "metric": {
                    "prompt_tokens_len": 160,
                    "completion_tokens_len": 12,
                    "llm_infer_ms": None,
                    "env_action_ms": None,
                    "stop_reason": "stop",
                    "cache_read_tokens": 120,
                },
                "started_at": "2026-11-16 03:00:02.500000",
                "finished_at": "2026-11-16 03:00:04.000000",
            },
        ],
        "started_at": "2026-11-16 03:00:00.000000",
        "finished_at": "2026-11-16 03:00:04.000000",
    }


def _convert(trajectory: dict, **options) -> tuple[dict, dict[str, bytes]]:
    return actf_to_atif(
        trajectory, agent_name="openclaw", agent_version="2026.7.1", **options
    )


class ActfToAtifTests(unittest.TestCase):
    def test_steps_follow_the_atif_layout(self) -> None:
        document, images = _convert(
            _trajectory(), model_name="gpt-test", session_id="run/model/0000_task"
        )

        self.assertEqual(images, {})
        self.assertEqual(document["schema_version"], "ATIF-v1.7")
        self.assertEqual(document["session_id"], "run/model/0000_task")
        self.assertEqual(
            document["agent"],
            {
                "name": "openclaw",
                "version": "2026.7.1",
                "model_name": "gpt-test",
                "tool_definitions": _trajectory()["tools"],
            },
        )
        self.assertEqual(
            document["steps"],
            [
                {"step_id": 1, "source": "system", "message": "Use tools carefully."},
                {
                    "step_id": 2,
                    "timestamp": "2026-11-16T03:00:00Z",
                    "source": "user",
                    "message": "Read input.txt",
                },
                {
                    "step_id": 3,
                    "timestamp": "2026-11-16T03:00:00Z",
                    "source": "agent",
                    "message": "",
                    "reasoning_content": "I need the file.",
                    "tool_calls": [
                        {
                            "tool_call_id": "call-1",
                            "function_name": "read",
                            "arguments": {"path": "input.txt"},
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "call-1",
                                "content": "line one\nline two",
                                "extra": {"toolName": "read", "isError": False},
                            }
                        ]
                    },
                    "metrics": {
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "cached_tokens": 100,
                        "extra": {"stop_reason": "toolUse"},
                    },
                    "extra": {"finished_at": "2026-11-16T03:00:02.500000Z"},
                },
                {
                    "step_id": 4,
                    "timestamp": "2026-11-16T03:00:02.500000Z",
                    "source": "agent",
                    "message": "Summary",
                    "metrics": {
                        "prompt_tokens": 160,
                        "completion_tokens": 12,
                        "cached_tokens": 120,
                        "extra": {"stop_reason": "stop"},
                    },
                    "extra": {"finished_at": "2026-11-16T03:00:04Z"},
                },
            ],
        )
        self.assertEqual(
            document["final_metrics"],
            {
                "total_prompt_tokens": 280,
                "total_completion_tokens": 42,
                "total_cached_tokens": 220,
                "total_steps": 4,
            },
        )
        self.assertNotIn("extra", document)

    def test_images_become_referenced_files(self) -> None:
        trajectory = _trajectory()
        encoded = base64.b64encode(PNG).decode("ascii")
        trajectory["steps"][0]["user_content"] = [
            {"type": "text", "text": "Describe this"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": encoded,
                },
            },
        ]
        trajectory["steps"][0]["observation"][0]["content"] = [
            {"type": "text", "text": "screenshot"},
            {"type": "image", "data": encoded, "mimeType": "image/png"},
        ]

        document, images = _convert(trajectory)

        part = {
            "type": "image",
            "source": {"media_type": "image/png", "path": PNG_PATH},
        }
        self.assertEqual(images, {PNG_PATH: PNG})
        self.assertEqual(
            document["steps"][1]["message"],
            [{"type": "text", "text": "Describe this"}, part],
        )
        self.assertEqual(
            document["steps"][2]["observation"]["results"][0]["content"],
            [{"type": "text", "text": "screenshot"}, part],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = write_atif(Path(tmp), document, images)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), document)
            self.assertEqual((Path(tmp) / PNG_PATH).read_bytes(), PNG)

    def test_result_without_a_call_in_its_step_keeps_its_id_in_extra(self) -> None:
        trajectory = _trajectory()
        trajectory["steps"][0]["observation"].append(
            {
                "role": "toolResult",
                "toolCallId": "call-9",
                "toolName": "read",
                "content": [{"type": "text", "text": "late"}],
                "isError": True,
            }
        )

        document, _ = _convert(trajectory)

        self.assertEqual(
            document["steps"][2]["observation"]["results"][1],
            {
                "content": "late",
                "extra": {"toolCallId": "call-9", "toolName": "read", "isError": True},
            },
        )

    def test_blocks_outside_atif_keep_their_json(self) -> None:
        trajectory = _trajectory()
        block = {"type": "resource", "uri": "file:///tmp/a.txt"}
        trajectory["steps"][0]["observation"][0]["content"] = [
            {"type": "text", "text": "see"},
            block,
        ]

        document, _ = _convert(trajectory)

        self.assertEqual(
            document["steps"][2]["observation"]["results"][0]["content"],
            "see\n" + json.dumps(block),
        )

    def test_codex_tool_calls_and_step_extras_are_kept(self) -> None:
        trajectory = _trajectory()
        trajectory["tools_format"] = "responses"
        step = trajectory["steps"][0]
        step["extra"] = {"model": "gpt-test-mini", "message_id": "msg-1"}
        step["metric"] = {
            "prompt_tokens_len": 50,
            "completion_tokens_len": 5,
            "reasoning_tokens_len": 2,
            "cache_read_tokens_len": 40,
        }
        step["assistant_content"]["tool_calls"] = [
            {
                "type": "toolCall",
                "id": "call-1",
                "name": "apply_patch",
                "arguments": {"input": "*** Begin Patch"},
                "native_type": "custom_tool_call",
                "input": "*** Begin Patch",
            }
        ]

        document, _ = _convert(trajectory)

        agent_step = document["steps"][2]
        self.assertEqual(document["agent"]["extra"], {"tools_format": "responses"})
        self.assertEqual(agent_step["model_name"], "gpt-test-mini")
        self.assertEqual(
            agent_step["tool_calls"],
            [
                {
                    "tool_call_id": "call-1",
                    "function_name": "apply_patch",
                    "arguments": {"input": "*** Begin Patch"},
                    "extra": {
                        "native_type": "custom_tool_call",
                        "input": "*** Begin Patch",
                    },
                }
            ],
        )
        self.assertEqual(
            agent_step["metrics"],
            {
                "prompt_tokens": 50,
                "completion_tokens": 5,
                "cached_tokens": 40,
                "extra": {"reasoning_tokens_len": 2},
            },
        )
        self.assertEqual(
            agent_step["extra"],
            {
                "model": "gpt-test-mini",
                "message_id": "msg-1",
                "finished_at": "2026-11-16T03:00:02.500000Z",
            },
        )

    def test_trajectory_without_steps_is_rejected(self) -> None:
        trajectory = _trajectory()
        trajectory["system_prompt"] = ""
        trajectory["steps"] = []

        with self.assertRaisesRegex(ValueError, "no steps"):
            _convert(trajectory)


class ExportRunTests(unittest.TestCase):
    def _run(self, root: Path) -> Path:
        run_dir = root / "demo-run"
        sample = run_dir / "samples" / "gpt-test" / "0000_read"
        sample.mkdir(parents=True)
        (sample / "trace_ACTF.json").write_text(
            json.dumps(_trajectory()), encoding="utf-8"
        )
        (sample / "info.json").write_text(
            json.dumps(
                {
                    "schema_version": "roller.sample/v1",
                    "sample_index": 0,
                    "task_id": "read",
                    "model": "gpt-test",
                    "harness": "claude-code",
                    "status": "ok",
                    "error": None,
                    "usage": {"cost_usd": 0.25},
                    "runtime": {"claude_code_version": "2.1.0"},
                }
            ),
            encoding="utf-8",
        )
        (sample / "score.json").write_text(
            json.dumps({"overall_score": 1.0}), encoding="utf-8"
        )
        failed = run_dir / "samples" / "gpt-test" / "0001_write"
        failed.mkdir()
        (failed / "info.json").write_text(
            json.dumps({"harness": "claude-code", "status": "error"}),
            encoding="utf-8",
        )
        return run_dir

    def test_export_run_writes_one_trajectory_per_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run(Path(tmp))
            output = Path(tmp) / "atif"

            written, skipped = export_run(run_dir, output)

            destination = output / "gpt-test" / "0000_read" / "trajectory.json"
            self.assertEqual(written, [destination])
            self.assertEqual(
                skipped, {"samples/gpt-test/0001_write": "no trace_ACTF.json"}
            )
            document = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(document["session_id"], "demo-run/gpt-test/0000_read")
            self.assertEqual(document["agent"]["name"], "claude-code")
            self.assertEqual(document["agent"]["version"], "2.1.0")
            self.assertEqual(document["agent"]["model_name"], "gpt-test")
            self.assertEqual(document["final_metrics"]["total_cost_usd"], 0.25)
            self.assertEqual(
                document["extra"],
                {
                    "roller": {
                        "task_id": "read",
                        "sample_index": 0,
                        "status": "ok",
                        "scores": {"overall_score": 1.0},
                    }
                },
            )

    def test_cli_exports_into_the_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run(Path(tmp))
            stdout = io.StringIO()

            with (
                contextlib.redirect_stdout(stdout),
                self.assertRaises(SystemExit) as exit,
            ):
                main(["export", str(run_dir)])

            self.assertEqual(exit.exception.code, 0)
            self.assertTrue(
                (
                    run_dir / "atif" / "gpt-test" / "0000_read" / "trajectory.json"
                ).is_file()
            )
            self.assertEqual(
                stdout.getvalue().splitlines(),
                [
                    f"Exported 1 ATIF trajectories to {run_dir / 'atif'}",
                    "Skipped 1 samples:",
                    "  samples/gpt-test/0001_write: no trace_ACTF.json",
                ],
            )


if __name__ == "__main__":
    unittest.main()
