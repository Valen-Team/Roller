from __future__ import annotations

import json
import re
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import urlopen

from roller.benches.report import sample_score
from roller.cli import _build_parser
from roller.viewer import RollerRun, make_server
from roller.viewer import app as viewer_app


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class ViewerTests(unittest.TestCase):
    def test_inline_image_blocks_preserve_order_and_reject_remote_sources(self):
        from roller.viewer.app import _blocks_view

        content = [
            {"type": "text", "text": "before"},
            {"type": "image", "data": "AA==", "mimeType": "image/png"},
            {"type": "text", "text": "between"},
            {"type": "image", "data": "AQ==", "mimeType": "image/jpeg"},
        ]
        blocks = _blocks_view(content, 40000)["blocks"]
        self.assertEqual(
            [b["type"] for b in blocks], ["text", "image", "text", "image"]
        )
        self.assertEqual(
            [b["url"] for b in blocks if b["type"] == "image"],
            ["data:image/png;base64,AA==", "data:image/jpeg;base64,AQ=="],
        )
        for block in (
            {"type": "image", "data": "https://example.com/image.png"},
            {"type": "image", "data": "AA==", "mimeType": "image/svg+xml"},
            {"type": "image", "data": "not-base64", "mimeType": "image/png"},
        ):
            self.assertNotIn("blocks", _blocks_view([block], 40000))

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "run"
        self.root.mkdir()
        write_json(
            self.root / "info.json",
            {
                "schema_version": "roller.run/v1",
                "roller_version": "0.2.0",
                "state": "running",
                "updated_at": "2026-08-21T10:00:03+00:00",
                "configuration": {
                    "harness": "openclaw",
                    "trajectory_count": 2,
                    "openclaw": {
                        "thinking": "high",
                        "keep_thinking": True,
                    },
                },
                "summary": {
                    "total": 3,
                    "processed": 2,
                    "pending": 1,
                    "usage": {"total_tokens": 17, "request_count": 2},
                },
                "samples": [
                    {
                        "task_id": "shared-task",
                        "model": "provider/model-a",
                        "status": "ok",
                        "path": "samples/model-a/00000000_shared-task",
                    },
                    {
                        "task_id": "shared-task",
                        "model": "provider/model-b",
                        "status": "error",
                        "path": "samples/model-b/00000000_shared-task",
                    },
                    {
                        "task_id": "escape",
                        "model": "bad/model",
                        "status": "ok",
                        "path": "../outside",
                    },
                ],
            },
        )
        self.sample_a = self.root / "samples/model-a/00000000_shared-task"
        self.sample_b = self.root / "samples/model-b/00000000_shared-task"
        write_json(
            self.sample_a / "info.json",
            {
                "schema_version": "roller.sample/v1",
                "request_index": 0,
                "source_index": 0,
                "task_id": "shared-task",
                "model": "provider/model-a",
                "harness": "openclaw",
                "status": "ok",
                "error": None,
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                "timing": {"elapsed_seconds": 2.5},
                "attempts": [{"attempt": 1, "status": "ok"}],
                "trajectory": {"actf": "trace_ACTF.json"},
            },
        )
        write_json(
            self.sample_a / "trace_ACTF.json",
            {
                "schema_version": "ACTF_v1.0",
                "system_prompt": "System prompt",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "read", "parameters": {"type": "object"}},
                    }
                ],
                "steps": [
                    {
                        "step_id": 1,
                        "user_content": "Read the file",
                        "assistant_content": {
                            "reasoning_content": "r" * 2_000,
                            "content": "done",
                            "tool_calls": [
                                {
                                    "type": "toolCall",
                                    "id": "call-1",
                                    "name": "read",
                                    "arguments": {"path": "/tmp/input.txt"},
                                }
                            ],
                        },
                        "observation": [
                            {
                                "role": "toolResult",
                                "toolCallId": "call-1",
                                "toolName": "read",
                                "isError": False,
                                "content": [{"type": "text", "text": "file content"}],
                            }
                        ],
                        "metric": {
                            "prompt_tokens_len": 5,
                            "completion_tokens_len": 3,
                            "stop_reason": "toolUse",
                        },
                        "started_at": "2026-08-21T10:00:00+00:00",
                        "finished_at": "2026-08-21T10:00:02+00:00",
                    }
                ],
                "started_at": "2026-08-21T10:00:00+00:00",
                "finished_at": "2026-08-21T10:00:02+00:00",
            },
        )
        write_json(
            self.sample_b / "info.json",
            {
                "schema_version": "roller.sample/v1",
                "request_index": 1,
                "source_index": 0,
                "task_id": "shared-task",
                "model": "provider/model-b",
                "harness": "openclaw",
                "status": "error",
                "error": "model failed",
                "usage": {"total_tokens": 9},
                "trajectory": {"actf": "../../../../outside.json"},
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_indexes_tasks_groups_models_and_ignores_escaping_paths(self) -> None:
        source = RollerRun(self.root)

        summary = source.run_summary()
        self.assertEqual(summary["task_count"], 1)
        self.assertEqual(summary["rollout_count"], 2)
        self.assertEqual(summary["status_counts"], {"ok": 1, "error": 1})
        self.assertEqual(summary["configuration"]["thinking"], "high")
        self.assertEqual(summary["state"], "running")
        self.assertEqual(
            summary["progress"], {"total": 3, "processed": 2, "pending": 1}
        )

        tasks = source.list_tasks(page_size=10)
        self.assertEqual(tasks["total"], 1)
        self.assertEqual(len(tasks["items"][0]["rollouts"]), 2)
        errors = source.list_tasks(status="error", query="model-b")
        self.assertEqual(errors["total"], 1)
        self.assertEqual(errors["items"][0]["rollouts"][0]["status"], "error")

    def test_run_checkpoint_status_overlays_finalized_sample_status(self) -> None:
        info_path = self.root / "info.json"
        run_info = json.loads(info_path.read_text(encoding="utf-8"))
        run_info["samples"][0]["status"] = "skipped"
        write_json(info_path, run_info)

        source = RollerRun(self.root)
        rollout = source.list_tasks(model="provider/model-a")["items"][0]["rollouts"][0]

        self.assertEqual(rollout["status"], "skipped")
        self.assertEqual(source.run_summary()["status_counts"]["skipped"], 1)

    def test_projects_codex_reasoning_configuration(self) -> None:
        info_path = self.root / "info.json"
        run_info = json.loads(info_path.read_text(encoding="utf-8"))
        run_info["configuration"]["harness"] = "codex"
        run_info["configuration"]["codex"] = {
            "reasoning_effort": "high",
            "reasoning_summary": "detailed",
            "keep_thinking": True,
            "supports_websockets": False,
        }
        write_json(info_path, run_info)

        configuration = RollerRun(self.root).run_summary()["configuration"]

        self.assertEqual(configuration["harness"], "codex")
        self.assertEqual(configuration["thinking"], "high")
        self.assertEqual(configuration["reasoning_summary"], "detailed")
        self.assertTrue(configuration["keep_thinking"])

    def test_projects_actf_summaries_and_truncates_large_step_fields(self) -> None:
        source = RollerRun(self.root)
        rollout = source.list_tasks(model="provider/model-a")["items"][0]["rollouts"][0]

        detail = source.rollout_detail(rollout["id"])
        trajectory = detail["trajectory"]
        assert trajectory is not None
        self.assertEqual(trajectory["step_count"], 1)
        self.assertEqual(trajectory["steps"][0]["reasoning_chars"], 2_000)
        self.assertEqual(trajectory["steps"][0]["tool_call_count"], 1)

        step = source.step_detail(rollout["id"], 0, text_limit=1_000)
        self.assertTrue(step["assistant"]["reasoning"]["truncated"])
        self.assertEqual(len(step["assistant"]["reasoning"]["text"]), 1_000)
        self.assertEqual(step["assistant"]["tool_calls"][0]["name"], "read")
        self.assertEqual(step["observations"][0]["tool_call_id"], "call-1")

        failed = source.list_tasks(model="provider/model-b")["items"][0]["rollouts"][0]
        self.assertFalse(failed["has_actf"])
        self.assertIsNone(source.rollout_detail(failed["id"])["trajectory"])

    def test_user_message_images_are_served_as_image_blocks(self) -> None:
        trace_path = self.sample_a / "trace_ACTF.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace["steps"][0]["user_content"] = [
            {"type": "text", "text": "Describe the chart"},
            {"type": "image", "data": "AA==", "mimeType": "image/png"},
        ]
        trace["steps"][0]["assistant_content"] = {"tool_calls": []}
        write_json(trace_path, trace)

        source = RollerRun(self.root)
        rollout = source.list_tasks(model="provider/model-a")["items"][0]["rollouts"][0]

        user = source.step_detail(rollout["id"], 0)["user"]
        self.assertEqual(
            user["blocks"],
            [
                {"type": "text", "text": "Describe the chart"},
                {"type": "image", "url": "data:image/png;base64,AA=="},
            ],
        )
        self.assertEqual(user["text"], "Describe the chart\n<image>")
        step = source.rollout_detail(rollout["id"])["trajectory"]["steps"][0]
        self.assertEqual(step["user_preview"], "Describe the chart <image>")

    def grade(self, sample: Path, **record: object) -> None:
        info_path = sample / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["runtime"] = {
            "bench": {
                "bench": "wildclawbench",
                "category": "01_Productivity_Flow",
                "score_key": "overall_score",
                "error": None,
                **record,
            }
        }
        write_json(info_path, info)

    def test_runs_without_bench_data_report_no_scores(self) -> None:
        source = RollerRun(self.root)

        rollouts = source.list_tasks()["items"][0]["rollouts"]
        self.assertEqual([item["bench"] for item in rollouts], [None, None])
        self.assertEqual(source.run_summary()["score_counts"], {})
        self.assertEqual(source.list_tasks(score="zero")["total"], 0)

    def test_rollout_summary_reports_the_score_of_the_bench_report(self) -> None:
        sample_c = self.root / "samples/model-a/00000001_other-task"
        write_json(
            sample_c / "info.json",
            {"task_id": "other-task", "model": "provider/model-a", "source_index": 1},
        )
        scores = {"overall_score": 0.75, "accuracy": 1.0, "passed": True}
        self.grade(self.sample_a, scores=scores)
        skipped = "agent error; benchmark grading skipped"
        self.grade(
            self.sample_b,
            scores={"overall_score": 0.0, "score_eligible": False, "error": skipped},
            error=skipped,
        )
        self.grade(sample_c, scores={"overall_score": 0.0}, category="")
        source = RollerRun(self.root)

        def rollouts(**filters: str) -> list[dict]:
            tasks = source.list_tasks(**filters)["items"]
            return [rollout for task in tasks for rollout in task["rollouts"]]

        self.assertEqual(
            [item["bench"] for item in rollouts()],
            [
                {
                    "name": "wildclawbench",
                    "category": "01_Productivity_Flow",
                    "score": sample_score(scores, "overall_score"),
                    "error": "",
                },
                {
                    "name": "wildclawbench",
                    "category": "01_Productivity_Flow",
                    "score": None,
                    "error": skipped,
                },
                {"name": "wildclawbench", "category": "", "score": 0.0, "error": ""},
            ],
        )
        self.assertEqual(
            source.run_summary()["score_counts"],
            {"positive": 1, "unscored": 1, "zero": 1},
        )
        for group, task_id in (
            ("positive", "shared-task"),
            ("unscored", "shared-task"),
            ("zero", "other-task"),
        ):
            self.assertEqual(
                [item["task_id"] for item in rollouts(score=group)], [task_id]
            )
        self.assertEqual(
            [item["bench"]["score"] for item in rollouts(sort="score")],
            [0.0, 0.75, None],
        )
        self.assertEqual(
            [item["bench"]["score"] for item in rollouts(sort="-score")],
            [0.75, None, 0.0],
        )
        self.assertEqual(len(rollouts(query="grading skipped")), 1)
        self.assertEqual(len(rollouts(query="productivity")), 2)
        with self.assertRaisesRegex(viewer_app.ViewerDataError, "sort"):
            source.list_tasks(sort="tokens")

    def test_score_follows_the_score_key_and_a_grader_error_in_the_scores(self) -> None:
        self.grade(
            self.sample_a, scores={"reward": 1, "speed": 0.2}, score_key="reward"
        )
        self.grade(
            self.sample_b, scores={"overall_score": 0.0, "error": "judge failed"}
        )
        source = RollerRun(self.root)

        first, second = source.list_tasks()["items"][0]["rollouts"]

        self.assertEqual(first["bench"]["score"], 1.0)
        self.assertIsNone(second["bench"]["score"])
        self.assertEqual(second["bench"]["error"], "judge failed")

    def test_ui_strings_are_english_by_default_and_all_defined(self) -> None:
        static = Path(viewer_app.__file__).with_name("static")
        html = (static / "index.html").read_text(encoding="utf-8")
        script = (static / "app.js").read_text(encoding="utf-8")
        start = script.index("const STRINGS = {")
        end = script.index("\n  };", start)
        defined = set(re.findall(r"^    (\w+): \[", script[start:end], re.MULTILINE))
        in_page = set(re.findall(r'data-i18n(?:-[a-z-]+)?="(\w+)"', html))
        in_script = set(re.findall(r'\bth?\("(\w+)"', script))

        self.assertIn('<html lang="en">', html)
        # Chinese text lives in the table of app.js and nowhere else.
        self.assertNotRegex(html + script[:start] + script[end:], "[\u4e00-\u9fff]")
        self.assertTrue(in_page and in_script)
        self.assertEqual((in_page | in_script) - defined, set())

    def test_serves_static_ui_and_read_only_api(self) -> None:
        server = make_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(f"{address}/", timeout=5) as response:
                html = response.read().decode("utf-8")
                self.assertIn("Roller Trajectory Viewer", html)
                self.assertIn(
                    "default-src 'self'", response.headers["Content-Security-Policy"]
                )
            with urlopen(f"{address}/style.css", timeout=5) as response:
                css = response.read().decode("utf-8")
                self.assertIn("[hidden] { display: none !important; }", css)
            with urlopen(f"{address}/api/v1/run", timeout=5) as response:
                payload = json.loads(response.read())
                self.assertEqual(payload["rollout_count"], 2)
            with urlopen(f"{address}/api/v1/tasks?page_size=1", timeout=5) as response:
                payload = json.loads(response.read())
                self.assertEqual(payload["page_size"], 1)
                self.assertEqual(payload["total"], 1)
            with urlopen(
                f"{address}/api/v1/tasks?score=zero&sort=-score", timeout=5
            ) as response:
                self.assertEqual(json.loads(response.read())["total"], 0)
            with self.assertRaises(HTTPError) as raised:
                urlopen(f"{address}/api/v1/tasks?sort=tokens", timeout=5)
            self.assertEqual(raised.exception.code, 422)
            with self.assertRaises(HTTPError) as raised:
                urlopen(f"{address}/api/v1/rollouts/not-found", timeout=5)
            self.assertEqual(raised.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_cli_exposes_view_command_with_loopback_defaults(self) -> None:
        args = _build_parser().parse_args(["view", str(self.root)])
        self.assertEqual(args.command, "view")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertIsNone(args.port)

    def test_busy_default_port_falls_back_to_a_free_port(self) -> None:
        with socket.socket() as blocker:
            blocker.bind(("127.0.0.1", 0))
            blocker.listen()
            busy = blocker.getsockname()[1]
            with mock.patch.object(viewer_app, "DEFAULT_PORT", busy):
                server = make_server(self.root)
            try:
                self.assertNotEqual(server.server_address[1], busy)
                self.assertGreater(server.server_address[1], 0)
            finally:
                server.server_close()

    def test_busy_explicit_port_is_an_error(self) -> None:
        with socket.socket() as blocker:
            blocker.bind(("127.0.0.1", 0))
            blocker.listen()
            busy = blocker.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "another --port"):
                make_server(self.root, port=busy)


if __name__ == "__main__":
    unittest.main()
