from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from roller.cli import main
from roller.hf_dataset import (
    STATS_END,
    STATS_START,
    Rules,
    export_run,
    hermes_session,
    load_rules,
    session_trace,
)
from roller.storage import safe_component
from roller.viewer.app import RollerRun

SECRET_HOST = "10.20.30.40"
KEY = "sk-live-0123456789abcdef"
NODE = "gpu-node-7.cluster.internal"
MODEL_A = "vllm/model-a"
MODEL_B = "vllm/model-b"
# Sample directories as Roller names them for a provider/model reference.
DIR_A = safe_component(MODEL_A)
DIR_B = safe_component(MODEL_B)
PART_A = "openclaw-model-a"
PART_B = "opencode-model-b"

# An OpenClaw session as ``trace.jsonl`` holds it, with a second session after it.
NATIVE = [
    {"type": "session", "version": 3, "id": "roller", "cwd": "/tmp_workspace"},
    {
        "type": "message",
        "message": {"role": "user", "content": f"读取 http://{SECRET_HOST}/brief"},
    },
    {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "先读文件。"},
                {
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "read",
                    "arguments": {"path": "a"},
                },
            ],
            "stopReason": "toolUse",
        },
    },
    {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolCallId": "call-1",
            "toolName": "read",
            "content": [{"type": "text", "text": f'the "session" file, key {KEY}'}],
            "isError": False,
        },
    },
    {"type": "session", "version": 3, "id": "roller-2", "cwd": "/tmp_workspace"},
    {"type": "message", "message": {"role": "user", "content": "继续"}},
]

ACTF = {
    "schema_version": "ACTF_v1.0",
    "system_prompt": "You are an agent.",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ],
    "steps": [
        {
            "step_id": 1,
            "user_content": f"读取 http://{SECRET_HOST}/brief",
            "assistant_content": {
                "content": "",
                "reasoning_content": "先读文件。",
                "tool_calls": [
                    {"id": "call-1", "name": "read", "arguments": {"path": "a"}},
                    {
                        "id": "call-2",
                        "function": {"name": "bash", "arguments": '{"c":"ls"}'},
                    },
                ],
            },
            "observation": [
                {
                    "toolCallId": "call-1",
                    "content": [
                        {"type": "text", "text": "42"},
                        {"type": "image", "data": "data:image/png;base64,AAAA"},
                    ],
                },
                {"toolCallId": "call-2", "content": "a"},
            ],
            "metric": {
                "prompt_tokens_len": 100,
                "completion_tokens_len": 20,
                "cache_read_tokens": 30,
                "cache_write_tokens": 10,
                "stop_reason": "toolUse",
            },
            "started_at": "2026-11-16 03:00:00.500000",
        },
        {"step_id": 2, "assistant_content": {"content": "完成"}},
    ],
}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _sample(
    run_dir: Path,
    model: str,
    index: int,
    task: str,
    *,
    harness: str,
    status: str = "ok",
    score: float | None = None,
    grader_error: str | None = None,
) -> Path:
    """Write one sample with the names and fields a collection gives it."""

    sample = run_dir / "samples" / safe_component(model) / f"{index:08d}_{task}"
    native = harness == "openclaw"
    bench = {
        "bench": "wildclawbench",
        "category": "03_Social_Interaction",
        "metadata": {"upstream_task_id": task.split("__")[0], "run_index": 1},
        "score_key": "overall_score",
        "error": grader_error,
        "scores": {"overall_score": score or 0.0},
    }
    info = {
        "schema_version": "roller.sample/v1",
        "request_index": index,
        "sample_index": 0,
        "sample_count": 1,
        "task_id": task,
        "source_index": index,
        "model": model,
        "model_label": model,
        "harness": harness,
        "status": status,
        "error": None if status == "ok" else "task exceeded 300s",
        "usage": {"input_tokens": 900, "output_tokens": 100, "total_tokens": 1000},
        "timing": {"elapsed_seconds": 12.5},
        "trajectory": {
            "raw_schema": "openclaw-session-jsonl"
            if native
            else f"roller.{harness}-events/v1",
            "sft": "trace_SFT.json" if status == "ok" else None,
        },
        "runtime": {"api_base": f"http://{SECRET_HOST}:8000/v1", "node": NODE},
        "provenance": {
            "api_base_urls": [f"http://{SECRET_HOST}:8000/v1"],
            "environment_names": [
                "BRAVE_API_KEY",
                "JUDGE_MODEL",
                "LOCAL_API_KEY",
                "NO_PROXY",
                "OPENROUTER_API_KEY",
            ],
        },
    }
    if status == "ok":
        info["runtime"]["bench"] = bench
    _write(sample / "info.json", info)
    if status != "ok":
        return sample
    sample.joinpath("trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in NATIVE)
        if native
        else '{"schema": "roller.opencode-events/v1"}\n',
        encoding="utf-8",
    )
    _write(sample / "trace_ACTF.json", ACTF)
    _write(
        sample / "trace_SFT.json",
        {
            "messages": [
                {"role": "user", "content": f"读取 http://{SECRET_HOST}/brief"},
                {"role": "assistant", "content": f"{model} 完成", "tool_calls": []},
            ],
            "tools": [{"type": "function", "function": {"name": "read"}}],
        },
    )
    _write(sample / "score.json", bench)
    (sample / "results").mkdir()
    (sample / "results" / "answer.md").write_text("42\n", encoding="utf-8")
    (sample / "results" / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    (sample / "wire_trace.jsonl").write_text(
        json.dumps({"url": f"http://{SECRET_HOST}:8000/v1"}) + "\n", encoding="utf-8"
    )
    return sample


def _run(root: Path, name: str = "collected-run") -> Path:
    """A run of two models, as one ``roller collect`` with two ``--model`` writes it."""

    run_dir = root / name
    _write(
        run_dir / "info.json",
        {
            "schema_version": "roller.run/v1",
            "state": "finished",
            "updated_at": "2026-11-16T07:00:00+00:00",
            "finished_at": "2026-11-16T07:00:00+00:00",
            "configuration": {
                "harness": "mixed",
                "models": [MODEL_A, MODEL_B],
                "tasks": "/data/alice/runs/collected-run/tasks.jsonl",
                "bench": {"name": "wildclawbench", "root": "/data/alice/WildClawBench"},
                "openclaw": {"model_config_path": "/data/alice/configs/models.json"},
            },
        },
    )
    (run_dir / "tasks.jsonl").write_text(
        '{"id": "chat", "workspace": "/data/alice/WildClawBench/workspace/chat"}\n',
        encoding="utf-8",
    )
    _write(run_dir / "bench-report.json", {"covers": "both models"})
    _sample(run_dir, MODEL_A, 0, "chat__run_01", harness="openclaw", score=1.0)
    _sample(run_dir, MODEL_B, 1, "chat__run_01", harness="opencode", score=0.25)
    _sample(run_dir, MODEL_A, 2, "chat__run_02", harness="openclaw", score=0.5)
    _sample(
        run_dir,
        MODEL_B,
        3,
        "chat__run_02",
        harness="opencode",
        grader_error="judge timed out",
    )
    _sample(run_dir, MODEL_B, 4, "plan__run_01", harness="opencode", status="error")
    return run_dir


def _export(
    run_dir: Path, output: Path, rules: Rules | None = None, runs: bool = False
) -> list[dict]:
    """Export with no credential value in the environment and no machine name."""

    return export_run(run_dir, output, rules, environ={}, runs=runs, machine="")[0]


def _rows(output: Path, part: str) -> list[dict]:
    lines = (output / "data" / f"{part}.jsonl").read_text("utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


class HuggingFaceExportTests(unittest.TestCase):
    def test_model_aliases_can_share_one_public_part(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root)
            _sample(run, "vllm/model-a-replica", 5, "plan__run_02", harness="openclaw")
            rules = Rules(
                replace=((re.compile("model-a(?:-replica)?"), "public-model"),)
            )
            output = root / "dataset"
            _export(run, output, rules)
            rows = _rows(output, "openclaw-public-model")
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["model"] for row in rows}, {"vllm/public-model"})
            self.assertEqual(len({row["trace"] for row in rows}), 3)
            self.assertTrue(all((output / row["trace"]).is_file() for row in rows))

    def test_model_alias_rules_also_apply_to_generated_names_and_sample_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root)
            rules = Rules(replace=((re.compile("model-a"), "public-model"),))
            rows, _ = export_run(
                run, root / "dataset", rules=rules, environ={}, machine=""
            )
            selected = [r for r in rows if r["harness"] == "openclaw"]
            self.assertEqual(len(selected), 2)
            for row in selected:
                self.assertEqual(row["run"], "openclaw-public-model")
                self.assertNotIn("model-a", row["sample"])
                self.assertTrue(row["files"])
                trace = root / "dataset" / row["trace"]
                self.assertTrue(trace.is_file())
                self.assertNotIn("model-a", trace.read_text())

    def test_task_keys_can_survive_export_without_collection_addresses(self) -> None:
        key = "sk-benchmark-fixture-0123456789abcdef"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root)
            sample = next(run.glob("samples/*/*/trace_SFT.json"))
            original = json.loads(sample.read_text())
            original["messages"].append({"role": "user", "content": key})
            sample.write_text(json.dumps(original))
            before = sample.read_bytes()
            rows, _ = export_run(
                run, root / "dataset", environ={}, machine="", preserve_task_keys=True
            )
            messages = "\n".join(row["messages"] or "" for row in rows)
            self.assertIn(key, messages)
            self.assertNotIn(SECRET_HOST, messages)
            self.assertEqual(sample.read_bytes(), before)

    def test_run_is_split_into_one_part_per_harness_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"

            rows = _export(_run(Path(tmp)), output)

            self.assertEqual(len(rows), 5)
            self.assertEqual(
                sorted(path.name for path in (output / "data").iterdir()),
                [f"{PART_A}.jsonl", f"{PART_B}.jsonl"],
            )
            # The folder holds what is published: the card, the tables, the sessions.
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["README.md", "data", "traces"],
            )
            first = _rows(output, PART_A)[0]
            self.assertEqual(first["run"], PART_A)
            self.assertEqual(first["sample"], f"samples/{DIR_A}/00000000_chat__run_01")
            self.assertEqual(
                first["trace"], f"traces/{PART_A}/00000000_chat__run_01.jsonl"
            )
            self.assertEqual(first["task_id"], "chat__run_01")
            self.assertEqual(first["upstream_task_id"], "chat")
            self.assertEqual(first["run_index"], 1)
            self.assertEqual(first["bench"], "wildclawbench")
            self.assertEqual(first["category"], "03_Social_Interaction")
            self.assertEqual((first["model"], first["harness"]), (MODEL_A, "openclaw"))
            self.assertEqual(first["score"], 1.0)
            self.assertEqual((first["steps"], first["tool_calls"]), (2, 2))
            self.assertEqual(first["total_tokens"], 1000)
            self.assertEqual(first["files"], ["answer.md", "chart.png"])
            self.assertEqual(
                json.loads(first["messages"])[1]["content"], f"{MODEL_A} 完成"
            )
            self.assertEqual(json.loads(first["tools"])[0]["function"]["name"], "read")

    def test_unscored_and_failed_samples_keep_their_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"

            _export(_run(Path(tmp)), output)

            rows = {row["task_id"]: row for row in _rows(output, PART_B)}
            unscored = rows["chat__run_02"]
            self.assertIsNone(unscored["score"])
            self.assertEqual(unscored["grader_error"], "judge timed out")
            failed = rows["plan__run_01"]
            self.assertEqual(failed["status"], "error")
            self.assertIsNone(failed["trace"])
            self.assertIsNone(failed["messages"])
            self.assertEqual(failed["steps"], 0)

    def test_run_directories_are_written_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"

            rows = _export(_run(Path(tmp)), output, runs=True)

            self.assertEqual(
                sorted(path.name for path in (output / "runs").iterdir()),
                [PART_A, PART_B],
            )
            self.assertEqual(rows[0]["files"], ["answer.md", "chart.png"])
            part = output / "runs" / PART_B
            info = json.loads((part / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [entry["path"] for entry in info["samples"]],
                [
                    f"samples/{DIR_B}/00000001_chat__run_01",
                    f"samples/{DIR_B}/00000003_chat__run_02",
                    f"samples/{DIR_B}/00000004_plan__run_01",
                ],
            )
            summary = info["summary"]
            self.assertEqual(
                (summary["total"], summary["ok"], summary["error"]), (3, 2, 1)
            )
            self.assertEqual(summary["usage"]["total_tokens"], 3000)
            # The times and the configuration of the collection are kept.
            self.assertEqual(info["finished_at"], "2026-11-16T07:00:00+00:00")
            self.assertEqual(info["configuration"]["models"], [MODEL_A, MODEL_B])
            self.assertEqual(
                [row["messages"][1]["content"] for row in _lines(part / "sft.jsonl")],
                [f"{MODEL_B} 完成", f"{MODEL_B} 完成"],
            )
            self.assertEqual(
                (part / "tasks.jsonl").read_text(),
                '{"id": "chat", "workspace": "<bench-root>/workspace/chat"}\n',
            )
            report = json.loads((part / "bench-report.json").read_text("utf-8"))
            self.assertNotIn("covers", report)
            self.assertFalse((part / "samples" / DIR_A).exists())

            summary = RollerRun(part).run_summary()
            self.assertEqual(summary["name"], PART_B)
            self.assertEqual(summary["models"], [MODEL_B])

    def test_native_session_is_published_with_its_harness_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            output = Path(tmp) / "dataset"

            _export(run_dir, output)

            traces = sorted(
                path.relative_to(output).as_posix()
                for path in (output / "traces").rglob("*.jsonl")
            )
            self.assertEqual(
                traces,
                [
                    f"traces/{PART_A}/00000000_chat__run_01.jsonl",
                    f"traces/{PART_A}/00000002_chat__run_02.jsonl",
                    f"traces/{PART_B}/00000001_chat__run_01.jsonl",
                    f"traces/{PART_B}/00000003_chat__run_02.jsonl",
                ],
            )
            native = run_dir / "samples" / DIR_A / "00000000_chat__run_01/trace.jsonl"
            source = native.read_text("utf-8").splitlines()
            lines = (output / traces[0]).read_text("utf-8").splitlines()
            self.assertEqual(
                json.loads(lines[0]),
                {
                    "type": "session",
                    "version": 3,
                    "id": f"{PART_A}/00000000_chat__run_01",
                    "cwd": "/tmp_workspace",
                    "harness": "openclaw",
                    "name": f"chat__run_01 · {MODEL_A}",
                    "model": MODEL_A,
                    "roller": {
                        "run": PART_A,
                        "sample": f"samples/{DIR_A}/00000000_chat__run_01",
                        "task_id": "chat__run_01",
                        "status": "ok",
                        "score": 1.0,
                    },
                },
            )
            self.assertEqual(
                json.loads(lines[4])["id"], f"{PART_A}/00000000_chat__run_01#2"
            )
            # Every other line is the native line; only the endpoint is replaced.
            native_lines = [
                line.replace(SECRET_HOST, "endpoint-1.example").replace(
                    KEY, "[REDACTED]"
                )
                for line in source
            ]
            self.assertEqual(
                lines[1:4] + lines[5:], native_lines[1:4] + native_lines[5:]
            )
            card = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("data_files: traces/**/*.jsonl\n  default: true", card)
            self.assertIn("data_files: data/*.jsonl", card)

            # A sample that left the run leaves no trace behind on the next export.
            shutil.rmtree(run_dir / "samples" / DIR_A / "00000002_chat__run_02")
            _export(run_dir, output)
            self.assertFalse((output / traces[1]).exists())

    def test_other_harnesses_get_a_session_converted_from_actf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"

            _export(_run(Path(tmp)), output)

            lines = _lines(output / "traces" / PART_B / "00000001_chat__run_01.jsonl")
            self.assertEqual(
                lines[0],
                {
                    "type": "session",
                    "version": 3,
                    "timestamp": "2026-11-16T03:00:00.500Z",
                    "cwd": "",
                    "harness": "opencode",
                    "id": f"{PART_B}/00000001_chat__run_01",
                    "name": f"chat__run_01 · {MODEL_B}",
                    "model": MODEL_B,
                    "roller": {
                        "run": PART_B,
                        "sample": f"samples/{DIR_B}/00000001_chat__run_01",
                        "task_id": "chat__run_01",
                        "status": "ok",
                        "score": 0.25,
                    },
                },
            )
            self.assertEqual(
                [line.get("message", {}).get("role") for line in lines[3:]],
                [
                    "system",
                    "user",
                    "assistant",
                    "toolResult",
                    "toolResult",
                    "assistant",
                ],
            )

    def test_failed_deepseek_sample_exports_its_redacted_native_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root)
            output = root / "dataset"
            sample = _sample(
                run,
                "deepseek-model",
                5,
                "chat__run_01",
                harness="deepseek",
                status="error",
            )
            events = [
                {"type": "metadata", "schema": "roller.deepseek-events/v1"},
                {"type": "request", "url": f"http://{SECRET_HOST}:8000/v1"},
            ]
            (sample / "trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
            )

            _export(run, output)

            row = _rows(output, "deepseek-deepseek-model")[0]
            self.assertEqual(row["status"], "error")
            self.assertIsNone(row["messages"])
            self.assertEqual(
                _lines(output / row["trace"]),
                [
                    events[0],
                    {"type": "request", "url": "http://endpoint-1.example/v1"},
                ],
            )

    def test_opencode_exports_native_fallback_with_redacted_proxy_assignments(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root)
            output = root / "dataset"
            sample = _sample(
                run, MODEL_B, 5, "chat__run_01", harness="opencode", status="error"
            )
            schema = "roller.opencode-events/v1"
            proxy = "grader.example,10.31.0.0/16"
            loopback = "localhost,127.0.0.1,::1"
            events = [
                {"type": "metadata", "schema": schema},
                {"type": "diagnostic", "text": f"NO_PROXY={loopback},{proxy}"},
                {"type": "diagnostic", "text": f"no_proxy={loopback}"},
                {
                    "type": "text",
                    "text": f"Task fixture: NO_PROXY={loopback},example.org",
                },
            ]
            (sample / "trace.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )

            export_run(run, output, environ={"NO_PROXY": proxy}, machine="", runs=False)

            row = next(
                row
                for row in _rows(output, PART_B)
                if row["status"] == "error" and row["trace"]
            )
            header, *trace = _lines(output / row["trace"])
            self.assertEqual(header["harness"], "opencode")
            self.assertEqual(header["source_schema"], schema)
            self.assertIsNone(row["messages"])
            self.assertEqual(
                trace,
                [
                    events[0],
                    {"type": "diagnostic", "text": "NO_PROXY=localhost,127.0.0.1"},
                    {"type": "diagnostic", "text": "no_proxy=localhost,127.0.0.1"},
                    events[-1],
                ],
            )

    def test_codex_trace_survives_hub_line_splitting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            output = Path(tmp) / "dataset"
            sample = run_dir / "samples" / DIR_A / "00000000_chat__run_01"
            info_path = sample / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["harness"] = "codex"
            info["trajectory"]["raw_schema"] = "codex-session-jsonl"
            info_path.write_text(json.dumps(info), encoding="utf-8")
            events = [
                {"type": "session_meta", "payload": {"id": "test"}},
                {
                    "type": "event_msg",
                    "payload": {"message": "前\u0085中\u2028后\u2029"},
                },
            ]
            (sample / "trace.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events),
                encoding="utf-8",
            )

            _export(run_dir, output)

            trace = output / "traces/codex-model-a/00000000_chat__run_01.jsonl"
            self.assertEqual(
                [json.loads(line) for line in trace.read_text("utf-8").splitlines()],
                events,
            )

    def test_hermes_journal_becomes_a_hermes_session(self) -> None:
        call = {"id": "c1", "type": "function", "function": {"name": "terminal"}}
        events = [
            {
                "type": "participant",
                "timestamp": "2026-11-20T17:51:37+00:00",
                "session_id": "native",
            },
            {
                "type": "request",
                "timestamp": "2026-11-20T17:51:38+00:00",
                "model": "m",
                "messages": [{"role": "system", "content": "You are Hermes."}],
                "tools": [],
            },
            {
                "type": "message",
                "timestamp": "2026-11-20T17:51:38+00:00",
                "message": {"role": "user", "content": "Run ls."},
            },
            {
                "type": "message",
                "timestamp": "2026-11-20T17:51:40+00:00",
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 9},
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "List it.",
                    "tool_calls": [call],
                },
            },
            {
                "type": "message",
                "timestamp": "2026-11-20T17:51:41+00:00",
                "message": {"role": "tool", "content": "a", "tool_call_id": "c1"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "trace.jsonl"
            raw.write_text("".join(json.dumps(event) + "\n" for event in events))
            session = hermes_session(raw, {"id": "s1", "model": "m"})

        start = 1795197097.0
        self.assertEqual(
            session,
            {
                "id": "s1",
                "model": "m",
                "source": "cli",
                "system_prompt": "You are Hermes.",
                "started_at": start,
                "ended_at": start + 4,
                "message_count": 3,
                "tool_call_count": 1,
                "messages": [
                    {"role": "user", "content": "Run ls.", "timestamp": start + 1},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "List it.",
                        "tool_calls": [call],
                        "timestamp": start + 3,
                        "finish_reason": "tool_calls",
                    },
                    {
                        "role": "tool",
                        "content": "a",
                        "tool_call_id": "c1",
                        "timestamp": start + 4,
                    },
                ],
            },
        )

    def test_session_trace_is_a_pi_session(self) -> None:
        lines = session_trace(
            ACTF, {"harness": "opencode", "id": "s1", "name": "demo", "model": "p/m"}
        )

        time = "2026-11-16T03:00:00.500Z"
        self.assertEqual(
            lines[:3],
            [
                {
                    "type": "session",
                    "version": 3,
                    "timestamp": time,
                    "cwd": "",
                    "harness": "opencode",
                    "id": "s1",
                    "name": "demo",
                    "model": "p/m",
                },
                {
                    "type": "model_change",
                    "id": "00000001",
                    "parentId": None,
                    "timestamp": time,
                    "provider": "p",
                    "modelId": "m",
                },
                {
                    "type": "session_info",
                    "id": "00000002",
                    "parentId": "00000001",
                    "timestamp": time,
                    "name": "demo",
                },
            ],
        )
        self.assertEqual(
            [(line["id"], line["parentId"]) for line in lines[3:]],
            [(f"{index:08x}", f"{index - 1:08x}") for index in range(3, 9)],
        )
        self.assertEqual({line["timestamp"] for line in lines[3:]}, {time})
        self.assertEqual({line["type"] for line in lines[3:]}, {"message"})
        messages = [line["message"] for line in lines[3:]]
        self.assertEqual(
            {message.pop("timestamp") for message in messages}, {1794798000500}
        )
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": "You are an agent.",
                    "toolsAdded": [
                        {
                            "name": "read",
                            "description": "Read a file",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"读取 http://{SECRET_HOST}/brief"}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "先读文件。"},
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "read",
                            "arguments": {"path": "a"},
                        },
                        {
                            "type": "toolCall",
                            "id": "call-2",
                            "name": "bash",
                            "arguments": {"c": "ls"},
                        },
                    ],
                    "provider": "p",
                    "model": "m",
                    "usage": {
                        "input": 60,
                        "output": 20,
                        "cacheRead": 30,
                        "cacheWrite": 10,
                        "totalTokens": 120,
                    },
                    "stopReason": "toolUse",
                },
                {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "toolName": "read",
                    "content": [
                        {"type": "text", "text": "42"},
                        {"type": "image", "data": "AAAA", "mimeType": "image/png"},
                    ],
                    "isError": False,
                },
                {
                    "role": "toolResult",
                    "toolCallId": "call-2",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "a"}],
                    "isError": False,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "完成"}],
                    "provider": "p",
                    "model": "m",
                },
            ],
        )

    def test_later_runs_add_their_parts_to_the_same_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = _run(Path(tmp))
            later = Path(tmp) / "later-run"
            _write(later / "info.json", {"schema_version": "roller.run/v1"})
            _sample(
                later, "vllm/model-c", 0, "chat__run_01", harness="codex", score=1.0
            )
            output = Path(tmp) / "dataset"

            _export(first, output)
            _export(later, output)

            self.assertEqual(
                sorted(path.name for path in (output / "data").iterdir()),
                ["codex-model-c.jsonl", f"{PART_A}.jsonl", f"{PART_B}.jsonl"],
            )
            card = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("| `codex-model-c` | codex | `model-c` | 1 |", card)
            self.assertIn(f"| `{PART_A}` | openclaw | `model-a` | 2 |", card)

            # Exporting one run again leaves the parts of the other run as they are.
            shutil.rmtree(first / "samples" / DIR_A / "00000002_chat__run_02")
            _export(first, output)
            card = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn(f"| `{PART_A}` | openclaw | `model-a` | 1 |", card)
            self.assertIn("| `codex-model-c` | codex | `model-c` | 1 |", card)
            self.assertTrue((output / "traces" / "codex-model-c").is_dir())

    def test_export_without_run_directories_replaces_an_earlier_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            output = Path(tmp) / "dataset"

            _export(run_dir, output, runs=True)
            self.assertIn("runs/", (output / "README.md").read_text("utf-8"))
            _export(run_dir, output)

            self.assertFalse((output / "runs").exists())
            self.assertEqual(len(_rows(output, PART_A) + _rows(output, PART_B)), 5)

    def test_parts_that_share_a_model_id_are_named_after_the_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _sample(run_dir, "east/model-a", 0, "chat", harness="codex", score=1.0)
            _sample(run_dir, "west/model-a", 1, "chat", harness="codex", score=1.0)

            rows = _export(run_dir, Path(tmp) / "dataset")

            self.assertEqual(
                sorted(row["run"] for row in rows),
                [
                    safe_component("codex-east/model-a"),
                    safe_component("codex-west/model-a"),
                ],
            )

    def test_card_reports_each_part_and_keeps_edits_on_reexport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            output = Path(tmp) / "dataset"

            _export(run_dir, output)

            card = output / "README.md"
            text = card.read_text(encoding="utf-8")
            self.assertIn(
                f"| `{PART_A}` | openclaw | `model-a` | 2 | 2 | 2 | 0.750 | 2.0 | 2,000 |",
                text,
            )
            # Three samples, one scored 0.25, one unscored and one failed.
            self.assertIn(
                f"| `{PART_B}` | opencode | `model-b` | 3 | 2 | 1 | 0.083 | 1.3 | 3,000 |",
                text,
            )

            self.assertNotIn("runs/", text)

            card.write_text(
                text.replace("# collected-run", "# My dataset"), encoding="utf-8"
            )
            shutil.rmtree(run_dir / "samples" / DIR_B / "00000004_plan__run_01")
            _export(run_dir, output)

            text = card.read_text(encoding="utf-8")
            self.assertIn("# My dataset", text)
            self.assertIn(
                f"| `{PART_B}` | opencode | `model-b` | 2 | 2 | 1 | 0.125 | 2.0 | 2,000 |",
                text,
            )
            self.assertEqual(text.count(STATS_START), 1)
            self.assertEqual(text.count(STATS_END), 1)

    def test_card_has_no_token_column_for_a_run_without_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            for path in run_dir.glob("samples/*/*/info.json"):
                info = json.loads(path.read_text(encoding="utf-8"))
                info["usage"] = {"input_tokens": 0, "output_tokens": 0}
                _write(path, info)
            output = Path(tmp) / "dataset"

            _export(run_dir, output)

            text = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("| Scored | Mean score | Mean steps |\n", text)
            self.assertIn(
                f"| `{PART_A}` | openclaw | `model-a` | 2 | 2 | 2 | 0.750 | 2.0 |\n",
                text,
            )

    def test_export_redacts_what_the_run_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            output = Path(tmp) / "dataset"
            environ = {
                "OPENROUTER_API_KEY": KEY,
                "JUDGE_MODEL": "model-a",
                # A stand-in value: replacing it would rewrite ordinary words.
                "LOCAL_API_KEY": "local",
            }

            _, redaction = export_run(
                run_dir, output, environ=environ, runs=True, machine=""
            )

            self.assertEqual(redaction.endpoints, (f"{SECRET_HOST}:8000",))
            self.assertEqual(
                redaction.paths,
                (
                    str(run_dir.resolve()),
                    "/data/alice/WildClawBench",
                    "/data/alice/runs/collected-run",
                    "/data/alice/configs",
                ),
            )
            self.assertEqual(redaction.credentials, ("OPENROUTER_API_KEY",))
            self.assertEqual(redaction.unset, ("BRAVE_API_KEY",))
            self.assertEqual(
                redaction.lines(),
                [
                    "Redacted: 1 endpoints, 4 host paths, "
                    "the values of OPENROUTER_API_KEY",
                    "Not set in this environment, so their values were not "
                    "searched for: BRAVE_API_KEY",
                ],
            )

            part = output / "runs" / PART_A
            sample = part / "samples" / DIR_A / "00000000_chat__run_01"
            info = json.loads((sample / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(
                info["runtime"]["api_base"], "http://endpoint-1.example/v1"
            )
            # The model name is a value of a passed variable, and no credential.
            self.assertEqual(info["model"], MODEL_A)
            configuration = json.loads((part / "info.json").read_text("utf-8"))[
                "configuration"
            ]
            self.assertEqual(configuration["tasks"], "<tasks-dir>/tasks.jsonl")
            self.assertEqual(configuration["bench"]["root"], "<bench-root>")
            self.assertEqual(
                configuration["openclaw"]["model_config_path"],
                "<config-dir>/models.json",
            )
            exported = [
                path
                for path in output.rglob("*")
                if path.is_file() and path.suffix != ".png"
            ]
            text = "".join(path.read_text("utf-8") for path in exported)
            self.assertNotIn(SECRET_HOST, text)
            self.assertNotIn(KEY, text)
            self.assertNotIn("/data/alice", text)
            trace = output / "traces" / PART_A / "00000000_chat__run_01.jsonl"
            self.assertIn("key [REDACTED]", trace.read_text("utf-8"))
            self.assertIn("http://endpoint-1.example/brief", trace.read_text("utf-8"))
            self.assertEqual(
                (sample / "results" / "chart.png").read_bytes(),
                b"\x89PNG\r\n\x1a\n\xff\xfe",
            )
            # The source run is read, never rewritten.
            source = run_dir / "samples" / DIR_A / "00000000_chat__run_01/info.json"
            self.assertIn(SECRET_HOST, source.read_text("utf-8"))

    def test_export_redacts_proxy_exceptions_machine_name_and_run_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            machine = "devbox.alice.team-ns.svc.corp.local"
            no_proxy = (
                "localhost,127.0.0.1,10.0.0.0/8,.svc,.corp.example.cn,"
                f"{SECRET_HOST},100.102.47.241,10.140.52.0/24"
            )
            seen = [
                f"NO_PROXY={no_proxy}",
                "Could not resolve mirror.corp.example.cn (100.102.47.241)",
                "public address 1100.102.47.2410 stays",
                f"Author: root <root@{machine}>",
                "search alice.team-ns.svc.corp.local svc.corp.local",
                "Runtime: host=devbox | os=Linux",
                "homepage https://devbox.github.io and devbox-2 stay",
                f"wrote {run_dir.resolve()}/samples/log.txt",
            ]
            trace = run_dir / "samples" / DIR_A / "00000000_chat__run_01/trace.jsonl"
            with trace.open("a", encoding="utf-8") as writer:
                for text in seen:
                    row = {
                        "type": "message",
                        "message": {"role": "user", "content": text},
                    }
                    writer.write(json.dumps(row) + "\n")
            output = Path(tmp) / "dataset"

            _, redaction = export_run(
                run_dir, output, environ={"NO_PROXY": no_proxy}, machine=machine
            )

            self.assertEqual(
                redaction.hosts,
                ("10.140.52.0/24", "100.102.47.241", "corp.example.cn"),
            )
            self.assertEqual(redaction.machine, machine)
            self.assertEqual(
                redaction.lines()[0],
                "Redacted: 1 endpoints, 3 NO_PROXY hosts, the name of this machine, "
                "4 host paths",
            )
            exported = _lines(
                output / "traces" / PART_A / "00000000_chat__run_01.jsonl"
            )
            self.assertEqual(
                [row["message"]["content"] for row in exported[-len(seen) :]],
                [
                    "NO_PROXY=localhost,127.0.0.1",
                    "Could not resolve mirror.internal-3.example (internal-2.example)",
                    "public address 1100.102.47.2410 stays",
                    "Author: root <root@collection-host.example>",
                    "search collection-domain.example svc.corp.local",
                    "Runtime: host=collection-host | os=Linux",
                    "homepage https://devbox.github.io and devbox-2 stay",
                    "wrote <run-dir>/samples/log.txt",
                ],
            )

    def test_redaction_can_be_turned_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            output = Path(tmp) / "dataset"

            _, redaction = export_run(
                run_dir,
                output,
                environ={"OPENROUTER_API_KEY": KEY},
                redact=False,
                runs=True,
            )

            self.assertEqual(redaction.lines(), [])
            part = output / "runs" / PART_A
            sample = part / "samples" / DIR_A / "00000000_chat__run_01"
            info = json.loads((sample / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(
                info["runtime"]["api_base"], f"http://{SECRET_HOST}:8000/v1"
            )
            trace = output / "traces" / PART_A / "00000000_chat__run_01.jsonl"
            self.assertIn(f"key {KEY}", trace.read_text("utf-8"))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit):
                main(
                    [
                        "export",
                        str(run_dir),
                        "--format",
                        "hf",
                        "--no-redact",
                        "--include-runs",
                    ]
                )
            self.assertEqual(len(stdout.getvalue().splitlines()), 2)
            configuration = json.loads(
                (run_dir / "hf" / "runs" / PART_A / "info.json").read_text("utf-8")
            )["configuration"]
            self.assertEqual(
                configuration["bench"]["root"], "/data/alice/WildClawBench"
            )

    def test_rules_add_to_the_redaction_of_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            output = Path(tmp) / "dataset"
            rules_path = Path(tmp) / "redact.json"
            _write(
                rules_path,
                {
                    "replace": [
                        {"pattern": r"gpu-node-\d+\.cluster\.internal", "with": "node"}
                    ],
                    "exclude": ["samples/*/wire_trace.jsonl"],
                    "forbid": [r"cluster\.internal"],
                },
            )

            _export(run_dir, output, load_rules(rules_path), runs=True)

            sample = output / "runs" / PART_A / "samples" / DIR_A
            sample = sample / "00000000_chat__run_01"
            self.assertFalse((sample / "wire_trace.jsonl").exists())
            info = json.loads((sample / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["runtime"]["node"], "node")
            self.assertEqual(
                info["runtime"]["api_base"], "http://endpoint-1.example/v1"
            )

    def test_forbidden_pattern_left_in_a_file_stops_the_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            rules = Rules(forbid=(re.compile(r"cluster\.internal"),))

            with self.assertRaisesRegex(
                ValueError, rf"samples/{DIR_A}/00000000_chat__run_01/info\.json:1"
            ):
                _export(run_dir, Path(tmp) / "dataset", rules)

    def test_default_output_inside_the_run_is_not_copied_into_itself(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            (run_dir / "tokens").mkdir()
            (run_dir / "tokens" / "tokens.jsonl").write_text("{}\n")
            stdout = io.StringIO()

            unset = dict.fromkeys(
                ("BRAVE_API_KEY", "LOCAL_API_KEY", "OPENROUTER_API_KEY"), ""
            )
            for _ in range(2):
                with (
                    mock.patch.dict(os.environ, unset),
                    mock.patch("roller.hf_dataset.machine_name", return_value=""),
                    contextlib.redirect_stdout(stdout),
                    self.assertRaises(SystemExit) as exit,
                ):
                    main(["export", str(run_dir), "--format", "hf", "--include-runs"])
                self.assertEqual(exit.exception.code, 0)

            output = run_dir / "hf"
            self.assertEqual(len(_rows(output, PART_A) + _rows(output, PART_B)), 5)
            self.assertFalse((output / "runs" / PART_A / "hf").exists())
            self.assertFalse((output / "runs" / PART_A / "tokens").exists())
            printed = stdout.getvalue().splitlines()
            self.assertEqual(
                printed[0], f"Exported 5 samples as a Hugging Face dataset to {output}"
            )
            self.assertEqual(printed[1], "Redacted: 1 endpoints, 4 host paths")
            self.assertEqual(
                printed[2],
                "Not set in this environment, so their values were not searched "
                "for: BRAVE_API_KEY, LOCAL_API_KEY, OPENROUTER_API_KEY",
            )


if __name__ == "__main__":
    unittest.main()


class ImportedRunRedactionTests(unittest.TestCase):
    def _session(self, path: Path, session_id: str, cwd: str, text: str) -> None:
        records = [
            {
                "type": "bridge-session",
                "sessionId": session_id,
                "ownerAccountUuid": "acct-1111-2222",
            },
            {
                "type": "user",
                "uuid": f"{session_id}-1",
                "timestamp": "2026-11-01T10:00:00Z",
                "sessionId": session_id,
                "cwd": cwd,
                "message": {"role": "user", "content": text},
            },
            {
                "type": "assistant",
                "uuid": f"{session_id}-2",
                "timestamp": "2026-11-01T10:00:01Z",
                "sessionId": session_id,
                "cwd": cwd,
                "message": {
                    "id": "m1",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "ok"}],
                },
            },
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )

    def test_home_paths_user_names_and_accounts_of_imported_sessions(self) -> None:
        from roller.importer import discover, import_sessions

        token = "tok-abcdefghijklmnop"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(
                root / "src" / "-Users-alice-work-demo" / "s1.jsonl",
                "s1",
                "/Users/alice/work/demo",
                "Read /Users/alice/notes.txt and "
                "~/.claude/projects/-Users-alice-work-demo/s1.jsonl. "
                f"Alice wrote it. ALICE SMITH, alice@example.org, {token}\n"
                "alice\n/tmp/pytest-of-alice/pytest-4, _Users_alice_.cache, "
                "alice.github.io, github.com/search?q=repo%3Aalice%2Fdemo, USERalice",
            )
            self._session(
                root / "src" / "-home-bob-proj" / "s2.jsonl",
                "s2",
                "/home/bob/proj",
                "Open /home/bob/proj/a.py; bob stays a word",
            )
            run = root / "run"
            import_sessions(run, discover([root / "src"])[0], parallel=1)
            output = root / "hf"
            with (
                mock.patch.object(Path, "home", return_value=Path("/Users/alice")),
                mock.patch(
                    "roller.hf_dataset._git_identity",
                    return_value=("Alice Smith", "alice@example.org"),
                ),
            ):
                _, redaction = export_run(
                    run,
                    output,
                    environ={"MY_API_TOKEN": token, "SHORT_KEY": "abc"},
                    machine="",
                    runs=True,
                )
            text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.is_file()
            )
            # A name joined to a word is left in place and reported.
            text = text.replace("USERalice", "")
            for leaked in ("alice", "/home/bob", "acct-1111-2222", token):
                self.assertNotIn(leaked, text.lower())
            self.assertIn("/Users/user/notes.txt", text)
            self.assertIn("-Users-user-work-demo", text)
            self.assertIn("/home/user-2/proj/a.py", text)
            self.assertIn("bob stays a word", text)
            self.assertEqual(redaction.users, ("alice", "bob"))
            self.assertEqual(redaction.identities, 1)
            self.assertEqual(redaction.credentials, ("MY_API_TOKEN",))
            report = json.loads(
                (root / "hf.redaction-report.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("home_path", report["counts"])
            self.assertNotIn("email", report["counts"])
            self.assertGreater(report["counts"]["user_name"], 0)
