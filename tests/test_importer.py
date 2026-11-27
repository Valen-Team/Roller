from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import zstandard

from roller.importer import discover, import_sessions

CWD = "/Users/alice/work/demo"
MODEL = "claude-test-1"
TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}
LATER_TOOL = {
    "name": "mcp__pr__status",
    "description": "PR status",
    "input_schema": {"type": "object", "properties": {}},
}


def _jsonl(path: Path, records: list[dict], tail: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records) + tail,
        encoding="utf-8",
    )
    return path


def _zst(path: Path, records: list[dict]) -> None:
    """Compress a session as two zstd frames."""

    lines = "".join(json.dumps(record) + "\n" for record in records).splitlines(True)
    compressor = zstandard.ZstdCompressor()
    path.write_bytes(
        b"".join(
            compressor.compress("".join(part).encode())
            for part in (lines[:3], lines[3:])
        )
    )


class Claude:
    """Records of one Claude Code session with increasing timestamps."""

    def __init__(self, session_id: str = "s1") -> None:
        self.session_id = session_id
        self.count = 0

    def _base(self, kind: str, **extra) -> dict:
        self.count += 1
        return {
            "type": kind,
            "uuid": f"{self.session_id}-{self.count}",
            "timestamp": f"2026-11-01T10:00:{self.count:02d}Z",
            "sessionId": self.session_id,
            "cwd": CWD,
            "gitBranch": "main",
            "version": "2.1.0",
            **extra,
        }

    def user(self, content, **extra) -> dict:
        return self._base("user", message={"role": "user", "content": content}, **extra)

    def assistant(self, message_id: str, content, model: str = MODEL) -> dict:
        return self._base(
            "assistant",
            message={
                "id": message_id,
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        )

    def tool_turn(self, message_id: str, name: str, result) -> list[dict]:
        call = f"toolu_{message_id}"
        return [
            self.assistant(
                message_id,
                [{"type": "tool_use", "id": call, "name": name, "input": {}}],
            ),
            self.user(
                [{"type": "tool_result", "tool_use_id": call, "content": result}]
            ),
        ]

    def shape(self, tools: list[dict]) -> dict:
        return {
            "type": "api-request-shape",
            "sessionId": self.session_id,
            "shape": {
                "system": [{"type": "text", "text": "You are Claude."}],
                "tools": tools,
            },
        }


def _codex(session_id: str, *, source="vscode", calls=True, tools=None) -> list[dict]:
    records = [
        {
            "timestamp": "2026-11-02T09:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "timestamp": "2026-11-02T09:00:00Z",
                "cwd": "/Users/alice/work/codexdemo",
                "cli_version": "0.162.1",
                "base_instructions": {"text": "You are Codex."},
                "git": {
                    "commit_hash": "abc123",
                    "branch": "dev",
                    "repository_url": "https://example.com/r.git",
                },
                "source": source,
                "thread_source": "subagent" if source != "vscode" else "user",
                "creator_user_id": "user-123456",
            },
        },
        {"type": "turn_context", "payload": {"model": "gpt-other"}},
        {"type": "turn_context", "payload": {"model": "gpt-main"}},
        {"type": "turn_context", "payload": {"model": "gpt-main"}},
        *(
            [
                {
                    "type": "response_item",
                    "payload": {"type": "additional_tools", **tools},
                }
            ]
            if tools
            else []
        ),
        {
            "timestamp": "2026-11-02T09:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "<environment_context>x"},
                    {"type": "input_text", "text": '<image name=[Image #1] path="a">'},
                    {"type": "input_text", "text": "</image>"},
                    {"type": "input_text", "text": "Fix the build"},
                ],
            },
        },
    ]
    if calls:
        records += [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": '{"cmd": "ls"}',
                    "call_id": "c1",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "ok",
                },
            },
            {"type": "event_msg", "payload": {"type": "token_count", "info": {}}},
        ]
    records += [
        {
            "timestamp": "2026-11-02T09:00:05Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
        {"type": "event_msg", "payload": {"type": "token_count", "info": {}}},
    ]
    return records


class ImporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.sources = self.root / "sources"
        self.run = self.root / "runs" / "imported"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_import(self, *paths: Path):
        sessions, subagents = discover(list(paths) or [self.sources])
        summary = import_sessions(self.run, sessions, parallel=2)
        summary.subagents = subagents
        return summary

    def samples(self) -> dict[str, tuple[Path, dict]]:
        return {
            info["task_id"]: (path.parent, info)
            for path in self.run.glob("samples/*/*/info.json")
            if (info := json.loads(path.read_text()))
        }

    def test_claude_session_becomes_a_sample(self) -> None:
        claude = Claude()
        thinking = claude.assistant("m1", [{"type": "thinking", "thinking": "plan"}])
        call = claude.assistant(
            "m1", [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]
        )
        _jsonl(
            self.sources / "proj" / "s1.jsonl",
            [
                {"type": "queue-operation", "timestamp": "2026-11-01T09:59:00Z"},
                claude.user("Read the file"),
                thinking,
                call,
                claude.user(
                    [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]
                ),
                claude.assistant("m2", [{"type": "text", "text": "Done"}]),
            ],
        )
        summary = self.run_import()
        self.assertEqual((summary.new, summary.failed), (1, []))
        directory, info = self.samples()["s1"]
        self.assertEqual(directory.parent.name, MODEL)
        self.assertEqual(directory.name, "00000001_s1")
        actf = json.loads((directory / "trace_ACTF.json").read_text())
        # Two records of message m1 are one step, and its usage counts once.
        self.assertEqual(len(actf["steps"]), 2)
        self.assertEqual(info["usage"]["input_tokens"], 200)
        self.assertEqual(info["usage"]["request_count"], 2)
        self.assertEqual(info["project"], "demo")
        self.assertEqual(info["title"], "Read the file")
        self.assertEqual(info["trajectory"]["raw_schema"], "claude-code-session-jsonl")
        self.assertEqual(info["trajectory"]["tools_source"], "inferred")
        self.assertEqual(info["trajectory"]["system_prompt_source"], "missing")
        source = info["provenance"]["import"]
        self.assertEqual(source["git"]["branch"], "main")
        self.assertNotIn("parent_session_id", source)
        self.assertTrue((directory / "trace_SFT.json").is_file())
        run_info = json.loads((self.run / "info.json").read_text())
        self.assertEqual(run_info["source"], "import")
        self.assertEqual(run_info["summary"]["ok"], 1)
        self.assertEqual(len((self.run / "sft.jsonl").read_text().splitlines()), 1)

    def test_subagents_and_empty_sessions_are_not_samples(self) -> None:
        claude = Claude("main")
        session = [claude.user("hi"), claude.assistant("m1", "hello")]
        _jsonl(self.sources / "proj" / "main.jsonl", session)
        _jsonl(self.sources / "proj" / "main" / "subagents" / "agent-a1.jsonl", session)
        _jsonl(self.sources / "proj" / "agent-old.jsonl", session)
        _jsonl(
            self.sources / "proj" / "queued.jsonl",
            [{"type": "queue-operation", "timestamp": "2026-11-01T10:00:00Z"}],
        )
        synthetic = Claude("synthetic")
        _jsonl(
            self.sources / "proj" / "synthetic.jsonl",
            [synthetic.user("hi"), synthetic.assistant("m1", "x", model="<synthetic>")],
        )
        summary = self.run_import()
        self.assertEqual(set(self.samples()), {"main"})
        self.assertEqual((summary.empty, summary.subagents), (2, 2))

    def test_claude_titles_and_parent_session(self) -> None:
        def session(name: str, *records, claude: Claude | None = None) -> None:
            claude = claude or Claude(name)
            _jsonl(
                self.sources / "proj" / f"{name}.jsonl",
                [*records, claude.user("fallback"), claude.assistant("m", "ok")],
            )

        session(
            "custom",
            {"type": "ai-title", "aiTitle": "AI"},
            {"type": "custom-title", "customTitle": "Old"},
            {"type": "custom-title", "customTitle": "Mine"},
        )
        session("ai", {"type": "ai-title", "aiTitle": "AI title"})
        claude = Claude("injected")
        session(
            "injected",
            claude.user("internal", isMeta=True),
            claude.user("<local-command-stdout>ls</local-command-stdout>"),
            claude.user(
                "<system-reminder>\nnote\n</system-reminder>\nReal\nwork " + "x" * 200
            ),
            claude=claude,
        )
        parent = Claude("parent")
        _jsonl(
            self.sources / "proj" / "child.jsonl",
            [parent.user("old"), parent.assistant("m", "ok")],
        )
        self.run_import()
        samples = self.samples()
        self.assertEqual(samples["custom"][1]["title"], "Mine")
        self.assertEqual(samples["ai"][1]["title"], "AI title")
        title = samples["injected"][1]["title"]
        self.assertTrue(title.startswith("Real work xxx"))
        self.assertEqual(len(title), 120)
        self.assertEqual(
            samples["child"][1]["provenance"]["import"]["parent_session_id"], "parent"
        )

    def test_reimport_skips_unchanged_and_updates_in_place(self) -> None:
        claude = Claude()
        path = _jsonl(
            self.sources / "proj" / "s1.jsonl",
            [claude.user("hi"), claude.assistant("m1", "hello")],
            tail='{"type": "user", "mess',
        )
        self.assertEqual(self.run_import().new, 1)
        directory, _ = self.samples()["s1"]
        actf = directory / "trace_ACTF.json"
        written = actf.stat().st_mtime_ns

        summary = self.run_import()
        self.assertEqual((summary.unchanged, summary.new, summary.updated), (1, 0, 0))
        self.assertEqual(actf.stat().st_mtime_ns, written)

        _jsonl(
            path,
            [
                claude.user("hi"),
                claude.assistant("m1", "hello"),
                claude.user("more"),
                claude.assistant("m2", "again"),
            ],
        )
        os.utime(path, (1, 1))
        self.assertEqual(self.run_import().updated, 1)
        self.assertEqual(list(self.samples()), ["s1"])
        self.assertEqual(self.samples()["s1"][0], directory)
        self.assertEqual(len(json.loads(actf.read_text())["steps"]), 2)

    def test_codex_discovery(self) -> None:
        sessions = self.sources / "sessions" / "2026" / "11" / "02"
        archived = self.sources / "archived_sessions"
        _jsonl(sessions / "rollout-a.jsonl", _codex("a"))
        _jsonl(archived / "rollout-a.jsonl", _codex("a", calls=False))
        _jsonl(sessions / "rollout-b.jsonl", _codex("b"))
        _zst(sessions / "rollout-b.jsonl.zst", _codex("b", calls=False))
        _zst(archived / "rollout-c.jsonl.zst", _codex("c"))
        _jsonl(
            sessions / "rollout-child.jsonl",
            _codex("child", source={"subagent": {"thread_spawn": {}}}),
        )
        summary = self.run_import()
        samples = self.samples()
        self.assertEqual(set(samples), {"a", "b", "c"})
        self.assertEqual(summary.subagents, 1)
        source = samples["a"][1]["provenance"]["import"]
        self.assertEqual(
            source["source_path"], str((sessions / "rollout-a.jsonl").resolve())
        )
        self.assertTrue(
            samples["b"][1]["provenance"]["import"]["source_path"].endswith(".jsonl")
        )
        # Every frame of the compressed file is read.
        directory, info = samples["c"]
        self.assertEqual(
            (directory / "trace.jsonl").read_text().splitlines(),
            [json.dumps(record) for record in _codex("c")],
        )
        self.assertEqual(
            len(json.loads((directory / "trace_ACTF.json").read_text())["steps"]), 2
        )

        self.assertEqual((info["model"], info["project"]), ("gpt-main", "codexdemo"))
        self.assertEqual(info["title"], "Fix the build")
        self.assertEqual(
            info["provenance"]["import"]["git"],
            {
                "branch": "dev",
                "commit": "abc123",
                "remote": "https://example.com/r.git",
            },
        )
        self.assertEqual(info["provenance"]["import"]["identity"], ["user-123456"])
        self.assertEqual(
            info["provenance"]["import"]["models"], ["gpt-main", "gpt-other"]
        )
        # Codex without recorded tools: inferred from its calls, and still SFT.
        self.assertEqual(info["trajectory"]["tools_source"], "inferred")
        self.assertEqual(info["trajectory"]["system_prompt_source"], "captured")
        self.assertTrue((directory / "trace_SFT.json").is_file())

    def test_codex_tools_from_additional_tools(self) -> None:
        tools = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "functions",
                    "tools": [{"type": "function", "name": "shell", "parameters": {}}],
                }
            ]
        }
        _jsonl(self.sources / "rollout-a.jsonl", _codex("a", tools=tools))
        self.run_import()
        directory, info = self.samples()["a"]
        self.assertEqual(info["trajectory"]["tools_source"], "captured")
        record = json.loads((directory / "trace_SFT.json").read_text())
        self.assertEqual(
            record["tools"], [{"type": "function", "name": "shell", "parameters": {}}]
        )

    def test_sft_tools_from_request_shapes(self) -> None:
        claude = Claude()
        _jsonl(
            self.sources / "proj" / "s1.jsonl",
            [
                claude.shape([TOOL]),
                claude.user("Find the PR tool"),
                *claude.tool_turn(
                    "m1",
                    "Read",
                    [{"type": "tool_reference", "tool_name": LATER_TOOL["name"]}],
                ),
                claude.shape([TOOL, LATER_TOOL]),
                *claude.tool_turn("m2", LATER_TOOL["name"], "open"),
                claude.assistant("m3", "Done"),
            ],
        )
        claude = Claude("single")
        _jsonl(
            self.sources / "proj" / "single.jsonl",
            [
                claude.shape([TOOL]),
                claude.user("Read it"),
                *claude.tool_turn("m1", "Read", "x"),
                claude.assistant("m2", "Done"),
            ],
        )
        self.run_import()
        samples = self.samples()
        directory, info = samples["s1"]
        self.assertEqual(info["trajectory"]["tools_source"], "captured_merged")
        self.assertEqual(info["trajectory"]["system_prompt_source"], "captured")
        record = json.loads((directory / "trace_SFT.json").read_text())
        self.assertEqual(record["messages"][0]["content"], "You are Claude.")
        self.assertEqual(
            [tool["function"]["name"] for tool in record["tools"]],
            ["Read", LATER_TOOL["name"]],
        )
        self.assertIn('"tool_reference"', record["messages"][3]["content"])
        self.assertEqual(samples["single"][1]["trajectory"]["tools_source"], "captured")
        index = [
            json.loads(line)
            for line in (self.run / "sft_index.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            sorted((row["tools_source"], row["system_prompt_source"]) for row in index),
            [("captured", "captured"), ("captured_merged", "captured")],
        )

    def test_opencode_discovery_and_import_respects_xdg_data_home_and_skips_subagents(
        self,
    ) -> None:
        from roller.importer import OPENCODE, default_roots

        xdg = self.root / "xdg"
        opencode_root = xdg / "opencode"
        opencode_root.mkdir(parents=True)
        old_xdg = os.environ.get("XDG_DATA_HOME")
        os.environ["XDG_DATA_HOME"] = str(xdg)
        try:
            self.assertEqual(default_roots(OPENCODE), [opencode_root])
        finally:
            if old_xdg is None:
                os.environ.pop("XDG_DATA_HOME", None)
            else:
                os.environ["XDG_DATA_HOME"] = old_xdg

        main_session = opencode_root / "sessions" / "oc-1.jsonl"
        _jsonl(
            main_session,
            [
                {
                    "type": "request",
                    "native_session_id": "oc-1",
                    "body": {
                        "messages": [{"role": "system", "content": "You are OpenCode."}],
                        "tools": [],
                    },
                },
                {
                    "type": "session_export",
                    "data": {
                        "info": {
                            "id": "oc-1",
                            "title": "Fix bug",
                            "directory": CWD,
                            "time": {"created": 1762000000000},
                        },
                        "messages": [
                            {
                                "info": {
                                    "id": "m1",
                                    "sessionID": "oc-1",
                                    "role": "user",
                                    "time": {"created": 1762000000100},
                                },
                                "parts": [
                                    {"id": "p1", "messageID": "m1", "sessionID": "oc-1", "type": "text", "text": "Fix it"}
                                ],
                            },
                            {
                                "info": {
                                    "id": "m2",
                                    "sessionID": "oc-1",
                                    "role": "assistant",
                                    "modelID": "claude-sonnet-4",
                                    "time": {"created": 1762000001000, "completed": 1762000002000},
                                    "tokens": {"input": 10, "output": 5, "total": 15},
                                    "cost": 0.01,
                                },
                                "parts": [
                                    {"id": "p2", "messageID": "m2", "sessionID": "oc-1", "type": "text", "text": "Fixed."}
                                ],
                            },
                        ],
                    },
                }
            ],
        )
        sub_session = opencode_root / "sessions" / "oc-sub.jsonl"
        _jsonl(
            sub_session,
            [
                {
                    "type": "session_export",
                    "data": {
                        "info": {
                            "id": "oc-sub",
                            "parentID": "oc-1",
                            "directory": CWD,
                            "time": {"created": 1762000000500},
                        },
                        "messages": [],
                    },
                }
            ],
        )

        sessions, subagents = discover([opencode_root], harness=OPENCODE)
        self.assertEqual([s.session_id for s in sessions], ["oc-1"])
        self.assertEqual(subagents, 1)

        summary = import_sessions(self.run, sessions, parallel=1)
        self.assertEqual(summary.new, 1)
        samples = self.samples()
        self.assertIn("oc-1", samples)
        _, info = samples["oc-1"]
        self.assertEqual(info["harness"], OPENCODE)
        self.assertEqual(info["model"], "claude-sonnet-4")
        self.assertEqual(info["title"], "Fix bug")
        self.assertEqual(info["trajectory"]["raw_schema"], "opencode-session-jsonl")


if __name__ == "__main__":
    unittest.main()
