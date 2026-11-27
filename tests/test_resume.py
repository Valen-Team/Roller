from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_gateway_ports import isolate_gateway_ports

from roller.actf import convert_openclaw_trace
from roller.cli import _build_parser, _collect
from roller.collector import CollectionConfig, Collector
from roller.conversation import ConversationManager
from roller.harness_registry import HARNESS_NAMES as HARNESSES
from roller.harnesses.base import InteractiveHarness, InteractiveSession, TurnResult
from roller.integrity import _validate_output_files, validate_resumable_sample
from roller.models import HarnessResult, RunRequest, Task, UserSimulatorSpec
from roller.sft import actf_to_sft


class RecordingSession(InteractiveSession):
    def __init__(self, harness, request, output_dir) -> None:
        self.harness = harness
        self.request = request
        self.output_dir = output_dir
        self.messages = []
        self.replies = []

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.harness.calls.append((self.harness.name, self.request.task.id))

    def send(self, message: str, timeout_seconds: float) -> TurnResult:
        if self.request.task.id.endswith("__user_simulator"):
            reply = (
                '{"action":"message","content":"say done"}'
                if not self.messages
                else '{"action":"finish"}'
            )
        else:
            reply = "done"
        self.messages.append(message)
        self.replies.append(reply)
        return TurnResult(reply, 0.01)

    def finish(self, status: str, error: str | None = None) -> HarnessResult:
        events = []
        for message, reply in zip(self.messages, self.replies):
            events.extend(
                [
                    {
                        "type": "message",
                        "message": {"role": "user", "content": message},
                    },
                    {
                        "type": "message",
                        "message": {"role": "assistant", "content": reply},
                    },
                ]
            )
        raw = self.output_dir / "trace.jsonl"
        raw.write_text("\n".join(json.dumps(row) for row in events) + "\n")
        actf = self.output_dir / "trace_ACTF.json"
        convert_openclaw_trace(raw, actf, user_messages=self.messages)
        (self.output_dir / "wire_trace.jsonl").write_text("{}\n")
        # Match actual artifact availability: the other clients emit no cache trace.
        if self.harness.name == "openclaw" and self.harness.capture_cache:
            (self.output_dir / "cache_trace.jsonl").write_text("{}\n")
        return HarnessResult(
            status=status,
            error=error,
            elapsed_seconds=0.01,
            trace_path=str(raw),
            trajectory_path=str(actf),
        )


class RecordingHarness(InteractiveHarness):
    def __init__(self, name, calls, capture_cache=True) -> None:
        self.name = name
        self.calls = calls
        self.capture_cache = capture_cache

    def create_session(self, request, output_dir, attempt, *, collect_outputs=True):
        return RecordingSession(self, request, output_dir)

    def run(self, request, output_dir, attempt):
        session = self.create_session(request, output_dir, attempt)
        session.start()
        for prompt in request.task.prompts:
            session.send(prompt, request.timeout_seconds)
        return session.finish("ok")


class CliResumeCompatibilityTests(unittest.TestCase):
    """Keep CLI lock and metadata writes real; replace only Docker/harness I/O."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        isolate_gateway_ports(self)
        self.calls = []
        patches = contextlib.ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch("roller.cli.collect._docker_preflight", return_value="test-image-id")
        )
        constructors = {
            "openclaw": "OpenClawHarness",
            "codex": "CodexHarness",
            "claude-code": "ClaudeCodeHarness",
            "hermes": "HermesHarness",
            "opencode": "OpenCodeHarness",
            "deepseek": "DeepSeekHarness",
            "pi": "PiHarness",
            "stirrup": "StirrupHarness",
        }
        patches.enter_context(
            patch("roller.cli.preflight._stirrup_installed", return_value=True)
        )
        for name, constructor in constructors.items():
            patches.enter_context(
                patch(
                    f"roller.cli.collect.{constructor}",
                    side_effect=lambda config, slot, name=name: RecordingHarness(
                        name, self.calls
                    ),
                )
            )

    def invoke(self, args, *, resume=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return _collect(
                _build_parser().parse_args(args + (["--resume"] if resume else []))
            )

    def seed(self, name, harness, simulator=None):
        tasks = self.root / f"{name}.jsonl"
        row = {"id": "task", "harness": harness, "model": "provider/model"}
        if simulator:
            row["user_simulator"] = {
                "harness": simulator,
                "instruction": "ask once",
                "max_turns": 1,
            }
        else:
            row["prompt"] = "say done"
        tasks.write_text(json.dumps(row) + "\n")
        args = [
            "collect",
            "--tasks",
            str(tasks),
            "--api-base",
            "http://example.test/v1",
            "--api-key-env",
            "TEST_KEY",
            "--output",
            str(self.root / "runs"),
            "--run-name",
            name,
        ]
        self.assertEqual(self.invoke(args), 0)
        return args, self.root / "runs" / name

    def grow(self, name, **row):
        """Four Codex tasks whose model comes from ``--model``."""

        tasks = self.root / f"{name}.jsonl"
        tasks.write_text(
            "".join(
                json.dumps(
                    {"id": f"task-{i}", "harness": "codex", "prompt": f"say {i}", **row}
                )
                + "\n"
                for i in range(4)
            )
        )
        args = [
            "collect",
            "--tasks",
            str(tasks),
            "--api-base",
            "http://example.test/v1",
            "--api-key-env",
            "TEST_KEY",
            "--output",
            str(self.root / "runs"),
            "--run-name",
            name,
        ]
        return args, self.root / "runs" / name

    @staticmethod
    def run_samples(run):
        info = json.loads((run / "info.json").read_text())
        return [
            (row["request_index"], row["path"], row["status"])
            for row in info["samples"]
        ]

    @staticmethod
    def sft_prompts(run):
        return [
            next(
                m["content"]
                for m in json.loads(line)["messages"]
                if m["role"] == "user"
            )
            for line in (run / "sft.jsonl").read_text().splitlines()
        ]

    def test_resume_grows_a_run_to_more_tasks(self):
        args, run = self.grow("more-tasks")
        self.assertEqual(self.invoke([*args, "--model", "m1", "--limit", "2"]), 0)
        finished = self.snapshot(run / "samples")
        recorded = json.loads((run / "lock.json").read_text())

        with self.assertLogs("roller.run_lock", "INFO") as logs:
            self.assertEqual(
                self.invoke([*args, "--model", "m1", "--limit", "4"], resume=True), 0
            )

        self.assertIn("lock.json now lists 4 requests (2 before)", logs.output[0])
        # Each task ran once: the finished samples were not run again.
        self.assertEqual(self.calls, [("codex", f"task-{index}") for index in range(4)])
        samples = self.snapshot(run / "samples")
        self.assertEqual({name: samples[name] for name in finished}, finished)
        self.assertEqual(
            self.run_samples(run),
            [
                (0, "samples/m1/00000000_task-0", "skipped"),
                (1, "samples/m1/00000001_task-1", "skipped"),
                (2, "samples/m1/00000002_task-2", "ok"),
                (3, "samples/m1/00000003_task-3", "ok"),
            ],
        )
        self.assertEqual(self.sft_prompts(run), [f"say {index}" for index in range(4)])
        lock = json.loads((run / "lock.json").read_text())
        self.assertEqual(lock["requests"][:2], recorded["requests"])
        self.assertEqual(
            [row["task_id"] for row in lock["requests"]],
            [f"task-{index}" for index in range(4)],
        )
        self.assertEqual({**lock, "requests": recorded["requests"]}, recorded)

    def test_resume_adds_a_model_and_keeps_finished_samples_in_place(self):
        args, run = self.grow("more-models")
        self.assertEqual(self.invoke([*args, "--model", "m1", "--limit", "2"]), 0)
        finished = self.snapshot(run / "samples")
        both = [*args, "--model", "m1", "--model", "m2", "--limit", "2"]

        self.assertEqual(self.invoke(both, resume=True), 0)

        self.assertEqual(len(self.calls), 4)
        # Requests are ordered by task, then model: the second m1 sample moves
        # from position 1 to position 2 and stays in its directory.
        expected = [
            (0, "samples/m1/00000000_task-0", "skipped"),
            (1, "samples/m2/00000000_task-0", "ok"),
            (2, "samples/m1/00000001_task-1", "skipped"),
            (3, "samples/m2/00000001_task-1", "ok"),
        ]
        self.assertEqual(self.run_samples(run), expected)
        self.assertEqual(self.sft_prompts(run), ["say 0", "say 0", "say 1", "say 1"])
        samples = self.snapshot(run / "samples")
        self.assertEqual(
            [name for name in finished if samples[name] != finished[name]],
            ["m1/00000001_task-1/info.json"],
        )
        for index, sample, _ in expected:
            info = json.loads((run / sample / "info.json").read_text())
            self.assertEqual((info["request_index"], info["status"]), (index, "ok"))
        self.assertEqual(
            [
                (row["request_index"], row["task_id"], row["model"])
                for row in json.loads((run / "lock.json").read_text())["requests"]
            ],
            [
                (0, "task-0", "m1"),
                (1, "task-0", "m2"),
                (2, "task-1", "m1"),
                (3, "task-1", "m2"),
            ],
        )

        # The grown run resumes like any other.
        self.assertEqual(self.invoke(both, resume=True), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(
            self.run_samples(run),
            [(index, sample, "skipped") for index, sample, _ in expected],
        )
        self.assertEqual(self.snapshot(run / "samples"), samples)

    def test_resume_grows_a_run_whose_tasks_have_their_own_images(self):
        args, run = self.grow("task-images", runtime={"image": "task-env:1"})
        with patch(
            "roller.cli.collect.prepare_task_images",
            side_effect=lambda binary, requests, images, **options: {
                request.task.id: f"sha256:{request.task.id}" for request in requests
            },
        ):
            self.assertEqual(self.invoke([*args, "--model", "m1", "--limit", "1"]), 0)
            self.assertEqual(
                self.invoke([*args, "--model", "m1", "--limit", "2"], resume=True), 0
            )
        self.assertEqual(self.calls, [("codex", "task-0"), ("codex", "task-1")])
        self.assertEqual(
            json.loads((run / "lock.json").read_text())["execution"]["task_images"],
            {
                f"task-{index}": {
                    "image": "task-env:1",
                    "digest": f"sha256:task-{index}",
                }
                for index in range(2)
            },
        )

    def test_resume_still_rejects_removed_or_changed_requests(self):
        args, run = self.grow("fewer")
        models = ["--model", "m1", "--model", "m2"]
        self.assertEqual(self.invoke([*args, *models]), 0)
        before = self.snapshot(run)
        for flags, message in (
            (
                [*models, "--limit", "3"],
                "2 recorded requests are not in the current command; "
                "first: task-3 | m1 (task index 3)",
            ),
            (
                ["--model", "m1"],
                "4 recorded requests are not in the current command; "
                "first: task-0 | m2 (task index 0)",
            ),
            (
                [*models, "--samples", "2"],
                "8 recorded requests differ from the current command; "
                "first: task-0 | m1: sample_count: recorded=1, current=2",
            ),
            (
                [*models, "--timeout", "999"],
                "8 recorded requests differ from the current command; "
                "first: task-0 | m1: timeout_seconds: recorded=600, current=999",
            ),
            ([*models, "--thinking", "high"], "$.execution.codex.reasoning_effort"),
        ):
            with self.subTest(flags=flags):
                with self.assertRaises(FileExistsError) as caught:
                    self.invoke([*args, *flags], resume=True)
                self.assertIn("lock mismatch", str(caught.exception))
                self.assertIn(message, str(caught.exception))
                self.assertEqual(self.snapshot(run), before)
        self.assertEqual(len(self.calls), 8)

    def test_resume_accepts_another_patch_release_only(self):
        args, run = self.seed("patch-release", "codex")
        lock = run / "lock.json"
        before = lock.read_bytes(), lock.stat().st_mtime_ns
        created = json.loads(lock.read_text())["roller_version"]
        major, minor = (int(part) for part in created.split(".")[:2])
        patch_release = f"{major}.{minor}.99"

        with (
            patch("roller.cli.collect.__version__", patch_release),
            self.assertLogs("roller.run_lock", "INFO") as logs,
        ):
            self.assertEqual(self.invoke(args, resume=True), 0)

        self.assertEqual(
            logs.output,
            [
                f"INFO:roller.run_lock:run was created by Roller {created}; "
                f"resuming with Roller {patch_release}"
            ],
        )
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(
            json.loads((run / "info.json").read_text())["summary"]["skipped"], 1
        )
        self.assertEqual((lock.read_bytes(), lock.stat().st_mtime_ns), before)

        with (
            patch("roller.cli.collect.__version__", f"{major}.{minor + 1}.0"),
            self.assertRaisesRegex(
                FileExistsError,
                f"created by Roller {created} and this is Roller {major}.{minor + 1}.0; "
                "a run resumes only within its minor series",
            ),
        ):
            self.invoke(args, resume=True)
        self.assertEqual((lock.read_bytes(), lock.stat().st_mtime_ns), before)

    def test_resume_rebuilds_sft_after_the_projection_changed(self):
        args, run = self.seed("projection", "codex")
        sample = next(run.glob("samples/*/*"))
        raw = (sample / "trace.jsonl").stat().st_mtime_ns

        def later_projection(trajectory):
            record = actf_to_sft(trajectory)
            record["messages"][0]["content"] = "written by a later patch release"
            return record

        with patch("roller.integrity.actf_to_sft", side_effect=later_projection):
            self.assertEqual(self.invoke(args, resume=True), 0)

        self.assertEqual(len(self.calls), 1)
        self.assertEqual((sample / "trace.jsonl").stat().st_mtime_ns, raw)
        record = json.loads((sample / "trace_SFT.json").read_text())
        self.assertEqual(
            record["messages"][0]["content"], "written by a later patch release"
        )
        self.assertEqual(
            [json.loads(line) for line in (run / "sft.jsonl").read_text().splitlines()],
            [record],
        )
        summary = json.loads((run / "info.json").read_text())["summary"]
        self.assertEqual((summary["ok"], summary["skipped"]), (0, 1))

    def test_resume_can_verify_output_digests(self):
        args, _ = self.seed("verify-outputs", "codex")
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            self.invoke([*args, "--resume-verify-outputs"])
        for flags, expected in (((), False), (("--resume-verify-outputs",), True)):
            with (
                self.subTest(flags=flags),
                patch(
                    "roller.collector.validate_resumable_sample",
                    wraps=validate_resumable_sample,
                ) as validate,
            ):
                self.assertEqual(self.invoke([*args, *flags], resume=True), 0)
                self.assertEqual(
                    validate.call_args.kwargs["verify_output_sha256"], expected
                )

    def test_output_digest_check_catches_a_same_size_change(self):
        sample = self.root / "sample"
        (sample / "results").mkdir(parents=True)
        output = sample / "results" / "answer.txt"
        output.write_bytes(b"right")
        info = {
            "output_collection": {
                "passed": True,
                "files": [
                    {
                        "path": "answer.txt",
                        "size": 5,
                        "sha256": hashlib.sha256(b"right").hexdigest(),
                    }
                ],
            }
        }
        _validate_output_files(sample, info, verify_sha256=True)
        output.write_bytes(b"wrong")
        # Path and size still match, so the default check accepts the file.
        _validate_output_files(sample, info)
        with self.assertRaisesRegex(ValueError, "content changed: results/answer.txt"):
            _validate_output_files(sample, info, verify_sha256=True)

    @staticmethod
    def snapshot(root):
        return {
            str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*")
            if path.is_file() and path.name != "collect.log"
        }

    def assert_disabled_legacy_run_resumes(self, args, run):
        path = run / "lock.json"
        recorded = json.loads(path.read_text())
        for name in HARNESSES:
            settings = recorded["execution"].get(name)
            if settings is None:
                continue
            settings["external_thinking"] = "off"
            if name == "claude-code":
                settings.update(
                    signature_recovery="off",
                    signature_recovery_attempts=1,
                    signature_recovery_max_tokens=8192,
                )
        path.write_text(json.dumps(recorded))
        for info_path in (run / "samples").rglob("info.json"):
            info = json.loads(info_path.read_text())
            info.setdefault("runtime", {})["external_thinking"] = {"mode": "off"}
            if info["harness"] == "claude-code":
                info["runtime"]["signature_recovery"] = {"mode": "off"}
            info_path.write_text(json.dumps(info))
        before_lock = path.read_bytes(), path.stat().st_mtime_ns
        before_samples = self.snapshot(run / "samples")
        before_calls = list(self.calls)

        self.assertEqual(self.invoke(args, resume=True), 0)

        self.assertEqual(
            json.loads((run / "info.json").read_text())["summary"]["skipped"], 1
        )
        self.assertEqual(self.calls, before_calls)
        self.assertEqual(self.snapshot(run / "samples"), before_samples)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before_lock)

    def test_cli_resumes_disabled_legacy_lock_for_every_harness(self):
        for harness in HARNESSES:
            with self.subTest(harness=harness):
                self.assert_disabled_legacy_run_resumes(*self.seed(harness, harness))

    def test_legacy_claude_sidechain_resume_stops_before_changing_run(self):
        args, run = self.seed("legacy-sidechain", "claude-code")
        sample = next(run.glob("samples/*/*"))
        actf_path = sample / "trace_ACTF.json"
        actf = json.loads(actf_path.read_text())
        child = json.loads(json.dumps(actf["steps"][0]))
        child["step_id"] = len(actf["steps"]) + 1
        child["extra"] = {"is_sidechain": True}
        actf["steps"].append(child)
        actf_path.write_text(json.dumps(actf))
        before = self.snapshot(run)
        before_calls = list(self.calls)

        with self.assertRaisesRegex(ValueError, "legacy Claude Code sidechain"):
            self.invoke(args, resume=True)

        self.assertEqual(self.snapshot(run), before)
        self.assertEqual(self.calls, before_calls)

    def test_claude_truncated_native_tail_still_resumes(self):
        args, run = self.seed("truncated-tail", "claude-code")
        sample = next(run.glob("samples/*/*"))
        with (sample / "trace.jsonl").open("a") as stream:
            stream.write('{"type":')
        before_calls = list(self.calls)

        self.assertEqual(self.invoke(args, resume=True), 0)
        self.assertEqual(self.calls, before_calls)
        self.assertEqual(
            json.loads((run / "info.json").read_text())["summary"]["skipped"], 1
        )

    def test_cli_resumes_disabled_legacy_lock_for_mixed_agent_and_simulator(self):
        for agent, simulator in (("codex", "claude-code"), ("claude-code", "codex")):
            with self.subTest(agent=agent, simulator=simulator):
                self.assert_disabled_legacy_run_resumes(
                    *self.seed(agent, agent, simulator)
                )

    def test_cli_resumes_legacy_claude_settings_fingerprint_lock(self):
        settings = self.root / "settings.json"
        settings.write_text('{"model":"one"}\n', encoding="utf-8")
        tasks = self.root / "claude-settings.jsonl"
        tasks.write_text(
            json.dumps(
                {
                    "id": "task",
                    "harness": "claude-code",
                    "model": "provider/model",
                    "prompt": "say done",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        args = [
            "collect",
            "--tasks",
            str(tasks),
            "--claude-settings",
            str(settings),
            "--api-base",
            "http://example.test/v1",
            "--api-key-env",
            "TEST_KEY",
            "--output",
            str(self.root / "runs"),
            "--run-name",
            "claude-settings",
        ]
        with patch(
            "roller.cli.collect.hashlib.blake2b",
            side_effect=AssertionError("new runs must use only SHA-256"),
        ):
            self.assertEqual(self.invoke(args), 0)
        run = self.root / "runs" / "claude-settings"
        lock_path = run / "lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        claude = lock["execution"]["claude-code"]
        content = settings.read_bytes()
        self.assertEqual(claude["config_sha256"], hashlib.sha256(content).hexdigest())
        self.assertNotIn("settings_fingerprint", claude)

        claude["settings_fingerprint"] = hashlib.blake2b(
            content, digest_size=16
        ).hexdigest()
        del claude["config_sha256"]
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        historical_lock = lock_path.read_bytes(), lock_path.stat().st_mtime_ns
        calls_before_resume = list(self.calls)

        self.assertEqual(self.invoke(args, resume=True), 0)

        self.assertEqual(self.calls, calls_before_resume)
        self.assertEqual(
            (lock_path.read_bytes(), lock_path.stat().st_mtime_ns),
            historical_lock,
        )

        settings.write_text('{"model":"two"}\n', encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "lock mismatch"):
            self.invoke(args, resume=True)

    def test_full_context_archive_is_backfilled_without_rerunning(self) -> None:
        full = self.root / "SYSTEM.md"
        hashed = self.root / "USER.md"
        full.write_text("persist me", encoding="utf-8")
        hashed.write_text("hash me", encoding="utf-8")
        tasks = self.root / "context.jsonl"
        tasks.write_text(
            json.dumps(
                {
                    "id": "task",
                    "harness": "codex",
                    "model": "provider/model",
                    "prompt": "say done",
                    "files": {
                        "context": [
                            {
                                "role": "system",
                                "source": "SYSTEM.md",
                                "capture": "full",
                            },
                            {
                                "role": "user",
                                "source": "USER.md",
                                "capture": "hash",
                            },
                        ]
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run = self.root / "runs" / "context"
        args = [
            "collect",
            "--tasks",
            str(tasks),
            "--api-base",
            "http://example.test/v1",
            "--api-key-env",
            "TEST_KEY",
            "--output",
            str(self.root / "runs"),
            "--run-name",
            "context",
        ]

        self.assertEqual(self.invoke(args), 0)

        full_digest = hashlib.sha256(b"persist me").hexdigest()
        hash_digest = hashlib.sha256(b"hash me").hexdigest()
        blob_root = run / "context" / "sha256"
        self.assertEqual(
            (blob_root / full_digest).read_text(encoding="utf-8"), "persist me"
        )
        self.assertFalse((blob_root / hash_digest).exists())
        sample_info_path = next((run / "samples").rglob("info.json"))
        context_digest = json.loads(sample_info_path.read_text(encoding="utf-8"))[
            "provenance"
        ]["context_manifest_digest"]
        archive_path = run / "context" / "manifests" / f"{context_digest}.json"
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
        self.assertEqual(
            archive["declarations"][0]["files"][0]["blob"],
            f"context/sha256/{full_digest}",
        )

        calls_before_resume = list(self.calls)
        shutil.rmtree(run / "context")
        self.assertEqual(self.invoke(args, resume=True), 0)

        self.assertEqual(self.calls, calls_before_resume)
        self.assertTrue((blob_root / full_digest).is_file())
        self.assertTrue(archive_path.is_file())

    def test_cli_still_recaptures_and_adopts_an_unlocked_native_run(self):
        args, run = self.seed("unlocked-native", "codex")
        (run / "lock.json").unlink()
        before_calls = len(self.calls)

        self.assertEqual(self.invoke(args, resume=True), 0)

        summary = json.loads((run / "info.json").read_text())["summary"]
        self.assertEqual((summary["ok"], summary["skipped"]), (1, 0))
        self.assertEqual(len(self.calls), before_calls + 1)
        self.assertTrue((run / "lock.json").is_file())
        self.assertFalse((run / ".legacy-resume-untrusted.json").exists())

    def test_cli_rejects_retired_samples_before_any_run_files_change(self):
        modes = (
            ("codex", "external_thinking", "external"),
            ("codex", "external_thinking", "external_always"),
            ("claude-code", "signature_recovery", "fill"),
            ("claude-code", "signature_recovery", "prefer"),
        )
        for harness, key, mode in modes:
            for participant in ("agent", "simulator"):
                for locked in (False, True):
                    name = f"{mode}-{participant}-{locked}"
                    with self.subTest(case=name):
                        args, run = self.seed(
                            name,
                            harness if participant == "agent" else "openclaw",
                            harness if participant == "simulator" else None,
                        )
                        sample = next(run.glob("samples/*/*"))
                        if participant == "simulator":
                            sample = sample / "user_simulator"
                        info_path = sample / "info.json"
                        info = json.loads(info_path.read_text())
                        info.setdefault("runtime", {})[key] = {"mode": mode}
                        info_path.write_text(json.dumps(info))
                        if not locked:
                            (run / "lock.json").unlink()
                        before = self.snapshot(run)
                        before_calls = list(self.calls)

                        with self.assertRaisesRegex(
                            ValueError, "is retired; use a new run directory"
                        ):
                            self.invoke(args, resume=True)

                        self.assertEqual(self.snapshot(run), before)
                        self.assertEqual(self.calls, before_calls)


class MixedResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calls = []
        self.keep_aux = True
        self.capture_cache = True
        self.override = None
        self.requests = []
        for agent in HARNESSES:
            for simulator in (None, *HARNESSES):
                index = len(self.requests)
                task = Task(
                    id=f"{agent}--{simulator or 'predefined'}",
                    prompts=() if simulator else ("say done",),
                    source_index=index,
                    harness=agent,
                    user_simulator=(
                        UserSimulatorSpec("ask once", harness=simulator, max_turns=1)
                        if simulator
                        else None
                    ),
                )
                self.requests.append(RunRequest(index, task, "provider/model", 30))

    def factory(self, slot, request):
        agent = RecordingHarness(
            self.override or request.task.harness, self.calls, self.capture_cache
        )
        spec = request.task.user_simulator
        if spec is None:
            return agent
        simulator = RecordingHarness(spec.harness, self.calls, self.capture_cache)
        return ConversationManager(agent, simulator, keep_aux_traces=self.keep_aux)

    def collector(self, resume=False):
        return Collector(
            CollectionConfig(
                run_dir=self.root,
                resume=resume,
                keep_aux_traces=self.keep_aux,
                capture_cache_trace=self.capture_cache,
            ),
            self.factory,
        )

    def collect(self, resume=False):
        return self.collector(resume).collect(self.requests)

    def participants(self):
        for request in self.requests:
            sample = self.collector()._sample_dir(request)
            yield request, sample, self.override or request.task.harness
            if request.task.user_simulator:
                yield (
                    request,
                    sample / "user_simulator",
                    request.task.user_simulator.harness,
                )

    def assert_reused(self):
        before = list(self.calls)
        raw_before = {
            str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.root.glob("samples/*/*/**/trace.jsonl")
        }
        self.assertEqual(
            [r["status"] for r in self.collect(True)], ["skipped"] * len(self.requests)
        )
        self.assertEqual(self.calls, before)
        for name, snapshot in raw_before.items():
            path = Path(name)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), snapshot)

    def assert_only_rerun(self, request):
        before = len(self.calls)
        results = self.collect(True)
        for result in results:
            expected = "ok" if result["task_id"] == request.task.id else "skipped"
            self.assertEqual(result["status"], expected)
        expected_calls = 2 if request.task.user_simulator else 1
        self.assertEqual(len(self.calls) - before, expected_calls)

    def test_full_mode_reuses_all_agents_and_simulator_pairs(self):
        self.assertEqual(
            [r["status"] for r in self.collect()], ["ok"] * len(self.requests)
        )
        for _, path, harness in self.participants():
            self.assertTrue((path / "wire_trace.jsonl").is_file())
            self.assertEqual(
                (path / "cache_trace.jsonl").is_file(), harness == "openclaw"
            )
        self.assert_reused()

    def test_missing_openclaw_cache_reruns_only_its_sample(self):
        self.collect()
        for request, path, harness in self.participants():
            if harness != "openclaw":
                continue
            with self.subTest(task=request.task.id, participant=path.name):
                cache = path / "cache_trace.jsonl"
                cache.unlink()
                self.assert_only_rerun(request)
                self.assertTrue(cache.is_file())

    def test_missing_wire_trace_still_reruns_each_harness_on_either_side(self):
        self.collect()
        for request, path, _ in self.participants():
            with self.subTest(task=request.task.id, participant=path.name):
                wire = path / "wire_trace.jsonl"
                wire.unlink()
                self.assert_only_rerun(request)
                self.assertTrue(wire.is_file())

    def test_stale_sft_is_rebuilt_from_the_trace_on_either_side(self):
        self.collect()
        dataset = (self.root / "sft.jsonl").read_bytes()
        stale = (
            lambda path: path.write_text('{"messages":[],"tools":[]}\n'),
            lambda path: path.write_text('{"messages":[],"tools":[],"extra":1}\n'),
            lambda path: path.write_text("{"),
            lambda path: path.unlink(),
        )
        originals = {}
        for index, (_, path, _) in enumerate(self.participants()):
            sft = path / "trace_SFT.json"
            originals[sft] = sft.read_bytes()
            stale[index % len(stale)](sft)

        with self.assertLogs("roller.collector", "INFO") as logs:
            self.assert_reused()

        for sft, original in originals.items():
            self.assertEqual(sft.read_bytes(), original)
        self.assertEqual((self.root / "sft.jsonl").read_bytes(), dataset)
        # One line for each sample, naming the files of both participants.
        rebuilt = [line for line in logs.output if "rebuilt" in line]
        self.assertEqual(len(rebuilt), len(self.requests))
        self.assertIn(
            "INFO:roller.collector:[openclaw--codex | provider/model] rebuilt "
            "trace_SFT.json, user_simulator/trace_SFT.json from the trace; "
            "it differed from the SFT this Roller version derives",
            rebuilt,
        )

    def test_invalid_actf_still_reruns_a_sample_with_a_stale_sft(self):
        self.collect()
        for request, path, _ in self.participants():
            if not request.task.id.startswith("codex--"):
                continue
            with self.subTest(task=request.task.id, participant=path.name):
                (path / "trace_SFT.json").write_text('{"messages":[],"tools":[]}\n')
                actf_path = path / "trace_ACTF.json"
                actf = json.loads(actf_path.read_text())
                actf["steps"] = []
                actf_path.write_text(json.dumps(actf))
                self.assert_only_rerun(request)

    def test_strict_validation_still_reports_a_stale_sft(self):
        self.requests = self.requests[:1]
        self.collect()
        request = self.requests[0]
        sample = self.collector()._sample_dir(request)
        info = json.loads((sample / "info.json").read_text())
        sft = sample / "trace_SFT.json"
        original = sft.read_bytes()
        options = {"require_wire_trace": True, "require_cache_trace": True}
        self.assertEqual(
            validate_resumable_sample(sample, request, info, **options), []
        )
        sft.write_text('{"messages":[],"tools":[]}\n')

        with self.assertRaisesRegex(ValueError, "canonical ACTF projection"):
            validate_resumable_sample(sample, request, info, **options)
        self.assertEqual(
            validate_resumable_sample(
                sample, request, info, rebuild_sft=True, **options
            ),
            [sft],
        )
        self.assertEqual(sft.read_bytes(), original)

    def test_a_reused_sample_takes_its_position_in_the_current_requests(self):
        self.requests = self.requests[:3]
        self.collect()
        self.requests = [
            dataclasses.replace(request, index=index)
            for index, request in enumerate(reversed(self.requests))
        ]
        before = list(self.calls)

        results = self.collect(True)

        self.assertEqual(self.calls, before)
        expected = [
            (request.index, request.task.id, "skipped") for request in self.requests
        ]
        self.assertEqual(
            [(r["request_index"], r["task_id"], r["status"]) for r in results],
            expected,
        )
        run = json.loads((self.root / "info.json").read_text())
        self.assertEqual(
            [(r["request_index"], r["task_id"], r["status"]) for r in run["samples"]],
            expected,
        )
        for request in self.requests:
            info = json.loads(
                (self.collector()._sample_dir(request) / "info.json").read_text()
            )
            self.assertEqual(
                (info["request_index"], info["status"]), (request.index, "ok")
            )

    def test_disabled_or_pruned_cache_traces_do_not_block_resume(self):
        for self.keep_aux, self.capture_cache in ((True, False), (False, True)):
            with self.subTest(full=self.keep_aux, cache=self.capture_cache):
                self.collect()
                for _, path, _ in self.participants():
                    self.assertFalse((path / "cache_trace.jsonl").exists())
                self.assert_reused()

    def test_executed_agent_harness_takes_precedence_over_task_declaration(self):
        for self.override in HARNESSES:
            with self.subTest(override=self.override):
                self.collect()
                self.assert_reused()
                if self.override == "openclaw":
                    request = next(
                        r for r in self.requests if r.task.harness == "codex"
                    )
                    (
                        self.collector()._sample_dir(request) / "cache_trace.jsonl"
                    ).unlink()
                    self.assert_only_rerun(request)

    def test_missing_or_unknown_harness_metadata_keeps_strict_cache_check(self):
        self.requests = self.requests[:1]
        for harness in (None, "unknown"):
            with self.subTest(harness=harness):
                self.collect()
                sample = self.collector()._sample_dir(self.requests[0])
                info_path = sample / "info.json"
                info = json.loads(info_path.read_text())
                if harness is None:
                    del info["harness"]
                else:
                    info["harness"] = harness
                info_path.write_text(json.dumps(info))
                (sample / "cache_trace.jsonl").unlink()
                self.assert_only_rerun(self.requests[0])


if __name__ == "__main__":
    unittest.main()
