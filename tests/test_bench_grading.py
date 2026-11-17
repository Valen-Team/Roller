from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from test_bench_adapters import PINCH_AUTOMATED, PINCH_HYBRID
from test_collector import FakeHarness
from test_gateway_ports import isolate_gateway_ports

from roller.benches import grading, grading_failure, missing_grader_env
from roller.benches.base import BenchBinding, GradeSpec
from roller.benches.grading import GradedHarness, GradeOutcome
from roller.benches.pinchbench import PinchBenchAdapter
from roller.benches.qwenclaw import JUDGE_ENV_NAMES
from roller.benches.run_config import BENCH_RUN_SCHEMA, load_bench_run_plan
from roller.cli import _build_parser, _collect, main
from roller.collector import CollectionConfig, Collector
from roller.harnesses.base import InteractiveHarness
from roller.models import HarnessResult, RunRequest, Task

GRADER = "def grade(transcript, workspace_path):\n    return {}\n"
JUDGE_NAMES = (
    "PINCHBENCH_JUDGE_API_KEY",
    "OPENROUTER_API_KEY",
    "QWENCLAW_JUDGE_BASE_URL",
    "QWENCLAW_JUDGE_API_KEY",
    "JUDGE_BASE_URL",
    "JUDGE_API_KEY",
    "JUDGE_MODEL",
)


def _pinch_root(root: Path) -> Path:
    tasks = root / "tasks"
    tasks.mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "assets" / "seed.txt").write_text("before\n", encoding="utf-8")
    (tasks / "manifest.yaml").write_text(
        "categories:\n  coding: [task_edit, task_sessions]\n", encoding="utf-8"
    )
    (tasks / "task_edit.md").write_text(PINCH_AUTOMATED, encoding="utf-8")
    (tasks / "task_sessions.md").write_text(PINCH_HYBRID, encoding="utf-8")
    return root


def _row(name: str, *, env=(), metadata=None, grade: bool = True) -> dict:
    binding = BenchBinding(
        name,
        grade=GradeSpec(GRADER, env_names=tuple(env)) if grade else None,
        metadata=metadata or {},
    )
    return {"id": "task", "prompt": "x", "bench": binding.as_dict()}


class GraderEnvironmentTests(unittest.TestCase):
    def test_pinchbench_judged_tasks_need_one_of_two_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = PinchBenchAdapter().load(_pinch_root(Path(temporary)))
        judged = [row for row in rows if row["id"] == "task_sessions"]
        automated = [row for row in rows if row["id"] == "task_edit"]

        self.assertEqual(
            missing_grader_env(rows, {}),
            ["PINCHBENCH_JUDGE_API_KEY or OPENROUTER_API_KEY (1 task)"],
        )
        self.assertEqual(missing_grader_env(rows, {"OPENROUTER_API_KEY": "key"}), [])
        self.assertEqual(
            missing_grader_env(judged, {"PINCHBENCH_JUDGE_API_KEY": "key"}), []
        )
        # An empty value is a missing one.
        self.assertEqual(len(missing_grader_env(judged, {"OPENROUTER_API_KEY": ""})), 1)
        # A selection without judged tasks asks for no judge key.
        self.assertEqual(missing_grader_env(automated, {}), [])

    def test_pinchbench_rows_are_unchanged_so_existing_runs_still_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = PinchBenchAdapter().load(_pinch_root(Path(temporary)))
        judged = next(row for row in rows if row["id"] == "task_sessions")

        self.assertNotIn("env", judged["bench"]["grade"])
        self.assertIn("OPENROUTER_API_KEY", judged["bench"]["grade"]["optional_env"])
        self.assertIn(
            "PINCHBENCH_JUDGE_API_KEY", judged["bench"]["grade"]["optional_env"]
        )

    def test_qwenclaw_needs_an_endpoint_and_a_key_under_either_name(self) -> None:
        # The exported rows list all seven names the judge reads.
        judged = _row(
            "qwenclawbench", env=JUDGE_ENV_NAMES, metadata={"grading_type": "hybrid"}
        )
        automated = _row("qwenclawbench", metadata={"grading_type": "automated"})

        self.assertEqual(
            missing_grader_env([judged, judged], {"JUDGE_API_KEY": "key"}),
            ["QWENCLAW_JUDGE_BASE_URL or JUDGE_BASE_URL (2 tasks)"],
        )
        self.assertEqual(
            missing_grader_env(
                [judged],
                {"QWENCLAW_JUDGE_BASE_URL": "http://judge", "JUDGE_API_KEY": "key"},
            ),
            [],
        )
        self.assertEqual(missing_grader_env([automated], {}), [])

    def test_every_name_a_task_declares_is_required_by_default(self) -> None:
        wildclaw = _row("wildclawbench", env=("OPENROUTER_API_KEY", "JUDGE_MODEL"))
        custom = _row("my-own-bench", env=("MY_JUDGE_KEY",))
        ungraded = _row("wildclawbench", grade=False)
        plain = {"id": "plain", "prompt": "x"}

        self.assertEqual(
            missing_grader_env(
                [wildclaw, custom, ungraded, plain], {"JUDGE_MODEL": "judge"}
            ),
            ["OPENROUTER_API_KEY (1 task)", "MY_JUDGE_KEY (1 task)"],
        )

    def test_the_grader_receives_every_variable_that_is_set(self) -> None:
        spec = GradeSpec(GRADER, env_names=("NEEDED",), optional_env_names=("EXTRA",))
        with mock.patch.dict(os.environ, {"NEEDED": "1", "EXTRA": ""}):
            self.assertEqual(
                grading.ContainerGrader(spec)._env_args(), ["-e", "NEEDED"]
            )


class GraderEnvironmentCliTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = _pinch_root(Path(temporary.name) / "pinchbench")
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in JUDGE_NAMES:
            os.environ.pop(name, None)

    def argv(self, *extra: str) -> list[str]:
        return [
            "collect",
            "--bench",
            "pinchbench",
            "--bench-root",
            str(self.root),
            "--model",
            "provider/model",
            *extra,
        ]

    def test_a_dry_run_stops_on_a_missing_judge_key(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "benchmark graders need environment variables that are not set: "
            r"PINCHBENCH_JUDGE_API_KEY or OPENROUTER_API_KEY \(1 task\)$",
        ):
            _collect(_build_parser().parse_args(self.argv("--dry-run")))

    def test_the_command_exits_2_with_one_error_line(self) -> None:
        with (
            self.assertLogs("roller.cli", level="ERROR") as logs,
            self.assertRaises(SystemExit) as exit_,
            mock.patch("logging.basicConfig"),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            main(self.argv("--dry-run"))

        self.assertEqual(exit_.exception.code, 2)
        self.assertEqual(len(logs.output), 1)
        self.assertNotIn("\n", logs.output[0])
        self.assertIn("PINCHBENCH_JUDGE_API_KEY or OPENROUTER_API_KEY", logs.output[0])
        self.assertEqual(stdout.getvalue(), "")

    def test_a_real_run_stops_before_docker_is_touched(self) -> None:
        with (
            mock.patch("roller.cli._docker_preflight") as docker,
            mock.patch("roller.cli.Collector") as collector,
            self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"),
        ):
            _collect(
                _build_parser().parse_args(
                    self.argv("--output", str(self.root.parent / "runs"))
                )
            )

        docker.assert_not_called()
        collector.assert_not_called()
        self.assertFalse((self.root.parent / "runs").exists())

    def test_a_selection_without_judged_tasks_needs_no_key(self) -> None:
        for selection in (
            ("--bench-variant", "automated-only"),
            ("--bench-task-id", "task_edit"),
        ):
            with self.subTest(selection=selection):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = _collect(
                        _build_parser().parse_args(self.argv("--dry-run", *selection))
                    )
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(stdout.getvalue())["tasks"], 1)

    def test_either_key_lets_the_run_through(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "key"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = _collect(_build_parser().parse_args(self.argv("--dry-run")))

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["grader_retries"], 2)


class FakeSession:
    container_started = True
    container = "live"

    def __init__(self) -> None:
        self.harness = mock.Mock()
        self.finished: list[tuple[str, str | None]] = []

    def start(self) -> None:
        pass

    def send(self, prompt: str, timeout: float) -> None:
        pass

    def finish(self, status: str, error: str | None = None) -> HarnessResult:
        self.finished.append((status, error))
        return HarnessResult(status=status, elapsed_seconds=0.1, error=error)


def _failure(message: str = "judge request failed: HTTP 429", elapsed: float = 0.2):
    return GradeOutcome({"overall_score": 0.0, "error": message}, elapsed)


class GraderRetryTests(unittest.TestCase):
    def _run(
        self,
        outcomes: list,
        *,
        retries: int = 2,
        spec: GradeSpec | None = None,
        session: FakeSession | None = None,
    ) -> tuple[dict, mock.Mock, list[float], FakeSession]:
        session = session or FakeSession()
        inner = mock.Mock(spec=InteractiveHarness)
        inner.name = "openclaw"
        inner.create_session.return_value = session
        binding = BenchBinding("wildclawbench", grade=spec or GradeSpec(GRADER))
        harness = GradedHarness(
            inner, binding, write_score_file=False, grader_retries=retries
        )
        request = RunRequest(0, Task("demo", ("do it",), 0), "model", 30)
        with (
            mock.patch.object(
                grading.ContainerGrader, "grade", side_effect=list(outcomes)
            ) as grade,
            mock.patch.object(grading.time, "sleep") as sleep,
            mock.patch.object(grading, "logger"),
        ):
            result = harness.run(request, Path("/unused"), 1)
        delays = [call.args[0] for call in sleep.call_args_list]
        return result.metadata["bench"], grade, delays, session

    def test_a_failed_judge_is_asked_again_in_the_same_container(self) -> None:
        bench, grade, delays, session = self._run(
            [_failure(), _failure(), GradeOutcome({"overall_score": 0.8}, 0.2)]
        )

        self.assertEqual(grade.call_count, 3)
        self.assertEqual({call.args[1] for call in grade.call_args_list}, {"live"})
        self.assertEqual(delays, [5.0, 10.0])
        self.assertEqual(bench["scores"], {"overall_score": 0.8})
        self.assertIsNone(bench["error"])
        self.assertEqual(bench["grade_attempts"], 3)
        # The container is torn down once, after the last attempt.
        self.assertEqual(session.finished, [("ok", None)])

    def test_a_score_is_never_retried_and_zero_is_a_score(self) -> None:
        bench, grade, delays, _ = self._run([GradeOutcome({"overall_score": 0.0}, 0.2)])

        self.assertEqual((grade.call_count, delays), (1, []))
        self.assertEqual(bench["scores"], {"overall_score": 0.0})
        self.assertNotIn("grade_attempts", bench)

    def test_the_failure_is_kept_once_the_retries_are_used_up(self) -> None:
        bench, grade, delays, _ = self._run(
            [_failure("first"), _failure("second"), _failure("third")]
        )

        self.assertEqual((grade.call_count, delays), (3, [5.0, 10.0]))
        self.assertEqual(bench["scores"]["error"], "third")
        self.assertEqual(bench["grade_attempts"], 3)

    def test_zero_retries_grades_once(self) -> None:
        bench, grade, delays, _ = self._run([_failure()], retries=0)

        self.assertEqual((grade.call_count, delays), (1, []))
        self.assertNotIn("grade_attempts", bench)

    def test_a_crashed_grader_is_retried(self) -> None:
        crash = GradeOutcome(
            {"overall_score": 0.0, "error": "grader failed: Traceback"},
            0.1,
            error="Traceback",
        )
        bench, grade, _, _ = self._run(
            [crash, GradeOutcome({"overall_score": 1.0}, 0.1)]
        )

        self.assertEqual(grade.call_count, 2)
        self.assertEqual(bench["scores"], {"overall_score": 1.0})

    def test_an_exception_while_grading_is_retried(self) -> None:
        bench, grade, _, _ = self._run(
            [OSError("docker exec failed"), GradeOutcome({"overall_score": 1.0}, 0.1)]
        )

        self.assertEqual(grade.call_count, 2)
        self.assertEqual(bench["scores"], {"overall_score": 1.0})

    def test_a_grader_that_failed_after_ten_minutes_is_not_run_again(self) -> None:
        bench, grade, delays, _ = self._run(
            [_failure("official test script timed out", elapsed=3300.0)]
        )

        self.assertEqual((grade.call_count, delays), (1, []))
        self.assertEqual(bench["scores"]["error"], "official test script timed out")

    def test_nothing_is_retried_without_a_container(self) -> None:
        session = FakeSession()
        session.container_started = False
        bench, grade, delays, _ = self._run([], session=session)

        self.assertEqual((grade.call_count, delays), (0, []))
        self.assertEqual(bench["error"], "container unavailable")

    def test_a_failed_agent_is_retried_only_where_the_bench_grades_it(self) -> None:
        def run(spec: GradeSpec, outcomes: list):
            session = FakeSession()
            session.send = mock.Mock(side_effect=TimeoutError())
            return self._run(outcomes, spec=spec, session=session)

        bench, grade, delays, session = run(GradeSpec(GRADER), [])
        self.assertEqual((grade.call_count, delays), (0, []))
        self.assertEqual(bench["error"], "agent timeout; benchmark grading skipped")
        self.assertEqual(session.finished[0][0], "timeout")

        bench, grade, delays, _ = run(
            GradeSpec(GRADER, grade_on_agent_failure=True),
            [_failure(), GradeOutcome({"overall_score": 0.5}, 0.1)],
        )
        self.assertEqual((grade.call_count, delays), (2, [5.0]))
        self.assertEqual(bench["scores"], {"overall_score": 0.5})

    def test_the_grader_slot_is_free_while_waiting_to_retry(self) -> None:
        semaphore = threading.BoundedSemaphore(1)
        held: list[bool] = []
        inner = mock.Mock(spec=InteractiveHarness)
        inner.name = "openclaw"
        inner.create_session.return_value = FakeSession()
        harness = GradedHarness(
            inner,
            BenchBinding("wildclawbench", grade=GradeSpec(GRADER)),
            write_score_file=False,
            grading_semaphore=semaphore,
            grader_retries=1,
        )

        def sleep(_seconds: float) -> None:
            held.append(not semaphore.acquire(blocking=False))
            semaphore.release()

        with (
            mock.patch.object(
                grading.ContainerGrader,
                "grade",
                side_effect=[_failure(), GradeOutcome({"overall_score": 1.0}, 0.1)],
            ),
            mock.patch.object(grading.time, "sleep", side_effect=sleep),
            mock.patch.object(grading, "logger"),
        ):
            harness.run(
                RunRequest(0, Task("demo", ("do it",), 0), "model", 30),
                Path("/unused"),
                1,
            )

        self.assertEqual(held, [False])


def _sample(status: str = "ok", **bench) -> dict:
    return {"status": status, "runtime": {"bench": bench} if bench else {}}


class GradingFailureTests(unittest.TestCase):
    def test_names_the_reason_a_finished_sample_has_no_score(self) -> None:
        scored = {"score_key": "overall_score", "scores": {"overall_score": 0.0}}
        judge = {"error": None, "scores": {"overall_score": 0.0, "error": "judge down"}}
        crash = {"error": "Traceback", "scores": {"overall_score": 0.0}}

        self.assertIsNone(grading_failure(_sample(**scored)))
        self.assertEqual(grading_failure(_sample(**judge)), "judge down")
        self.assertEqual(grading_failure(_sample(**crash)), "Traceback")
        # A sample that Resume kept is still unscored.
        self.assertEqual(grading_failure(_sample("skipped", **judge)), "judge down")
        self.assertEqual(
            grading_failure(_sample(scores={"notes": "no number"})),
            "grader returned no score",
        )

    def test_a_failed_agent_or_an_ungraded_task_is_not_a_grading_failure(self) -> None:
        skipped = {
            "error": "agent timeout; benchmark grading skipped",
            "scores": {"overall_score": 0.0},
        }

        self.assertIsNone(grading_failure(_sample("timeout", **skipped)))
        self.assertIsNone(grading_failure(_sample("error")))
        self.assertIsNone(grading_failure(_sample("ok")))
        self.assertIsNone(grading_failure({"status": "ok"}))


class JudgedHarness(FakeHarness):
    """A harness whose grader fails while ``state['judge']`` names an error."""

    def run(self, request: RunRequest, output_dir: Path, attempt: int) -> HarnessResult:
        result = super().run(request, output_dir, attempt)
        error = self.state["judge"].get(request.task.id)
        result.metadata["bench"] = {
            "bench": "wildclawbench",
            "score_key": "overall_score",
            "error": None,
            "scores": (
                {"overall_score": 0.0, "error": error}
                if error
                else {"overall_score": self.state.get("score", 1.0)}
            ),
        }
        return result


class ResumeUnscoredTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name) / "run"
        self.state = {
            "lock": threading.Lock(),
            "calls": {},
            "active": 0,
            "max_active": 0,
            "judge": {"task-1": "judge request failed"},
            "score": 0.0,
        }
        self.requests = [
            RunRequest(index, Task(f"task-{index}", ("hello",), index), "model", 30)
            for index in range(2)
        ]

    def collect(self, **config) -> list[dict]:
        return Collector(
            CollectionConfig(run_dir=self.run_dir, **config),
            lambda slot: JudgedHarness(self.state, slot),
        ).collect(self.requests)

    def calls(self) -> list[int]:
        return [self.state["calls"][(f"task-{index}", "model")] for index in range(2)]

    def test_resume_keeps_an_unscored_sample_unless_asked(self) -> None:
        first = self.collect()
        self.assertEqual([row["status"] for row in first], ["ok", "ok"])
        self.assertEqual(grading_failure(first[1]), "judge request failed")
        self.state["judge"] = {}

        kept = self.collect(resume=True)

        self.assertEqual([row["status"] for row in kept], ["skipped", "skipped"])
        self.assertEqual(self.calls(), [1, 1])
        self.assertEqual(grading_failure(kept[1]), "judge request failed")

    def test_resume_unscored_reruns_only_the_sample_whose_grading_failed(self) -> None:
        self.collect()
        self.state["judge"] = {}

        with self.assertLogs("roller.collector", level="INFO") as logs:
            again = self.collect(resume=True, resume_unscored=True)

        # task-0 scored a real zero and is kept; task-1 runs again.
        self.assertEqual([row["status"] for row in again], ["skipped", "ok"])
        self.assertEqual(self.calls(), [1, 2])
        self.assertIsNone(grading_failure(again[1]))
        self.assertEqual(again[1]["runtime"]["bench"]["scores"], {"overall_score": 0.0})
        self.assertIn(
            "[task-1 | model] grading failed (judge request failed); rerunning",
            "\n".join(logs.output),
        )

        settled = self.collect(resume=True, resume_unscored=True)
        self.assertEqual([row["status"] for row in settled], ["skipped", "skipped"])
        self.assertEqual(self.calls(), [1, 2])


class GradingFailureCliTests(unittest.TestCase):
    """Run ``roller collect`` for real up to the collector, which is replaced."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        isolate_gateway_ports(self)
        self.bench_root = _pinch_root(self.root / "pinchbench")
        self.tasks = self.root / "tasks.jsonl"
        self.tasks.write_text(
            json.dumps(
                {
                    "id": "task",
                    "prompt": "hi",
                    "bench": {"name": "my-own-bench", "grade": {"code": GRADER}},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.results: list[dict] = []
        self.configs: list[CollectionConfig] = []
        self.harnesses: list = []
        patches = contextlib.ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            mock.patch("roller.cli._docker_preflight", return_value="image-id")
        )
        patches.enter_context(
            mock.patch("roller.cli.Collector", side_effect=self._collector)
        )

    def _collector(self, config, harness_factory):
        self.configs.append(config)
        for result in self.results:
            sample = config.run_dir / "samples" / "model" / result["task_id"]
            sample.mkdir(parents=True, exist_ok=True)
            (sample / "info.json").write_text(json.dumps(result), encoding="utf-8")
        collector = mock.MagicMock()

        def collect(requests):
            self.harnesses.append(harness_factory(0, requests[0]))
            return self.results

        collector.collect.side_effect = collect
        return collector

    def _result(self, task: str, *, error: str | None = None, status: str = "ok"):
        return {
            "task_id": task,
            "model": "provider/model",
            "harness": "openclaw",
            "status": status,
            "runtime": {
                "bench": {
                    "bench": "pinchbench",
                    "score_key": "overall_score",
                    "error": None,
                    "scores": (
                        {"overall_score": 0.0, "error": error}
                        if error
                        else {"overall_score": 1.0}
                    ),
                }
            },
        }

    def invoke(self, *extra: str, bench: bool = True) -> tuple[int, str]:
        source = (
            [
                "--bench",
                "pinchbench",
                "--bench-root",
                str(self.bench_root),
                "--bench-variant",
                "automated-only",
            ]
            if bench
            else ["--tasks", str(self.tasks)]
        )
        args = [
            "collect",
            *source,
            "--model",
            "provider/model",
            "--output",
            str(self.root / "runs"),
            "--run-name",
            "run",
            *extra,
        ]
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = _collect(_build_parser().parse_args(args))
        return code, stdout.getvalue()

    def test_a_run_with_unscored_samples_says_so_and_exits_non_zero(self) -> None:
        rate_limit = (
            "LLM judge failed: no parseable score after 1 attempt(s): "
            "HTTP Error 429: Too Many Requests"
        )
        crash = "grader failed: Traceback (most recent call last):\nKeyError: 'total'"
        self.results = [
            self._result("a"),
            self._result("b", error=rate_limit),
            self._result("c", error=rate_limit),
            self._result("d", error=crash),
        ]

        code, output = self.invoke()

        self.assertEqual(code, 1)
        self.assertIn("Collected 4 trajectories (0 failed).", output)
        self.assertIn(
            "Grading failed for 3 of 4 samples; they have no score. "
            "Most common reasons:\n"
            f"  2x {rate_limit}\n"
            "  1x KeyError: 'total'\n"
            "Rerun them with --resume --resume-unscored",
            output,
        )
        report = json.loads(
            (self.root / "runs" / "run" / "bench-report.json").read_text("utf-8")
        )
        self.assertEqual((report["scored_count"], report["unscored_count"]), (1, 3))

    def test_a_fully_scored_run_exits_zero_and_says_nothing(self) -> None:
        self.results = [self._result("a"), self._result("b")]

        code, output = self.invoke()

        self.assertEqual(code, 0)
        self.assertNotIn("Grading failed", output)

    def test_a_failed_agent_is_not_reported_as_a_grading_failure(self) -> None:
        self.results = [
            self._result("a"),
            self._result(
                "b", error="agent timeout; benchmark grading skipped", status="timeout"
            ),
        ]

        code, output = self.invoke()

        self.assertEqual(code, 1)
        self.assertIn("Collected 2 trajectories (1 failed).", output)
        self.assertNotIn("Grading failed", output)

    def test_a_task_list_with_graders_is_treated_the_same(self) -> None:
        self.results = [self._result("task", error="judge down")]

        code, output = self.invoke(bench=False)

        self.assertEqual(code, 1)
        self.assertIn("Grading failed for 1 of 1 samples", output)

    def test_the_options_reach_the_grader_and_the_collector(self) -> None:
        self.invoke(bench=False)
        self.assertEqual(self.harnesses[-1].grader_retries, 2)
        self.assertFalse(self.configs[-1].resume_unscored)

        self.invoke(
            "--resume", "--resume-unscored", "--grader-retries", "0", bench=False
        )
        self.assertEqual(self.harnesses[-1].grader_retries, 0)
        self.assertTrue(self.configs[-1].resume_unscored)

    def test_the_options_are_recorded_but_do_not_enter_the_lock(self) -> None:
        self.invoke(bench=False)
        run = self.root / "runs" / "run"
        lock = (run / "lock.json").read_bytes()
        configuration = json.loads((run / "info.json").read_text())["configuration"]
        self.assertEqual(
            (configuration["grader_retries"], configuration["resume_unscored"]),
            (2, False),
        )
        for name in ("grader_retries", "resume_unscored", "grade_attempts"):
            self.assertNotIn(name.encode(), lock)

        # A resumed run may change both; the lock stays byte for byte the same.
        self.invoke(
            "--resume", "--resume-unscored", "--grader-retries", "5", bench=False
        )
        self.assertEqual((run / "lock.json").read_bytes(), lock)
        configuration = json.loads((run / "info.json").read_text())["configuration"]
        self.assertEqual(
            (configuration["grader_retries"], configuration["resume_unscored"]),
            (5, True),
        )

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "--resume-unscored requires --resume"):
            self.invoke("--resume-unscored", bench=False)
        with self.assertRaisesRegex(ValueError, "--grader-retries cannot be negative"):
            self.invoke("--grader-retries", "-1", bench=False)

    def test_a_run_configuration_sets_both_options(self) -> None:
        config = self.root / "run.yaml"
        config.write_text(
            json.dumps(
                {
                    "schema_version": BENCH_RUN_SCHEMA,
                    "benchmark": {"id": "pinchbench", "root": str(self.bench_root)},
                    "model": {
                        "id": "model",
                        "base_url": "http://127.0.0.1:1/v1",
                        "api_key_env": "MODEL_API_KEY",
                    },
                    "harness": {"id": "codex"},
                    "execution": {
                        "resume": True,
                        "resume_unscored": True,
                        "grader_retries": 0,
                    },
                }
            ),
            encoding="utf-8",
        )

        plan = load_bench_run_plan(
            config, environ={"MODEL_API_KEY": "key"}, temporary_root=self.root
        )
        args = _build_parser().parse_args(list(plan.collect_argv))

        self.assertEqual((args.grader_retries, args.resume_unscored), (0, True))


if __name__ == "__main__":
    unittest.main()
