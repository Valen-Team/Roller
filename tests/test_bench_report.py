from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from roller.benches import build_report, format_report, mean_aggregate
from roller.benches.qwenclaw import QwenClawBenchAdapter
from roller.benches.skillsbench import SkillsBenchAdapter
from roller.cli import _bench_report
from roller.tasks import TaskFormatError

GRADER = "def grade(transcript, workspace_path):\n    return {}\n"


def _row(
    task: str,
    model: str = "model-a",
    score: float | None = None,
    *,
    status: str = "ok",
    error: str | None = None,
    grader_error: str | None = None,
    category: str | None = "demo",
    harness: str | None = "openclaw",
    metadata: dict | None = None,
    passed: bool | None = None,
) -> dict:
    scores: dict | None = None
    if grader_error:
        scores = {"overall_score": 0.0, "score_eligible": False, "error": grader_error}
    elif score is not None:
        scores = {"overall_score": score}
        if passed is not None:
            scores["passed"] = passed
    return {
        "task_id": task,
        "model": model,
        "harness": harness,
        "status": status,
        "error": error,
        "bench_scores": scores,
        "bench_score_key": "overall_score",
        "bench_category": category,
        "bench_metadata": metadata,
        "bench_error": grader_error,
    }


def _blended_keys(report: dict) -> set[str]:
    """Top-level keys that would describe several models with one number."""

    return set(report) - {
        "schema_version",
        "task_count",
        "scored_count",
        "unscored_count",
        "planned_count",
        "missing_count",
        "models",
        "tasks",
    }


class PerModelReportTests(unittest.TestCase):
    def test_two_models_are_reported_side_by_side_and_never_averaged(self) -> None:
        rows = [
            _row("t1", "model-a", 1.0),
            _row("t2", "model-a", 1.0),
            _row("t1", "model-b", 0.0),
            _row("t2", "model-b", 0.0),
        ]

        report = build_report(rows)

        self.assertEqual(_blended_keys(report), set())
        self.assertEqual(
            (report["task_count"], report["scored_count"], report["unscored_count"]),
            (4, 4, 0),
        )
        self.assertEqual(
            [(entry["model"], entry["harness"]) for entry in report["models"]],
            [("model-a", "openclaw"), ("model-b", "openclaw")],
        )
        self.assertEqual([entry["average"] for entry in report["models"]], [1.0, 0.0])
        self.assertEqual([entry["task_count"] for entry in report["models"]], [2, 2])
        self.assertEqual(
            [entry["categories"]["demo"]["average"] for entry in report["models"]],
            [1.0, 0.0],
        )
        # The rows of every model stay at the top level; entries hold no copy.
        self.assertEqual(
            [(task["task_id"], task["model"]) for task in report["tasks"]],
            [
                ("t1", "model-a"),
                ("t2", "model-a"),
                ("t1", "model-b"),
                ("t2", "model-b"),
            ],
        )
        self.assertTrue(all("tasks" not in entry for entry in report["models"]))

        text = format_report("wildclawbench", report)
        self.assertNotIn("0.5000", text)
        self.assertIn("2 models, 4 samples  (scored 4, unscored 0)", text)
        self.assertIn("model-a (openclaw)  average 1.0000  scored 2/2", text)
        self.assertIn("model-b (openclaw)  average 0.0000  scored 2/2", text)
        self.assertEqual(text.count("  demo  (2 tasks)"), 2)

    def test_one_model_keeps_the_keys_earlier_readers_use(self) -> None:
        rows = [_row("t1", score=1.0), _row("t2", score=0.5)]

        report = build_report(rows)

        aggregate = mean_aggregate(rows)
        for key, value in aggregate.items():
            self.assertEqual(report[key], value, key)
        self.assertEqual(report["average"], 0.75)
        self.assertEqual(len(report["models"]), 1)
        entry = report["models"][0]
        self.assertEqual((entry["model"], entry["harness"]), ("model-a", "openclaw"))
        for key in ("task_count", "scored_count", "average", "categories"):
            self.assertEqual(entry[key], report[key], key)

        text = format_report("wildclawbench", report)
        self.assertIn("  model  model-a (openclaw)", text)
        self.assertIn("  scored 2/2  (unscored 0)", text)
        self.assertIn("0.7500", text)

    def test_a_label_that_ran_under_two_harnesses_is_split_by_harness(self) -> None:
        rows = [
            _row("t1", "model-a", 1.0, harness="codex"),
            _row("t1", "model-a", 0.0, harness="claude-code"),
            _row("t1", "model-b", 0.5, harness="codex"),
        ]

        report = build_report(rows)

        self.assertEqual(
            [
                (entry["model"], entry["harness"], entry["average"])
                for entry in report["models"]
            ],
            [
                ("model-a", "codex", 1.0),
                ("model-a", "claude-code", 0.0),
                ("model-b", "codex", 0.5),
            ],
        )
        self.assertEqual(_blended_keys(report), set())

    def test_a_launch_failure_without_a_harness_stays_with_its_model(self) -> None:
        # A sample that failed before its Harness was built records "unknown".
        rows = [
            _row("t1", score=1.0),
            _row("t2", status="error", error="image build failed", harness="unknown"),
        ]

        report = build_report(rows)

        self.assertEqual(len(report["models"]), 1)
        self.assertEqual(report["models"][0]["harness"], "openclaw")
        self.assertEqual((report["task_count"], report["average"]), (2, 0.5))

    def test_skillsbench_reports_a_pass_rate_per_model_only(self) -> None:
        rows = [
            _row("t1", "model-a", 1.0, passed=True),
            _row("t2", "model-a", 1.0, passed=True),
            _row("t1", "model-b", 0.0, passed=False),
            _row("t2", "model-b", status="timeout", error="task exceeded 30s"),
        ]

        report = build_report(rows, SkillsBenchAdapter().aggregate)

        self.assertEqual(_blended_keys(report), set())
        self.assertEqual([entry["pass_rate"] for entry in report["models"]], [1.0, 0.0])
        self.assertEqual(
            [entry["infra_failed_count"] for entry in report["models"]], [0, 1]
        )
        text = format_report("skillsbench", report)
        self.assertIn(
            "model-a (openclaw)  pass 1.0000  reward 1.0000  scored 2/2", text
        )
        self.assertIn(
            "model-b (openclaw)  pass 0.0000  reward 0.0000  scored 1/2", text
        )
        self.assertNotIn("0.5000", text)

    def test_a_report_without_samples_says_so(self) -> None:
        report = build_report([])

        self.assertEqual((report["task_count"], report["models"]), (0, []))
        self.assertIn("no samples", format_report("demo", report))


class DenominatorTests(unittest.TestCase):
    def test_unscored_samples_count_as_zero_in_headline_and_categories(self) -> None:
        rows = [
            _row("t1", score=1.0, category="alpha"),
            _row("t2", status="error", error="gateway did not start", category="alpha"),
            _row("t3", grader_error="judge request failed", category="beta"),
            _row("t4", status="timeout", error="task exceeded 30s", category="beta"),
        ]

        report = mean_aggregate(rows)

        self.assertEqual(
            (report["task_count"], report["scored_count"], report["unscored_count"]),
            (4, 1, 3),
        )
        self.assertEqual((report["average"], report["scored_average"]), (0.25, 1.0))
        self.assertEqual(
            report["categories"],
            {
                "alpha": {
                    "task_count": 2,
                    "scored_count": 1,
                    "average": 0.5,
                    "scored_average": 1.0,
                },
                "beta": {
                    "task_count": 2,
                    "scored_count": 0,
                    "average": 0.0,
                    "scored_average": None,
                },
            },
        )
        # The categories add up to the headline.
        categories = report["categories"].values()
        self.assertEqual(
            sum(row["average"] * row["task_count"] for row in categories)
            / sum(row["task_count"] for row in categories),
            report["average"],
        )
        # An unscored row shows the sample's status and what went wrong.
        self.assertEqual(
            [
                (task["status"], task["score"], task["error"])
                for task in report["tasks"]
            ],
            [
                ("ok", 1.0, None),
                ("error", None, "gateway did not start"),
                ("ok", None, "judge request failed"),
                ("timeout", None, "task exceeded 30s"),
            ],
        )

        text = format_report("demo", build_report(rows))
        self.assertIn("0.5000  alpha  (2 tasks, 1 scored)", text)
        self.assertIn("0.0000  beta  (2 tasks, 0 scored)", text)
        self.assertIn("- t2 (status error): gateway did not start", text)
        self.assertIn("- t3 (status ok): judge request failed", text)

    def test_skillsbench_categories_follow_the_score_eligible_headline(self) -> None:
        rows = [
            _row("t1", score=1.0, passed=True, category="alpha"),
            _row("t2", status="timeout", error="task exceeded 30s", category="alpha"),
            _row("t3", status="error", error="image build failed", category="beta"),
        ]

        report = SkillsBenchAdapter().aggregate(rows)

        # Upstream leaves ineligible trials out of the pass rate; they still count.
        self.assertEqual(
            (report["task_count"], report["scored_count"], report["pass_rate"]),
            (3, 1, 1.0),
        )
        self.assertEqual(report["infra_failed_count"], 2)
        self.assertEqual(
            report["categories"]["alpha"],
            {
                "task_count": 2,
                "scored_count": 1,
                "pass_rate": 1.0,
                "average": 1.0,
                "scored_average": 1.0,
            },
        )
        self.assertEqual(
            report["categories"]["beta"],
            {
                "task_count": 1,
                "scored_count": 0,
                "pass_rate": None,
                "average": None,
                "scored_average": None,
            },
        )
        text = format_report(
            "skillsbench", build_report(rows, SkillsBenchAdapter().aggregate)
        )
        self.assertIn("pass 1.0000  reward 1.0000  alpha  (2 trials, 1 scored)", text)
        self.assertIn("pass n/a  reward n/a  beta  (1 trials, 0 scored)", text)

    def test_qwenclaw_categories_weight_upstream_tasks_like_the_headline(self) -> None:
        def run(task: str, score: float | None, **extra) -> dict:
            return _row(
                f"{task}__run",
                score=score,
                metadata={"upstream_task_id": task},
                **extra,
            )

        # task_a ran once, task_b three times; one run of task_b never started.
        rows = [
            run("task_a", 1.0),
            run("task_b", 0.0),
            run("task_b", 0.0),
            run("task_b", None, status="error", error="gateway did not start"),
        ]

        report = QwenClawBenchAdapter().aggregate(rows)

        self.assertEqual(report["average"], 0.5)
        self.assertEqual(report["categories"]["demo"]["average"], report["average"])
        self.assertEqual(
            (
                report["categories"]["demo"]["task_count"],
                report["categories"]["demo"]["scored_count"],
            ),
            (4, 3),
        )
        self.assertEqual(report["runs_per_task"], {"task_a": 1, "task_b": 3})


class BenchReportCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name) / "run"
        self.run_dir.mkdir()
        self.index = 0

    def _tasks(self, *tasks: tuple[str, str], graded: bool = True) -> None:
        with (self.run_dir / "tasks.jsonl").open("a", encoding="utf-8") as stream:
            for task, category in tasks:
                bench = {
                    "schema_version": "roller.bench/v1",
                    "name": "wildclawbench",
                    "category": category,
                    "metadata": {"name": task},
                }
                if graded:
                    bench["grade"] = {"code": GRADER}
                row = {"id": task, "harness": "openclaw", "prompt": "x", "bench": bench}
                stream.write(json.dumps(row) + "\n")

    def _info(self, *, total: int | None = None, bench: bool = True, **extra) -> None:
        configuration = {"harness": "openclaw", **extra}
        if bench:
            configuration["bench"] = {"name": "wildclawbench"}
        payload: dict = {
            "schema_version": "roller.run/v1",
            "configuration": configuration,
        }
        if total is not None:
            payload["summary"] = {"total": total}
        (self.run_dir / "info.json").write_text(json.dumps(payload), encoding="utf-8")

    def _sample(
        self,
        task: str,
        model: str = "model-a",
        *,
        score: float | None = None,
        status: str = "ok",
        error: str | None = None,
        category: str = "alpha",
        harness: str = "openclaw",
        label: str | None = None,
    ) -> None:
        directory = (
            self.run_dir / "samples" / (label or model) / f"{self.index:08d}_{task}"
        )
        directory.mkdir(parents=True)
        info: dict = {
            "schema_version": "roller.sample/v1",
            "request_index": self.index,
            "task_id": task,
            "model": model,
            "harness": harness,
            "status": status,
            "error": error,
            "runtime": {},
        }
        if label:
            info["model_label"] = label
        if score is not None:
            info["runtime"]["bench"] = {
                "bench": "wildclawbench",
                "category": category,
                "score_key": "overall_score",
                "error": None,
                "scores": {"overall_score": score},
            }
        (directory / "info.json").write_text(json.dumps(info), encoding="utf-8")
        self.index += 1

    def _report(self) -> tuple[dict, str]:
        output = self.run_dir / "bench-report.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = _bench_report(
                SimpleNamespace(run_dir=str(self.run_dir), output=str(output))
            )
        self.assertEqual(code, 0)
        return json.loads(output.read_text(encoding="utf-8")), stdout.getvalue()

    def test_a_run_of_two_models_writes_no_average_over_both(self) -> None:
        self._tasks(("t1", "alpha"), ("t2", "alpha"))
        self._info(total=4)
        for model, score in (("model-a", 1.0), ("model-b", 0.0)):
            for task in ("t1", "t2"):
                self._sample(task, model, score=score)

        report, text = self._report()

        self.assertEqual(_blended_keys(report), set())
        self.assertEqual(
            [(entry["model"], entry["average"]) for entry in report["models"]],
            [("model-a", 1.0), ("model-b", 0.0)],
        )
        self.assertNotIn("0.5000", text)
        self.assertNotIn(
            "0.5", json.dumps({k: v for k, v in report.items() if k != "tasks"})
        )

    def test_samples_that_failed_before_grading_stay_in_the_denominator(self) -> None:
        self._tasks(("t1", "alpha"), ("t2", "alpha"), ("t3", "beta"), ("t4", "beta"))
        self._info(total=4)
        self._sample("t1", score=1.0)
        for task in ("t2", "t3", "t4"):
            self._sample(task, status="error", error="OpenClaw gateway did not start")

        report, text = self._report()

        self.assertEqual(
            (report["task_count"], report["scored_count"], report["unscored_count"]),
            (4, 1, 3),
        )
        self.assertEqual((report["average"], report["scored_average"]), (0.25, 1.0))
        # The category of an ungraded sample comes from its task row.
        self.assertEqual(report["categories"]["alpha"]["task_count"], 2)
        self.assertEqual(report["categories"]["beta"]["average"], 0.0)
        self.assertEqual(
            report["tasks"][3],
            {
                "task_id": "t4",
                "model": "model-a",
                "harness": "openclaw",
                "category": "beta",
                "status": "error",
                "score": None,
                "error": "OpenClaw gateway did not start",
            },
        )
        self.assertNotIn("missing_count", report)
        self.assertIn("scored 1/4  (unscored 3)", text)
        self.assertIn("- t3 (status error): OpenClaw gateway did not start", text)

    def test_a_run_whose_every_sample_failed_to_launch_still_reports(self) -> None:
        self._tasks(("t1", "alpha"))
        self._info(total=1)
        self._sample("t1", status="error", error="image not found", harness="unknown")

        report, _ = self._report()

        self.assertEqual((report["task_count"], report["average"]), (1, 0.0))

    def test_rollouts_without_a_sample_are_counted_but_not_averaged(self) -> None:
        self._tasks(("t1", "alpha"), ("t2", "alpha"), ("t3", "alpha"))
        self._info(total=6)
        self._sample("t1", score=1.0)
        self._sample("t2", score=0.0)

        report, text = self._report()

        self.assertEqual((report["planned_count"], report["missing_count"]), (6, 4))
        self.assertEqual((report["task_count"], report["average"]), (2, 0.5))
        self.assertEqual(list(report)[-2:], ["models", "tasks"])
        self.assertIn(
            "4 of 6 planned rollouts have no sample yet and are not in the averages",
            text,
        )

    def test_a_task_without_a_grader_is_not_an_unscored_sample(self) -> None:
        self._tasks(("t1", "alpha"))
        self._tasks(("t2", "alpha"), graded=False)
        self._info(total=2)
        self._sample("t1", score=1.0)
        self._sample("t2")

        report, _ = self._report()

        self.assertEqual((report["task_count"], report["average"]), (1, 1.0))

    def test_an_exported_task_list_names_the_bench_samples_of_a_plain_run(self) -> None:
        # ``roller bench export`` followed by ``roller collect --tasks``: the run
        # has no tasks.jsonl of its own, and other tasks may share the list.
        tasks = self.run_dir.parent / "exported.jsonl"
        tasks.write_text(
            json.dumps(
                {
                    "id": "t1",
                    "prompt": "x",
                    "bench": {"name": "wildclawbench", "grade": {"code": GRADER}},
                }
            )
            + "\n"
            + json.dumps({"id": "plain", "prompt": "x"})
            + "\n",
            encoding="utf-8",
        )
        self._info(total=3, bench=False, tasks=str(tasks))
        self._sample("t1", score=1.0)
        self._sample("t1", "model-b", status="error", error="gateway did not start")
        self._sample("plain", status="error", error="gateway did not start")

        report, _ = self._report()

        self.assertEqual(report["task_count"], 2)
        self.assertEqual(
            [(entry["model"], entry["average"]) for entry in report["models"]],
            [("model-a", 1.0), ("model-b", 0.0)],
        )

    def test_a_missing_task_list_is_reported_not_guessed(self) -> None:
        # The exported list is gone, so nothing says what the failed sample was.
        self._info(total=2, bench=False, tasks=str(self.run_dir.parent / "gone.jsonl"))
        self._sample("t1", score=1.0)
        self._sample("t2", status="error", error="gateway did not start")

        with self.assertLogs("roller.benches.run_report", level="WARNING") as logs:
            report, _ = self._report()

        self.assertEqual(report["task_count"], 1)
        self.assertIn("1 samples have no grading record", logs.output[0])

    def test_models_are_named_by_their_label(self) -> None:
        self._info(total=1)
        self._sample("t1", "provider/served-id-2026", score=1.0, label="glm")

        report, text = self._report()

        self.assertEqual(report["models"][0]["model"], "glm")
        self.assertEqual(report["tasks"][0]["model"], "glm")
        self.assertIn("model  glm (openclaw)", text)

    def test_a_run_without_bench_samples_is_an_error(self) -> None:
        self._info(total=1, bench=False)
        self._sample("plain")

        with self.assertRaisesRegex(TaskFormatError, "no bench samples"):
            self._report()


if __name__ == "__main__":
    unittest.main()
