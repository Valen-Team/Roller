from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

from test_collector import FakeHarness

from roller import hf_dataset
from roller.cli import main
from roller.collector import CollectionConfig, Collector
from roller.models import HarnessResult, RunRequest, Task
from roller.sft_dataset import bench_score, write_sft_dataset
from roller.storage import safe_component

MODEL_A = "vllm/model-a"
MODEL_B = "vllm/model-b"


def _bench(score: float, *, bench: str = "wildclawbench", error: str | None = None):
    """The grade of a sample as grading records it in ``runtime.bench``."""

    return {
        "bench": bench,
        "category": "01_Productivity",
        "metadata": {},
        "score_key": "overall_score",
        "error": error,
        "scores": {"overall_score": score},
    }


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def _cli(*argv: str) -> tuple[int, list[str]]:
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            main(list(argv))
    except SystemExit as exit:
        return exit.code, stdout.getvalue().splitlines()
    raise AssertionError("main() returned without exiting")


def _report_scores(run_dir: Path) -> dict[tuple[str, str], float | None]:
    """The score of every graded sample in the report ``roller bench report`` writes."""

    report = run_dir / "bench-report.json"
    code, _ = _cli("bench", "report", str(run_dir), "--output", str(report))
    assert code == 0
    return {
        (row["model"], row["task_id"]): row["score"]
        for row in json.loads(report.read_text("utf-8"))["tasks"]
    }


def _index_scores(path: Path) -> dict[tuple[str, str], float | None]:
    return {(row["model"], row["task_id"]): row["score"] for row in _lines(path)}


class GradedHarness(FakeHarness):
    """A FakeHarness whose samples carry a grade, and that fails chosen tasks."""

    def run(self, request: RunRequest, output_dir: Path, attempt: int) -> HarnessResult:
        if request.task.id in self.state["failed"]:
            return HarnessResult.error_result("agent crashed")
        result = super().run(request, output_dir, attempt)
        grade = self.state["grades"].get(request.task.id)
        if grade is not None:
            result.metadata["bench"] = grade
        return result


def _collect(run_dir: Path, state: dict, resume: bool = False) -> list[dict]:
    """Collect four tasks whose records differ, so a misplaced index line shows."""

    state.update(lock=threading.Lock(), calls={}, active=0, max_active=0)
    requests = [
        RunRequest(
            index=index,
            task=Task(
                id=f"task-{index}", prompts=(f"prompt {index}",), source_index=index
            ),
            model="provider/model",
            timeout_seconds=30,
        )
        for index in range(4)
    ]
    return Collector(
        CollectionConfig(run_dir=run_dir, resume=resume),
        lambda slot: GradedHarness(state, slot),
    ).collect(requests)


class CollectedIndexTests(unittest.TestCase):
    def test_index_follows_the_records_and_carries_the_report_scores(self) -> None:
        state = {
            "grades": {
                "task-0": _bench(1.0),
                "task-1": _bench(0.0),
                "task-2": _bench(0.0, error="judge timed out"),
            },
            "failed": {"task-3"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            _collect(run_dir, state)

            records = _lines(run_dir / "sft.jsonl")
            index = _lines(run_dir / "sft_index.jsonl")
            model_dir = safe_component("provider/model")
            self.assertEqual(
                index[0],
                {
                    "task_id": "task-0",
                    "model": "provider/model",
                    "model_label": "provider/model",
                    "harness": "fake",
                    "sample": f"samples/{model_dir}/00000000_task-0",
                    "sample_index": 0,
                    "bench": "wildclawbench",
                    "category": "01_Productivity",
                    "score": 1.0,
                    "score_key": "overall_score",
                },
            )
            self.assertEqual(list(index[0]), list(index[1]))
            self.assertEqual(
                [row["task_id"] for row in index], ["task-0", "task-1", "task-2"]
            )
            # Line N of the index describes line N of the dataset.
            self.assertEqual(
                [
                    json.loads((run_dir / row["sample"] / "trace_SFT.json").read_text())
                    for row in index
                ],
                records,
            )
            # The records keep their shape: nothing but messages and tools.
            self.assertEqual(
                {key for row in records for key in row}, {"messages", "tools"}
            )
            # A zero is a score; a failed grading is not.
            self.assertEqual([row["score"] for row in index], [1.0, 0.0, None])
            self.assertEqual(
                _index_scores(run_dir / "sft_index.jsonl"), _report_scores(run_dir)
            )
            self.assertFalse((run_dir / "sft_segments_index.jsonl").exists())

    def test_resuming_a_run_without_an_index_writes_the_same_index(self) -> None:
        # SkillsBench scores only samples whose status is "ok"; a resumed sample
        # is "skipped" in memory and must keep the score of its info.json.
        state = {
            "grades": {
                f"task-{n}": _bench(float(n % 2), bench="skillsbench") for n in range(4)
            },
            "failed": set(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _collect(run_dir, state)
            dataset = (run_dir / "sft.jsonl").read_bytes()
            index = (run_dir / "sft_index.jsonl").read_bytes()
            # A run collected before the index existed.
            (run_dir / "sft_index.jsonl").unlink()

            resumed = _collect(run_dir, state, resume=True)

            self.assertEqual([row["status"] for row in resumed], ["skipped"] * 4)
            self.assertEqual(state["calls"], {})
            self.assertEqual((run_dir / "sft.jsonl").read_bytes(), dataset)
            self.assertEqual((run_dir / "sft_index.jsonl").read_bytes(), index)
            self.assertEqual(
                [row["score"] for row in _lines(run_dir / "sft_index.jsonl")],
                [0.0, 1.0, 0.0, 1.0],
            )

    def test_samples_without_a_grade_have_null_bench_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            _collect(run_dir, {"grades": {}, "failed": set()})

            index = _lines(run_dir / "sft_index.jsonl")
            self.assertEqual(len(index), 4)
            for row in index:
                self.assertEqual(
                    [row[key] for key in ("bench", "category", "score", "score_key")],
                    [None] * 4,
                )


class BenchScoreTests(unittest.TestCase):
    def _score(self, bench: dict, **result) -> float | None:
        return bench_score({"status": "ok", "runtime": {"bench": bench}, **result})[1]

    def test_score_follows_the_adapter_of_the_bench(self) -> None:
        # WildClawBench: the declared key, else the mean of the numeric fields.
        self.assertEqual(self._score(_bench(0.75)), 0.75)
        self.assertEqual(
            self._score({**_bench(0.0), "scores": {"accuracy": 1.0, "style": 0.5}}),
            0.75,
        )
        self.assertIsNone(self._score(_bench(0.0, error="judge timed out")))
        self.assertIsNone(
            self._score({**_bench(0.0), "scores": {"overall_score": 0.0, "error": "x"}})
        )
        # A bench without an adapter is averaged the same way.
        self.assertEqual(self._score(_bench(0.5, bench="in-house")), 0.5)
        self.assertEqual(bench_score({"status": "ok", "runtime": {}}), ({}, None))

    def test_skillsbench_does_not_score_a_verifier_that_could_not_run(self) -> None:
        broken = _bench(0.0, bench="skillsbench")
        broken["scores"]["verifier_output"] = "bash: uvx: command not found"

        self.assertIsNone(self._score(broken))
        self.assertEqual(self._score(_bench(0.0, bench="skillsbench")), 0.0)
        self.assertEqual(
            self._score(
                _bench(1.0, bench="skillsbench"),
                status="skipped",
                resumed_from_status="ok",
            ),
            1.0,
        )


def _sample(
    run_dir: Path,
    request_index: int,
    source_index: int,
    model: str,
    *,
    status: str = "ok",
    bench: dict | None = None,
    eligibility: dict | None = None,
    segments: int = 0,
) -> Path:
    """Write one sample as a collection before the index existed left it on disk."""

    task = f"task-{source_index}"
    sample = run_dir / "samples" / safe_component(model) / f"{source_index:08d}_{task}"
    sample.mkdir(parents=True)
    message = [
        {"role": "user", "content": task},
        {"role": "assistant", "content": f"{model} did {task}"},
    ]
    trajectory: dict = {"sft": None}
    if status == "ok" and segments:
        (sample / "sft_segments").mkdir()
        trajectory["sft_segments"] = []
        for window in range(segments):
            path = f"sft_segments/{window + 1:04d}.json"
            record = {"messages": message, "tools": [], "target_start_index": 1}
            (sample / path).write_text(json.dumps(record), encoding="utf-8")
            trajectory["sft_segments"].append({"path": path, "window_index": window})
    elif status == "ok":
        trajectory["sft"] = "trace_SFT.json"
        (sample / "trace_SFT.json").write_text(
            json.dumps({"messages": message, "tools": []}), encoding="utf-8"
        )
    runtime: dict = {}
    if bench is not None:
        runtime["bench"] = bench
    if eligibility is not None:
        runtime["sft_eligibility"] = eligibility
    info = {
        "schema_version": "roller.sample/v1",
        "request_index": request_index,
        "sample_index": 0,
        "sample_count": 1,
        "task_id": task,
        "source_index": source_index,
        "model": model,
        "model_label": model,
        "harness": "openclaw",
        "status": status,
        "error": None if status == "ok" else "task exceeded 300s",
        "usage": {},
        "trajectory": trajectory,
        "runtime": runtime,
    }
    (sample / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return sample


def _run(root: Path, bench: str = "wildclawbench") -> Path:
    """A graded run of two models; requests alternate between the models."""

    run_dir = root / "run"
    _sample(run_dir, 0, 0, MODEL_A, bench=_bench(1.0, bench=bench))
    _sample(run_dir, 1, 0, MODEL_B, bench=_bench(0.25, bench=bench))
    _sample(run_dir, 2, 1, MODEL_A, bench=_bench(0.0, bench=bench, error="no judge"))
    _sample(run_dir, 3, 1, MODEL_B, status="error")
    _sample(
        run_dir,
        4,
        2,
        MODEL_A,
        bench=_bench(1.0, bench=bench),
        eligibility={"eligible": False, "reason": "token_limit"},
    )
    _sample(run_dir, 5, 2, MODEL_B, bench=_bench(0.5, bench=bench))
    return run_dir


def _tasks(path: Path) -> list[tuple[str, str]]:
    return [(row["model"], row["task_id"]) for row in _lines(path)]


def _finish_collection(run_dir: Path) -> None:
    """Write the run-level SFT files as the end of a collection does."""

    results = sorted(
        (
            json.loads(path.read_text())
            for path in run_dir.glob("samples/*/*/info.json")
        ),
        key=lambda info: info["request_index"],
    )
    Collector(CollectionConfig(run_dir=run_dir), lambda slot: None)._write_sft_dataset(
        results
    )


class ExportTests(unittest.TestCase):
    def test_export_rebuilds_the_files_of_collection_from_the_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))

            code, printed = _cli("export", str(run_dir), "--format", "sft")

            output = run_dir / "sft"
            self.assertEqual(code, 0)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["sft.jsonl", "sft_index.jsonl"],
            )
            # Request order, not directory order.
            self.assertEqual(
                _tasks(output / "sft_index.jsonl"),
                [
                    (MODEL_A, "task-0"),
                    (MODEL_B, "task-0"),
                    (MODEL_A, "task-1"),
                    (MODEL_B, "task-2"),
                ],
            )
            self.assertEqual(
                [row["messages"][1]["content"] for row in _lines(output / "sft.jsonl")],
                [
                    f"{model} did {task}"
                    for model, task in _tasks(output / "sft_index.jsonl")
                ],
            )
            self.assertEqual(
                printed,
                [
                    f"Exported 4 records to {output / 'sft.jsonl'} (index: sft_index.jsonl)",
                    "Kept 4 of 6 samples; dropped 2:",
                    "  execution_failed: 1",
                    "  token_limit: 1",
                ],
            )
            # The same bytes that collection writes into the run.
            _finish_collection(run_dir)
            for name in ("sft.jsonl", "sft_index.jsonl"):
                self.assertEqual(
                    (output / name).read_bytes(), (run_dir / name).read_bytes()
                )

    def test_min_score_keeps_the_samples_that_reach_it(self) -> None:
        for bench in ("wildclawbench", "skillsbench"):
            with self.subTest(bench=bench), tempfile.TemporaryDirectory() as tmp:
                run_dir = _run(Path(tmp), bench)
                output = Path(tmp) / "good"

                code, printed = _cli(
                    "export",
                    str(run_dir),
                    "--format",
                    "sft",
                    "--min-score",
                    "0.5",
                    "-o",
                    str(output),
                )

                self.assertEqual(code, 0)
                index = _lines(output / "sft_index.jsonl")
                self.assertEqual(
                    [(row["model"], row["task_id"], row["score"]) for row in index],
                    [(MODEL_A, "task-0", 1.0), (MODEL_B, "task-2", 0.5)],
                )
                self.assertEqual(len(_lines(output / "sft.jsonl")), 2)
                self.assertEqual(
                    printed,
                    [
                        f"Exported 2 records to {output / 'sft.jsonl'} (index: sft_index.jsonl)",
                        "Kept 2 of 6 samples; dropped 4:",
                        "  below_min_score: 1",
                        "  execution_failed: 1",
                        "  token_limit: 1",
                        "  ungraded: 1",
                        "--min-score 0.5 is compared with the score that `roller bench "
                        "report` lists for each sample; in this run they range from "
                        "0.25 to 1.",
                    ],
                )
                # The scores are those of the report of the run.
                full = Path(tmp) / "full"
                _cli("export", str(run_dir), "--format", "sft", "-o", str(full))
                report = _report_scores(run_dir)
                for sample, score in _index_scores(full / "sft_index.jsonl").items():
                    self.assertEqual(score, report[sample])

    def test_min_score_leaves_out_a_run_without_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _sample(run_dir, 0, 0, MODEL_A)

            code, printed = _cli(
                "export", str(run_dir), "--format", "sft", "--min-score", "0.5"
            )

            self.assertEqual(code, 0)
            self.assertEqual((run_dir / "sft" / "sft.jsonl").read_text(), "")
            self.assertEqual((run_dir / "sft" / "sft_index.jsonl").read_text(), "")
            self.assertEqual(
                printed[1:],
                [
                    "Kept 0 of 1 sample; dropped 1:",
                    "  ungraded: 1",
                    "--min-score 0.5 is compared with the score that `roller bench "
                    "report` lists for each sample; no sample of this run has one.",
                ],
            )

    def test_segments_have_their_own_dataset_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _sample(run_dir, 0, 0, MODEL_A, bench=_bench(1.0))
            _sample(run_dir, 1, 1, MODEL_A, bench=_bench(0.25), segments=2)
            output = Path(tmp) / "out"

            code, printed = _cli(
                "export", str(run_dir), "--format", "sft", "-o", str(output)
            )

            self.assertEqual(code, 0)
            self.assertEqual(_tasks(output / "sft_index.jsonl"), [(MODEL_A, "task-0")])
            segments = _lines(output / "sft_segments.jsonl")
            index = _lines(output / "sft_segments_index.jsonl")
            self.assertEqual([row["target_start_index"] for row in segments], [1, 1])
            self.assertEqual(
                [
                    (row["task_id"], row["segment"], row["window_index"], row["score"])
                    for row in index
                ],
                [
                    ("task-1", "sft_segments/0001.json", 0, 0.25),
                    ("task-1", "sft_segments/0002.json", 1, 0.25),
                ],
            )
            self.assertEqual(
                [
                    json.loads((run_dir / row["sample"] / row["segment"]).read_text())
                    for row in index
                ],
                segments,
            )
            self.assertEqual(
                printed,
                [
                    f"Exported 1 record to {output / 'sft.jsonl'} (index: sft_index.jsonl)",
                    f"Exported 2 records to {output / 'sft_segments.jsonl'} "
                    "(index: sft_segments_index.jsonl)",
                    "Kept 2 of 2 samples",
                ],
            )
            # Collection writes the same four files into the run.
            _finish_collection(run_dir)
            for file in sorted(output.iterdir()):
                self.assertEqual(
                    file.read_bytes(), (run_dir / file.name).read_bytes(), file.name
                )
            self.assertEqual(len(list(output.iterdir())), 4)

            # A run with segments keeps both segment files when none reaches the score.
            _cli(
                "export",
                str(run_dir),
                "--format",
                "sft",
                "--min-score",
                "0.5",
                "-o",
                str(output),
            )
            self.assertEqual((output / "sft_segments.jsonl").read_text(), "")
            self.assertEqual((output / "sft_segments_index.jsonl").read_text(), "")
            self.assertEqual(len(_lines(output / "sft_index.jsonl")), 1)

    def test_segment_files_of_an_earlier_write_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            sample = _sample(run_dir, 0, 0, MODEL_A)
            info = json.loads((sample / "info.json").read_text())
            for name in ("sft_segments.jsonl", "sft_segments_index.jsonl"):
                (run_dir / name).write_text("{}\n", encoding="utf-8")

            written = write_sft_dataset(run_dir, run_dir, [(sample, info)])

            self.assertEqual(written.records, {"sft.jsonl": 1})
            self.assertFalse((run_dir / "sft_segments.jsonl").exists())
            self.assertFalse((run_dir / "sft_segments_index.jsonl").exists())

    def test_min_score_applies_to_the_sft_format_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))

            code, _ = _cli("export", str(run_dir), "--min-score", "0.5")

            self.assertEqual(code, 2)
            self.assertFalse((run_dir / "atif").exists())

    def test_help_describes_the_format_and_the_minimum_score(self) -> None:
        code, printed = _cli("export", "--help")

        self.assertEqual(code, 0)
        self.assertIn("{atif,hf,sft,tokens,xtuner}", "\n".join(printed))
        self.assertIn("--min-score X", "\n".join(printed))
        self.assertIn("--inline-images", "\n".join(printed))

    def test_sft_export_upgrades_legacy_placeholder_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            sft_path = sorted((run_dir / "samples").glob("*/*/trace_SFT.json"))[0]
            uri = "data:image/png;base64,AA=="
            legacy = {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "Inspect\n<image>"},
                    {
                        "role": "assistant",
                        "reasoning_content": "",
                        "content": "done",
                        "tool_calls": [],
                    },
                ],
                "tools": [],
                "images": [uri],
            }
            sft_path.write_text(json.dumps(legacy), encoding="utf-8")
            output = Path(tmp) / "exported"
            code, _ = _cli("export", "--format", "sft", str(run_dir), "-o", str(output))
            self.assertEqual(code, 0)
            rows = _lines(output / "sft.jsonl")
            upgraded = next(
                row for row in rows if isinstance(row["messages"][1]["content"], list)
            )
            self.assertNotIn("images", upgraded)
            self.assertEqual(
                upgraded["messages"][1]["content"],
                [
                    {"type": "text", "text": "Inspect"},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            )

    def test_hugging_face_run_directories_index_their_own_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run(Path(tmp))
            # The run-level files as collection left them, covering both models.
            _finish_collection(run_dir)
            self.assertEqual(len(_lines(run_dir / "sft_index.jsonl")), 4)
            output = Path(tmp) / "dataset"

            hf_dataset.export_run(run_dir, output, environ={}, runs=True, machine="")

            for part, model in (
                ("openclaw-model-a", MODEL_A),
                ("openclaw-model-b", MODEL_B),
            ):
                root = output / "runs" / part
                index = _lines(root / "sft_index.jsonl")
                self.assertEqual({row["model"] for row in index}, {model})
                self.assertEqual(
                    [
                        json.loads(
                            (root / row["sample"] / "trace_SFT.json").read_text()
                        )
                        for row in index
                    ],
                    _lines(root / "sft.jsonl"),
                )


if __name__ == "__main__":
    unittest.main()
