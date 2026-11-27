"""ZeroBench: both splits, the images as workspace files, strict grading, avg@k/pass@k."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from roller.benches import get_bench, missing_grader_env
from roller.benches.base import BenchFormatError, GradeSpec
from roller.benches.grading import transcript_container_path
from roller.benches.report import build_report, format_report
from roller.benches.zerobench import (
    CACHE_VARIABLE,
    DEFAULT_REVISION,
    SPLIT_SIZES,
    ZeroBenchAdapter,
    datasets_cache,
    grader,
    task_files,
    task_prompt,
)
from roller.docker_runtime import DockerRuntime
from roller.harnesses.claude_code import (
    GRADING_TRANSCRIPT_CONTAINER_PATH,
    ClaudeCodeConfig,
    ClaudeCodeHarness,
    ClaudeCodeSession,
)
from roller.harnesses.claude_code_sandbox import CLAUDE_HOME, SandboxAgent
from roller.models import RunRequest, Task

PNG = bytes.fromhex("89504e470d0a1a0a") + b"png-body"
JPEG = b"\xff\xd8\xff" + b"jpeg-body"
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"webp-body"
# The prompt the EvalKit zerobench-claude-code protocol sends, verbatim.
SINGLE_PROMPT = (
    "The image for this question is at /task/image.png. You are working inside a "
    "Linux container without internet access. You can run shell commands and Python "
    "to inspect and analyze the image files, for example to crop, zoom, enhance or "
    "measure them, and you can view any image file, including ones you create. These "
    "Python libraries are installed: numpy, scipy, sympy, pillow, imageio, "
    "opencv-python-headless, scikit-image, matplotlib, pandas, networkx, pytesseract. "
    "You may write scratch files under /task/. Give your final answer in your last "
    "message.\n\nHow many cells?\n\n\nLet's think step by step and give the final "
    "answer in curly braces, like this: {final answer}"
)
MULTI_LOCATION = (
    "The 2 images for this question are, in order, at /task/image_1.jpg, "
    "/task/image_2.webp."
)


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.casts: list[tuple[str, Any]] = []

    def cast_column(self, name: str, feature: Any) -> FakeDataset:
        self.casts.append((name, feature))
        return self

    def __iter__(self):
        return iter(self.rows)


def rows(split: str) -> list[dict[str, Any]]:
    """The split's rows as ``load_dataset`` returns them once decoding is off."""

    images = [
        [{"bytes": PNG, "path": "a.png"}],
        [{"bytes": JPEG, "path": "b.jpg"}, {"bytes": WEBP, "path": "c.webp"}],
    ]
    return [
        {
            "question_id": f"{split}/{index}",
            "question_text": "How many cells?" if index == 1 else f"Question {index}?",
            "question_answer": "12" if index == 1 else str(index),
            "question_images_decoded": images[min(index - 1, 1)],
        }
        for index in range(1, SPLIT_SIZES[split] + 1)
    ]


@pytest.fixture
def datasets(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A stand-in ``datasets`` module recording what the adapter asked for."""

    calls: dict[str, Any] = {}

    def load_dataset(dataset, *, split, revision, cache_dir):
        calls.update(
            dataset=dataset, split=split, revision=revision, cache_dir=cache_dir
        )
        calls["dataset_object"] = FakeDataset(rows(split))
        return calls["dataset_object"]

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    module.Image = lambda decode: ("Image", decode)  # type: ignore[attr-defined]
    module.Sequence = lambda feature: ("Sequence", feature)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def test_both_variants_export_their_questions(
    tmp_path: Path, datasets: dict[str, Any]
) -> None:
    main = get_bench("zerobench").load(tmp_path)
    assert len(main) == 100 == len({row["id"] for row in main})
    assert datasets["dataset"] == "jonathan-roberts1/zerobench"
    assert datasets["split"] == "zerobench" and datasets["revision"] == DEFAULT_REVISION
    # Nothing is passed, so datasets reads the cache HF_HOME points at. A
    # directory under --bench-root is a new, empty cache: an offline host that
    # holds only the prepared dataset cache cannot rebuild one from the Hub, so
    # pointing cache_dir there fails the load outright.
    assert datasets["cache_dir"] is None
    assert main[0]["bench"]["metadata"]["dataset"]["cache"] == datasets_cache()
    assert datasets["dataset_object"].casts == [
        ("question_images_decoded", ("Sequence", ("Image", False)))
    ]
    subquestions = ZeroBenchAdapter().load(tmp_path, variant="zerobench_subquestions")
    assert len(subquestions) == 334 == len({row["id"] for row in subquestions})
    assert datasets["split"] == "zerobench_subquestions"
    assert subquestions[0]["bench"]["metadata"]["dataset"]["split"] == (
        "zerobench_subquestions"
    )
    with pytest.raises(BenchFormatError, match="variant must be one of"):
        ZeroBenchAdapter().load(tmp_path, variant="test")
    with pytest.raises(BenchFormatError, match="no categories"):
        ZeroBenchAdapter().load(tmp_path, categories=["geometry"])


def test_an_explicit_cache_is_the_only_one_passed_through(
    tmp_path: Path, datasets: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_VARIABLE, str(tmp_path / "prepared"))
    row = ZeroBenchAdapter().load(tmp_path, task_ids=["zerobench/1"])[0]
    assert datasets["cache_dir"] == str(tmp_path / "prepared")
    assert row["bench"]["metadata"]["dataset"]["cache"] == str(tmp_path / "prepared")
    monkeypatch.setenv(CACHE_VARIABLE, "   ")
    ZeroBenchAdapter().load(tmp_path, task_ids=["zerobench/1"])
    assert datasets["cache_dir"] is None


def test_an_unexpected_question_count_is_refused(
    tmp_path: Path, datasets: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys.modules["datasets"],
        "load_dataset",
        lambda dataset, **kwargs: FakeDataset(rows("zerobench")[:99]),
    )
    with pytest.raises(BenchFormatError, match="100 unique ZeroBench zerobench"):
        ZeroBenchAdapter().load(tmp_path)


def test_images_become_workspace_files_the_prompt_locates(
    tmp_path: Path, datasets: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZEROBENCH_REVISION", "b" * 40)
    rows_out = ZeroBenchAdapter().load(
        tmp_path, task_ids=["zerobench/1", "zerobench/2"]
    )
    single, multi = rows_out
    assert [row["id"] for row in rows_out] == ["zerobench/1", "zerobench/2"]
    assert single["harness"] == "claude-code"
    assert single["runtime"] == {"workspace_path": "/task"}
    assert single["timeout_seconds"] == 10800
    assert single["files"]["input_mode"] == "copy"
    # A null output directory is Stirrup's alone; every other Harness, this
    # bench's claude-code included, builds its workspace around one.
    assert single["files"]["output_dir"] == "outputs"
    assert [item["target"] for item in single["files"]["inputs"]] == ["image.png"]
    assert [item["target"] for item in multi["files"]["inputs"]] == [
        "image_1.jpg",
        "image_2.webp",
    ]
    stored = tmp_path.resolve() / ("b" * 40) / "zerobench" / "images"
    assert (stored / "zerobench_1_1.png").read_bytes() == PNG
    assert (stored / "zerobench_2_1.jpg").read_bytes() == JPEG
    assert (stored / "zerobench_2_2.webp").read_bytes() == WEBP
    assert single["file_root"] == str(tmp_path.resolve())
    assert [item["source"] for item in multi["files"]["inputs"]] == [
        f"{'b' * 40}/zerobench/images/zerobench_2_1.jpg",
        f"{'b' * 40}/zerobench/images/zerobench_2_2.webp",
    ]
    assert MULTI_LOCATION in multi["prompt"]
    assert multi["prompt"].startswith(MULTI_LOCATION)
    assert task_files([Path("x.png")]) == ["image.png"]
    assert ZeroBenchAdapter().load(tmp_path, task_ids=["zerobench/1"]) == [single]
    with pytest.raises(BenchFormatError, match="unknown task id"):
        ZeroBenchAdapter().load(tmp_path, task_ids=["nope"])


def test_prompt_matches_the_evalkit_protocol(
    tmp_path: Path, datasets: dict[str, Any]
) -> None:
    single = ZeroBenchAdapter().load(tmp_path, task_ids=["zerobench/1"])[0]
    assert single["prompt"] == SINGLE_PROMPT
    assert task_prompt("How many cells?", ["image.png"]) == SINGLE_PROMPT
    # The reference travels with the grader, never with the task.
    grade = GradeSpec.from_dict(single["bench"]["grade"])
    assert grade.kwargs == {"reference": "12"} and grade.needs_transcript is True
    # An ending the model brought about is the sample's result, so it is scored
    # rather than left unjudged: a last turn that only called a tool, or turns
    # or time used up, all count as wrong under the official protocol.
    assert grade.grade_on_agent_failure is True
    assert "12" not in single["prompt"]
    assert missing_grader_env([single], {}) == []


def message(role: str, content: Any, wrapped: bool = True) -> dict[str, Any]:
    body = {"role": role, "content": content}
    return {"type": "message", "message": body} if wrapped else body


@pytest.mark.parametrize(
    ("text", "reference", "answer", "score"),
    [
        ("so it is {7} then {42}", "42", "42", 1.0),
        ("{Blue Whale}", "blue whale", "blue whale", 1.0),
        ("{ 42 }", "42", " 42 ", 0.0),
        ("{42}", " 42 ", "42", 1.0),
        ("{421}", "42", "421", 0.0),
        ("the answer is 42", "42", "", 0.0),
    ],
)
def test_the_official_curly_brace_match(
    text: str, reference: str, answer: str, score: float
) -> None:
    result = grader.grade([message("assistant", text)], "/task", reference)
    assert (result["answer"], result["overall_score"]) == (answer, score)


def test_the_last_visible_assistant_text_is_the_answer() -> None:
    transcript = [
        message("user", [{"type": "text", "text": "{42}"}]),
        message("assistant", [{"type": "text", "text": "draft {1}"}]),
        message(
            "assistant",
            [
                {"type": "thinking", "thinking": "maybe {9}"},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
            ],
        ),
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "final {42}"},
        },
        message("assistant", [{"type": "tool_use", "id": "t2", "name": "Bash"}]),
    ]
    assert grader.final_text(transcript) == "final {42}"
    assert grader.grade(transcript, "/task", "42")["overall_score"] == 1.0
    assert grader.final_text([message("assistant", "plain", wrapped=False)]) == "plain"


@pytest.mark.parametrize("transcript", [[], [message("user", "{42}")], None])
def test_no_final_message_scores_zero_and_says_so(transcript: Any) -> None:
    result = grader.grade(transcript, "/task", "42")
    assert result == {
        "overall_score": 0.0,
        "outcome": "no_final_message",
        "answer": None,
        "reference": "42",
    }


def sample(task: str, score: float | None) -> dict[str, Any]:
    return {
        "task_id": task,
        "model": "m",
        "harness": "claude-code",
        "status": "ok",
        "bench_scores": {"overall_score": score} if score is not None else None,
        "bench_error": None if score is not None else "container unavailable",
        "bench_metadata": {"upstream_task_id": task},
    }


def test_avg_at_k_and_pass_at_k() -> None:
    adapter = ZeroBenchAdapter()
    rows_in = [sample("a", 1.0), sample("a", 0.0), sample("b", 0.0), sample("b", 0.0)]
    report = build_report(rows_in, adapter.aggregate)
    assert (report["k"], report["avg_at_k"], report["pass_at_k"]) == (2, 25.0, 50.0)
    assert report["unit"] == "%" and report["average"] == 0.25
    assert report["metrics"] == {"avg@2": 0.25, "pass@2": 0.5}
    assert "avg@2    " in format_report("zerobench", report)
    uneven = adapter.aggregate([*rows_in, sample("b", 1.0)])
    assert (uneven["k"], uneven["avg_at_k"], uneven["pass_at_k"]) == (None, None, None)
    assert "metrics" not in uneven
    unscored = adapter.aggregate([*rows_in[:3], sample("b", None)])
    assert (unscored["k"], unscored["avg_at_k"], unscored["pass_at_k"]) == (
        2,
        None,
        None,
    )
    assert "metrics" not in unscored


# The claude-code harness has no fixed transcript path a grader could read:
# the CLI names its session files after the project directory and the session
# id, and subagents write their own. ``prepare_transcript_for_grading`` stages
# the ACTF conversion's visible assistant turns instead, on both backends.
TEXTS = ("Let me crop it {1}", "The answer is {42}")


def claude_session_files(root: Path, texts: Sequence[str]) -> None:
    """A native Claude Code session directory with ``texts`` as visible turns."""

    project = root / "-task"
    project.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    for index, text in enumerate(texts, start=1):
        events.append(
            {
                "uuid": f"a{index}",
                "type": "assistant",
                "timestamp": f"2026-01-01T00:00:{index:02d}Z",
                "message": {
                    "id": f"msg_{index}",
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [
                        {"type": "thinking", "thinking": "reasoning holds {7}"},
                        {"type": "text", "text": text},
                        {
                            "type": "tool_use",
                            "id": f"t{index}",
                            "name": "Bash",
                            "input": {"command": "python crop.py"},
                        },
                    ],
                },
            }
        )
        events.append(
            {
                "uuid": f"r{index}",
                "type": "user",
                "timestamp": f"2026-01-01T00:00:{index:02d}Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"t{index}",
                            "content": "done",
                        }
                    ],
                },
            }
        )
    # A last turn that only calls a tool, as a run out of turns ends.
    events.append(
        {
            "uuid": "a9",
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:50Z",
            "message": {
                "id": "msg_9",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "tool_use", "id": "t9", "name": "Bash"}],
            },
        }
    )
    (project / "session.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    subagents = project / "session" / "subagents"
    subagents.mkdir(parents=True, exist_ok=True)
    # A subagent answers last in wall-clock order and must not be graded.
    (subagents / "agent.jsonl").write_text(
        json.dumps(
            {
                "uuid": "s1",
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:40Z",
                "isSidechain": True,
                "message": {
                    "id": "msg_s",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "subagent guesses {99}"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def claude_session(
    tmp_path: Path, texts: Sequence[str] = TEXTS
) -> tuple[ClaudeCodeSession, list[tuple[str, str]]]:
    """A started claude-code session on the Docker backend, plus its uploads."""

    runtime = DockerRuntime()
    runtime.image_id = MagicMock(return_value=None)  # type: ignore[method-assign]
    uploads: list[tuple[str, str]] = []

    def docker_cp(args: list[str], timeout: float | None = None) -> Any:
        if args[1].startswith("container:"):
            claude_session_files(Path(args[2]), texts)
        else:
            target = args[2].removeprefix("container:")
            uploads.append((target, Path(args[1]).read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(args, 0, "", "")

    runtime.run = MagicMock(side_effect=docker_cp)  # type: ignore[method-assign]
    harness = ClaudeCodeHarness(ClaudeCodeConfig(wire_trace_mode="off"), 0, runtime)
    request = RunRequest(
        index=0,
        task=Task(id="zerobench/1", prompts=(SINGLE_PROMPT,), source_index=0),
        model="claude-opus-5",
        timeout_seconds=30,
    )
    session = ClaudeCodeSession(harness, request, tmp_path / "sample", 1)
    session.container = "container"
    session.container_started = True
    session.messages.append(SINGLE_PROMPT)
    return session, uploads


class FakeSandbox:
    """A remote sandbox's upload/download channel, without a cluster."""

    version = "v1"

    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = texts
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, bool]] = []

    def download(
        self, source: str, target: Path, *, contents: bool = False, timeout: float
    ) -> None:
        self.downloads.append((source, contents))
        claude_session_files(Path(target), self.texts)

    def upload(
        self, source: Path, target: str, *, contents: bool = False, timeout: float
    ) -> None:
        self.uploads.append((target, Path(source).read_text(encoding="utf-8")))


def test_the_docker_backend_stages_the_final_assistant_text(tmp_path: Path) -> None:
    session, uploads = claude_session(tmp_path)
    assert (
        transcript_container_path(session, GradeSpec(code=""))
        == GRADING_TRANSCRIPT_CONTAINER_PATH
    )
    assert [target for target, _ in uploads] == [GRADING_TRANSCRIPT_CONTAINER_PATH]
    transcript = json.loads(uploads[0][1])
    assert grader.final_text(transcript) == "The answer is {42}"
    assert grader.grade(transcript, "/task", "42")["overall_score"] == 1.0
    # The graded text is the one Roller records as the run's final reply.
    assert session._latest_reply() == "The answer is {42}"


def test_the_sandbox_backend_stages_the_same_transcript(tmp_path: Path) -> None:
    session, _ = claude_session(tmp_path)
    box = FakeSandbox(TEXTS)
    session.agent = SandboxAgent(
        session.harness, session.request, session.output_dir, "box", MagicMock()
    )
    session.agent.box = box  # type: ignore[assignment]
    # The grader itself runs through this sandbox; there is no docker CLI.
    assert session.sandbox is box
    assert (
        transcript_container_path(session, GradeSpec(code=""))
        == GRADING_TRANSCRIPT_CONTAINER_PATH
    )
    assert box.downloads == [(f"{CLAUDE_HOME}/projects", True)]
    assert [target for target, _ in box.uploads] == [GRADING_TRANSCRIPT_CONTAINER_PATH]
    assert (
        grader.grade(json.loads(box.uploads[0][1]), "/task", "42")["overall_score"]
        == 1.0
    )
    session.harness.runtime.run.assert_not_called()  # type: ignore[attr-defined]


def test_a_turn_limit_grades_the_text_that_still_stands(tmp_path: Path) -> None:
    session, uploads = claude_session(tmp_path, texts=TEXTS[:1])
    transcript_container_path(session, GradeSpec(code=""))
    result = grader.grade(json.loads(uploads[0][1]), "/task", "42")
    assert (result["answer"], result["overall_score"]) == ("1", 0.0)


def test_a_run_without_a_visible_message_scores_zero(tmp_path: Path) -> None:
    session, uploads = claude_session(tmp_path, texts=())
    assert (
        transcript_container_path(session, GradeSpec(code=""))
        == GRADING_TRANSCRIPT_CONTAINER_PATH
    )
    assert json.loads(uploads[0][1]) == []
    result = grader.grade(json.loads(uploads[0][1]), "/task", "42")
    assert (result["outcome"], result["overall_score"]) == ("no_final_message", 0.0)
