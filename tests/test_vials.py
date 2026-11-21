"""VIALS: task prompts, the upstream grader, grading in the collector, pass^k."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from roller.benches import get_bench, missing_grader_env
from roller.benches.base import GradeSpec
from roller.benches.grading import ContainerGrader
from roller.benches.report import build_report, format_report
from roller.benches.vials import VIALSAdapter, grader, task_files, task_prompt

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
    "AAAABJRU5ErkJggg=="
)


def questions(root: Path) -> list[dict]:
    one = root / "rev" / "images" / "q1_1.png"
    two = [root / "rev" / "images" / f"q2_{i}.jpg" for i in (1, 2)]
    for path in (one, *two):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG)
    return [
        {
            "id": "q1",
            "question": "How many cells?",
            "answer": "12",
            "domain": "Microscopy",
            "images": [one],
            "answer_lower_bound": 11.0,
            "answer_upper_bound": 13.0,
        },
        {
            "id": "q2",
            "question": "Which lane?",
            "answer": "lane 3",
            "domain": "Gels",
            "images": two,
            "answer_lower_bound": None,
            "answer_upper_bound": None,
        },
    ]


def test_task_rows_follow_the_upstream_agent_prompt(tmp_path: Path) -> None:
    with (
        patch("roller.benches.vials.load_questions", return_value=questions(tmp_path)),
        patch.dict(os.environ, {"VIALS_REVISION": "a" * 40}),
    ):
        rows = get_bench("vials").load(tmp_path)
        gels = VIALSAdapter().load(tmp_path, categories=["Gels"])
    single, multi = rows
    assert [row["id"] for row in gels] == ["q2"]
    assert single["harness"] == "stirrup"
    assert single["runtime"] == {"workspace_path": "/task"}
    assert single["files"]["input_mode"] == "copy"
    assert [item["target"] for item in single["files"]["inputs"]] == ["image.png"]
    assert "The task image is available at /task/image.png." in single["prompt"]
    assert single["prompt"].endswith("Question\n\nHow many cells?\n")
    assert multi["prompt"] == task_prompt("Which lane?", ["image_1.jpg", "image_2.jpg"])
    assert (
        "This task has 2 image panels in canonical order at /task/image_1.jpg, "
        "/task/image_2.jpg. Treat them as parts of a single figure that must be "
        "answered together.\n\nWhich lane?" in multi["prompt"]
    )
    grade = GradeSpec.from_dict(single["bench"]["grade"])
    assert grade.collector_files == ("answer.txt",)
    assert grade.kwargs["reference"] == "12"
    assert "12" not in single["prompt"]
    assert single["bench"]["metadata"]["dataset"]["revision"] == "a" * 40
    assert task_files([Path("x.webp")]) == ["image.webp"]
    assert missing_grader_env([single], {}) == [
        "VIALS_JUDGE_BASE_URL or JUDGE_BASE_URL (1 task)",
        "VIALS_JUDGE_API_KEY or JUDGE_API_KEY (1 task)",
    ]


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("work\nANSWER: 42", "42"),
        ("ANSWER: 1\nmore\n**ANSWER:** $12$", "12"),
        ("- ANSWER: \\[ 3 \\]", "3"),
        ("The answer: 5\nANSWER: 6", "6"),
        ("note answer: blue", "blue"),
        ("ANSWER:", None),
    ],
)
def test_answer_line_extraction(text: str, answer: str | None) -> None:
    assert grader.extract_answer(text) == answer


class Judge(BaseHTTPRequestHandler):
    replies: list[tuple[int, dict]] = []
    requests: list[dict] = []

    def log_message(self, *args):
        return None

    def do_POST(self):
        type(self).requests.append(
            json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        )
        status, body = type(self).replies.pop(0)
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def verdict(reasoning: str, equivalent: bool) -> tuple[int, dict]:
    content = json.dumps({"reasoning": reasoning, "equivalent": equivalent})
    return 200, {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
    }


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Judge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    Judge.replies, Judge.requests = [], []
    monkeypatch.setenv("JUDGE_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("VIALS_JUDGE_API_KEY", "key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(grader.time, "sleep", lambda seconds: None)
    yield Judge
    server.shutdown()
    server.server_close()


def workspace(tmp_path: Path, answer: str | None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if answer is not None:
        (tmp_path / "answer.txt").write_text(answer)
    (tmp_path / "images").mkdir()
    (tmp_path / "images/image.png").write_bytes(PNG)
    return tmp_path


def test_missing_answer_scores_zero_without_a_judge(tmp_path: Path, judge) -> None:
    for answer in (None, "I could not tell."):
        result = grader.grade(str(workspace(tmp_path / str(answer), answer)), "q", "12")
        assert result["overall_score"] == 0.0 and result["outcome"] == "no_answer"
    assert judge.requests == []


def test_judge_request_bounds_and_multimodal_regrade(tmp_path: Path, judge) -> None:
    judge.replies = [
        (
            200,
            {
                "choices": [
                    {"message": {"content": "not json"}, "finish_reason": "stop"}
                ]
            },
        ),
        verdict("UNVERIFIABLE: labels differ", False),
        verdict("Lane 3 is the red lane.", True),
    ]
    result = grader.grade(
        str(workspace(tmp_path, "ANSWER: red lane")),
        "Which lane?",
        "lane 3",
        11.0,
        None,
        ["images/image.png"],
    )
    assert result["overall_score"] == 1.0
    assert result["verdict"]["multimodal"] is True
    text, _, image = judge.requests
    assert list(text) == [
        "model",
        "messages",
        "response_format",
        "max_completion_tokens",
        "reasoning_effort",
    ]
    assert text["model"] == "gpt-6-luna" and text["reasoning_effort"] == "medium"
    assert text["max_completion_tokens"] == 4096 and "temperature" not in text
    prompt = text["messages"][1]["content"]
    assert (
        "lower=11.0, upper=n/a" in prompt
        and "Candidate final answer:\nred lane" in prompt
    )
    assert "You do NOT see the image." in text["messages"][0]["content"]
    parts = image["messages"][1]["content"]
    assert parts[0]["image_url"] == {
        "url": "data:image/png;base64," + base64.b64encode(PNG).decode(),
        "detail": "auto",
    }
    assert "Using the image only" in parts[1]["text"]


def test_a_refused_judge_leaves_the_sample_unscored(tmp_path: Path, judge) -> None:
    judge.replies = [(400, {"error": "bad request"})]
    result = grader.grade(str(workspace(tmp_path, "ANSWER: 12")), "q", "12")
    assert result["score_eligible"] is False and "judge failed" in result["error"]
    assert len(judge.requests) == 1


class Execution:
    """Docker exec of the answer-file read, as the container would answer it."""

    def __init__(self, answer: bytes | None):
        self.answer = answer

    def exec(self, container, command, *, workdir=None, options=(), timeout=None):
        if self.answer is None:
            return subprocess.CompletedProcess(command, 3, "", "")
        return subprocess.CompletedProcess(
            command, 0, base64.encodebytes(self.answer).decode(), ""
        )


def test_collector_side_grading_reads_the_answer_and_stages_files(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(PNG)
    code = f"""
import os
from pathlib import Path

def grade(workspace_path, expected):
    root = Path(workspace_path)
    answer = root / "answer.txt"
    return {{
        "overall_score": float(answer.is_file() and answer.read_text() == expected),
        "image": (root / "images/image.png").read_bytes() == {PNG!r},
        "key": os.environ.get("JUDGE_API_KEY"),
    }}
"""
    spec = GradeSpec(
        code=code,
        kwargs={"expected": "ANSWER: 12\n"},
        needs_transcript=False,
        stage=((str(image), "images/image.png"),),
        collector_files=("answer.txt",),
    )
    assert GradeSpec.from_dict(spec.as_dict()) == spec
    with patch.dict(os.environ, {"JUDGE_API_KEY": "here"}):
        scored = ContainerGrader(spec).grade(
            Execution(b"ANSWER: 12\n"),
            "c",
            task_id="t",
            workspace_path="/task",
            transcript_container_path=None,
        )
        missing = ContainerGrader(spec).grade(
            Execution(None),
            "c",
            task_id="t",
            workspace_path="/task",
            transcript_container_path=None,
        )
    assert scored.error is None
    assert scored.scores == {"overall_score": 1.0, "image": True, "key": "here"}
    assert missing.scores["overall_score"] == 0.0


def sample(task: str, score: float | None, model: str = "m") -> dict:
    return {
        "task_id": task,
        "model": model,
        "harness": "stirrup",
        "status": "ok",
        "bench_scores": {"overall_score": score} if score is not None else None,
        "bench_error": None if score is not None else "judge failed",
        "bench_metadata": {"upstream_task_id": task},
    }


def test_pass_k_avg_k_and_pass_at_k() -> None:
    adapter = VIALSAdapter()
    rows = [sample("a", 1.0)] * 3 + [
        sample("b", 1.0),
        sample("b", 0.0),
        sample("b", 0.0),
    ]
    report = build_report(rows, adapter.aggregate)
    assert report["metrics"] == {"pass^3": 0.5, "avg@3": 0.666667, "pass@3": 1.0}
    assert "pass^3   " in format_report("vials", report)
    unscored = adapter.aggregate([*rows[:5], sample("b", None)])
    assert "metrics" not in unscored and unscored["average"] == 0.666667
