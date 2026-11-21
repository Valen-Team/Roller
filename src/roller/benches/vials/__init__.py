"""VIALS with tools: scientific image questions answered by an agent in a container.

The task follows the upstream tool-assisted agent adapter
(https://github.com/Handshake-AI-Research/VIALS, adapters/vials): its prompt
is the task message, the images are files under ``/task``, and the answer is
the last ``ANSWER:`` line of ``/task/answer.txt``. The dataset is read from
Hugging Face at a pinned revision and cached under ``--bench-root``.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
import urllib.request
from collections.abc import Iterable, Sequence
from hashlib import sha256
from importlib import resources
from pathlib import Path
from typing import Any

from roller.benches.base import (
    BenchAdapter,
    BenchBinding,
    BenchFormatError,
    GradeSpec,
    validate_timeout_multiplier,
)
from roller.benches.report import mean_aggregate

DATASET = "handshake-ai-research/VIALS"
SPLIT = "test"
DEFAULT_REVISION = "76be0fc34a7b2fbd27de0bac2aa58e07973d1095"
REVISION_VARIABLE = "VIALS_REVISION"
WORKDIR = "/task"
OUTPUT_DIR_NAME = ".roller_outputs"
# The longest reference episodes ran about two and a half hours.
TIMEOUT_SECONDS = 14400
GRADE_TIMEOUT_SECONDS = 1800
JUDGE_ENV_NAMES = tuple(
    f"{prefix}{name}"
    for name in ("BASE_URL", "API_KEY", "MODEL", "REASONING_EFFORT", "MAX_TOKENS")
    for prefix in ("VIALS_JUDGE_", "JUDGE_")
)
JUDGE_REQUIRED_ENV = (
    ("VIALS_JUDGE_BASE_URL", "JUDGE_BASE_URL"),
    ("VIALS_JUDGE_API_KEY", "JUDGE_API_KEY"),
)
_GRADER_SOURCE = (
    resources.files("roller.benches.vials")
    .joinpath("grader.py")
    .read_text(encoding="utf-8")
)
_EXTENSIONS = ((b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"))
PROMPT = """You are an expert scientist working inside a Linux container.

{images} You have access to a shell and may use any software available in the container to inspect and \
analyze the provided files. The following Python libraries are already installed:

- numpy, scipy, sympy
- pillow, imageio, opencv-python-headless, scikit-image
- matplotlib, pandas, networkx
- pytesseract (backed by the tesseract-ocr CLI)

Do not use the internet, external databases, or information outside the files provided for this task. \
You may write scratch files under /task/ while you work. You may also inspect the original image or \
cropped regions as additional vision inputs.

Answer the question below. When finished, write your response to /task/answer.txt, with the final line \
in the following format:

ANSWER: <your final answer>

Question

{question}
"""


def _extension(data: bytes) -> str:
    for magic, extension in _EXTENSIONS:
        if data.startswith(magic):
            return extension
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise BenchFormatError("VIALS image is not PNG, JPEG or WebP")


def _open(url: str):
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=600
    )


def _hub(path: str) -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + path


def _resolve_revision(revision: str) -> str:
    """A branch or tag as its commit, so a run records the data it used."""

    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    with _open(_hub(f"/api/datasets/{DATASET}/revision/{revision}")) as response:
        return json.load(response)["sha"]


def _download(root: Path, revision: str) -> list[Path]:
    """The test split's parquet files at ``revision``, downloaded once."""

    directory = root / revision
    files = sorted(directory.glob(f"{SPLIT}-*.parquet"))
    if files:
        return files
    with _open(_hub(f"/api/datasets/{DATASET}/tree/{revision}/data/tasks")) as response:
        names = [
            item["path"]
            for item in json.load(response)
            if item["path"].rsplit("/", 1)[-1].startswith(f"{SPLIT}-")
        ]
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = directory / name.rsplit("/", 1)[-1]
        partial = target.with_suffix(".partial")
        with (
            _open(_hub(f"/datasets/{DATASET}/resolve/{revision}/{name}")) as response,
            partial.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        partial.replace(target)
    return sorted(directory.glob(f"{SPLIT}-*.parquet"))


def load_questions(root: Path, revision: str) -> list[dict[str, Any]]:
    """The test split as questions, with each image written once under ``root``."""

    try:
        import pyarrow.parquet as parquet
    except ImportError:
        raise BenchFormatError(
            "VIALS reads the dataset with pyarrow: pip install 'roller-agent[vials]'"
        ) from None
    images_dir = root / revision / "images"
    questions = []
    for path in _download(root, revision):
        for row in parquet.read_table(path).to_pylist():
            images = []
            for index, image in enumerate(row["images"], start=1):
                data = image["bytes"]
                file = images_dir / f"{row['id']}_{index}{_extension(data)}"
                if not file.is_file():
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(data)
                images.append(file)
            metadata = row["metadata"]
            questions.append(
                {
                    "id": row["id"],
                    "question": html.unescape(row["question"]),
                    "answer": html.unescape(row["answer"]),
                    "domain": row["domain"],
                    "images": images,
                    "answer_lower_bound": _bound(metadata["answer_lower_bound"]),
                    "answer_upper_bound": _bound(metadata["answer_upper_bound"]),
                }
            )
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise BenchFormatError("expected VIALS tasks with unique ids")
    return questions


def _bound(value: Any) -> float | None:
    return None if value is None or math.isnan(float(value)) else float(value)


def task_files(images: Sequence[Path]) -> list[str]:
    """``image.<ext>``, or ``image_<n>.<ext>`` in order when there are several."""

    return [
        f"image{'' if len(images) == 1 else f'_{index}'}{path.suffix}"
        for index, path in enumerate(images, start=1)
    ]


def task_prompt(question: str, names: Sequence[str]) -> str:
    paths = [f"{WORKDIR}/{name}" for name in names]
    if len(paths) == 1:
        located = f"The task image is available at {paths[0]}."
    else:
        located = f"The task images are available under {WORKDIR}/."
        question = (
            f"This task has {len(paths)} image panels in canonical order at "
            f"{', '.join(paths)}. Treat them as parts of a single figure that must "
            f"be answered together.\n\n{question}"
        )
    return PROMPT.format(images=located, question=question)


class VIALSAdapter(BenchAdapter):
    name = "vials"
    description = "VIALS scientific image questions with code execution (Stirrup)"

    def load(
        self,
        root: Path,
        *,
        categories: Sequence[str] | None = None,
        task_ids: Sequence[str] | None = None,
        variant: str | None = None,
        timeout_multiplier: float = 1.0,
    ) -> list[dict[str, Any]]:
        if variant is not None:
            raise BenchFormatError(f"{self.name}: this bench has no variants")
        multiplier = validate_timeout_multiplier(timeout_multiplier)
        root = root.expanduser().resolve()
        revision = _resolve_revision(
            os.environ.get(REVISION_VARIABLE, "").strip() or DEFAULT_REVISION
        )
        rows = []
        for question in load_questions(root, revision):
            if categories and question["domain"] not in categories:
                continue
            if task_ids and question["id"] not in task_ids:
                continue
            names = task_files(question["images"])
            staged = tuple(
                (str(path), f"images/{name}")
                for path, name in zip(question["images"], names)
            )
            binding = BenchBinding(
                name=self.name,
                category=question["domain"],
                metadata={
                    "upstream_task_id": question["id"],
                    "dataset": {"id": DATASET, "split": SPLIT, "revision": revision},
                },
                grade=GradeSpec(
                    code=_GRADER_SOURCE,
                    kwargs={
                        "question": question["question"],
                        "reference": question["answer"],
                        "answer_lower_bound": question["answer_lower_bound"],
                        "answer_upper_bound": question["answer_upper_bound"],
                        "images": [target for _, target in staged],
                    },
                    optional_env_names=JUDGE_ENV_NAMES,
                    timeout_seconds=GRADE_TIMEOUT_SECONDS,
                    needs_transcript=False,
                    stage=staged,
                    collector_files=("answer.txt",),
                ),
            )
            rows.append(
                {
                    "schema_version": "roller.task/v1",
                    "id": question["id"],
                    "harness": "stirrup",
                    "prompt": task_prompt(question["question"], names),
                    "timeout_seconds": max(1, math.ceil(TIMEOUT_SECONDS * multiplier)),
                    "file_root": str(root),
                    "files": {
                        "inputs": [
                            {
                                "source": path.relative_to(root).as_posix(),
                                "target": name,
                                "sha256": sha256(path.read_bytes()).hexdigest(),
                            }
                            for path, name in zip(question["images"], names)
                        ],
                        "input_mode": "copy",
                        "output_dir": OUTPUT_DIR_NAME,
                    },
                    "runtime": {"workspace_path": WORKDIR},
                    "bench": binding.as_dict(),
                }
            )
        return rows

    def required_env(self, binding: BenchBinding) -> tuple[tuple[str, ...], ...]:
        return JUDGE_REQUIRED_ENV

    def aggregate(self, samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """The mean score, and pass^k, avg@k and pass@k once every sample is scored.

        k is the number of samples per question; a run with unscored samples
        or uneven sample counts reports only the mean.
        """

        rows = list(samples)
        report = mean_aggregate(rows)
        questions: dict[str, list[float | None]] = {}
        for row, task in zip(rows, report["tasks"]):
            metadata = row.get("bench_metadata") or {}
            key = metadata.get("upstream_task_id") or row.get("task_id")
            questions.setdefault(str(key), []).append(task["score"])
        counts = {len(scores) for scores in questions.values()}
        scores = [score for values in questions.values() for score in values]
        if len(counts) == 1 and None not in scores:
            k = counts.pop()
            report["metrics"] = {
                f"pass^{k}": round(
                    sum(all(s == 1 for s in v) for v in questions.values())
                    / len(questions),
                    6,
                ),
                f"avg@{k}": round(sum(scores) / len(scores), 6),
                f"pass@{k}": round(
                    sum(any(s == 1 for s in v) for v in questions.values())
                    / len(questions),
                    6,
                ),
            }
        return report
