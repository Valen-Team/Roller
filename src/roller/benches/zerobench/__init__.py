"""ZeroBench with tools: hard visual reasoning answered by an agent in a container.

The task follows the ``zerobench-claude-code`` protocol: the official question
and answer instruction are the task message, the question's images are files
under ``/task`` rather than message parts, and the answer is the last
``{final answer}`` of the agent's last message. The dataset is read from
Hugging Face at a pinned revision, through the ``datasets`` cache ``HF_HOME``
points at; ``--bench-root`` holds the question images exported from it.

The two splits are exported as variants: ``zerobench`` (100 main questions,
the official headline) and ``zerobench_subquestions`` (334 easier
sub-questions).
"""

from __future__ import annotations

import io
import math
import os
import re
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

DATASET = "jonathan-roberts1/zerobench"
DEFAULT_REVISION = "8fcdfd9138c802cac52b774b36946e70f738f500"
REVISION_VARIABLE = "ZEROBENCH_REVISION"
# Set to read the dataset from a cache other than the one HF_HOME points at.
CACHE_VARIABLE = "ZEROBENCH_DATASETS_CACHE"
IMAGE_COLUMN = "question_images_decoded"
# A split with a different size silently changes both metric denominators.
SPLIT_SIZES = {"zerobench": 100, "zerobench_subquestions": 334}
WORKDIR = "/task"
# Workspace-relative directory the Harness builds for collected output. The
# protocol asks for nothing from the workspace, so it is normally empty.
OUTPUT_DIR = "outputs"
# EvalKit's wall-clock bound of one episode; the harness turn limit ends almost
# all of them first.
TIMEOUT_SECONDS = 10800
SCORING = "zerobench-official-readme-strict-curly-v1"
_GRADER_SOURCE = (
    resources.files("roller.benches.zerobench")
    .joinpath("grader.py")
    .read_text(encoding="utf-8")
)
_EXTENSIONS = ((b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"))
_UNSAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")
# The official answer instruction, appended to every question text.
INSTRUCTION = (
    "\n\n\nLet's think step by step and give the final answer in curly "
    "braces, like this: {final answer}"
)
PROMPT = """{images} You are working inside a Linux container without internet access. You can run shell \
commands and Python to inspect and analyze the image files, for example to crop, zoom, enhance or measure \
them, and you can view any image file, including ones you create. These Python libraries are installed: \
numpy, scipy, sympy, pillow, imageio, opencv-python-headless, scikit-image, matplotlib, pandas, networkx, \
pytesseract. You may write scratch files under {workdir}/. Give your final answer in your last message.

{question}"""


def _to_png(data: bytes) -> bytes:
    """Re-encode an image the sandbox tooling may not read; PNG keeps every pixel."""

    try:
        from PIL import Image
    except ImportError:
        raise BenchFormatError(
            "ZeroBench converts an image that is not PNG, JPEG or WebP with Pillow: "
            "pip install Pillow"
        ) from None
    buffer = io.BytesIO()
    with Image.open(io.BytesIO(data)) as picture:
        picture.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def image_file(image: dict[str, Any]) -> tuple[bytes, str]:
    """One undecoded dataset image as ``(bytes, extension)``.

    PNG, JPEG and WebP keep their original bytes: re-encoding a benchmark image
    would change what the model sees. The format is read from the magic bytes
    rather than from the dataset's own metadata, which the column does not carry
    once decoding is turned off.
    """

    data = image.get("bytes")
    if data is None:
        data = Path(image["path"]).read_bytes()
    for magic, extension in _EXTENSIONS:
        if data.startswith(magic):
            return data, extension
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return data, ".webp"
    return _to_png(data), ".png"


def cache_override() -> str | None:
    """An explicitly requested ``datasets`` cache directory, if any.

    Nothing is passed by default, so ``datasets`` reads the cache ``HF_HOME``
    or ``HF_DATASETS_CACHE`` points at. A directory under ``--bench-root``
    would be a new, empty cache: an offline host that holds only the prepared
    dataset cache has no Hub blobs to rebuild one from, so ``load_dataset``
    then fails with ``OfflineModeIsEnabled`` instead of reading the dataset.
    ``--bench-root`` still holds the question images this bench exports.
    """

    value = os.environ.get(CACHE_VARIABLE, "").strip()
    return value or None


def datasets_cache() -> str | None:
    """The cache ``load_dataset`` actually reads, recorded with every task."""

    override = cache_override()
    if override:
        return override
    try:
        from datasets import config
    except ImportError:
        config = None  # type: ignore[assignment]
    resolved = getattr(config, "HF_DATASETS_CACHE", None)
    if resolved:
        return str(resolved)
    return os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME") or None


def _load_split(
    split: str, revision: str, cache_dir: str | None
) -> list[dict[str, Any]]:
    """The split's rows with the image column left undecoded."""

    try:
        from datasets import Image as HFImage
        from datasets import Sequence as HFSequence
        from datasets import load_dataset
    except ImportError:
        raise BenchFormatError(
            "ZeroBench reads the dataset with datasets: pip install datasets"
        ) from None
    dataset = load_dataset(DATASET, split=split, revision=revision, cache_dir=cache_dir)
    # Read original bytes rather than recompressing the benchmark images.
    dataset = dataset.cast_column(IMAGE_COLUMN, HFSequence(HFImage(decode=False)))
    return list(dataset)


def load_questions(root: Path, revision: str, split: str) -> list[dict[str, Any]]:
    """One split as questions, with each image written once under ``root``."""

    expected = SPLIT_SIZES[split]
    images_dir = root / revision / split / "images"
    questions = []
    for row in _load_split(split, revision, cache_override()):
        question_id = str(row["question_id"])
        stem = _UNSAFE_ID.sub("_", question_id)
        images = []
        for index, image in enumerate(row[IMAGE_COLUMN], start=1):
            data, extension = image_file(image)
            file = images_dir / f"{stem}_{index}{extension}"
            if not file.is_file():
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(data)
                file.chmod(0o644)
            images.append(file)
        if not images:
            raise BenchFormatError(
                f"ZeroBench question {question_id} has no image to put in /task"
            )
        questions.append(
            {
                "id": question_id,
                "question": row["question_text"],
                "answer": row["question_answer"],
                "images": images,
            }
        )
    if (
        len(questions) != expected
        or len({question["id"] for question in questions}) != expected
    ):
        raise BenchFormatError(
            f"expected the {expected} unique ZeroBench {split} questions, "
            f"got {len(questions)}"
        )
    return questions


def task_files(images: Sequence[Path]) -> list[str]:
    """``image.<ext>``, or ``image_<n>.<ext>`` in order when there are several."""

    return [
        f"image{'' if len(images) == 1 else f'_{index}'}{path.suffix}"
        for index, path in enumerate(images, start=1)
    ]


def task_prompt(question: str, names: Sequence[str]) -> str:
    """The prompt of one episode: where the images are, then the official question."""

    paths = [f"{WORKDIR}/{name}" for name in names]
    located = (
        f"The image for this question is at {paths[0]}."
        if len(paths) == 1
        else f"The {len(paths)} images for this question are, in order, at "
        f"{', '.join(paths)}."
    )
    # The instruction's own braces reach the prompt through the substitution,
    # which format() does not expand again.
    return PROMPT.format(
        images=located, workdir=WORKDIR, question=question + INSTRUCTION
    )


class ZeroBenchAdapter(BenchAdapter):
    name = "zerobench"
    description = "ZeroBench visual reasoning with code execution (Claude Code)"
    variants = tuple(SPLIT_SIZES)
    default_variant = "zerobench"

    def load(
        self,
        root: Path,
        *,
        categories: Sequence[str] | None = None,
        task_ids: Sequence[str] | None = None,
        variant: str | None = None,
        timeout_multiplier: float = 1.0,
    ) -> list[dict[str, Any]]:
        if categories:
            raise BenchFormatError(f"{self.name}: this bench has no categories")
        split = variant or self.default_variant
        if split not in self.variants:
            raise BenchFormatError(
                f"{self.name}: variant must be one of {', '.join(self.variants)}"
            )
        multiplier = validate_timeout_multiplier(timeout_multiplier)
        root = root.expanduser().resolve()
        # Offline runs cannot ask the Hub what a branch points at, so the
        # revision is used as given and recorded with every task.
        revision = os.environ.get(REVISION_VARIABLE, "").strip() or DEFAULT_REVISION
        questions = load_questions(root, revision, split)
        # The cache the rows came from, so an offline run records where its
        # dataset was read rather than where this bench wrote its images.
        cache = datasets_cache()
        requested = set(task_ids or ())
        unknown = requested - {question["id"] for question in questions}
        if unknown:
            raise BenchFormatError(
                f"{self.name}: unknown task id(s): {', '.join(sorted(unknown))}"
            )
        rows = []
        for question in questions:
            if requested and question["id"] not in requested:
                continue
            names = task_files(question["images"])
            binding = BenchBinding(
                name=self.name,
                metadata={
                    "upstream_task_id": question["id"],
                    "variant": split,
                    "scoring": SCORING,
                    "dataset": {
                        "id": DATASET,
                        "split": split,
                        "revision": revision,
                        "cache": cache,
                    },
                },
                grade=GradeSpec(
                    code=_GRADER_SOURCE,
                    # The reference never enters the container's workspace: the
                    # grader reads only the transcript.
                    kwargs={"reference": question["answer"]},
                    needs_transcript=True,
                    # ZeroBench scores whatever the agent last said, so an
                    # ending the model brought about -- its turns or its time
                    # used up, a final turn that only called a tool -- is that
                    # sample's result and counts as wrong. Grading a sandbox
                    # fault instead fails in the grader, which Roller records
                    # apart from a zero, so such a sample stays redoable.
                    grade_on_agent_failure=True,
                ),
            )
            rows.append(
                {
                    "schema_version": "roller.task/v1",
                    "id": question["id"],
                    "harness": "claude-code",
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
                        # The answer is the agent's last message, so nothing has
                        # to be collected from the workspace. The prompt still
                        # points the agent at /task for its scratch files, as
                        # the protocol words it, and this directory stays empty
                        # unless the agent finds it: a null output directory is
                        # Stirrup's alone, and every other Harness builds the
                        # workspace around one.
                        "output_dir": OUTPUT_DIR,
                    },
                    "runtime": {"workspace_path": WORKDIR},
                    "bench": binding.as_dict(),
                }
            )
        return rows

    def aggregate(self, samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """The mean score, plus the official avg@k and pass@k as percentages.

        k is the number of sample slots per question, read from the run rather
        than assumed to be the official five. Uneven slot counts or an unscored
        slot leave both official metrics ``None``: they are defined over a
        complete k-sample grid, and a partial run would understate them.
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
        k = counts.pop() if len(counts) == 1 else None
        complete = k is not None and None not in scores
        report["k"] = k
        report["unit"] = "%"
        report["avg_at_k"] = (
            round(100 * sum(score or 0.0 for score in scores) / len(scores), 6)
            if complete
            else None
        )
        report["pass_at_k"] = (
            round(
                100
                * sum(
                    any(score == 1 for score in values) for values in questions.values()
                )
                / len(questions),
                6,
            )
            if complete
            else None
        )
        if complete:
            # The same two numbers as fractions, which is the only scale the
            # report formatter prints and draws its bars on.
            report["metrics"] = {
                f"avg@{k}": round(report["avg_at_k"] / 100, 6),
                f"pass@{k}": round(report["pass_at_k"] / 100, 6),
            }
        return report
