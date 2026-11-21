"""The run-level SFT datasets and the index of each.

``sft.jsonl`` and ``sft_segments.jsonl`` hold training records and nothing
else. Beside each is an index, ``sft_index.jsonl`` and
``sft_segments_index.jsonl``: one line per record, in the same order, naming
the task, the model and the bench score of the sample the record came from.
Collection writes them into the run directory. ``roller export --format sft``
writes them again from the samples on disk, and can keep only the samples
that reach a bench score.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roller.benches import get_bench, mean_aggregate
from roller.blobs import BLOBS_DIR, load_blobs
from roller.capture import _safe_file
from roller.compaction_sft import SFT_SEGMENTS_DIR
from roller.sft import normalize_sft_record
from roller.storage import write_jsonl
from roller.trajectory_audit import SEGMENTED_SFT_REASON, sft_exclusion_reason

SFT_FILE = "sft.jsonl"
SEGMENTS_FILE = "sft_segments.jsonl"
INDEX_FILES = {
    SFT_FILE: "sft_index.jsonl",
    SEGMENTS_FILE: "sft_segments_index.jsonl",
}
# Why a minimum score left a sample out, beside the reasons of
# ``sft_exclusion_reason``.
UNGRADED = "ungraded"
BELOW_MIN_SCORE = "below_min_score"

# A sample directory and the content of its ``info.json``.
Sample = tuple[Path, Mapping[str, Any]]


def bench_score(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], float | None]:
    """Return the bench record of a sample and its score in ``roller bench report``.

    The record is empty for a sample that no bench graded. The score is
    ``None`` for such a sample and for one whose grading failed.
    """

    runtime = result.get("runtime")
    bench = runtime.get("bench") if isinstance(runtime, Mapping) else None
    if not isinstance(bench, Mapping):
        return {}, None
    # The row that ``roller bench report`` aggregates, so that the adapter of
    # the bench gives the sample the same number here and in the report.
    row = {
        "task_id": result.get("task_id"),
        "model": result.get("model"),
        # A resumed sample is "skipped" in memory and "ok" in its info.json.
        "status": result.get("resumed_from_status") or result.get("status"),
        "error": result.get("error"),
        "bench_scores": bench.get("scores"),
        "bench_score_key": bench.get("score_key"),
        "bench_category": bench.get("category"),
        "bench_metadata": bench.get("metadata"),
        "bench_error": bench.get("error"),
    }
    try:
        report = get_bench(str(bench.get("bench"))).aggregate([row])
    except KeyError:
        report = mean_aggregate([row])
    return bench, report["tasks"][0]["score"]


def index_row(
    run_dir: Path, sample_dir: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe the sample that an SFT record came from."""

    bench, score = bench_score(result)
    trajectory = result.get("trajectory") or {}
    source = (result.get("provenance") or {}).get("import") or {}
    return {
        "task_id": result.get("task_id"),
        "model": result.get("model"),
        "model_label": result.get("model_label", result.get("model")),
        "harness": result.get("harness"),
        "sample": sample_dir.relative_to(run_dir).as_posix(),
        "sample_index": result.get("sample_index", 0),
        "bench": bench.get("bench"),
        "category": bench.get("category"),
        "score": score,
        "score_key": bench.get("score_key"),
        # Imported sessions say where their tools and system prompt came from
        # and which session they continue.
        **{
            key: trajectory[key]
            for key in ("tools_source", "system_prompt_source")
            if key in trajectory
        },
        **(
            {"parent_session_id": source["parent_session_id"]}
            if "parent_session_id" in source
            else {}
        ),
    }


def _record(path: Path, blobs: Path) -> dict[str, Any]:
    try:
        record = load_blobs(json.loads(path.read_text(encoding="utf-8")), blobs)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot aggregate SFT sample {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise RuntimeError(f"invalid SFT sample shape: {path}")
    try:
        return normalize_sft_record(record)
    except ValueError as exc:
        raise RuntimeError(f"invalid SFT sample shape ({exc}): {path}") from exc


def sample_records(
    sample_dir: Path, result: Mapping[str, Any], row: dict[str, Any], blobs: Path
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield every SFT record (linear or compaction segments) of one kept sample.

    ``blobs`` is the run's ``blobs/``, which holds the images of the records of
    a run collected with ``--training-files-only``.
    """
    segments = (result.get("trajectory") or {}).get("sft_segments") or []
    if not segments:
        path = sample_dir / "trace_SFT.json"
        record = _record(path, blobs)
        if "target_start_index" in record:
            raise RuntimeError(f"legacy SFT record has a target mask: {path}")
        yield record, row
        return
    for item in segments:
        relative = item.get("path") if isinstance(item, dict) else None
        parts = Path(relative).parts if isinstance(relative, str) else ()
        if (
            len(parts) != 2
            or parts[0] != SFT_SEGMENTS_DIR
            or not parts[1].endswith(".json")
        ):
            raise RuntimeError(f"invalid SFT segment path: {relative!r}")
        try:
            path = _safe_file(sample_dir, relative)
        except ValueError as exc:
            raise RuntimeError(
                f"cannot aggregate SFT segment {relative}: {exc}"
            ) from exc
        yield (
            _record(path, blobs),
            {**row, "segment": relative, "window_index": item.get("window_index")},
        )


def _records(
    samples: Iterable[tuple[Path, Mapping[str, Any], dict[str, Any]]],
    segmented: bool,
    blobs: Path,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield the records of the linear or the segmented samples with their index rows."""

    for sample_dir, result, row in samples:
        segments = (result.get("trajectory") or {}).get("sft_segments") or []
        # Keep masked segments separate from legacy linear records.
        if bool(segments) != segmented:
            continue
        yield from sample_records(sample_dir, result, row, blobs)


def _write(
    directory: Path,
    name: str,
    rows: Iterable[tuple[dict[str, Any], dict[str, Any]]],
) -> int:
    """Write a dataset and, line for line, its index; return the record count."""

    index: list[dict[str, Any]] = []

    def records() -> Iterator[dict[str, Any]]:
        for record, row in rows:
            index.append(row)
            yield record

    index_path = directory / INDEX_FILES[name]
    # An index must never lie beside a dataset it does not describe, so the old
    # one goes before the dataset is replaced.
    index_path.unlink(missing_ok=True)
    write_jsonl(directory / name, records())
    write_jsonl(index_path, index)
    return len(index)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


@dataclass(frozen=True)
class Written:
    """What one write of the datasets kept and what it left out."""

    # Records written, by dataset file.
    records: dict[str, int]
    samples: int
    # Samples without a record, by reason.
    dropped: dict[str, int]
    # The lowest and highest bench score of the samples collection keeps.
    score_range: tuple[float, float] | None

    def lines(self, output: Path, min_score: float | None = None) -> list[str]:
        lines = [
            f"Exported {_count(count, 'record')} to {output / name} "
            f"(index: {INDEX_FILES[name]})"
            for name, count in self.records.items()
        ]
        dropped = sum(self.dropped.values())
        lines.append(
            f"Kept {self.samples - dropped} of {_count(self.samples, 'sample')}"
            + (f"; dropped {dropped}:" if dropped else "")
        )
        lines += [
            f"  {reason}: {count}" for reason, count in sorted(self.dropped.items())
        ]
        if min_score is not None:
            lines.append(
                f"--min-score {min_score:g} is compared with the score that "
                "`roller bench report` lists for each sample; "
                + (
                    f"in this run they range from {self.score_range[0]:g} "
                    f"to {self.score_range[1]:g}."
                    if self.score_range
                    else "no sample of this run has one."
                )
            )
        return lines


def select_samples(
    run_dir: Path,
    samples: Iterable[Sample],
    min_score: float | None = None,
) -> tuple[
    list[tuple[Path, Mapping[str, Any], dict[str, Any]]],
    dict[str, int],
    tuple[float, float] | None,
    int,
]:
    """Filter samples by SFT eligibility and optional minimum bench score."""
    sample_list = list(samples)
    kept: list[tuple[Path, Mapping[str, Any], dict[str, Any]]] = []
    dropped: Counter[str] = Counter()
    scores: list[float] = []
    for sample_dir, result in sample_list:
        reason = sft_exclusion_reason(result)
        if reason not in (None, SEGMENTED_SFT_REASON):
            dropped[reason] += 1
            continue
        row = index_row(run_dir, sample_dir, result)
        score = row["score"]
        if score is not None:
            scores.append(score)
        if min_score is None or (score is not None and score >= min_score):
            kept.append((sample_dir, result, row))
        else:
            dropped[UNGRADED if score is None else BELOW_MIN_SCORE] += 1
    score_range = (min(scores), max(scores)) if scores else None
    return kept, dict(dropped), score_range, len(sample_list)


def load_run_samples(run_dir: Path) -> list[Sample]:
    """Load all samples of a run directory in request order."""
    samples = [
        (path.parent, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(run_dir.glob("samples/*/*/info.json"))
    ]
    samples.sort(key=lambda sample: sample[1].get("request_index") or 0)
    return samples


def write_sft_dataset(
    directory: Path,
    run_dir: Path,
    samples: Iterable[Sample],
    min_score: float | None = None,
) -> Written:
    """Write the SFT datasets of ``samples`` and their indexes into ``directory``.

    A sample has a record when ``sft_exclusion_reason`` keeps it; a compacted
    sample has one per segment, in ``sft_segments.jsonl``, which is written when
    a sample of the run has segments. With ``min_score``, a sample also needs a
    bench score of at least that value.
    """

    sample_list = list(samples)
    kept, dropped, score_range, total = select_samples(run_dir, sample_list, min_score)

    blobs = run_dir / BLOBS_DIR
    records = {SFT_FILE: _write(directory, SFT_FILE, _records(kept, False, blobs))}
    if any(
        isinstance(result.get("trajectory"), dict)
        and result["trajectory"].get("sft_segments")
        for _, result in sample_list
    ):
        records[SEGMENTS_FILE] = _write(
            directory, SEGMENTS_FILE, _records(kept, True, blobs)
        )
    else:
        for name in (SEGMENTS_FILE, INDEX_FILES[SEGMENTS_FILE]):
            (directory / name).unlink(missing_ok=True)
    return Written(
        records=records,
        samples=total,
        dropped=dropped,
        score_range=score_range,
    )


def export_run(run_dir: Path, output: Path, min_score: float | None = None) -> Written:
    """Write the SFT datasets of a run from its samples on disk, in request order.

    Without ``min_score`` the files equal those collection wrote into the run.
    """

    return write_sft_dataset(output, run_dir, load_run_samples(run_dir), min_score)
