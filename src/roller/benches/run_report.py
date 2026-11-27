"""Build and persist benchmark reports from a finished or partial run directory."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roller.benches import get_bench
from roller.benches.report import build_report, mean_aggregate
from roller.storage import atomic_write_json
from roller.tasks import TaskFormatError

logger = logging.getLogger(__name__)


def task_rows(
    run_dir: Path, configuration: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Task rows of a run by id, from its snapshot or the task list it was given."""

    paths = [run_dir / "tasks.jsonl"]
    if isinstance(configuration.get("tasks"), str):
        paths.append(Path(configuration["tasks"]))
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            payload = json.loads(text)
            rows = payload if isinstance(payload, list) else [payload]
        except json.JSONDecodeError:
            try:
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError:
                continue
        return {
            row["id"]: row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
    return {}


def ungraded_bench_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """The bench record of a sample that failed before its grader ran.

    Such a sample has no record of its own, so its task row says whether it is
    graded, and under which category. ``None`` for a task without a grader.
    """

    block = row.get("bench")
    if not isinstance(block, dict) or block.get("grade") is None:
        return None
    return {
        "bench": block.get("name"),
        "category": block.get("category"),
        "metadata": block.get("metadata"),
    }


def generate_bench_report(
    run_dir: Path, *, output: Path | None = None
) -> tuple[str, dict[str, Any], Path | None]:
    """Read sample info files under ``run_dir`` and build the aggregated bench report."""

    resolved_run = run_dir.expanduser().resolve()
    samples_root = resolved_run / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(f"run directory has no samples/: {resolved_run}")
    try:
        run_info = json.loads((resolved_run / "info.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        run_info = {}
    configuration = run_info.get("configuration") if isinstance(run_info, dict) else {}
    configuration = configuration if isinstance(configuration, dict) else {}
    bench = configuration.get("bench")
    rows_by_task = task_rows(resolved_run, configuration)
    rows: list[dict[str, Any]] = []
    bench_names: set[str] = set()
    sample_count = unidentified = 0
    for info_path in sorted(samples_root.glob("*/*/info.json")):
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("skipping unreadable sample info: %s", info_path)
            continue
        if not isinstance(info, dict):
            continue
        sample_count += 1
        runtime = info.get("runtime")
        payload = runtime.get("bench") if isinstance(runtime, dict) else None
        if not isinstance(payload, dict):
            # Dropping a sample that never reached its grader would shrink the
            # denominator; it counts as unscored, like one whose agent crashed.
            row = rows_by_task.get(str(info.get("task_id")))
            if row is not None:
                payload = ungraded_bench_payload(row)
            elif isinstance(bench, dict):
                # ``collect --bench`` runs nothing but the tasks of its bench.
                payload = {"bench": bench.get("name")}
            else:
                payload = None
                unidentified += 1
            if payload is None:
                continue
        name = payload.get("bench")
        if isinstance(name, str) and name:
            bench_names.add(name)
        rows.append(
            {
                "task_id": info.get("task_id"),
                "model": info.get("model_label") or info.get("model"),
                "harness": info.get("harness"),
                "status": info.get("status"),
                "error": info.get("error"),
                "bench_scores": payload.get("scores"),
                "bench_score_key": payload.get("score_key"),
                "bench_category": payload.get("category"),
                "bench_metadata": payload.get("metadata"),
                "bench_error": payload.get("error"),
            }
        )
    if not rows:
        raise TaskFormatError(f"no bench samples found under {samples_root}")
    if unidentified:
        logger.warning(
            "%d samples have no grading record and the task list of the run is no "
            "longer available, so the report cannot tell whether they are bench "
            "samples; they are left out",
            unidentified,
        )
    label = ", ".join(sorted(bench_names)) or "unknown"
    aggregate = mean_aggregate
    if len(bench_names) == 1:
        try:
            aggregate = get_bench(sorted(bench_names)[0]).aggregate
        except KeyError:
            pass
    report = build_report(rows, aggregate)
    summary = run_info.get("summary") if isinstance(run_info, dict) else None
    planned = summary.get("total") if isinstance(summary, dict) else None
    if isinstance(planned, int) and planned > sample_count:
        # An interrupted run: say so, and keep the two long lists last.
        report["planned_count"] = planned
        report["missing_count"] = planned - sample_count
        for key in ("models", "tasks"):
            report[key] = report.pop(key)
    written: Path | None = None
    if output is not None:
        written = output.expanduser().resolve()
        written.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(written, report)
    return label, report, written
