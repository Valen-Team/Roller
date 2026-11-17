"""QwenClawBench sample aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from roller.benches.report import mean_aggregate


def _task_average(task_scores: dict[str, list[float]]) -> float | None:
    """Mean over upstream tasks of each task's mean over its runs."""

    means = [sum(values) / len(values) for values in task_scores.values()]
    return round(sum(means) / len(means), 6) if means else None


def aggregate_samples(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(samples)
    report = mean_aggregate(rows)
    grouped: dict[str, list[float]] = {}
    by_category: dict[str, dict[str, list[float]]] = {}
    for row, task in zip(rows, report["tasks"]):
        metadata = row.get("bench_metadata")
        upstream_id = (
            metadata.get("upstream_task_id")
            if isinstance(metadata, dict)
            else row.get("task_id")
        )
        if not isinstance(upstream_id, str):
            continue
        # An unscored run counts as zero, in a category as in the headline.
        value = task["score"] or 0.0
        grouped.setdefault(upstream_id, []).append(value)
        category = task["category"]
        if isinstance(category, str) and category:
            by_category.setdefault(category, {}).setdefault(upstream_id, []).append(
                value
            )
    task_means = {
        task_id: sum(values) / len(values) for task_id, values in grouped.items()
    }
    perfect = sum(value >= 1.0 - 1e-9 for value in task_means.values())
    report["average"] = _task_average(grouped)
    for name, task_scores in by_category.items():
        report["categories"][name]["average"] = _task_average(task_scores)
    report["upstream_task_count"] = len(task_means)
    report["strict_accuracy"] = (
        round(perfect / len(task_means), 6) if task_means else None
    )
    report["perfect_task_count"] = perfect
    report["runs_per_task"] = {
        task_id: len(values) for task_id, values in sorted(grouped.items())
    }
    return report
