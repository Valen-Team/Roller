from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

SCORE_FILE = "score.json"
REPORT_SCHEMA = "roller.bench_report/v1"


def _numeric(scores: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in scores.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def sample_score(
    scores: dict[str, Any], score_key: str = "overall_score"
) -> float | None:
    """Reduce one grader payload to a single number.

    Follows WildClawBench's rule: prefer the declared aggregate key, otherwise
    average whatever numeric fields the grader returned.
    """

    numeric = _numeric(scores)
    if not numeric:
        return None
    if score_key in numeric:
        return numeric[score_key]
    return sum(numeric.values()) / len(numeric)


def read_scores(sample_dir: Path) -> dict[str, Any] | None:
    path = sample_dir / SCORE_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def brief_reason(text: Any, limit: int = 200) -> str:
    """The last line of an error, which for a traceback names the exception."""

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    reason = lines[-1] if lines else "unknown"
    return reason if len(reason) <= limit else reason[: limit - 3] + "..."


def grading_failure(sample: Mapping[str, Any]) -> str | None:
    """Why a sample whose agent finished has no score, or ``None``.

    ``sample`` is a sample's ``info.json``. A sample whose agent failed is not a
    grading failure: it already counts as failed, and Resume runs it again.
    """

    if sample.get("status") not in {"ok", "skipped"}:
        return None
    runtime = sample.get("runtime")
    bench = runtime.get("bench") if isinstance(runtime, dict) else None
    if not isinstance(bench, dict):
        return None
    scores = bench.get("scores")
    error = bench.get("error")
    if not error and isinstance(scores, dict):
        error = scores.get("error")
    if error:
        return str(error)
    score_key = bench.get("score_key") or "overall_score"
    if not isinstance(scores, dict) or sample_score(scores, score_key) is None:
        return "grader returned no score"
    return None


def mean_aggregate(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Average scored samples over *all* samples, counting failures as zero.

    An agent that crashed or timed out earned no score; dropping it from the
    denominator would silently reward failure. Unscored samples are reported
    separately so a broken grader is not mistaken for a weak model. Categories
    follow the same rule, so they add up to the headline.
    """

    rows = list(samples)
    per_task: list[dict[str, Any]] = []
    per_category: dict[str, list[float | None]] = {}
    total = 0.0
    scored = 0
    for row in rows:
        scores = row.get("bench_scores")
        score_key = row.get("bench_score_key") or "overall_score"
        grader_error = row.get("bench_error")
        if not grader_error and isinstance(scores, dict):
            grader_error = scores.get("error")
        # A synthetic zero returned alongside a grader/infrastructure error is
        # not a model score. Keep it unscored so broken grading cannot depress
        # benchmark results while still surfacing the failure in the report.
        value = (
            sample_score(scores, score_key)
            if isinstance(scores, dict) and not grader_error
            else None
        )
        category = row.get("bench_category")
        if value is not None:
            total += value
            scored += 1
        if isinstance(category, str) and category:
            per_category.setdefault(category, []).append(value)
        per_task.append(
            {
                "task_id": row.get("task_id"),
                "model": row.get("model"),
                "harness": row.get("harness"),
                "category": category,
                "status": row.get("status"),
                "score": value,
                # A sample that never reached its grader has only its own error.
                "error": grader_error or (row.get("error") if value is None else None),
            }
        )
    task_count = len(rows)
    categories: dict[str, dict[str, Any]] = {}
    for name, values in sorted(per_category.items()):
        earned = [value for value in values if value is not None]
        categories[name] = {
            "task_count": len(values),
            "scored_count": len(earned),
            "average": round(sum(earned) / len(values), 6),
            "scored_average": round(sum(earned) / len(earned), 6) if earned else None,
        }
    return {
        "schema_version": REPORT_SCHEMA,
        "task_count": task_count,
        "scored_count": scored,
        "unscored_count": task_count - scored,
        "average": round(total / task_count, 6) if task_count else None,
        "scored_average": round(total / scored, 6) if scored else None,
        "categories": categories,
        "tasks": per_task,
    }


def _model_groups(
    rows: list[dict[str, Any]],
) -> list[tuple[str, str | None, list[dict[str, Any]]]]:
    """Split rows by model label, and by Harness where a label ran under several.

    A sample that failed before its Harness was built records none; it stays
    with its model's only Harness so that the failure is not split off.
    """

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row.get("model")), []).append(row)
    groups: list[tuple[str, str | None, list[dict[str, Any]]]] = []
    for model, items in by_model.items():
        harnesses = list(
            dict.fromkeys(
                row["harness"]
                for row in items
                if row.get("harness") not in (None, "unknown")
            )
        )
        if len(harnesses) <= 1:
            groups.append((model, harnesses[0] if harnesses else None, items))
            continue
        for harness in harnesses:
            groups.append(
                (
                    model,
                    harness,
                    [row for row in items if row.get("harness") == harness],
                )
            )
        rest = [row for row in items if row.get("harness") not in harnesses]
        if rest:
            groups.append((model, None, rest))
    return groups


def build_report(
    samples: Iterable[dict[str, Any]],
    aggregate: Callable[[list[dict[str, Any]]], dict[str, Any]] = mean_aggregate,
) -> dict[str, Any]:
    """Aggregate each model on its own; never average across models.

    ``models`` holds one entry per model, the bench's ``aggregate`` over that
    model's samples. A run of one model also keeps that aggregate at the top
    level. A run of several models has only counts there: an average over
    different models describes none of them.
    """

    reports = [
        (model, harness, aggregate(rows))
        for model, harness, rows in _model_groups(list(samples))
    ]
    entries = [
        {
            "model": model,
            "harness": harness,
            **{
                key: value
                for key, value in report.items()
                if key not in {"schema_version", "tasks"}
            },
        }
        for model, harness, report in reports
    ]
    if len(reports) == 1:
        summary = {key: value for key, value in reports[0][2].items() if key != "tasks"}
    else:
        summary = {"schema_version": REPORT_SCHEMA}
        for key in ("task_count", "scored_count", "unscored_count"):
            summary[key] = sum(entry.get(key) or 0 for entry in entries)
    tasks = [task for _, _, report in reports for task in report.get("tasks", [])]
    return {**summary, "models": entries, "tasks": tasks}


def _bar(value: float) -> str:
    filled = max(0, min(10, int(round(value * 10))))
    return "█" * filled + "░" * (10 - filled)


def _number(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def _model_name(entry: Mapping[str, Any]) -> str:
    harness = entry.get("harness")
    return f"{entry.get('model')} ({harness})" if harness else str(entry.get("model"))


def _summary_lines(entry: Mapping[str, Any]) -> list[str]:
    lines = [
        f"  scored {entry['scored_count']}/{entry['task_count']}"
        f"  (unscored {entry['unscored_count']})"
    ]
    average = entry.get("average")
    pass_rate = entry.get("pass_rate")
    if "pass_rate" in entry:
        if isinstance(pass_rate, (int, float)):
            lines.append(f"  pass rate   {_bar(pass_rate)} {pass_rate:.4f}")
        if isinstance(average, (int, float)):
            lines.append(f"  mean reward {_bar(average)} {average:.4f}")
        return lines
    if isinstance(average, (int, float)):
        lines.append(f"  average  {_bar(average)} {average:.4f}")
    scored_average = entry.get("scored_average")
    if scored_average is not None and entry["unscored_count"]:
        lines.append(f"  average over scored samples only: {scored_average:.4f}")
    return lines


def _row_metrics(entry: Mapping[str, Any]) -> str:
    """One model's result on one line, for the table of a run of several models."""

    scored = f"scored {entry['scored_count']}/{entry['task_count']}"
    if "pass_rate" in entry:
        return (
            f"pass {_number(entry.get('pass_rate'))}  "
            f"reward {_number(entry.get('average'))}  {scored}"
        )
    text = f"average {_number(entry.get('average'))}  {scored}"
    if entry["unscored_count"] and entry.get("scored_average") is not None:
        text += f"  (scored only {entry['scored_average']:.4f})"
    return text


def _category_lines(entry: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for name, row in (entry.get("categories") or {}).items():
        count = row["task_count"]
        scored = row.get("scored_count", count)
        unit = "trials" if "pass_rate" in row else "tasks"
        size = f"{count} {unit}" + (f", {scored} scored" if scored != count else "")
        if "pass_rate" in row:
            lines.append(
                f"    pass {_number(row['pass_rate'])}  reward {_number(row['average'])}"
                f"  {name}  ({size})"
            )
        else:
            lines.append(f"    {_number(row['average'])}  {name}  ({size})")
    return lines


def format_report(bench_name: str, report: dict[str, Any]) -> str:
    lines = [
        "",
        "=" * 62,
        f"  Bench report — {bench_name}",
        "=" * 62,
    ]
    # A report written before ``models`` existed is one model's aggregate.
    models = report.get("models") or ([report] if report.get("task_count") else [])
    several = len(models) > 1
    if not models:
        lines.append("  no samples")
    elif not several:
        if models[0].get("model"):
            lines.append(f"  model  {_model_name(models[0])}")
        lines.extend(_summary_lines(models[0]))
        categories = _category_lines(models[0])
        if categories:
            lines.extend(["", *categories])
    else:
        lines.append(
            f"  {len(models)} models, {report['task_count']} samples"
            f"  (scored {report['scored_count']}, unscored {report['unscored_count']})"
        )
        lines.append("")
        width = max(len(_model_name(entry)) for entry in models)
        for entry in models:
            lines.append(f"  {_model_name(entry):<{width}}  {_row_metrics(entry)}")
        for entry in models:
            categories = _category_lines(entry)
            if categories:
                lines.extend(["", f"  {_model_name(entry)}", *categories])
    missing = report.get("missing_count")
    if missing:
        lines.append("")
        lines.append(
            f"  {missing} of {report.get('planned_count')} planned rollouts have no "
            "sample yet and are not in the averages"
        )
    failures = [row for row in report.get("tasks", []) if row.get("score") is None]
    if failures:
        lines.append("")
        lines.append(f"  unscored samples ({len(failures)}):")
        for row in failures[:20]:
            name = f"{row.get('task_id')}"
            if several:
                name += f" | {row.get('model')}"
            status = row.get("status") or "unknown"
            detail = brief_reason(row.get("error")) if row.get("error") else ""
            lines.append(
                f"    - {name} (status {status})" + (f": {detail}" if detail else "")
            )
        if len(failures) > 20:
            lines.append(f"    ... and {len(failures) - 20} more")
    lines.append("=" * 62)
    return "\n".join(lines)
