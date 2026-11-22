"""Benchmark adapters: turn upstream benches into Roller tasks, then score them.

Adding a bench means writing one :class:`BenchAdapter` and registering it in
``BENCHES`` below. Nothing else in Roller needs to change: the adapter emits
ordinary ``roller.task/v1`` rows whose optional ``bench`` block carries the
task's scorer, and the collect path already ignores row keys it does not know.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

from roller.benches.base import (
    BENCH_SCHEMA,
    BenchAdapter,
    BenchBinding,
    BenchFormatError,
    GradeSpec,
    binding_from_task_row,
    required_env,
)
from roller.benches.pinchbench import PinchBenchAdapter
from roller.benches.qwenclaw import QwenClawBenchAdapter
from roller.benches.report import (
    SCORE_FILE,
    build_report,
    format_report,
    grading_failure,
    mean_aggregate,
    read_scores,
    sample_score,
)
from roller.benches.skillsbench import SkillsBenchAdapter
from roller.benches.swebench_pro import SWEBenchProAdapter
from roller.benches.vials import VIALSAdapter
from roller.benches.wildclaw import WildClawBenchAdapter
from roller.benches.zerobench import ZeroBenchAdapter

BENCHES: dict[str, BenchAdapter] = {
    adapter.name: adapter
    for adapter in (
        PinchBenchAdapter(),
        QwenClawBenchAdapter(),
        SkillsBenchAdapter(),
        SWEBenchProAdapter(),
        VIALSAdapter(),
        WildClawBenchAdapter(),
        ZeroBenchAdapter(),
    )
}


def get_bench(name: str) -> BenchAdapter:
    try:
        return BENCHES[name]
    except KeyError:
        known = ", ".join(sorted(BENCHES)) or "none registered"
        raise KeyError(f"unknown bench {name!r}; available: {known}") from None


def list_benches() -> list[BenchAdapter]:
    return [BENCHES[name] for name in sorted(BENCHES)]


def missing_grader_env(
    rows: Iterable[Mapping[str, Any]], environ: Mapping[str, str] | None = None
) -> list[str]:
    """Name the variables the graders of task ``rows`` need and ``environ`` lacks.

    A row without a grader needs nothing, and a bench says per task what its
    grader needs, so a selection of automatically graded tasks does not ask for
    a judge key.
    """

    env = os.environ if environ is None else environ
    missing: dict[tuple[str, ...], int] = {}
    for row in rows:
        binding = binding_from_task_row(dict(row))
        if binding is None or binding.grade is None:
            continue
        adapter = BENCHES.get(binding.name)
        groups = adapter.required_env(binding) if adapter else required_env(binding)
        for group in groups:
            if not any(env.get(name) for name in group):
                missing[group] = missing.get(group, 0) + 1
    return [
        f"{' or '.join(group)} ({count} task{'s' if count != 1 else ''})"
        for group, count in missing.items()
    ]


__all__ = [
    "BENCHES",
    "BENCH_SCHEMA",
    "SCORE_FILE",
    "BenchAdapter",
    "BenchBinding",
    "BenchFormatError",
    "GradeSpec",
    "PinchBenchAdapter",
    "QwenClawBenchAdapter",
    "SkillsBenchAdapter",
    "SWEBenchProAdapter",
    "VIALSAdapter",
    "WildClawBenchAdapter",
    "ZeroBenchAdapter",
    "binding_from_task_row",
    "build_report",
    "format_report",
    "get_bench",
    "grading_failure",
    "list_benches",
    "mean_aggregate",
    "missing_grader_env",
    "read_scores",
    "sample_score",
]
