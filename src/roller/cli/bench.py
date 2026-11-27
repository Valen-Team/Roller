"""Handlers for ``roller bench`` subcommands."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from roller.benches import format_report, get_bench, list_benches
from roller.benches.run_config import load_bench_run_plan, temporary_environment
from roller.benches.run_report import generate_bench_report
from roller.cli.collect import _collect
from roller.cli.preflight import repeat_bench_rows
from roller.tasks import TaskFormatError


def _bench(args: argparse.Namespace) -> int:
    if args.bench_command == "list":
        for adapter in list_benches():
            variants = (
                f" [variants: {', '.join(adapter.variants)}]"
                if adapter.variants
                else ""
            )
            print(f"{adapter.name:<16} {adapter.description}{variants}")
        return 0
    if args.bench_command == "run":
        with tempfile.TemporaryDirectory(prefix="roller-bench-run-") as temporary:
            plan = load_bench_run_plan(
                args.config,
                args.extra_config,
                temporary_root=Path(temporary),
            )
            collect_args = plan.collect_args
            collect_args.dry_run = bool(args.dry_run)
            collect_args._bench_run_config = plan.provenance
            with temporary_environment(plan.environment):
                return _collect(collect_args)
    if args.bench_command == "export":
        adapter = get_bench(args.name)
        root = Path(args.root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"bench root is not a directory: {root}")
        rows = adapter.load(
            root,
            categories=args.category or None,
            task_ids=args.task_id or None,
            variant=args.variant,
            timeout_multiplier=args.timeout_multiplier,
        )
        if not rows:
            raise TaskFormatError(f"{adapter.name}: selection matched no tasks")
        rows = repeat_bench_rows(rows, args.runs_per_task, adapter.name)
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        categories = sorted({row["bench"].get("category") or "" for row in rows} - {""})
        print(f"Exported {len(rows)} {adapter.name} tasks to {output}")
        if categories:
            print(f"Categories: {', '.join(categories)}")
        print(
            f"Run with: roller collect --tasks {output} --model <model> --task-timeouts"
        )
        return 0
    return _bench_report(args)


def _bench_report(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    label, report, written = generate_bench_report(Path(args.run_dir), output=output)
    use_json = (
        bool(getattr(args, "json", False)) or getattr(args, "format", None) == "json"
    )
    if use_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report(label, report))
        if written is not None:
            print(f"Report written to {written}")
    return 0
