"""Top-level command dispatch and auxiliary CLI commands."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from roller import hf_dataset, hf_redaction, sft_dataset, token_dataset, xtuner_dataset
from roller import importer as importer_module
from roller.atif import export_run
from roller.cleanup import clean
from roller.cli.bench import _bench
from roller.cli.collect import _collect
from roller.cli.parser import _build_parser
from roller.docker_runtime import DockerRuntime
from roller.tasks import TaskFormatError
from roller.viewer import serve as serve_viewer

logger = logging.getLogger(__name__)


def _init(args: argparse.Namespace) -> int:
    """Copy the starter examples to ``<directory>/examples``, keeping existing files."""

    package = Path(__file__).resolve().parents[1]
    # Wheels ship the examples inside the package; a source checkout keeps them at the repository root.
    source = (
        package / "examples"
        if (package / "examples").is_dir()
        else package.parents[1] / "examples"
    )
    target = Path(args.directory) / "examples"
    written = kept = 0
    for name in ("tasks", "config", "benchmarks", "env"):
        for path in sorted((source / name).rglob("*")):
            if not path.is_file() or path.suffix == ".env":
                continue
            destination = target / path.relative_to(source)
            if destination.exists():
                kept += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            written += 1
    print(
        f"Wrote {written} files to {target}"
        + (f" ({kept} existing files kept)" if kept else "")
    )
    print(
        f"Next: cp {target / 'env' / 'api.env.example'} {target / 'env' / 'api.env'} and set the model service URL and API key"
    )
    return 0


def _view(args: argparse.Namespace) -> int:
    return serve_viewer(Path(args.run_dir), host=args.host, port=args.port)


def _export(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not (run_dir / "samples").is_dir():
        raise FileNotFoundError(f"{run_dir} has no samples directory")
    output = Path(args.output) if args.output else run_dir / args.format
    if args.min_score is not None and args.format not in {"sft", "xtuner"}:
        raise ValueError("--min-score applies to --format sft and --format xtuner")
    if args.inline_images and args.format != "xtuner":
        raise ValueError("--inline-images applies to --format xtuner")
    if args.format == "hf":
        rules = (
            hf_redaction.load_rules(Path(args.redact_rules))
            if args.redact_rules
            else None
        )
        rows, redaction = hf_dataset.export_run(
            run_dir,
            output,
            rules,
            redact=args.redact,
            runs=args.include_runs,
            preserve_task_keys=args.preserve_task_keys,
        )
        print(f"Exported {len(rows)} samples as a Hugging Face dataset to {output}")
        for line in redaction.lines():
            print(line)
        report_path = output.with_name(output.name + ".redaction-report.json")
        print(f"Redaction review: {report_path}")
        return 0
    if args.redact_rules or args.include_runs or args.preserve_task_keys:
        raise ValueError(
            "--redact-rules, --include-runs and --preserve-task-keys apply to --format hf"
        )
    if args.format == "tokens":
        samples, joined, tokens = token_dataset.export_run(run_dir, output)
        print(
            f"Exported {samples} samples as {joined} sequences of {tokens} tokens "
            f"to {output / token_dataset.TOKENS_FILE}"
        )
        return 0
    if args.format == "sft":
        summary = sft_dataset.export_run(run_dir, output, args.min_score)
        for line in summary.lines(output, args.min_score):
            print(line)
        return 0
    if args.format == "xtuner":
        summary = xtuner_dataset.export_run(
            run_dir, output, args.min_score, inline_images=args.inline_images
        )
        for line in summary.lines(output, args.min_score):
            print(line)
        return 0
    written, skipped = export_run(run_dir, output)
    print(f"Exported {len(written)} ATIF trajectories to {output}")
    if skipped:
        print(f"Skipped {len(skipped)} samples:")
        for sample, reason in skipped.items():
            print(f"  {sample}: {reason}")
    return 0


def _import(args: argparse.Namespace) -> int:
    sessions, subagents = importer_module.discover(
        [Path(path).expanduser() for path in args.paths]
        or importer_module.default_roots(args.harness),
        args.harness,
        args.since,
        args.project,
    )
    if args.dry_run:
        for line in importer_module.dry_run_lines(sessions):
            print(line)
        return 0
    run_dir = Path(args.output) / args.run_name
    summary = importer_module.import_sessions(run_dir, sessions, args.parallel)
    summary.subagents = subagents
    for line in summary.lines():
        print(line)
    print(f"Run: {run_dir}")
    return 1 if summary.failed else 0


def _clean(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve() if args.path else None
    if path is not None and not path.is_dir():
        raise FileNotFoundError(f"not a directory: {path}")
    for line in clean(
        DockerRuntime(args.docker_binary), path, dry_run=args.dry_run, force=args.force
    ):
        print(line)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        if args.command == "collect":
            exit_code = _collect(args)
        elif args.command == "bench":
            exit_code = _bench(args)
        elif args.command == "view":
            exit_code = _view(args)
        elif args.command == "export":
            exit_code = _export(args)
        elif args.command == "import":
            exit_code = _import(args)
        elif args.command == "init":
            exit_code = _init(args)
        elif args.command == "clean":
            exit_code = _clean(args)
        else:
            parser.error(f"unsupported command: {args.command}")
            return
    except (
        FileNotFoundError,
        FileExistsError,
        TaskFormatError,
        ValueError,
        RuntimeError,
    ) as exc:
        logger.error("%s", exc)
        exit_code = 2
    except KeyboardInterrupt:
        logger.error("interrupted")
        exit_code = 130
    if exit_code > 128 and args.command in {"collect", "bench"}:
        # A signal stopped the run. Rollout threads that lost their containers
        # may still be unwinding, and a normal exit would wait for them.
        logging.shutdown()
        sys.stdout.flush()
        os._exit(exit_code)
    raise SystemExit(exit_code)
