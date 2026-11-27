"""Sample path, error-summary, and legacy-migration helpers for collection."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roller.capture import recover_capture_retention
from roller.compaction_sft import SFT_SEGMENTS_DIR
from roller.models import RunRequest
from roller.sample_metadata import has_legacy_claude_sidechain
from roller.storage import safe_component

HEARTBEAT_SECONDS = 60.0


# What --training-files-only keeps of a finished sample. The images of its SFT
# records go to the run's ``blobs/``, shared by every sample of the run.
TRAINING_FILES = {
    "info.json",
    "score.json",
    "trace_SFT.json",
    SFT_SEGMENTS_DIR,
    "token_ids.jsonl",
}


def _clear_sft_segments(sample_dir: Path) -> None:
    path = sample_dir / SFT_SEGMENTS_DIR
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def error_line(error: object, limit: int = 300) -> str:
    """The first line of an error, short enough for one log line."""

    text = str(error or "").strip()
    line = text.splitlines()[0].strip() if text else "no error message"
    return line if len(line) <= limit else line[: limit - 3] + "..."


def failure_groups(
    results: list[dict[str, Any]], limit: int = 5
) -> list[tuple[int, str, str]]:
    """Failed results by reason, most frequent first: (count, reason, example task)."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") in {"error", "timeout"}:
            # Ids, ports and counts differ between rollouts that fail alike.
            key = re.sub(r"[0-9a-f]{8,}|\d+", "#", error_line(result.get("error")))
            groups.setdefault(key, []).append(result)
    return [
        (len(group), error_line(group[0].get("error")), str(group[0].get("task_id")))
        for group in sorted(groups.values(), key=len, reverse=True)[:limit]
    ]


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    if not minutes:
        return f"{seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def _remove_path(path: Path) -> None:
    # Never follow a symlink out of the sample directory.
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _sample_directory(run_dir: Path, request: RunRequest) -> Path:
    model_dir = safe_component(request.display_model)
    task_dir = f"{request.task.source_index:08d}_{safe_component(request.task.id)}"
    if request.sample_count > 1:
        task_dir += f"__sample_{request.sample_index + 1:03d}"
    return run_dir / "samples" / model_dir / task_dir


def check_retired_cot_samples(run_dir: Path, requests: list[RunRequest]) -> None:
    """Recover capture publications and reject incompatible old samples before collection."""

    for request in requests:
        sample_dir = _sample_directory(run_dir, request)
        for base in (sample_dir, sample_dir / "user_simulator"):
            info_path = base / "info.json"
            capture_root = base / "capture"
            if capture_root.is_dir():
                interrupted = {
                    match.group(1)
                    for entry in capture_root.iterdir()
                    if (
                        match := re.fullmatch(
                            r"\.(attempt-\d+)\.retention-(?:stage|backup)", entry.name
                        )
                    )
                    is not None
                }
                for attempt in interrupted:
                    recover_capture_retention(
                        capture_root / attempt, prefer_raw=not info_path.is_file()
                    )
                if interrupted and not info_path.is_file():
                    raise ValueError(
                        f"{base}: capture retention was interrupted; a valid capture attempt was recovered, "
                        "but the sample has no finalized info.json. Resume stopped before rerunning the model."
                    )
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                isinstance(info, dict)
                and info.get("harness") == "claude-code"
                and info.get("status") == "ok"
            ):
                actf_path = base / "trace_ACTF.json"
                try:
                    actf = json.loads(actf_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    actf = None
                if isinstance(actf, dict) and has_legacy_claude_sidechain(actf):
                    raise ValueError(
                        f"{actf_path}: legacy Claude Code sidechain steps are mixed into root ACTF/SFT; "
                        "resume stopped before changing the run or calling the model. "
                        "Reproject the existing raw trace offline or use a new run directory."
                    )
            runtime = info.get("runtime") if isinstance(info, dict) else None
            if not isinstance(runtime, dict):
                continue
            for key in ("external_thinking", "signature_recovery"):
                value = runtime.get(key)
                mode = value.get("mode") if isinstance(value, dict) else value
                if mode in ("external", "external_always", "fill", "prefer"):
                    raise ValueError(
                        f"{info_path}: {key}={mode} is retired; use a new run "
                        "directory. Existing trajectory files were not changed."
                    )
