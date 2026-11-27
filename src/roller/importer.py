"""Import the local sessions of Claude Code and Codex as a Roller run.

Each session becomes one sample: its native session as ``trace.jsonl``, the
ACTF trajectory and, when one can be made, the SFT record, so ``roller view``
and ``roller export`` read it like a collected sample. A session whose file has
not changed since the last import is left as it is.
"""

from __future__ import annotations

import fnmatch
import io
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import zstandard

from roller.actf.claude_code import convert_claude_code_trace
from roller.actf.codex import convert_codex_trace
from roller.actf.opencode import (
    convert_opencode_trace,
    extract_opencode_usage,
    read_opencode_events,
)
from roller.compaction_sft import has_main_compaction
from roller.opencode_capture import native_messages
from roller.run_outputs import write_run_dataset, write_run_info
from roller.sft import SFTConversionError, write_sft
from roller.storage import atomic_write_json, safe_component
from roller.usage import extract_claude_code_usage, extract_codex_usage

CLAUDE_CODE = "claude-code"
CODEX = "codex"
OPENCODE = "opencode"
HARNESSES = (CLAUDE_CODE, CODEX, OPENCODE)
RAW_SCHEMAS = {
    CLAUDE_CODE: "claude-code-session-jsonl",
    CODEX: "codex-session-jsonl",
    OPENCODE: "opencode-session-jsonl",
}
STAGING_DIR = ".import"
TITLE_LENGTH = 120
_REMINDER_START, _REMINDER_END = "<system-reminder>", "</system-reminder>"
# Text the harness put into a user message; never the title of a session.
_INJECTED = (
    "<local-command-",
    "<task-notification>",
    "<command-name>",
    "<command-message>",
    "<bash-input>",
    "This session is being continued",
    "[Request interrupted",
    "<environment_context>",
    "<INSTRUCTIONS>",
    "<user_instructions>",
    "# AGENTS.md",
    "<turn_aborted>",
    "<skill>",
    "<subagent_notification>",
    "<image name=",
    "</image>",
    "Codex could not read the local image",
    "<recommended_plugins>",
    "<external_codex_apps_",
)
# The request below the files that a Codex IDE message mentions.
_IDE_REQUEST = "## My request:"


@dataclass(frozen=True)
class Session:
    harness: str
    path: Path
    session_id: str
    started_at: str
    project: str
    size: int
    mtime: float


@dataclass
class Summary:
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    empty: int = 0
    subagents: int = 0
    sft: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = [
            f"Imported {self.new} new, updated {self.updated}, unchanged "
            f"{self.unchanged}, empty {self.empty}, failed {len(self.failed)}",
            f"Skipped {self.subagents} subagent sessions",
            f"{self.sft} samples have an SFT record",
        ]
        lines += [f"  failed {session}: {error}" for session, error in self.failed]
        return lines


def default_roots(harness: str | None) -> list[Path]:
    claude = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    codex = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    xdg_data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    opencode = Path(os.environ.get("OPENCODE_DATA_DIR") or xdg_data / "opencode")
    roots = {
        CLAUDE_CODE: [claude / "projects"],
        CODEX: [codex / "sessions", codex / "archived_sessions"],
        OPENCODE: [opencode],
    }
    return [
        root
        for name in ([harness] if harness else HARNESSES)
        for root in roots[name]
        if root.is_dir()
    ]


def _lines(path: Path) -> Iterator[str]:
    with path.open("rb") as raw:
        stream = (
            zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
            if path.name.endswith(".zst")
            else raw
        )
        yield from io.TextIOWrapper(stream, encoding="utf-8", errors="replace")


def _records(path: Path) -> Iterator[dict[str, Any]]:
    for line in _lines(path):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def _project(cwd: Any) -> str:
    return Path(cwd).name if isinstance(cwd, str) and cwd else ""


def _opencode_iso(ms: Any) -> str:
    if isinstance(ms, (int, float)) and not isinstance(ms, bool):
        return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(
            timespec="seconds"
        )
    return ""


def _probe(path: Path) -> Session | str | None:
    """Describe one session file; ``"subagent"``, or ``None`` for no session."""

    records = _records(path)
    first = next(records, None)
    if first is None:
        return None
    stat = path.stat()
    if first.get("type") == "session_meta" and isinstance(first.get("payload"), dict):
        meta = first["payload"]
        source = meta.get("source")
        if not path.name.startswith("rollout-"):
            return None
        if meta.get("thread_source") == "subagent" or "subagent" in (
            source if isinstance(source, (dict, str)) else ()
        ):
            return "subagent"
        return Session(
            CODEX,
            path,
            str(meta.get("id")),
            str(meta.get("timestamp") or first.get("timestamp") or ""),
            _project(meta.get("cwd")),
            stat.st_size,
            stat.st_mtime,
        )
    if first.get("type") in {"session_export", "event", "request"}:
        info: dict[str, Any] = {}
        fallback_id = ""
        for record in (first, *records):
            rtype = record.get("type")
            if rtype == "session_export" and isinstance(record.get("data"), dict):
                info = {**info, **(record["data"].get("info") or {})}
                break
            if rtype == "event" and isinstance(record.get("event"), dict):
                ev = record["event"]
                if ev.get("type") in {"session.created", "session.updated"}:
                    candidate = (ev.get("properties") or {}).get("info")
                    if isinstance(candidate, dict):
                        info.update(candidate)
                        if info.get("id") and info.get("directory"):
                            break
            elif rtype == "request" and not fallback_id and record.get("native_session_id"):
                fallback_id = str(record["native_session_id"])
        if info.get("parentID"):
            return "subagent"
        return Session(
            OPENCODE,
            path,
            str(info.get("id") or fallback_id or path.stem),
            _opencode_iso((info.get("time") or {}).get("created")),
            _project(info.get("directory")),
            stat.st_size,
            stat.st_mtime,
        )
    if path.name.endswith(".zst"):
        return None
    started, cwd = "", None
    for record in (first, *records):
        started = started or str(record.get("timestamp") or "")
        if record.get("cwd"):
            cwd = record["cwd"]
            break
    return Session(
        CLAUDE_CODE,
        path,
        path.stem,
        started,
        _project(cwd),
        stat.st_size,
        stat.st_mtime,
    )


def discover(
    paths: list[Path],
    harness: str | None = None,
    since: str | None = None,
    projects: list[str] | None = None,
) -> tuple[list[Session], int]:
    """Find the sessions under ``paths`` and count the subagent sessions left out."""

    files: list[Path] = []
    for root in paths:
        files += (
            [root]
            if root.is_file()
            else sorted(
                path
                for path in root.rglob("*")
                if path.name.endswith((".jsonl", ".jsonl.zst"))
            )
        )
    subagents = 0
    chosen: dict[tuple[str, str], Session] = {}
    for path in files:
        if "subagents" in path.parts or path.name.startswith("agent-"):
            subagents += 1
            continue
        session = _probe(path)
        if session == "subagent":
            subagents += 1
        if not isinstance(session, Session):
            continue
        key = (session.harness, session.session_id)
        # A live session over its compressed or archived copy; the longest
        # file of a Claude Code session that was continued in another project.
        rank = (
            "archived_sessions" in path.parts,
            path.name.endswith(".zst"),
            -session.size,
        )
        current = chosen.get(key)
        if current is None or rank < (
            "archived_sessions" in current.path.parts,
            current.path.name.endswith(".zst"),
            -current.size,
        ):
            chosen[key] = session
    sessions = [
        session
        for session in chosen.values()
        if (harness is None or session.harness == harness)
        and (since is None or session.started_at[:10] >= since)
        and (
            not projects
            or any(fnmatch.fnmatch(session.project, glob) for glob in projects)
        )
    ]
    return sorted(sessions, key=lambda session: session.started_at), subagents


def _title_text(text: str) -> str:
    text = text.strip()
    if text.startswith("# Files mentioned by the user"):
        text = text.partition(_IDE_REQUEST)[2].strip()
    while text.startswith(_REMINDER_START):
        end = text.find(_REMINDER_END)
        if end < 0:
            return ""
        text = text[end + len(_REMINDER_END) :].strip()
    if text.startswith(_INJECTED):
        return ""
    return " ".join(text.split())[:TITLE_LENGTH]


def _user_texts(record: dict[str, Any], harness: str) -> list[str]:
    if harness == CODEX:
        payload = record.get("payload") or {}
        if not (
            record.get("type") == "response_item"
            and payload.get("type") == "message"
            and payload.get("role") == "user"
        ):
            return []
        content = payload.get("content")
    else:
        if (
            record.get("type") != "user"
            or record.get("isMeta")
            or record.get("isCompactSummary")
        ):
            return []
        content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        return [content]
    return [
        block["text"]
        for block in content or []
        if isinstance(block, dict)
        and block.get("type") in {"text", "input_text"}
        and isinstance(block.get("text"), str)
    ]


def _metadata(session: Session, trace: Path) -> dict[str, Any]:
    """The model, title and provenance of a session from its native records."""

    models: Counter[str] = Counter()
    titles: dict[str, str] = {}
    first_text = ""
    cwd = cli_version = parent = None
    git: dict[str, Any] = {"branch": None, "commit": None, "remote": None}
    identity: list[str] = []
    shapes: set[str] = set()
    if session.harness == OPENCODE:
        events = read_opencode_events(trace)
        info: dict[str, Any] = {}
        for event in events:
            if event.get("type") == "session_export":
                info = {**info, **(event.get("data", {}).get("info") or {})}
            elif event.get("type") == "event" and (event.get("event") or {}).get("type") in {
                "session.created",
                "session.updated",
            }:
                info = {
                    **info,
                    **((event["event"].get("properties") or {}).get("info") or {}),
                }
        for message in native_messages(events, root_only=True):
            msg_info = message.get("info") or {}
            if msg_info.get("role") == "assistant" and msg_info.get("modelID"):
                models[str(msg_info["modelID"])] += 1
            elif msg_info.get("role") == "user" and not first_text:
                for part in message.get("parts") or []:
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        title = _title_text(part["text"])
                        if title:
                            first_text = title
                            break
        return {
            "model": models.most_common(1)[0][0] if models else "unknown",
            "models": sorted(models),
            "title": _title_text(info["title"]) if info.get("title") else first_text or None,
            "cwd": info.get("directory"),
            "cli_version": info.get("version"),
            "git": git,
            "parent_session_id": info.get("parentID"),
            "identity": [],
            "tools_source": None,
        }
    for record in _records(trace):
        kind = record.get("type")
        payload = record.get("payload") or {}
        if not first_text:
            first_text = next(
                (
                    title
                    for text in _user_texts(record, session.harness)
                    if (title := _title_text(text))
                ),
                "",
            )
        if session.harness == CODEX:
            if kind == "session_meta" and cwd is None:
                cwd = payload.get("cwd")
                cli_version = payload.get("cli_version")
                parent = payload.get("forked_from_id")
                meta_git = payload.get("git") or {}
                git = {
                    "branch": meta_git.get("branch"),
                    "commit": meta_git.get("commit_hash"),
                    "remote": meta_git.get("repository_url"),
                }
                identity += [
                    payload[key]
                    for key in ("creator_user_id", "creator_account_id")
                    if payload.get(key)
                ]
            elif kind == "turn_context" and payload.get("model"):
                models[payload["model"]] += 1
            continue
        cwd = cwd or record.get("cwd")
        cli_version = cli_version or record.get("version")
        git["branch"] = git["branch"] or record.get("gitBranch")
        if parent is None and record.get("sessionId"):
            parent = record["sessionId"]
        if kind == "assistant":
            model = (record.get("message") or {}).get("model")
            if model and model != "<synthetic>":
                models[model] += 1
        elif kind in {"custom-title", "ai-title"}:
            value = record.get("customTitle" if kind == "custom-title" else "aiTitle")
            if value:
                titles[kind] = value
        elif kind == "bridge-session":
            identity += [
                record[key]
                for key in ("ownerAccountUuid", "ownerOrganizationUuid")
                if record.get(key)
            ]
        elif kind == "api-request-shape":
            shapes.add(json.dumps((record.get("shape") or {}).get("tools")))
    if session.harness == CLAUDE_CODE and parent == session.session_id:
        parent = None
    title = titles.get("custom-title") or titles.get("ai-title")
    return {
        "model": models.most_common(1)[0][0] if models else "unknown",
        "models": sorted(models),
        "title": _title_text(title) if title else first_text or None,
        "cwd": cwd,
        "cli_version": cli_version,
        "git": git,
        "parent_session_id": parent,
        "identity": sorted(set(identity)),
        "tools_source": None
        if session.harness == CODEX
        else "captured_merged"
        if len(shapes) > 1
        else "captured"
        if shapes
        else "inferred",
    }


def _iso(value: str) -> str | None:
    if not value:
        return None
    return (
        datetime.fromisoformat(value)
        .replace(tzinfo=timezone.utc)
        .isoformat(timespec="seconds")
    )


def import_session(
    session: Session, index: int, run_dir: Path, previous: Path | None
) -> dict[str, Any] | None:
    """Write one session as a sample; ``None`` when it has no step."""

    staging = run_dir / STAGING_DIR / safe_component(session.session_id)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    trace = staging / "trace.jsonl"
    with session.path.open("rb") as source, trace.open("wb") as target:
        shutil.copyfileobj(
            zstandard.ZstdDecompressor().stream_reader(source, read_across_frames=True)
            if session.path.name.endswith(".zst")
            else source,
            target,
        )
    convert = {
        CLAUDE_CODE: convert_claude_code_trace,
        CODEX: convert_codex_trace,
        OPENCODE: convert_opencode_trace,
    }[session.harness]
    trajectory = convert(trace, staging / "trace_ACTF.json")
    if not trajectory["steps"]:
        shutil.rmtree(staging)
        return None
    metadata = _metadata(session, trace)
    runtime: dict[str, Any] = {}
    sft = staging / "trace_SFT.json"
    try:
        if has_main_compaction(trace, session.harness):
            raise SFTConversionError("context reset SFT requires captured Calls")
        write_sft(sft, trajectory)
    except SFTConversionError as exc:
        runtime["trajectory_conversion"] = {
            "status": "unavailable",
            "target": "sft",
            "error": str(exc),
        }
    started, finished = _iso(trajectory["started_at"]), _iso(trajectory["finished_at"])
    usage_extractor = {
        CLAUDE_CODE: extract_claude_code_usage,
        CODEX: extract_codex_usage,
        OPENCODE: extract_opencode_usage,
    }[session.harness]
    usage = usage_extractor(trace)
    model = metadata["model"]
    result = {
        "schema_version": "roller.sample/v1",
        "request_index": index,
        "sample_index": 0,
        "sample_count": 1,
        "task_id": session.session_id,
        "source_index": index,
        "model": model,
        "model_label": model,
        "harness": session.harness,
        "status": "ok",
        "error": None,
        "usage": usage,
        "attempts": [],
        "timing": {
            "elapsed_seconds": round(
                (
                    datetime.fromisoformat(finished) - datetime.fromisoformat(started)
                ).total_seconds(),
                3,
            )
            if started and finished
            else 0,
            "started_at": started,
            "finished_at": finished,
        },
        "trajectory": {
            "raw": "trace.jsonl",
            "raw_schema": RAW_SCHEMAS[session.harness],
            "actf": "trace_ACTF.json",
            "actf_schema": "ACTF_v1.0",
            "sft": "trace_SFT.json" if sft.is_file() else None,
            "tools_source": metadata["tools_source"]
            or (trajectory.get("tools_capture") or {}).get("status"),
            "system_prompt_source": "captured"
            if trajectory["system_prompt"]
            else "missing",
        },
        "conversation": None,
        "output_collection": None,
        "runtime": runtime,
        "provenance": {
            "import": {
                "source_path": str(session.path.resolve()),
                "source_size": session.size,
                "source_mtime": session.mtime,
                "session_id": session.session_id,
                "cwd": metadata["cwd"],
                "cli_version": metadata["cli_version"],
                "git": metadata["git"],
                **(
                    {"parent_session_id": metadata["parent_session_id"]}
                    if metadata["parent_session_id"]
                    else {}
                ),
                "models": metadata["models"],
                "identity": metadata["identity"],
            }
        },
        "title": metadata["title"],
        "project": _project(metadata["cwd"]),
    }
    atomic_write_json(staging / "info.json", result)
    target = (
        run_dir
        / "samples"
        / safe_component(model)
        / f"{index:08d}_{safe_component(session.session_id)}"
    )
    for stale in {previous, target} - {None}:
        if stale.exists():
            shutil.rmtree(stale)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(target)
    return result


def _existing(run_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    samples = {}
    for path in run_dir.glob("samples/*/*/info.json"):
        info = json.loads(path.read_text(encoding="utf-8"))
        samples[info["task_id"]] = (path.parent, info)
    return samples


def import_sessions(
    run_dir: Path, sessions: list[Session], parallel: int = 4
) -> Summary:
    """Import ``sessions`` into ``run_dir`` and rewrite its run-level files."""

    summary = Summary()
    existing = _existing(run_dir)
    next_index = 1 + max(
        (int(info.get("source_index", 0)) for _, info in existing.values()), default=0
    )
    jobs: list[tuple[Session, int, Path | None]] = []
    for session in sessions:
        previous = existing.get(session.session_id)
        if previous is None:
            jobs.append((session, next_index, None))
            next_index += 1
            continue
        source = (previous[1].get("provenance") or {}).get("import") or {}
        if (source.get("source_size"), source.get("source_mtime")) == (
            session.size,
            session.mtime,
        ):
            summary.unchanged += 1
        else:
            jobs.append((session, int(previous[1]["source_index"]), previous[0]))
    run_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(import_session, session, index, run_dir, previous): (
                session,
                previous,
            )
            for session, index, previous in jobs
        }
        for future in as_completed(futures):
            session, previous = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                summary.failed.append(
                    (session.session_id, f"{type(exc).__name__}: {exc}")
                )
                continue
            if result is None:
                summary.empty += 1
            elif previous is None:
                summary.new += 1
            else:
                summary.updated += 1
    shutil.rmtree(run_dir / STAGING_DIR, ignore_errors=True)

    if not (run_dir / "info.json").is_file():
        atomic_write_json(
            run_dir / "info.json",
            {"schema_version": "roller.run/v1", "source": "import"},
        )
    results = sorted(
        (info for _, info in _existing(run_dir).values()),
        key=lambda info: info["request_index"],
    )
    write_run_info(run_dir, results, total_requests=len(results), state="finished")
    write_run_dataset(run_dir, results)
    summary.sft = sum(bool(info["trajectory"].get("sft")) for info in results)
    return summary


def dry_run_lines(sessions: list[Session]) -> list[str]:
    projects: dict[str, list[Session]] = {}
    for session in sessions:
        projects.setdefault(session.project or "-", []).append(session)
    lines = [f"{'project':<32} {'sessions':>8} {'size':>10}  first       last"]
    for name, group in sorted(projects.items()):
        size = sum(session.size for session in group) / 1e6
        dates = sorted(session.started_at[:10] for session in group)
        lines.append(
            f"{name[:32]:<32} {len(group):>8} {size:>8.1f}MB  {dates[0]:<10}  {dates[-1]}"
        )
    lines.append(f"{len(sessions)} sessions")
    return lines
