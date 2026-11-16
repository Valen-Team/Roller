"""Package a Roller run as a Hugging Face dataset folder.

A run is written as one part per Harness and model, named ``<harness>-<model>``.
Each part holds a Roller run directory with the samples of that model, so
``roller view`` opens it after a download, the session of every sample for the
trace viewer of the Hub, and one JSON Lines table with a row per sample. Every
part has its own paths, so further runs can be exported into the same folder.

The export replaces what is private to the collection environment: the model
endpoints the run recorded, the host paths of its configuration, and the values
of the credentials it passed to its containers.
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from roller.benches.report import read_scores, sample_score
from roller.collector import CollectionConfig, Collector
from roller.redaction import REDACTED
from roller.sft import SFTConversionError, _image_data_uri
from roller.storage import safe_component

DATA_DIR = "data"
TRACES_DIR = "traces"
# Native session schemas of ``trace.jsonl`` that the Hub reads as they are.
NATIVE_TRACES = {
    "openclaw-session-jsonl",
    "codex-session-jsonl",
    "claude-code-session-jsonl",
}
STATS_START = "<!-- roller:stats:start -->"
STATS_END = "<!-- roller:stats:end -->"

_CARD = """\
---
pretty_name: {name}
tags:
- roller
- agent
- trajectories
- tool-use
configs:
- config_name: traces
  data_files: {traces_dir}/**/*.jsonl
  default: true
- config_name: trajectories
  data_files: {data_dir}/*.jsonl
---

# {name}

Agent trajectories collected with [Roller](https://github.com/Valen-Team/Roller).
Each sample is one task run by one model in a disposable Docker container, with
its reasoning, tool calls, tool results, output files and run metadata.

{stats}

## Read the traces on the Hub

`{traces_dir}/` has the agent session of every sample. Open **Data Studio** on
this page and click a row to read the prompts, reasoning, tool calls and tool
results of a sample in the trace viewer.

## Load the table

`{data_dir}/` has one table per Harness and model, with one row per sample.
`messages` and `tools` are JSON strings in the format of an OpenAI chat request.

```python
import json

from datasets import load_dataset

rows = load_dataset("OWNER/NAME", "trajectories", split="train")
messages = json.loads(rows[0]["messages"])
```

## Browse the trajectories

`runs/{part}/` is a Roller run directory. Download it and open it in the Roller
Viewer to read every step of a sample:

```bash
hf download OWNER/NAME --repo-type dataset --include "runs/{part}/*" --local-dir {name}
uvx --from roller-agent roller view {name}/runs/{part}
```

Replace `OWNER/NAME` with the id of this dataset repository.

## Files

| Path | Content |
| --- | --- |
| `{traces_dir}/<harness>-<model>/<index>_<task>.jsonl` | Agent session of one sample, which the Hub opens in its [trace viewer](https://huggingface.co/docs/hub/agent-traces) |
| `{data_dir}/<harness>-<model>.jsonl` | One row per sample: task, model, status, score, steps, tokens, `messages`, `tools` |
| `runs/<harness>-<model>/info.json` | Run configuration, progress and total usage |
| `runs/<harness>-<model>/sft.jsonl` | SFT records of the successful samples |
| `runs/<harness>-<model>/samples/<model>/<index>_<task>/` | Native session, unified trajectory, SFT record, scores and output files of one sample |

The files of a sample are described in the
[trajectory outputs guide](https://valen-team.github.io/roller/traces/trajectory-outputs/).
"""


@dataclass(frozen=True)
class Rules:
    """What to change, leave out and refuse while copying a run."""

    replace: tuple[
        tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...
    ] = ()
    exclude: tuple[str, ...] = ()
    forbid: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class Redaction:
    """What an export replaced on its own, and what it could not look for."""

    endpoints: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    credentials: tuple[str, ...] = ()
    # Credential variables without a value in the exporting environment.
    unset: tuple[str, ...] = ()

    def lines(self) -> list[str]:
        lines = []
        replaced = [
            f"{len(self.endpoints)} endpoints" if self.endpoints else "",
            f"{len(self.paths)} host paths" if self.paths else "",
            f"the values of {', '.join(self.credentials)}" if self.credentials else "",
        ]
        if any(replaced):
            lines.append("Redacted: " + ", ".join(text for text in replaced if text))
        if self.unset:
            lines.append(
                "Not set in this environment, so their values were not searched for: "
                + ", ".join(self.unset)
            )
        return lines


def load_rules(path: Path) -> Rules:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Rules(
        replace=tuple(
            (re.compile(rule["pattern"]), rule["with"])
            for rule in payload.get("replace", [])
        ),
        exclude=tuple(payload.get("exclude", [])),
        forbid=tuple(re.compile(pattern) for pattern in payload.get("forbid", [])),
    )


# A shorter value of a credential variable is a stand-in such as ``local``;
# replacing it would rewrite ordinary words of the trajectories.
_CREDENTIAL_LENGTH = 12
_CREDENTIAL_NAMES = ("key", "token", "secret", "password", "credential")
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
# Paths of the run configuration: its key, whether it names a file, placeholder.
_CONFIGURED_PATHS = (
    ("bench", "root", False, "<bench-root>"),
    (None, "workspace", False, "<workspace>"),
    (None, "tasks", True, "<tasks-dir>"),
    (None, "thinking_capabilities_path", True, "<config-dir>"),
    ("*", "model_config_path", True, "<config-dir>"),
    ("*", "config_path", True, "<config-dir>"),
    ("*", "settings_path", True, "<config-dir>"),
)


def _host_path(value: Any, is_file: bool) -> str | None:
    """The directory of a configured host path, when it is specific enough."""

    if not isinstance(value, str) or not value.startswith("/"):
        return None
    path = PurePosixPath(value).parent if is_file else PurePosixPath(value)
    return str(path) if len(path.parts) >= 3 else None


def environment_redaction(
    run_dir: Path, environ: Mapping[str, str]
) -> tuple[Rules, Redaction]:
    """Derive from a run what its export replaces without being told.

    The samples record the model endpoints and the names of the variables their
    containers received, and the run records the host paths of its configuration.
    An endpoint host becomes ``endpoint-<n>.example`` and a path its placeholder.
    A credential variable that has a value in ``environ`` has that value replaced;
    one without a value is reported, because the export cannot look for it.
    """

    urls: set[str] = set()
    names: set[str] = set()
    for path in run_dir.glob("samples/*/*/info.json"):
        provenance = _json(path).get("provenance") or {}
        urls.update(url for url in provenance.get("api_base_urls") or [] if url)
        names.update(name for name in provenance.get("environment_names") or [] if name)

    literals: dict[str, str] = {}
    credentials: list[str] = []
    unset: list[str] = []
    for name in sorted(names):
        value = environ.get(name, "")
        if value.startswith(("http://", "https://")):
            urls.add(value)
        elif name.lower().replace("_", "").endswith(_CREDENTIAL_NAMES):
            if not value:
                unset.append(name)
            elif len(value) >= _CREDENTIAL_LENGTH:
                literals[value] = REDACTED
                credentials.append(name)

    endpoints: list[str] = []
    for url in sorted(urls):
        parts = urlsplit(url)
        host = parts.hostname or ""
        netloc = parts.netloc.rpartition("@")[2]
        if not host or host in _LOOPBACK or netloc in literals:
            continue
        placeholder = f"endpoint-{len(endpoints) + 1}.example"
        specific = "." in host or ":" in host
        if netloc != host or specific:
            literals[netloc] = placeholder
            endpoints.append(netloc)
        # A bare name such as ``inference`` is an ordinary word elsewhere.
        if specific:
            literals.setdefault(host, placeholder)

    configuration = _json(run_dir / "info.json").get("configuration") or {}
    paths: list[str] = []
    for section, key, is_file, placeholder in _CONFIGURED_PATHS:
        blocks = (
            [configuration]
            if section is None
            else [block for block in configuration.values() if isinstance(block, dict)]
            if section == "*"
            else [configuration.get(section) or {}]
        )
        for block in blocks:
            path = _host_path(block.get(key), is_file)
            if path and path not in literals:
                literals[path] = placeholder
                paths.append(path)

    if not literals:
        return Rules(), Redaction(unset=tuple(unset))
    pattern = re.compile(
        "|".join(re.escape(text) for text in sorted(literals, key=len, reverse=True))
    )
    return (
        Rules(replace=((pattern, lambda match: literals[match.group(0)]),)),
        Redaction(tuple(endpoints), tuple(paths), tuple(credentials), tuple(unset)),
    )


def _copy_file(source: Path, target: Path, rules: Rules, label: str) -> list[str]:
    """Copy one file and return where a forbidden pattern is still present."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if rules.replace or rules.forbid:
        found: list[str] = []
        try:
            with (
                source.open(encoding="utf-8", newline="") as reader,
                target.open("w", encoding="utf-8", newline="") as writer,
            ):
                for number, line in enumerate(reader, 1):
                    for pattern, text in rules.replace:
                        line = pattern.sub(text, line)
                    if any(pattern.search(line) for pattern in rules.forbid):
                        found.append(f"{label}:{number}")
                    writer.write(line)
            return found
        except UnicodeDecodeError:
            pass
    shutil.copyfile(source, target)
    return []


def _copy_files(
    run_dir: Path, files: Iterable[Path], destination: Path, rules: Rules
) -> list[str]:
    """Copy files of a run, keeping their paths, and return forbidden matches."""

    found: list[str] = []
    for source in files:
        relative = source.relative_to(run_dir).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in rules.exclude):
            continue
        found += _copy_file(source, destination / relative, rules, relative)
    return found


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _grade(
    sample_dir: Path, info: Mapping[str, Any]
) -> tuple[dict[str, Any], float | None, str | None]:
    """Return the bench record of a sample, its score and the grader error.

    The score is ``None`` when the sample was not graded or grading failed.
    """

    bench = read_scores(sample_dir) or (info.get("runtime") or {}).get("bench") or {}
    scores = bench.get("scores")
    if not isinstance(scores, dict):
        scores = bench
    error = bench.get("error") or scores.get("error")
    if not bench or error:
        return bench, None, error
    return (
        bench,
        sample_score(scores, bench.get("score_key") or "overall_score"),
        None,
    )


def _blocks(value: Any) -> list[dict[str, Any]]:
    """ACTF content as the text and image blocks of a Pi message."""

    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    blocks: list[dict[str, Any]] = []
    for block in value or []:
        if isinstance(block, str):
            blocks.append({"type": "text", "text": block})
        elif isinstance(block, Mapping) and block.get("type") == "image":
            try:
                header, _, data = _image_data_uri(block, "image").partition(",")
            except SFTConversionError:
                blocks.append({"type": "text", "text": "<image>"})
                continue
            media_type = header[len("data:") : -len(";base64")]
            blocks.append({"type": "image", "data": data, "mimeType": media_type})
        elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
            blocks.append({"type": "text", "text": block["text"]})
        else:
            blocks.append(
                {"type": "text", "text": json.dumps(block, ensure_ascii=False)}
            )
    return blocks


def _time(value: Any) -> datetime | None:
    """Parse an ACTF time (``YYYY-MM-DD HH:MM:SS.ffffff``, UTC)."""

    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    fields = function if isinstance(function, Mapping) else call
    arguments = fields.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    if not isinstance(arguments, Mapping):
        arguments = {"value": arguments}
    return {
        "type": "toolCall",
        "id": str(call.get("id", "")),
        "name": str(fields.get("name", "")),
        "arguments": dict(arguments),
    }


def _usage(metric: Mapping[str, Any]) -> dict[str, int]:
    def count(*names: str) -> int:
        return next((int(metric[name]) for name in names if metric.get(name)), 0)

    prompt, output = count("prompt_tokens_len"), count("completion_tokens_len")
    read = count("cache_read_tokens", "cache_read_tokens_len")
    write = count("cache_write_tokens", "cache_write_tokens_len")
    return {
        "input": max(prompt - read - write, 0),
        "output": output,
        "cacheRead": read,
        "cacheWrite": write,
        "totalTokens": prompt + output,
    }


def _tool_definitions(tools: Any) -> list[dict[str, Any]]:
    """The function tools of an ACTF trajectory as Pi tool declarations."""

    definitions: list[dict[str, Any]] = []
    for tool in tools if isinstance(tools, list) else []:
        function = tool.get("function", tool) if isinstance(tool, Mapping) else None
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            definitions.append(
                {
                    "name": function["name"],
                    "description": function.get("description") or "",
                    "parameters": function.get("parameters") or {},
                }
            )
    return definitions


def session_trace(
    trajectory: Mapping[str, Any], header: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Convert one ACTF trajectory to a session file the Hub reads as a trace.

    Used for the Harnesses whose native trace is not a session format the Hub
    knows. The lines follow Pi's session format, version 3, which the Hub
    accepts from any harness that names itself in the header. ``header`` gives
    the session ``id`` and ``harness`` and may give ``name``, ``model`` and
    further metadata. ``cwd`` is empty, because ACTF does not record the working
    directory; the Hub reads a session as Pi's format only when the header has
    the field. The system prompt and the tools become a ``system``
    message. Every ACTF step becomes a ``user`` message, when it has user input,
    an ``assistant`` message with thinking, text and tool-call blocks, and one
    ``toolResult`` message per tool result.
    """

    steps = trajectory.get("steps") or []
    clock = next(
        (time for step in steps if (time := _time(step.get("started_at")))),
        datetime.fromtimestamp(0, timezone.utc),
    )

    def stamp() -> str:
        return clock.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    lines: list[dict[str, Any]] = [
        {"type": "session", "version": 3, "timestamp": stamp(), "cwd": "", **header}
    ]

    def entry(kind: str, **fields: Any) -> None:
        parent = lines[-1]["id"] if len(lines) > 1 else None
        lines.append(
            {
                "type": kind,
                "id": f"{len(lines):08x}",
                "parentId": parent,
                "timestamp": stamp(),
                **fields,
            }
        )

    def message(**fields: Any) -> None:
        entry("message", message={**fields, "timestamp": int(clock.timestamp() * 1000)})

    model = str(header.get("model") or "")
    provider, _, model_id = model.rpartition("/")
    if model:
        entry("model_change", provider=provider, modelId=model_id)
    if header.get("name"):
        entry("session_info", name=header["name"])
    tools = _tool_definitions(trajectory.get("tools"))
    if trajectory.get("system_prompt") or tools:
        system = "\n".join(
            block["text"] for block in _blocks(trajectory.get("system_prompt") or "")
        )
        message(
            role="system", content=system, **({"toolsAdded": tools} if tools else {})
        )

    names: dict[str, str] = {}
    for step in steps:
        clock = _time(step.get("started_at")) or clock
        user = _blocks(step.get("user_content", ""))
        if user:
            message(role="user", content=user)
        assistant = step.get("assistant_content") or {}
        content: list[dict[str, Any]] = []
        if assistant.get("reasoning_content"):
            content.append(
                {"type": "thinking", "thinking": str(assistant["reasoning_content"])}
            )
        content += _blocks(assistant.get("content", ""))
        calls = [_tool_call(call) for call in assistant.get("tool_calls") or []]
        names.update((call["id"], call["name"]) for call in calls)
        reply: dict[str, Any] = {
            "role": "assistant",
            "content": content + calls,
            "provider": provider,
            "model": model_id,
        }
        metric = step.get("metric") or {}
        if metric:
            reply["usage"] = _usage(metric)
        if metric.get("stop_reason"):
            reply["stopReason"] = metric["stop_reason"]
        message(**reply)
        for result in step.get("observation") or []:
            call_id = str(result.get("toolCallId", ""))
            message(
                role="toolResult",
                toolCallId=call_id,
                toolName=str(result.get("toolName") or names.get(call_id, "")),
                content=_blocks(result.get("content", "")),
                isError=bool(result.get("isError")),
            )
    return lines


def _write_trace(
    path: Path,
    sample_dir: Path,
    info: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    header: Mapping[str, Any],
) -> bool:
    """Write the session trace of one sample and tell whether it has one.

    A native OpenClaw, Codex or Claude Code session is published as it is; an
    OpenClaw session header also gets ``header``, which names the harness and
    gives the session an id that is unique in the run. Any other sample is
    converted from its ACTF.
    """

    raw = sample_dir / "trace.jsonl"
    schema = (info.get("trajectory") or {}).get("raw_schema")
    if schema in NATIVE_TRACES and raw.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        if schema != "openclaw-session-jsonl":
            shutil.copyfile(raw, path)
            return True
        sessions = 0
        with (
            raw.open(encoding="utf-8", newline="") as reader,
            path.open("w", encoding="utf-8", newline="") as writer,
        ):
            for line in reader:
                if '"session"' in line:
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("type") == "session":
                        # Roller starts every OpenClaw session under the same id.
                        sessions += 1
                        suffix = f"#{sessions}" if sessions > 1 else ""
                        row = {**row, **header, "id": header["id"] + suffix}
                        line = json.dumps(row, ensure_ascii=False) + "\n"
                writer.write(line)
        return True
    if not trajectory.get("steps"):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for row in session_trace(trajectory, header):
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def sample_row(
    run_name: str,
    run_root: Path,
    sample_dir: Path,
    trace_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Describe one sample as a table row.

    With ``trace_path``, also write the session of the sample there and record
    the path, relative to ``root``, in the row.
    """

    info = _json(sample_dir / "info.json")
    bench, score, grader_error = _grade(sample_dir, info)
    metadata = bench.get("metadata") or {}
    trajectory = _json(sample_dir / "trace_ACTF.json")
    steps = trajectory.get("steps") or []
    sft = _json(sample_dir / "trace_SFT.json")
    usage = info.get("usage") or {}
    task_id = str(info.get("task_id") or sample_dir.name)
    model = info.get("model") or sample_dir.parent.name
    sample = sample_dir.relative_to(run_root).as_posix()
    results = sample_dir / "results"
    trace = None
    if trace_path is not None and root is not None:
        header = {
            "harness": info.get("harness"),
            "id": f"{run_name}/{sample_dir.name}",
            "name": f"{task_id} · {model}",
            "model": model,
            "roller": {
                "run": run_name,
                "sample": sample,
                "task_id": task_id,
                "status": info.get("status") or "partial",
                "score": score,
            },
        }
        if _write_trace(trace_path, sample_dir, info, trajectory, header):
            trace = trace_path.relative_to(root).as_posix()
    return {
        "run": run_name,
        "sample": sample,
        "trace": trace,
        "task_id": task_id,
        "upstream_task_id": str(metadata.get("upstream_task_id") or task_id),
        "run_index": int(metadata.get("run_index") or 1),
        "bench": bench.get("bench"),
        "category": bench.get("category"),
        "model": model,
        "harness": info.get("harness"),
        "status": info.get("status") or "partial",
        "error": info.get("error"),
        "score": score,
        "grader_error": grader_error,
        "steps": len(steps),
        "tool_calls": sum(
            len((step.get("assistant_content") or {}).get("tool_calls") or [])
            for step in steps
        ),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "elapsed_seconds": float(
            (info.get("timing") or {}).get("elapsed_seconds") or 0.0
        ),
        "files": sorted(
            path.relative_to(results).as_posix()
            for path in results.rglob("*")
            if path.is_file()
        ),
        "messages": json.dumps(sft["messages"], ensure_ascii=False) if sft else None,
        "tools": json.dumps(sft["tools"], ensure_ascii=False) if sft else None,
    }


def _parts(run_dir: Path) -> dict[str, list[Path]]:
    """Group the sample directories of a run by Harness and model.

    A part is named ``<harness>-<model>``, with the model id without its
    provider: ``openclaw-glm-5`` for ``vllm/glm-5`` on OpenClaw. Parts of a run
    that share that name are named after the whole model reference.
    """

    groups: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(run_dir.glob("samples/*/*/info.json")):
        info = _json(path)
        harness = str(info.get("harness") or "agent")
        model = str(info.get("model") or path.parent.parent.name)
        groups.setdefault((harness, model), []).append(path.parent)
    names = {
        key: safe_component(f"{key[0]}-{key[1].rsplit('/', 1)[-1]}") for key in groups
    }
    shared = Counter(names.values())
    return {
        names[key]
        if shared[names[key]] == 1
        else safe_component(f"{key[0]}-{key[1]}"): samples
        for key, samples in sorted(groups.items())
    }


def _rewrite_run_files(run_root: Path, had_report: bool) -> None:
    """Make the run-level files of a part describe the samples it holds.

    ``info.json``, ``sft.jsonl`` and ``sft_segments.jsonl`` are written by the
    code that writes them during collection, and ``bench-report.json`` by the
    code of ``roller bench report``.
    """

    results = sorted(
        (_json(path) for path in run_root.glob("samples/*/*/info.json")),
        key=lambda result: result.get("request_index") or 0,
    )
    before = _json(run_root / "info.json")
    collector = Collector(CollectionConfig(run_dir=run_root), lambda slot: None)
    collector._write_run_info(
        results, total_requests=len(results), state=before.get("state") or "finished"
    )
    info = _json(run_root / "info.json")
    for key in ("updated_at", "finished_at"):
        if key in before:
            info[key] = before[key]
        else:
            info.pop(key, None)
    (run_root / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    collector._write_sft_dataset(results)
    report = run_root / "bench-report.json"
    report.unlink(missing_ok=True)
    if had_report and any((r.get("runtime") or {}).get("bench") for r in results):
        from roller.cli import _bench_report  # imported here: cli imports this module

        with contextlib.redirect_stdout(io.StringIO()):
            _bench_report(SimpleNamespace(run_dir=str(run_root), output=str(report)))


def stats_table(rows: Iterable[Mapping[str, Any]]) -> str:
    """Summarize the samples of each part, a Harness and a model, as Markdown.

    The mean score divides by every sample of the model, so a sample without a
    score counts as zero, as it does in ``roller bench report``.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["run"], []).append(row)
    graded = any(
        row["score"] is not None or row["grader_error"]
        for group in groups.values()
        for row in group
    )
    header = ["Trajectories", "Harness", "Model", "Samples", "Completed"]
    if graded:
        header += ["Scored", "Mean score"]
    header += ["Mean steps", "Total tokens"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| --- | --- | --- |" + " ---: |" * (len(header) - 3),
    ]
    for run, group in sorted(groups.items()):
        scores = [row["score"] for row in group if row["score"] is not None]
        cells = [
            f"`{run}`",
            str(group[0]["harness"] or ""),
            f"`{str(group[0]['model']).rsplit('/', 1)[-1]}`",
            str(len(group)),
            str(sum(row["status"] == "ok" for row in group)),
        ]
        if graded:
            cells += [str(len(scores)), f"{sum(scores) / len(group):.3f}"]
        cells += [
            f"{sum(row['steps'] for row in group) / len(group):.1f}",
            f"{sum(row['total_tokens'] for row in group):,}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tables(output: Path) -> Iterator[dict[str, Any]]:
    """Read the rows of every part exported into a dataset folder."""

    for table in sorted((output / DATA_DIR).glob("*.jsonl")):
        with table.open(encoding="utf-8") as reader:
            for line in reader:
                yield json.loads(line)


def _write_card(output: Path, name: str, part: str, table: str) -> None:
    """Create the dataset card, or refresh the statistics of an existing one."""

    card = output / "README.md"
    stats = f"{STATS_START}\n{table}\n{STATS_END}"
    if not card.is_file():
        text = _CARD.format(
            name=name, part=part, data_dir=DATA_DIR, traces_dir=TRACES_DIR, stats=stats
        )
    else:
        text = card.read_text(encoding="utf-8")
        start, end = text.find(STATS_START), text.find(STATS_END)
        if start < 0 or end < start:
            return
        text = text[:start] + stats + text[end + len(STATS_END) :]
    card.write_text(text, encoding="utf-8")


def export_run(
    run_dir: Path,
    output: Path,
    rules: Rules | None = None,
    environ: Mapping[str, str] | None = None,
    redact: bool = True,
) -> tuple[list[dict[str, Any]], Redaction]:
    """Write a run into a dataset folder; return its table rows and what was redacted.

    The run is split into one part per Harness and model. A part named
    ``<harness>-<model>`` gets ``runs/<part>/``, a Roller run directory with the
    samples of that model, ``traces/<part>/`` and ``data/<part>.jsonl``, so the
    folder can hold the parts of further runs; exporting a part again replaces
    these three. The statistics of the card cover every part in the folder.

    With ``redact``, the copied files have the endpoints, host paths and
    credential values of the collection replaced, see ``environment_redaction``;
    ``environ`` supplies the credential values, ``os.environ`` by default.
    ``rules`` are applied in either case.

    Raises ``ValueError`` when a forbidden pattern is left in a copied file.
    """

    run_dir, output = run_dir.resolve(), output.resolve()
    derived, redaction = (
        environment_redaction(run_dir, os.environ if environ is None else environ)
        if redact
        else (Rules(), Redaction())
    )
    extra = rules or Rules()
    rules = Rules(derived.replace + extra.replace, extra.exclude, extra.forbid)
    parts = _parts(run_dir)
    shared = [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and output != path
        and output not in path.parents
        and (run_dir / "samples") not in path.parents
    ]
    found: list[str] = []
    for name, samples in parts.items():
        for stale in (output / "runs" / name, output / TRACES_DIR / name):
            if stale.exists():
                shutil.rmtree(stale)
        files = shared + [
            path
            for sample in samples
            for path in sorted(sample.rglob("*"))
            if path.is_file()
        ]
        found += _copy_files(run_dir, files, output / "runs" / name, rules)
    if found:
        shown = ", ".join(sorted(set(found))[:20])
        raise ValueError(f"{len(found)} lines still match a forbidden pattern: {shown}")
    rows: list[dict[str, Any]] = []
    for name in parts:
        destination = output / "runs" / name
        _rewrite_run_files(destination, (run_dir / "bench-report.json").is_file())
        part = [
            sample_row(
                name,
                destination,
                path.parent,
                output / TRACES_DIR / name / f"{path.parent.name}.jsonl",
                output,
            )
            for path in sorted(destination.glob("samples/*/*/info.json"))
        ]
        table = output / DATA_DIR / f"{name}.jsonl"
        table.parent.mkdir(parents=True, exist_ok=True)
        with table.open("w", encoding="utf-8") as writer:
            for row in part:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows += part
    if parts:
        _write_card(
            output, run_dir.name, next(iter(parts)), stats_table(_tables(output))
        )
    return rows, redaction
