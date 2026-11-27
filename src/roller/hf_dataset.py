"""Package a Roller run as a Hugging Face dataset folder.

A run is written as one part per Harness and model, named ``<harness>-<model>``.
Each part holds the session of every sample for the trace viewer of the Hub and
one JSON Lines table with a row per sample. On request it also holds a Roller
run directory with the samples of that model, which ``roller view`` opens after
a download. Every part has its own paths, so further runs can be exported into
the same folder.

The export replaces what is private to the collection environment: the model
endpoints the run recorded, the hosts its ``NO_PROXY`` named, the name of the
machine, the host paths of the run, and the values of the credentials it passed
to its containers.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import roller.export_audit as export_audit
from roller.benches.report import SCORE_FILE, read_scores, sample_score
from roller.benches.run_report import generate_bench_report
from roller.hf_redaction import (
    _USER_LENGTH,
    Redaction,
    Rules,
    _copy_files,
    _export_path,
    _json,
    environment_redaction,
    machine_name,
)
from roller.run_outputs import write_run_dataset, write_run_info
from roller.sft import SFTConversionError, _image_data_uri
from roller.storage import safe_component

DATA_DIR = "data"
TRACES_DIR = "traces"
# The Hub splits traces with str.splitlines(), which treats these JSON string
# characters as line breaks even though they are part of a JSONL record.
_TRACE_LINE_BREAKS = re.compile("[\\u0085\\u2028\\u2029]")
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
dataset_info:
- config_name: trajectories
  features:
  - {{name: run, dtype: string}}
  - {{name: sample, dtype: string}}
  - {{name: trace, dtype: string}}
  - {{name: task_id, dtype: string}}
  - {{name: upstream_task_id, dtype: string}}
  - {{name: run_index, dtype: int64}}
  - {{name: bench, dtype: string}}
  - {{name: category, dtype: string}}
  - {{name: model, dtype: string}}
  - {{name: harness, dtype: string}}
  - {{name: status, dtype: string}}
  - {{name: error, dtype: string}}
  - {{name: score, dtype: float64}}
  - {{name: grader_error, dtype: string}}
  - {{name: steps, dtype: int64}}
  - {{name: tool_calls, dtype: int64}}
  - {{name: input_tokens, dtype: int64}}
  - {{name: output_tokens, dtype: int64}}
  - {{name: total_tokens, dtype: int64}}
  - {{name: elapsed_seconds, dtype: float64}}
  - {{name: files, sequence: string}}
  - {{name: messages, dtype: string}}
  - {{name: tools, dtype: string}}
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

Replace `OWNER/NAME` with the id of this dataset repository.
{browse}
## Files

| Path | Content |
| --- | --- |
| `{traces_dir}/<harness>-<model>/<index>_<task>.jsonl` | Agent session of one sample, which the Hub opens in its [trace viewer](https://huggingface.co/docs/hub/agent-traces) |
| `{data_dir}/<harness>-<model>.jsonl` | One row per sample: task, model, status, score, steps, tokens, `messages`, `tools` |
{run_files}"""

_CARD_BROWSE = """
## Browse the trajectories

`runs/{part}/` is a Roller run directory. Download it and open it in the Roller
Viewer to read every step of a sample:

```bash
hf download OWNER/NAME --repo-type dataset --include "runs/{part}/*" --local-dir {name}
uvx --from roller-agent roller view {name}/runs/{part}
```
"""

_CARD_RUN_FILES = """\
| `runs/<harness>-<model>/info.json` | Run configuration, progress and total usage |
| `runs/<harness>-<model>/sft.jsonl` | SFT records of the successful samples |
| `runs/<harness>-<model>/sft_index.jsonl` | Task, model and score for each SFT record |
| `runs/<harness>-<model>/samples/<model>/<index>_<task>/` | Native session, unified trajectory, SFT record, scores and output files of one sample |

The files of a sample are described in the
[trajectory outputs guide](https://valen-team.github.io/roller/traces/trajectory-outputs/).
"""

# The files of a sample that its session and its table row are built from.
_SAMPLE_FILES = (
    "info.json",
    SCORE_FILE,
    "trace.jsonl",
    "trace_ACTF.json",
    "trace_SFT.json",
)


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


def hermes_session(raw: Path, header: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a Hermes journal to one session as ``hermes sessions export`` writes it.

    The Hub reads such a line as a Hermes trace. ``header`` gives the session
    ``id`` and further metadata. The messages are the journaled ones, from
    before Hermes compressed its history; the system prompt is the one of the
    first request.
    """

    system_prompt, messages, times = "", [], []
    with raw.open(encoding="utf-8") as reader:
        for line in reader:
            event = json.loads(line)
            times.append(datetime.fromisoformat(event["timestamp"]).timestamp())
            if event["type"] == "request" and not system_prompt:
                system_prompt = next(
                    (
                        message["content"]
                        for message in event["messages"]
                        if message["role"] == "system"
                    ),
                    "",
                )
            elif event["type"] == "message":
                message = {**event["message"], "timestamp": times[-1]}
                if event.get("finish_reason"):
                    message["finish_reason"] = event["finish_reason"]
                messages.append(message)
    return {
        **header,
        "source": "cli",
        "system_prompt": system_prompt,
        "started_at": times[0],
        "ended_at": times[-1],
        "message_count": len(messages),
        "tool_call_count": sum(len(m.get("tool_calls") or []) for m in messages),
        "messages": messages,
    }


def _trace_line(line: str) -> str:
    return _TRACE_LINE_BREAKS.sub(lambda match: f"\\u{ord(match[0]):04x}", line)


def _write_trace(
    path: Path,
    sample_dir: Path,
    info: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    header: Mapping[str, Any],
) -> bool:
    """Write the session trace of one sample and tell whether it has one.

    Preserve native OpenClaw, Codex and Claude Code records, escaping Unicode
    line separators for the Hub. An OpenClaw session header also gets
    ``header``, which names the harness and gives the session an id that is
    unique in the run. Write a Hermes journal as a Hermes session. Convert
    other samples from ACTF. DeepSeek and OpenCode samples without ACTF keep
    their native journals.
    """

    raw = sample_dir / "trace.jsonl"
    schema = (info.get("trajectory") or {}).get("raw_schema")
    opencode_native = schema == "roller.opencode-events/v1" and not trajectory.get(
        "steps"
    )
    preserve_native = (
        schema in NATIVE_TRACES
        or (schema == "roller.deepseek-events/v1" and not trajectory.get("steps"))
        or opencode_native
    )
    if preserve_native and raw.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        sessions = 0
        with (
            raw.open(encoding="utf-8", newline="") as reader,
            path.open("w", encoding="utf-8", newline="") as writer,
        ):
            if opencode_native:
                envelope = {**header, "type": "session", "source_schema": schema}
                writer.write(
                    _trace_line(json.dumps(envelope, ensure_ascii=False)) + "\n"
                )
            for line in reader:
                if schema == "openclaw-session-jsonl" and '"session"' in line:
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("type") == "session":
                        # Roller starts every OpenClaw session under the same id.
                        sessions += 1
                        suffix = f"#{sessions}" if sessions > 1 else ""
                        row = {**row, **header, "id": header["id"] + suffix}
                        line = json.dumps(row, ensure_ascii=False) + "\n"
                writer.write(_trace_line(line))
        return True
    if schema == "roller.hermes-events/v1" and raw.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        session = json.dumps(hermes_session(raw, header), ensure_ascii=False)
        path.write_text(_trace_line(session) + "\n", encoding="utf-8")
        return True
    if not trajectory.get("steps"):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for row in session_trace(trajectory, header):
            writer.write(_trace_line(json.dumps(row, ensure_ascii=False)) + "\n")
    return True


def sample_row(
    run_name: str,
    run_root: Path,
    sample_dir: Path,
    trace_path: Path | None = None,
    root: Path | None = None,
    results: Path | None = None,
) -> dict[str, Any]:
    """Describe one sample as a table row.

    With ``trace_path``, also write the session of the sample there and record
    the path, relative to ``root``, in the row. ``results`` is the directory
    whose file names fill ``files``, the ``results`` of the sample by default.
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
    results = results or sample_dir / "results"
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

    ``info.json``, ``sft.jsonl``, ``sft_index.jsonl`` and ``sft_segments.jsonl`` are written by the
    code that writes them during collection, and ``bench-report.json`` by the
    code of ``roller bench report``.
    """

    results = sorted(
        (_json(path) for path in run_root.glob("samples/*/*/info.json")),
        key=lambda result: result.get("request_index") or 0,
    )
    before = _json(run_root / "info.json")
    write_run_info(
        run_root,
        results,
        total_requests=len(results),
        state=before.get("state") or "finished",
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
    write_run_dataset(run_root, results)
    report = run_root / "bench-report.json"
    report.unlink(missing_ok=True)
    if had_report and any((r.get("runtime") or {}).get("bench") for r in results):
        generate_bench_report(run_root, output=report)


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
    # An endpoint that reports no usage leaves every count at zero.
    metered = any(row["total_tokens"] for group in groups.values() for row in group)
    header = ["Trajectories", "Harness", "Model", "Samples", "Completed"]
    if graded:
        header += ["Scored", "Mean score"]
    header += ["Mean steps"] + ["Total tokens"] * metered
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
        cells.append(f"{sum(row['steps'] for row in group) / len(group):.1f}")
        if metered:
            cells.append(f"{sum(row['total_tokens'] for row in group):,}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tables(output: Path) -> Iterator[dict[str, Any]]:
    """Read the rows of every part exported into a dataset folder."""

    for table in sorted((output / DATA_DIR).glob("*.jsonl")):
        with table.open(encoding="utf-8") as reader:
            for line in reader:
                yield json.loads(line)


def _write_card(output: Path, name: str, part: str, table: str, runs: bool) -> None:
    """Create the dataset card, or refresh the statistics of an existing one."""

    card = output / "README.md"
    stats = f"{STATS_START}\n{table}\n{STATS_END}"
    if not card.is_file():
        text = _CARD.format(
            name=name,
            data_dir=DATA_DIR,
            traces_dir=TRACES_DIR,
            stats=stats,
            browse=_CARD_BROWSE.format(name=name, part=part) if runs else "",
            run_files=_CARD_RUN_FILES if runs else "",
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
    runs: bool = False,
    machine: str | None = None,
    preserve_task_keys: bool = False,
) -> tuple[list[dict[str, Any]], Redaction]:
    """Write a run into a dataset folder; return its table rows and what was redacted.

    The run is split into one part per Harness and model. A part named
    ``<harness>-<model>`` gets ``traces/<part>/`` and ``data/<part>.jsonl``, so
    the folder can hold the parts of further runs; exporting a part again
    replaces them. With ``runs``, a part also gets ``runs/<part>/``, a Roller run
    directory with every file of the samples of that model. The statistics of
    the card cover every part in the folder.

    With ``redact``, the exported files have the endpoints, hosts, host paths
    and credential values of the collection replaced, see
    ``environment_redaction``; ``environ`` supplies the credential values,
    ``os.environ`` by default, and ``machine`` the name of the collection
    machine, the name of this machine by default. ``rules`` are applied in
    either case.

    ``preserve_task_keys`` skips credential-shape replacements for benchmarks
    with deliberate key fixtures. Known environment credentials and collection
    paths are still replaced when ``redact`` is enabled.

    Raises ``ValueError`` when a forbidden pattern is left in an exported file.
    """

    derived, redaction = (
        environment_redaction(
            run_dir,
            os.environ if environ is None else environ,
            machine_name() if machine is None else machine,
        )
        if redact
        else (Rules(), Redaction())
    )
    run_dir, output = run_dir.resolve(), output.resolve()
    extra = rules or Rules()
    rules = Rules(
        derived.replace
        + (export_audit.replacements() if redact and not preserve_task_keys else ())
        + extra.replace,
        extra.exclude,
        extra.forbid,
    )
    parts: dict[str, list[Path]] = {}
    source_samples: dict[str, Path] = {}
    for name, samples in _parts(run_dir).items():
        for pattern, replacement in rules.replace:
            name = pattern.sub(replacement, name)
        name = safe_component(name)
        parts.setdefault(name, []).extend(samples)
        source_samples.update(
            {
                _export_path(sample.relative_to(run_dir).as_posix(), rules): sample
                for sample in samples
            }
        )
    shared = [
        path
        for path in sorted(run_dir.rglob("*"))
        if runs
        and path.is_file()
        and output != path
        and output not in path.parents
        and (run_dir / "samples") not in path.parents
        and path.relative_to(run_dir).parts[0]
        not in {
            "atif",
            "hf",
            "sft",
            "tokens",
            "xtuner",
        }
        and not path.name.endswith(".redaction-report.json")
    ]
    with contextlib.ExitStack() as stack:
        # The sessions and the tables are built from redacted copies. Without
        # ``runs`` the copies hold only the files these need and are not kept.
        copies = (
            output / "runs"
            if runs
            else Path(stack.enter_context(tempfile.TemporaryDirectory()))
        )
        found: list[str] = []
        for name, samples in parts.items():
            for stale in (output / "runs" / name, output / TRACES_DIR / name):
                if stale.exists():
                    shutil.rmtree(stale)
            files = shared + [
                path
                for sample in samples
                for path in (
                    sorted(sample.rglob("*"))
                    if runs
                    else [sample / file for file in _SAMPLE_FILES]
                )
                if path.is_file()
            ]
            found += _copy_files(run_dir, files, copies / name, rules)
        if found:
            shown = ", ".join(sorted(set(found))[:20])
            raise ValueError(
                f"{len(found)} lines still match a forbidden pattern: {shown}"
            )
        rows: list[dict[str, Any]] = []
        for name in parts:
            destination = copies / name
            if runs:
                _rewrite_run_files(
                    destination, (run_dir / "bench-report.json").is_file()
                )
            part = [
                sample_row(
                    name,
                    destination,
                    path.parent,
                    output / TRACES_DIR / name / f"{path.parent.name}.jsonl",
                    output,
                    None
                    if runs
                    else source_samples[path.parent.relative_to(destination).as_posix()]
                    / "results",
                )
                for path in sorted(destination.glob("samples/*/*/info.json"))
            ]
            table = output / DATA_DIR / f"{name}.jsonl"
            table.parent.mkdir(parents=True, exist_ok=True)
            with table.open("w", encoding="utf-8") as writer:
                for row in part:
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += part
    if not runs:
        # An earlier export with ``runs`` may have left the directory empty.
        with contextlib.suppress(OSError):
            (output / "runs").rmdir()
    if parts:
        _write_card(
            output, run_dir.name, next(iter(parts)), stats_table(_tables(output)), runs
        )
    if output.is_dir():
        names = [user for user in redaction.users if len(user) >= _USER_LENGTH]
        export_audit.write_report(output, names)
    return rows, redaction
