"""Export ACTF trajectories as ATIF, the trajectory format Harbor reads.

ATIF is specified in https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roller.benches.report import read_scores
from roller.sft import _image_data_uri
from roller.storage import atomic_write_json

# The newest version this export needs: v1.7 adds ``extra`` on tool calls and
# observation results, and v1.8 only adds audio parts.
ATIF_SCHEMA_VERSION = "ATIF-v1.7"
TRAJECTORY_FILE = "trajectory.json"

_IMAGE_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_METRIC_FIELDS = {
    "prompt_tokens_len": "prompt_tokens",
    "completion_tokens_len": "completion_tokens",
    # Claude Code and Codex ACTF spell the cache-read counter differently.
    "cache_read_tokens": "cached_tokens",
    "cache_read_tokens_len": "cached_tokens",
}
_TOTAL_FIELDS = {
    "prompt_tokens": "total_prompt_tokens",
    "completion_tokens": "total_completion_tokens",
    "cached_tokens": "total_cached_tokens",
}


def _timestamp(value: Any) -> str | None:
    """An ACTF time (``YYYY-MM-DD HH:MM:SS.ffffff``, UTC) as ISO 8601."""

    if not isinstance(value, str) or not value:
        return None
    parsed = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def _image_part(block: Mapping[str, Any], images: dict[str, bytes]) -> dict[str, Any]:
    """Move one inline ACTF image to a file referenced by its content hash."""

    header, _, payload = _image_data_uri(block, "image").partition(",")
    media_type = header[len("data:") : -len(";base64")]
    data = base64.b64decode(payload)
    path = (
        f"images/{hashlib.sha256(data).hexdigest()[:16]}{_IMAGE_SUFFIXES[media_type]}"
    )
    images[path] = data
    return {"type": "image", "source": {"media_type": media_type, "path": path}}


def _content(value: Any, images: dict[str, bytes]) -> str | list[dict[str, Any]]:
    """ACTF content as an ATIF message: a string, or content parts when it has images."""

    if isinstance(value, str):
        return value
    parts: list[dict[str, Any]] = []
    for block in value:
        if isinstance(block, str):
            parts.append({"type": "text", "text": block})
        elif isinstance(block, Mapping) and block.get("type") == "image":
            parts.append(_image_part(block, images))
        elif (
            isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            parts.append({"type": "text", "text": block["text"]})
        else:
            # ATIF parts are text, image or audio; any other block keeps its JSON.
            parts.append(
                {"type": "text", "text": json.dumps(block, ensure_ascii=False)}
            )
    if all(part["type"] == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


def _tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    fields = function if isinstance(function, Mapping) else call
    arguments = fields.get("arguments", {})
    result: dict[str, Any] = {
        "tool_call_id": str(call.get("id", "")),
        "function_name": str(fields.get("name", "")),
        "arguments": dict(arguments)
        if isinstance(arguments, Mapping)
        else {"value": arguments},
    }
    extra = {
        key: value
        for key, value in call.items()
        if key not in {"type", "id", "name", "arguments", "function"}
    }
    if extra:
        result["extra"] = extra
    return result


def _metrics(metric: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in metric.items():
        if key in _METRIC_FIELDS:
            result[_METRIC_FIELDS[key]] = value
        elif value is not None and value != "":
            extra[key] = value
    if extra:
        result["extra"] = extra
    return result


def _observation(
    rows: list[Mapping[str, Any]], call_ids: set[str], images: dict[str, bytes]
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for row in rows:
        call_id = row.get("toolCallId")
        # ATIF only lets a result point at a call of its own step.
        linked = call_id in call_ids
        result: dict[str, Any] = {"source_call_id": call_id} if linked else {}
        result["content"] = _content(row.get("content", []), images)
        extra = {
            key: value
            for key, value in row.items()
            if key not in {"role", "content"} and not (linked and key == "toolCallId")
        }
        if extra:
            result["extra"] = extra
        results.append(result)
    return {"results": results}


def actf_to_atif(
    trajectory: Mapping[str, Any],
    *,
    agent_name: str,
    agent_version: str,
    model_name: str | None = None,
    session_id: str | None = None,
    cost_usd: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Convert one ACTF trajectory to ATIF.

    Returns the ATIF document and its image files, keyed by their path relative
    to the document. The system prompt becomes a ``system`` step; every ACTF
    step becomes a ``user`` step, when it has user input, and an ``agent`` step
    carrying the reply, its tool calls and their results.
    """

    images: dict[str, bytes] = {}
    steps: list[dict[str, Any]] = []
    system_prompt = trajectory.get("system_prompt")
    if system_prompt:
        steps.append({"step_id": 1, "source": "system", "message": system_prompt})

    for row in trajectory["steps"]:
        started_at = _timestamp(row.get("started_at"))
        time = {"timestamp": started_at} if started_at else {}
        user = _content(row.get("user_content", ""), images)
        if user:
            steps.append(
                {"step_id": len(steps) + 1, **time, "source": "user", "message": user}
            )

        assistant = row["assistant_content"]
        step_extra = dict(row.get("extra") or {})
        step: dict[str, Any] = {"step_id": len(steps) + 1, **time, "source": "agent"}
        if isinstance(step_extra.get("model"), str):
            step["model_name"] = step_extra["model"]
        step["message"] = assistant.get("content", "")
        if assistant.get("reasoning_content"):
            step["reasoning_content"] = assistant["reasoning_content"]
        calls = [_tool_call(call) for call in assistant.get("tool_calls", [])]
        if calls:
            step["tool_calls"] = calls
        if row.get("observation"):
            step["observation"] = _observation(
                row["observation"], {call["tool_call_id"] for call in calls}, images
            )
        if row.get("metric"):
            step["metrics"] = _metrics(row["metric"])
        finished_at = _timestamp(row.get("finished_at"))
        if finished_at:
            step_extra["finished_at"] = finished_at
        if step_extra:
            step["extra"] = step_extra
        steps.append(step)

    if not steps:
        raise ValueError("trajectory has no steps")

    agent: dict[str, Any] = {"name": agent_name, "version": agent_version}
    if model_name:
        agent["model_name"] = model_name
    if trajectory.get("tools"):
        agent["tool_definitions"] = trajectory["tools"]
    if trajectory.get("tools_format"):
        agent["extra"] = {"tools_format": trajectory["tools_format"]}

    final_metrics: dict[str, Any] = {}
    for field, total in _TOTAL_FIELDS.items():
        values = [
            step["metrics"][field] for step in steps if field in step.get("metrics", {})
        ]
        if values:
            final_metrics[total] = sum(values)
    if cost_usd is not None:
        final_metrics["total_cost_usd"] = cost_usd
    final_metrics["total_steps"] = len(steps)

    document: dict[str, Any] = {"schema_version": ATIF_SCHEMA_VERSION}
    if session_id:
        document["session_id"] = session_id
    document.update(agent=agent, steps=steps, final_metrics=final_metrics)
    if extra:
        document["extra"] = dict(extra)
    return document, images


def write_atif(
    directory: Path, document: Mapping[str, Any], images: Mapping[str, bytes]
) -> Path:
    """Write ``trajectory.json`` and the image files it references."""

    for relative, data in images.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    destination = directory / TRAJECTORY_FILE
    atomic_write_json(destination, document)
    return destination


def export_sample(run_dir: Path, sample_dir: Path, directory: Path) -> Path:
    """Export the Task Agent trajectory of one finalized sample."""

    info = json.loads((sample_dir / "info.json").read_text(encoding="utf-8"))
    trajectory = json.loads(
        (sample_dir / "trace_ACTF.json").read_text(encoding="utf-8")
    )
    harness = info["harness"]
    runtime = info.get("runtime") or {}
    usage = info.get("usage") or {}
    roller = {
        key: info[key]
        for key in ("task_id", "sample_index", "status", "error")
        if info.get(key) is not None
    }
    scores = read_scores(sample_dir)
    if scores is not None:
        roller["scores"] = scores
    if trajectory.get("sft_unavailable_reason"):
        roller["sft_unavailable_reason"] = trajectory["sft_unavailable_reason"]
    document, images = actf_to_atif(
        trajectory,
        agent_name=harness,
        agent_version=str(
            runtime.get(f"{harness.replace('-', '_')}_version") or "unknown"
        ),
        model_name=info.get("model"),
        session_id=f"{run_dir.name}/{sample_dir.relative_to(run_dir / 'samples').as_posix()}",
        cost_usd=usage.get("cost_usd") or None,
        extra={"roller": roller},
    )
    return write_atif(directory, document, images)


def export_run(run_dir: Path, output: Path) -> tuple[list[Path], dict[str, str]]:
    """Export every sample of a run that has an ACTF trajectory.

    Returns the written files and, for each sample that was not exported, the
    reason keyed by its path relative to the run.
    """

    written: list[Path] = []
    skipped: dict[str, str] = {}
    samples = run_dir / "samples"
    for info_path in sorted(samples.glob("*/*/info.json")):
        sample_dir = info_path.parent
        relative = sample_dir.relative_to(run_dir).as_posix()
        if not (sample_dir / "trace_ACTF.json").is_file():
            skipped[relative] = "no trace_ACTF.json"
            continue
        try:
            written.append(
                export_sample(
                    run_dir, sample_dir, output / sample_dir.relative_to(samples)
                )
            )
        except ValueError as exc:
            skipped[relative] = str(exc)
    return written, skipped
