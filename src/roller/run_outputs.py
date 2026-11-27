"""Run-level summary and dataset writers shared by collection, import, and export."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roller.capture import read_jsonl, validate_capture
from roller.sft_dataset import write_sft_dataset
from roller.storage import atomic_write_json, safe_component, write_jsonl
from roller.trajectory_audit import training_summary


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_dir_from_result(run_dir: Path, result: Mapping[str, Any]) -> Path:
    model = safe_component(
        str(result.get("model_label", result.get("model", "unknown")))
    )
    source_index = int(result.get("source_index", result.get("request_index", 0)))
    task = safe_component(str(result.get("task_id", "unknown")))
    task_dir = f"{source_index:08d}_{task}"
    sample_count = int(result.get("sample_count", 1))
    if sample_count > 1:
        sample_index = int(result.get("sample_index", 0))
        task_dir += f"__sample_{sample_index + 1:03d}"
    return run_dir / "samples" / model / task_dir


def usage_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }
    for result in results:
        usage = result.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key, 0)
            if isinstance(value, int | float) and not isinstance(value, bool):
                totals[key] += value
    totals["cost_usd"] = round(float(totals["cost_usd"]), 6)
    return totals


def write_run_info(
    run_dir: Path,
    results: list[dict[str, Any]],
    *,
    total_requests: int,
    state: str,
) -> None:
    info_path = run_dir / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        info = {"schema_version": "roller.run/v1"}
    summary = {
        "total": total_requests,
        "processed": len(results),
        "pending": max(0, total_requests - len(results)),
        "ok": sum(result.get("status") == "ok" for result in results),
        "error": sum(result.get("status") == "error" for result in results),
        "timeout": sum(result.get("status") == "timeout" for result in results),
        "skipped": sum(result.get("status") == "skipped" for result in results),
        "usage": usage_summary(results),
        "training": training_summary(results),
    }
    summary["completed"] = summary["ok"] + summary["skipped"]
    summary["failed"] = summary["error"] + summary["timeout"]
    info.update(
        {
            "state": state,
            "summary": summary,
            "samples": [
                {
                    "request_index": result.get("request_index"),
                    "sample_index": result.get("sample_index", 0),
                    "sample_count": result.get("sample_count", 1),
                    "task_id": result.get("task_id"),
                    "model": result.get("model"),
                    "model_label": result.get("model_label", result.get("model")),
                    "status": result.get("status"),
                    "error": result.get("error"),
                    "path": sample_dir_from_result(run_dir, result)
                    .relative_to(run_dir)
                    .as_posix(),
                    "usage": result.get("usage", {}),
                }
                for result in results
            ],
            "updated_at": _now(),
        }
    )
    if state == "finished":
        info["finished_at"] = info["updated_at"]
    else:
        info.pop("finished_at", None)
    atomic_write_json(info_path, info)


def write_run_dataset(
    run_dir: Path,
    results: Sequence[dict[str, Any]],
    *,
    training_files_only: bool = False,
) -> None:
    # With --training-files-only, roller export --format sft writes them.
    if not training_files_only:
        write_sft_dataset(
            run_dir,
            run_dir,
            [(sample_dir_from_result(run_dir, result), result) for result in results],
        )

    def calls() -> Iterator[dict[str, Any]]:
        for result in results:
            capture = result.get("runtime", {}).get("capture")
            if not isinstance(capture, dict) or not capture.get("manifest"):
                continue
            sample = sample_dir_from_result(run_dir, result)
            validate_capture(sample, capture)
            manifest_path = sample / capture["manifest"]
            for call in read_jsonl(manifest_path.parent / "calls.jsonl"):
                if call.get("export_status") == "available":
                    # The request and response stay in the sample's capture.
                    call.pop("context")
                    call.pop("output")
                    yield {
                        **call,
                        "schema_version": "roller.call-index/v1",
                        "sample": sample.relative_to(run_dir).as_posix(),
                        "sample_status": result["status"],
                        "sample_capture_status": capture.get("capture_status"),
                        "capture_manifest": capture["manifest"],
                        "task_id": result["task_id"],
                        "model_label": result.get("model_label"),
                        "sample_index": result.get("sample_index", 0),
                    }

    # A failed conversation can still contain valid individual HTTP calls.
    # SFT eligibility is decided separately for the whole conversation.
    if any(isinstance(r.get("runtime", {}).get("capture"), dict) for r in results):
        write_jsonl(run_dir / "calls.jsonl", calls())
