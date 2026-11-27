"""Integrity checks for finalized samples considered for resume."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roller.capture import validate_capture
from roller.compaction_sft import (
    SFT_SEGMENTS_DIR,
    build_compaction_segments,
    has_main_compaction,
)
from roller.harness_registry import (
    allows_raw_only_actf,
    compaction_mode,
    requires_cache_trace,
)
from roller.models import RunRequest
from roller.sample_metadata import (
    has_legacy_claude_sidechain,
    keeps_thinking,
    trajectory_conversion_unavailable,
)
from roller.sft import actf_to_sft, validate_sft_record
from roller.storage import atomic_write_json


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _has_json_event(path: Path) -> bool:
    """Return whether a harness-native JSONL trace contains one JSON object."""

    try:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    return True
    except OSError as exc:
        raise ValueError(f"{path.name} cannot be read: {exc}") from exc
    return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output_files(
    sample_dir: Path, info: Mapping[str, Any], *, verify_sha256: bool = False
) -> None:
    output = info.get("output_collection")
    if not isinstance(output, Mapping) or not output.get("passed"):
        return
    rows = output.get("files", [])
    if not isinstance(rows, list):
        raise ValueError("info.json output_collection.files must be a list")
    results_root = sample_dir / "results"
    if results_root.is_symlink():
        raise ValueError("sample results/ directory must not be a symlink")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"output_collection.files[{index}] must be an object")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"output_collection.files[{index}].path must be non-empty")
        relative_path = Path(relative)
        if relative_path.is_absolute() or any(
            part in {"", ".", ".."} for part in relative_path.parts
        ):
            raise ValueError(
                f"output_collection.files[{index}].path is not canonical: {relative}"
            )
        unresolved = results_root / relative_path
        current = results_root
        for part in relative_path.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"recorded output path contains a symlink: {relative}")
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(results_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"output_collection.files[{index}].path escapes results/: {relative}"
            ) from exc
        if not candidate.is_file():
            raise ValueError(f"recorded output file is missing: results/{relative}")
        expected_size = row.get("size")
        if isinstance(expected_size, int) and not isinstance(expected_size, bool):
            try:
                actual_size = candidate.stat().st_size
            except OSError as exc:
                raise ValueError(
                    f"recorded output file cannot be inspected: results/{relative}"
                ) from exc
            if actual_size != expected_size:
                raise ValueError(
                    f"recorded output file size changed: results/{relative}"
                )
        expected_sha256 = row.get("sha256")
        if verify_sha256 and isinstance(expected_sha256, str):
            try:
                actual_sha256 = _file_sha256(candidate)
            except OSError as exc:
                raise ValueError(
                    f"recorded output file cannot be read: results/{relative}"
                ) from exc
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"recorded output file content changed: results/{relative}"
                )


def _check_sft_file(
    path: Path, expected: dict[str, Any], mismatch: str, rebuilt: list[Path] | None
) -> None:
    """Compare an SFT file with the record this version derives from the trace.

    A file that differs is an error. When ``rebuilt`` is a list, the file is
    rewritten and added to it.
    """

    try:
        actual = _json_object(path)
        try:
            validate_sft_record(actual)
        except ValueError as exc:
            raise ValueError(f"{path.name} is not a valid SFT record: {exc}") from exc
        if actual != expected:
            raise ValueError(mismatch)
    except ValueError:
        if rebuilt is None:
            raise
        validate_sft_record(expected)
        atomic_write_json(path, expected)
        rebuilt.append(path)


def _validate_trajectory_files(
    base: Path,
    info: Mapping[str, Any],
    *,
    require_wire_trace: bool,
    require_cache_trace: bool,
    rebuild_sft: bool = False,
) -> list[Path]:
    """Validate the trajectory files of one participant.

    SFT files are derived from the trace. With ``rebuild_sft``, one that differs
    from what this version derives from the validated evidence is rewritten
    and returned; without it, the difference is an error.
    """

    raw_path = base / "trace.jsonl"
    actf_path = base / "trace_ACTF.json"
    sft_path = base / "trace_SFT.json"
    segment_dir = base / SFT_SEGMENTS_DIR
    trajectory = info.get("trajectory")
    segments = (
        trajectory.get("sft_segments") if isinstance(trajectory, Mapping) else None
    )
    if segments is not None and (not isinstance(segments, list) or not segments):
        raise ValueError("SFT segment metadata must be a non-empty list")
    if not segments and segment_dir.exists():
        raise ValueError("unreferenced SFT segment directory is present")
    if segments and trajectory.get("sft") is not None:
        raise ValueError("compacted sample cannot also reference a linear SFT")
    if not raw_path.is_file():
        raise ValueError(f"required trajectory file is missing: {raw_path.name}")
    runtime = info.get("runtime")
    conversion_target = trajectory_conversion_unavailable(
        runtime if isinstance(runtime, Mapping) else None
    )
    raw_only = allows_raw_only_actf(info.get("harness")) and conversion_target == "actf"
    sft_unavailable = conversion_target == "sft"
    if segments and sft_unavailable:
        raise ValueError("SFT-unavailable sample cannot reference SFT segments")
    if raw_only:
        if segments:
            raise ValueError("raw-only sample cannot contain SFT segments")
        trajectory = info.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise ValueError("info.json trajectory must be an object")
        if trajectory.get("actf") is not None or trajectory.get("sft") is not None:
            raise ValueError(
                "raw-only trajectory metadata must not reference ACTF/SFT artifacts"
            )
        if actf_path.exists() or sft_path.exists():
            raise ValueError("raw-only sample contains unexpected ACTF/SFT artifacts")
    else:
        if not actf_path.is_file():
            raise ValueError(f"required trajectory file is missing: {actf_path.name}")
        if sft_unavailable:
            trajectory = info.get("trajectory")
            if not isinstance(trajectory, Mapping) or trajectory.get("sft") is not None:
                raise ValueError(
                    "SFT-unavailable trajectory metadata must have null SFT"
                )
            if sft_path.exists():
                raise ValueError(
                    "SFT-unavailable sample contains an unexpected SFT artifact"
                )
        elif not segments and not sft_path.is_file() and not rebuild_sft:
            raise ValueError(f"required trajectory file is missing: {sft_path.name}")
        elif segments and sft_path.exists():
            raise ValueError("compacted sample must not contain a linear SFT")
    if require_wire_trace and not (base / "wire_trace.jsonl").is_file():
        raise ValueError("required full-mode wire_trace.jsonl is missing")
    # Only OpenClaw emits cache traces. Finalized metadata records the actual
    # harness, including CLI overrides, for each participant independently.
    # Keep the strict check for legacy or unrecognized harness metadata.
    cache_trace_required = require_cache_trace and requires_cache_trace(
        info.get("harness")
    )
    if cache_trace_required and not (base / "cache_trace.jsonl").is_file():
        raise ValueError("required full-mode cache_trace.jsonl is missing")
    if not _has_json_event(raw_path):
        raise ValueError("trace.jsonl contains no valid JSON object event")
    capture = runtime.get("capture") if isinstance(runtime, Mapping) else None
    if isinstance(capture, dict):
        validate_capture(base, capture)
    if raw_only:
        return []
    manifest = capture.get("manifest") if isinstance(capture, Mapping) else None
    try:
        # Captured Calls can reveal a rewrite that the native trace did not log.
        expected = (
            build_compaction_segments(
                base,
                info.get("harness"),
                manifest,
                keep_thinking=keeps_thinking(runtime),
            )
            if not sft_unavailable and isinstance(manifest, str)
            else []
        )
        # Like the collector: without Calls, any native reset rules out linear SFT.
        uncaptured_reset = not isinstance(manifest, str) and has_main_compaction(
            raw_path, info.get("harness")
        )
        if not sft_unavailable and not segments and (expected or uncaptured_reset):
            raise ValueError("context reset requires per-call export, not linear SFT")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        AttributeError,
    ) as exc:
        raise ValueError(f"compaction evidence cannot be validated: {exc}") from exc

    rebuilt: list[Path] | None = [] if rebuild_sft else None
    actf = _json_object(actf_path)
    if compaction_mode(info.get("harness")) == "claude" and has_legacy_claude_sidechain(
        actf
    ):
        raise ValueError(
            "legacy Claude Code sidechain steps are mixed into root ACTF/SFT; "
            "the sample cannot be safely reused or silently rerun"
        )
    if actf.get("schema_version") != "ACTF_v1.0":
        raise ValueError("trace_ACTF.json has an unsupported schema_version")
    steps = actf.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trace_ACTF.json steps must be a non-empty list")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping) or step.get("step_id") != index:
            raise ValueError("trace_ACTF.json step_id values must be sequential from 1")

    if segments and not sft_unavailable:
        if not isinstance(manifest, str):
            raise ValueError("SFT segments lack their captured Calls")
        if segments != [metadata for metadata, _ in expected]:
            raise ValueError(
                "SFT segment metadata does not match captured boundaries and Calls"
            )
        expected_paths = {item["path"] for item in segments}
        if not segment_dir.is_dir() or segment_dir.is_symlink():
            raise ValueError("SFT segment directory is missing or invalid")
        files = list(segment_dir.iterdir())
        if any(not path.is_file() or path.is_symlink() for path in files):
            raise ValueError("SFT segment directory contains an invalid entry")
        actual_paths = {path.relative_to(base).as_posix() for path in files}
        if actual_paths != expected_paths:
            raise ValueError("SFT segment files do not match metadata")
        for metadata, record in expected:
            _check_sft_file(
                base / metadata["path"],
                record,
                "SFT segment does not match captured Call context",
                rebuilt,
            )
    elif not sft_unavailable:
        _check_sft_file(
            sft_path,
            actf_to_sft(actf),
            "trace_SFT.json does not match the canonical ACTF projection",
            rebuilt,
        )
    return rebuilt or []


def validate_resumable_sample(
    sample_dir: Path,
    request: RunRequest,
    info: Mapping[str, Any],
    *,
    require_wire_trace: bool = False,
    require_cache_trace: bool = False,
    verify_output_sha256: bool = False,
    rebuild_sft: bool = False,
) -> list[Path]:
    """Raise ``ValueError`` when a finalized sample is unsafe to reuse.

    Output files are checked by path and size. ``verify_output_sha256`` also
    rereads each file and compares it with the recorded digest.

    The request's position in the run is not part of a sample's identity: a run
    that grows moves its requests. ``rebuild_sft`` rewrites SFT files that
    differ from what this version derives from the trace, and returns them.

    Cache-trace requirements are resolved separately for each participant's
    recorded harness; wire-trace requirements apply to every harness.
    """

    if info.get("schema_version") != "roller.sample/v1":
        raise ValueError("info.json has an unsupported schema_version")
    if info.get("status") != "ok":
        raise ValueError("info.json status is not ok")
    expected_identity = {
        "sample_index": request.sample_index,
        "sample_count": request.sample_count,
        "source_index": request.task.source_index,
        "task_id": request.task.id,
        "model": request.model,
    }
    legacy_defaults = {"sample_index": 0, "sample_count": 1}
    for key, expected in expected_identity.items():
        if info.get(key, legacy_defaults.get(key)) != expected:
            raise ValueError(
                f"info.json {key} does not match current request: "
                f"{info.get(key)!r} != {expected!r}"
            )

    if info.get("runtime", {}).get("training_files_only"):
        # Collected with --training-files-only: only its training files remain.
        return []
    rebuilt = _validate_trajectory_files(
        sample_dir,
        info,
        require_wire_trace=require_wire_trace,
        require_cache_trace=require_cache_trace,
        rebuild_sft=rebuild_sft,
    )
    _validate_output_files(sample_dir, info, verify_sha256=verify_output_sha256)

    if request.task.user_simulator is None:
        return rebuilt
    conversation = _json_object(sample_dir / "conversation.json")
    if conversation.get("schema_version") != "roller.conversation/v1":
        raise ValueError("conversation.json has an unsupported schema_version")
    if conversation.get("status") != "ok":
        raise ValueError("conversation.json status is not ok")
    simulator_dir = sample_dir / "user_simulator"
    simulator_info = _json_object(simulator_dir / "info.json")
    if simulator_info.get("schema_version") != "roller.participant/v1":
        raise ValueError("user_simulator/info.json has an unsupported schema_version")
    if simulator_info.get("status") != "ok":
        raise ValueError("user_simulator/info.json status is not ok")
    return rebuilt + _validate_trajectory_files(
        simulator_dir,
        simulator_info,
        require_wire_trace=require_wire_trace,
        require_cache_trace=require_cache_trace,
        rebuild_sft=rebuild_sft,
    )
