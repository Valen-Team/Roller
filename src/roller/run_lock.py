"""Resolved, secret-safe run lock used to make resume reproducible."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from roller.harness_registry import HARNESS_NAMES
from roller.models import RunRequest
from roller.redaction import redact_sensitive_values
from roller.storage import atomic_write_json

logger = logging.getLogger(__name__)
RUN_LOCK_SCHEMA = "roller.run-lock/v1"
LEGACY_MIGRATION_FILENAME = ".legacy-resume-untrusted.json"
URL_KEY_NAMES = {"baseurl", "apiurl", "endpoint", "url"}
SENSITIVE_QUERY_NAMES = {
    "accesskey",
    "accesstoken",
    "apikey",
    "auth",
    "authorization",
    "code",
    "credential",
    "key",
    "password",
    "refreshtoken",
    "secret",
    "sig",
    "signature",
    "token",
}


@dataclass(frozen=True)
class RunLockDecision:
    reuse_existing_samples: bool
    created: bool
    legacy_run: bool


def _sanitize_lock_values(value: Any, key_name: str = "") -> Any:
    """Redact credentials while retaining non-secret URL routing semantics."""

    redacted = redact_sensitive_values(value)
    if isinstance(redacted, dict):
        return {
            key: _sanitize_lock_values(item, str(key)) for key, item in redacted.items()
        }
    if isinstance(redacted, list):
        return [_sanitize_lock_values(item, key_name) for item in redacted]
    normalized = "".join(
        character for character in key_name.lower() if character.isalnum()
    )
    if isinstance(redacted, str) and normalized in URL_KEY_NAMES:
        try:
            parsed = urlsplit(redacted)
            port = parsed.port
        except ValueError:
            return "<invalid-url>"
        hostname = parsed.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        netloc = hostname
        if port is not None:
            netloc = f"{netloc}:{port}"
        query: list[tuple[str, str]] = []
        for key, item in parse_qsl(parsed.query, keep_blank_values=True):
            query_key = "".join(
                character for character in key.lower() if character.isalnum()
            )
            sensitive = (
                query_key in SENSITIVE_QUERY_NAMES
                or query_key.endswith("apikey")
                or query_key.endswith("token")
                or query_key.endswith("signature")
            )
            query.append((key, "<redacted>" if sensitive else item))
        return urlunsplit(
            (
                parsed.scheme,
                netloc if parsed.netloc else "",
                parsed.path,
                urlencode(query),
                "",
            )
        )
    return redacted


def build_run_lock(
    requests: Sequence[RunRequest],
    *,
    roller_version: str,
    execution: Mapping[str, Any],
    model_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a stable lock without persisting credentials.

    Task/input/context identity reuses the content digests already calculated
    by ``TaskBundleStore``. Small configuration objects are compared directly;
    hashing them again would add no useful property.
    """

    locked_requests: list[dict[str, Any]] = []
    for request in requests:
        if request.bundle is None:
            raise ValueError("run lock requires materialized task bundles")
        locked_requests.append(
            {
                "request_index": request.index,
                "sample_index": request.sample_index,
                "sample_count": request.sample_count,
                "source_index": request.task.source_index,
                "task_id": request.task.id,
                "model": request.model,
                "model_label": request.display_model,
                "task_harness": getattr(request.task, "harness", "openclaw"),
                "timeout_seconds": request.timeout_seconds,
                "user_simulator": (
                    {
                        "harness": request.task.user_simulator.harness,
                        "model": request.task.user_simulator.model,
                        "instruction": request.task.user_simulator.instruction,
                        "max_turns": request.task.user_simulator.max_turns,
                        "thinking": request.task.user_simulator.thinking,
                    }
                    if request.task.user_simulator is not None
                    else None
                ),
                "task_bundle": {
                    "manifest_digest": request.bundle.manifest_digest,
                    "input_tree_digest": request.bundle.input_tree_digest,
                    "input_manifest_digest": request.bundle.input_manifest_digest,
                    "context_manifest_digest": request.bundle.context_manifest_digest,
                    "resume_safe": request.bundle.resume_safe,
                },
            }
        )
    return {
        "schema_version": RUN_LOCK_SCHEMA,
        "roller_version": roller_version,
        "execution": _sanitize_lock_values(dict(execution)),
        "model_config": _sanitize_lock_values(
            dict(model_config) if model_config is not None else None
        ),
        "requests": locked_requests,
    }


def _short(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _differences(
    recorded: Any, current: Any, path: str = "$", limit: int = 8
) -> list[str]:
    if recorded == current:
        return []
    if limit <= 0:
        return [f"{path}: additional differences omitted"]
    if isinstance(recorded, dict) and isinstance(current, dict):
        rows: list[str] = []
        for key in sorted(set(recorded) | set(current), key=str):
            child = f"{path}.{key}"
            if key not in recorded:
                rows.append(f"{child}: only in current lock")
            elif key not in current:
                rows.append(f"{child}: missing from current lock")
            else:
                rows.extend(
                    _differences(recorded[key], current[key], child, limit - len(rows))
                )
            if len(rows) >= limit:
                break
        return rows
    if isinstance(recorded, list) and isinstance(current, list):
        rows = []
        for index in range(max(len(recorded), len(current))):
            child = f"{path}[{index}]"
            if index >= len(recorded):
                rows.append(f"{child}: only in current lock")
            elif index >= len(current):
                rows.append(f"{child}: missing from current lock")
            else:
                rows.extend(
                    _differences(
                        recorded[index], current[index], child, limit - len(rows)
                    )
                )
            if len(rows) >= limit:
                break
        return rows
    return [f"{path}: recorded={_short(recorded)}, current={_short(current)}"]


def _comparable_lock(lock: Any) -> Any:
    """Ignore retired no-ops and host-specific placement in the comparison."""

    if not isinstance(lock, dict) or not isinstance(lock.get("execution"), dict):
        return lock
    execution = dict(lock["execution"])
    for name in HARNESS_NAMES:
        settings = execution.get(name)
        if not isinstance(settings, dict):
            continue
        settings = dict(settings)
        if settings.get("external_thinking") == "off":
            settings.pop("external_thinking")
        if name == "claude-code" and settings.get("signature_recovery") == "off":
            for key in (
                "signature_recovery",
                "signature_recovery_attempts",
                "signature_recovery_max_tokens",
            ):
                settings.pop(key, None)
        if name == "openclaw":
            # Host-specific placement, not run semantics: the gateway range is
            # picked per host when the run starts, and which worker slot (so
            # which port) serves a sample is already decided by whichever
            # worker frees up first.
            settings.pop("gateway_port_start", None)
            settings.pop("gateway_port_step", None)
        execution[name] = settings
    return {**lock, "execution": execution}


def _release_series(version: Any) -> tuple[int, int] | None:
    """Major and minor of a version; patch, dev, rc and local parts are ignored."""

    match = re.match(r"(\d+)\.(\d+)", version) if isinstance(version, str) else None
    return (int(match[1]), int(match[2])) if match else None


def _request_key(row: Any) -> str:
    """Identity of a locked request; its position in the list is not part of it."""

    row = row if isinstance(row, dict) else {}
    names = ("source_index", "task_id", "model_label", "sample_index")
    return json.dumps([row.get(name) for name in names])


def _without_position(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    return {key: value for key, value in row.items() if key != "request_index"}


def _request_label(row: Any) -> str:
    if not isinstance(row, dict):
        return _short(row)
    label = f"{row.get('task_id')} | {row.get('model_label')}"
    index, count = row.get("sample_index"), row.get("sample_count")
    if isinstance(index, int) and isinstance(count, int) and count > 1:
        label += f" | sample {index + 1} of {count}"
    return label


def _task_images(lock: Mapping[str, Any]) -> Any:
    execution = lock.get("execution")
    return execution.get("task_images") if isinstance(execution, dict) else None


def _request_mismatches(recorded: Any, current: Any) -> list[str]:
    """Every recorded request must be in the current list with equal fields."""

    if not isinstance(recorded, list) or not isinstance(current, list):
        return _differences(recorded, current, "$.requests")
    by_key = {_request_key(row): row for row in current}
    missing, changed = [], []
    for row in recorded:
        counterpart = by_key.get(_request_key(row))
        if counterpart is None:
            missing.append(row)
        elif _without_position(row) != _without_position(counterpart):
            changed.append((row, counterpart))
    rows = []
    if missing:
        first = missing[0]
        noun = "request is" if len(missing) == 1 else "requests are"
        # The task index tells a removed task from one that moved in the task list.
        index = (
            f" (task index {first.get('source_index')})"
            if isinstance(first, dict)
            else ""
        )
        rows.append(
            f"{len(missing)} recorded {noun} not in the current command; "
            f"first: {_request_label(first)}{index}"
        )
    if changed:
        row, counterpart = changed[0]
        noun = "request differs" if len(changed) == 1 else "requests differ"
        details = _differences(
            _without_position(row), _without_position(counterpart), "", 3
        )
        rows.append(
            f"{len(changed)} recorded {noun} from the current command; "
            f"first: {_request_label(row)}: "
            + "; ".join(detail.lstrip(".") for detail in details)
        )
    return rows


def _mismatches(recorded: Any, current: Mapping[str, Any]) -> list[str]:
    """Explain why ``current`` cannot continue the run that ``recorded`` locked.

    A run continues under another patch release of its minor series, and with
    more requests than it recorded. Every other field must be equal.
    """

    recorded = _comparable_lock(recorded)
    current = _comparable_lock(dict(current))
    if not isinstance(recorded, dict):
        return _differences(recorded, current)
    rows = []
    created, running = recorded.get("roller_version"), current.get("roller_version")
    series = _release_series(created)
    if series is None:
        rows.extend(_differences(created, running, "$.roller_version"))
    elif series != _release_series(running):
        rows.append(
            f"the run was created by Roller {created} and this is Roller {running}; "
            f"a run resumes only within its minor series ({series[0]}.{series[1]}.x)"
        )
    recorded_settings, current_settings = (
        {
            key: value
            for key, value in lock.items()
            if key not in {"roller_version", "requests"}
        }
        for lock in (recorded, current)
    )
    # New requests bring the images of their tasks; the recorded ones stay equal.
    recorded_images, current_images = _task_images(recorded), _task_images(current)
    if isinstance(recorded_images, dict) and isinstance(current_images, dict):
        current_settings["execution"] = {
            **current["execution"],
            "task_images": {
                task: image
                for task, image in current_images.items()
                if task in recorded_images
            },
        }
    rows.extend(_differences(recorded_settings, current_settings))
    rows.extend(_request_mismatches(recorded.get("requests"), current.get("requests")))
    return rows


def ensure_run_lock(
    path: Path,
    expected: Mapping[str, Any],
    *,
    resume: bool,
    existing_run: bool,
    compatible_expected: Sequence[Mapping[str, Any]] = (),
) -> RunLockDecision:
    """Write a new lock or reject resume against a different resolved run.

    An existing run continues under another patch release of its minor series
    and with more requests than it recorded; the lock then keeps the version
    and settings it was written with and lists the current requests.

    Legacy runs without a lock are adopted, but their old samples are not
    reused on that first adoption because their execution configuration cannot
    be proven equivalent. ``compatible_expected`` contains exact historical
    encodings that may be accepted for Resume but are never written for new runs.
    """

    expected_row = dict(expected)
    migration_path = path.with_name(LEGACY_MIGRATION_FILENAME)
    if path.is_file():
        try:
            recorded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"existing run lock cannot be parsed; refusing to overwrite {path}: {exc}"
            ) from exc
        mismatches = [
            _mismatches(recorded, candidate)
            for candidate in (expected_row, *compatible_expected)
        ]
        if all(mismatches):
            details = "\n".join(f"- {row}" for row in mismatches[0])
            raise FileExistsError(
                "run directory cannot be resumed with different resolved inputs; "
                f"lock mismatch at {path}:\n{details}"
            )
        created, running = (
            recorded.get("roller_version"),
            expected_row.get("roller_version"),
        )
        if created != running:
            logger.info(
                "run was created by Roller %s; resuming with Roller %s",
                created,
                running,
            )
        requests = expected_row.get("requests")
        if recorded.get("requests") != requests:
            # The run grew: keep what was recorded and list the current requests.
            grown = {**recorded, "requests": requests}
            images = _task_images(expected_row)
            if images != _task_images(recorded):
                grown["execution"] = {**recorded["execution"], "task_images": images}
            atomic_write_json(path, grown)
            logger.info(
                "lock.json now lists %d requests (%d before)",
                len(requests),
                len(recorded["requests"]),
            )
        return RunLockDecision(
            reuse_existing_samples=bool(resume and not migration_path.is_file()),
            created=False,
            legacy_run=migration_path.is_file(),
        )

    legacy = bool(resume and existing_run)
    if legacy:
        # Write the marker first so a hard stop cannot leave an apparently
        # trusted lock in front of samples created under an unknown old config.
        atomic_write_json(
            migration_path,
            {
                "schema_version": "roller.legacy-resume-untrusted/v1",
                "reason": "existing run was created without lock.json",
            },
        )
    atomic_write_json(path, expected_row)
    return RunLockDecision(
        reuse_existing_samples=bool(resume and not legacy),
        created=True,
        legacy_run=legacy,
    )


def complete_legacy_adoption(path: Path) -> None:
    """Mark an unlocked legacy run trusted after every request was re-executed."""

    path.with_name(LEGACY_MIGRATION_FILENAME).unlink(missing_ok=True)
