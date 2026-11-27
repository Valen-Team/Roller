"""Layered configuration for the standard benchmark run command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from roller.benches import get_bench
from roller.harness_registry import HARNESS_NAMES
from roller.harnesses import HARNESS_API_PROTOCOL, HARNESS_CLI_OPTIONS
from roller.option_specs import (
    BENCH_SELECTION_OPTIONS,
    COLLECT_ONLY_FLAGS,
    EXECUTION_CORE_OPTIONS,
    EXECUTION_OPTIONS,
    EXECUTION_RUNTIME_OPTIONS,
    OptionSpec,
    default_namespace,
    option_map,
)
from roller.thinking import normalize_thinking

BENCH_RUN_SCHEMA = "roller.bench-run/v1"
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ENV_REFERENCE_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
ANY_ENV_REFERENCE_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


@dataclass(frozen=True)
class BenchRunPlan:
    """Resolved benchmark configuration mapped directly to ``collect`` arguments."""

    collect_args: argparse.Namespace
    environment: dict[str, str]
    provenance: dict[str, Any]
    resolved: dict[str, Any]


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML/JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object: {path}")
    return payload


def _load_extra(value: str, index: int) -> tuple[dict[str, Any], str]:
    candidate = Path(value).expanduser()
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        resolved = candidate.resolve()
        return _load_mapping(resolved, "extra config"), str(resolved)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise FileNotFoundError(
            f"extra config is neither a file nor a JSON object: {value}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("inline --extra-config must be a JSON object")
    return payload, f"<inline:{index}>"


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_environment(
    value: Any,
    *,
    environ: Mapping[str, str],
    path: str = "$",
) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_environment(item, environ=environ, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_environment(item, environ=environ, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value
    match = ENV_REFERENCE_RE.fullmatch(value)
    if match:
        name = match.group(1)
        resolved = environ.get(name)
        if resolved is None or resolved == "":
            raise ValueError(f"{path} references missing environment variable {name}")
        return resolved
    if ANY_ENV_REFERENCE_RE.search(value):
        raise ValueError(
            f"{path}: environment references must occupy the complete value"
        )
    return value


def _mapping(
    config: Mapping[str, Any], key: str, *, required: bool = False
) -> dict[str, Any]:
    value = config.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(unknown)}")


def _required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(mapping: Mapping[str, Any], key: str, label: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _string_list(mapping: Mapping[str, Any], key: str, label: str) -> list[str]:
    value = mapping.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{label}.{key} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _default_collect_values() -> dict[str, Any]:
    values: dict[str, Any] = {
        "command": "collect",
        "tasks": None,
        "bench": None,
        "model": [],
        "model_label": None,
        "harness": None,
        "prompt_key": "prompt",
        "id_key": "id",
        "workspace": None,
        "dry_run": False,
        "task_timeouts": False,
        "log_level": "INFO",
        "docker_image": None,
        "openclaw_image": None,
        "codex_image": None,
        "hermes_image": None,
        "opencode_image": None,
        "deepseek_image": None,
        "pi_image": None,
        "stirrup_image": None,
        "claude_code_image": None,
        "api_base": None,
        "hermes_api_base": None,
        "hermes_api_key_env": None,
        "opencode_api_base": None,
        "deepseek_api_base": None,
        "pi_api_base": None,
        "opencode_api_key_env": None,
        "deepseek_api_key_env": None,
        "pi_api_key_env": None,
        "stirrup_api_base": None,
        "stirrup_api_key_env": None,
        "codex_api_base": None,
        "claude_api_base": None,
        "api_key_env": None,
        "codex_api_key_env": None,
        "claude_api_key_env": None,
    }
    values.update(default_namespace(BENCH_SELECTION_OPTIONS))
    values.update(default_namespace(EXECUTION_CORE_OPTIONS))
    values.update(default_namespace(EXECUTION_RUNTIME_OPTIONS))
    for options in HARNESS_CLI_OPTIONS.values():
        values.update(default_namespace(options))
    return values


def _apply_option_map(
    target: dict[str, Any],
    specs: Mapping[str, OptionSpec],
    values: Mapping[str, Any],
    label: str,
) -> None:
    _reject_unknown(values, set(specs), label)
    for name, raw in values.items():
        spec = specs[name]
        if raw is None:
            continue
        target[spec.resolved_dest] = spec.coerce_yaml(raw, f"{label}.{name}")


def _benchmark_option_map(benchmark: str) -> dict[str, OptionSpec]:
    try:
        adapter = get_bench(benchmark)
    except KeyError:
        return {}
    return option_map(adapter.cli_options)


def _openclaw_model_config(
    model: Mapping[str, Any], temporary_root: Path
) -> tuple[str, Path]:
    model_id = _required_string(model, "id", "model")
    provider_override = _optional_string(model, "provider", "model")
    if provider_override is not None:
        provider_id = provider_override
        upstream_model_id = model_id
    elif "/" in model_id:
        # Backward compatibility for the historical provider/model spelling.
        provider_id, upstream_model_id = model_id.split("/", 1)
    else:
        provider_id = "roller"
        upstream_model_id = model_id
    if not provider_id or not upstream_model_id or "/" in provider_id:
        raise ValueError("model.provider must be a non-empty OpenClaw provider name")
    base_url = _required_string(model, "base_url", "model")
    api_key_env = _required_string(model, "api_key_env", "model")
    protocol = str(model.get("api_protocol") or "openai-completions")
    params = model.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("model.params must be an object")
    forbidden = sorted(set(params) & {"id", "name"})
    if forbidden:
        raise ValueError(
            "model.params cannot replace generated OpenClaw field(s): "
            + ", ".join(forbidden)
        )
    payload = {
        "mode": "merge",
        "providers": {
            provider_id: {
                "baseUrl": base_url,
                "apiKey": "${" + api_key_env + "}",
                "api": protocol,
                "models": [
                    {"id": upstream_model_id, "name": upstream_model_id, **params}
                ],
            }
        },
    }
    path = temporary_root / "openclaw-models.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"{provider_id}/{upstream_model_id}", path


def _validate_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("model.base_url must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("model.base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("model.base_url must not contain a query or fragment")


def _safe_url(value: str | None) -> str | None:
    """Remove URL credentials and query parameters before recording provenance."""

    if value is None:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = hostname if port is None else f"{hostname}:{port}"
    return parsed._replace(netloc=netloc, query="", fragment="").geturl()


def _redact(value: Any, key: str = "") -> Any:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    sensitive = any(
        marker in normalized
        for marker in ("apikey", "token", "password", "secret", "authorization")
    ) and not normalized.endswith("env")
    if sensitive and isinstance(value, str):
        return "<redacted>"
    if isinstance(value, dict):
        return {name: _redact(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    return value


def load_bench_run_plan(
    config_path: str | Path,
    extra_configs: Sequence[str] = (),
    *,
    environ: Mapping[str, str] | None = None,
    temporary_root: Path,
) -> BenchRunPlan:
    """Load layered YAML/JSON and resolve it into structured ``collect`` arguments."""

    environment = os.environ if environ is None else environ
    base_path = Path(config_path).expanduser().resolve()
    merged = _load_mapping(base_path, "benchmark run config")
    sources = [str(base_path)]
    for index, value in enumerate(extra_configs, start=1):
        overlay, source = _load_extra(value, index)
        merged = _deep_merge(merged, overlay)
        sources.append(source)
    resolved = _resolve_environment(merged, environ=environment)

    _reject_unknown(
        resolved,
        {"schema_version", "benchmark", "model", "harness", "execution"},
        "configuration",
    )
    if resolved.get("schema_version") != BENCH_RUN_SCHEMA:
        raise ValueError(f"schema_version must be {BENCH_RUN_SCHEMA!r}")

    benchmark = _mapping(resolved, "benchmark", required=True)
    model = _mapping(resolved, "model", required=True)
    harness = _mapping(resolved, "harness", required=True)
    execution = _mapping(resolved, "execution")
    execution_specs = option_map(EXECUTION_OPTIONS)
    _reject_unknown(
        benchmark,
        {
            "id",
            "root",
            "variant",
            "categories",
            "task_ids",
            "timeout_multiplier",
            "environment",
            "extra",
        },
        "benchmark",
    )
    _reject_unknown(
        model,
        {
            "id",
            "label",
            "provider",
            "base_url",
            "api_key_env",
            "api_protocol",
            "thinking",
            "params",
        },
        "model",
    )
    _reject_unknown(harness, {"id", "image", "extra"}, "harness")
    _reject_unknown(execution, set(execution_specs), "execution")

    benchmark_id = _required_string(benchmark, "id", "benchmark")
    benchmark_root = _required_string(benchmark, "root", "benchmark")
    harness_id = _required_string(harness, "id", "harness")
    if harness_id not in HARNESS_NAMES:
        raise ValueError(f"harness.id must be one of: {', '.join(HARNESS_NAMES)}")
    model_id = _required_string(model, "id", "model")
    model_provider = _optional_string(model, "provider", "model")
    base_url = _optional_string(model, "base_url", "model")
    api_key_env = _optional_string(model, "api_key_env", "model")
    api_protocol = _optional_string(model, "api_protocol", "model")
    thinking = normalize_thinking(_optional_string(model, "thinking", "model"))
    bench_environment = benchmark.get("environment", {})
    if not isinstance(bench_environment, dict):
        raise ValueError("benchmark.environment must be an object")
    exported_environment: dict[str, str] = {}
    for name, value in bench_environment.items():
        if not isinstance(name, str) or not ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"benchmark.environment has invalid name: {name!r}")
        if not isinstance(value, str):
            raise ValueError(f"benchmark.environment.{name} must be a string")
        exported_environment[name] = value
    if base_url is not None:
        _validate_url(base_url)
    if api_key_env is not None:
        if not ENV_NAME_RE.fullmatch(api_key_env):
            raise ValueError("model.api_key_env must be an environment variable name")
        if not (exported_environment.get(api_key_env) or environment.get(api_key_env)):
            raise ValueError(
                f"model.api_key_env references missing environment variable {api_key_env}"
            )

    collect_values = _default_collect_values()
    collect_values["bench"] = benchmark_id
    collect_values["bench_root"] = benchmark_root
    collect_values["model"] = [model_id]
    collect_values["harness"] = harness_id
    collect_values["bench_category"] = _string_list(
        benchmark, "categories", "benchmark"
    )
    collect_values["bench_task_id"] = _string_list(benchmark, "task_ids", "benchmark")
    collect_values["bench_variant"] = _optional_string(
        benchmark, "variant", "benchmark"
    )
    if (
        "timeout_multiplier" in benchmark
        and benchmark["timeout_multiplier"] is not None
    ):
        multiplier = benchmark["timeout_multiplier"]
        if isinstance(multiplier, bool):
            raise ValueError("benchmark.timeout_multiplier must not be a boolean")
        collect_values["bench_timeout_multiplier"] = float(multiplier)

    benchmark_extra = _mapping(benchmark, "extra")
    _apply_option_map(
        collect_values,
        _benchmark_option_map(benchmark_id),
        benchmark_extra,
        "benchmark.extra",
    )

    model_params = model.get("params", {})
    if not isinstance(model_params, dict):
        raise ValueError("model.params must be an object")
    harness_extra = _mapping(harness, "extra")
    if harness_id == "openclaw":
        custom_model_config = harness_extra.get("model_config")
        if custom_model_config is not None:
            if (
                base_url is not None
                or api_key_env is not None
                or api_protocol is not None
                or model_params
            ):
                raise ValueError(
                    "harness.extra.model_config cannot be combined with model.base_url, "
                    "model.api_key_env, model.api_protocol, or model.params"
                )
        else:
            if base_url is None or api_key_env is None:
                raise ValueError(
                    "OpenClaw requires model.base_url and model.api_key_env when "
                    "harness.extra.model_config is not set"
                )
            model_id, generated = _openclaw_model_config(model, temporary_root)
            collect_values["model"] = [model_id]
            harness_extra["model_config"] = str(generated)
    else:
        if base_url is None or api_key_env is None:
            raise ValueError(
                f"{harness_id} requires model.base_url and model.api_key_env"
            )
        expected_protocol = HARNESS_API_PROTOCOL[harness_id]
        if api_protocol is not None and api_protocol != expected_protocol:
            raise ValueError(
                f"model.api_protocol must be {expected_protocol!r} for {harness_id}"
            )
        if model_params:
            raise ValueError(
                f"model.params is only supported by generated OpenClaw model config; "
                f"use harness.extra for {harness_id}"
            )
        collect_values["api_base"] = base_url
        collect_values["api_key_env"] = api_key_env

    collect_values["model_label"] = _optional_string(model, "label", "model")
    collect_values["thinking"] = thinking
    collect_values["docker_image"] = _optional_string(harness, "image", "harness")
    _apply_option_map(
        collect_values,
        option_map(HARNESS_CLI_OPTIONS[harness_id]),
        harness_extra,
        "harness.extra",
    )

    pass_env = _string_list(execution, "pass_env", "execution")
    for name in pass_env:
        if not ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"execution.pass_env contains invalid name: {name}")
    _apply_option_map(collect_values, execution_specs, execution, "execution")
    collect_values["pass_env"] = pass_env

    redacted = _redact(resolved)
    digest = hashlib.sha256(
        json.dumps(
            redacted, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    provenance = {
        "schema_version": BENCH_RUN_SCHEMA,
        "sources": sources,
        "resolved_sha256": digest,
        "environment_names": sorted(exported_environment),
        "effective": {
            "benchmark": {
                "id": benchmark_id,
                "root": benchmark_root,
                "variant": benchmark.get("variant"),
                "categories": benchmark.get("categories", []),
                "task_ids": benchmark.get("task_ids", []),
                "timeout_multiplier": benchmark.get("timeout_multiplier", 1),
                "extra": _redact(benchmark_extra),
            },
            "model": {
                "id": model_id,
                "label": model.get("label"),
                "provider": model_provider,
                "base_url": _safe_url(base_url),
                "api_key_env": api_key_env,
                "api_protocol": (
                    api_protocol
                    or HARNESS_API_PROTOCOL.get(harness_id)
                    or ("openai-completions" if base_url is not None else None)
                ),
                "thinking": thinking,
                "params": _redact(model_params),
            },
            "harness": {
                "id": harness_id,
                "image": harness.get("image"),
                "extra": _redact(_mapping(harness, "extra")),
            },
            "execution": _redact(execution),
        },
    }
    return BenchRunPlan(
        argparse.Namespace(**collect_values),
        exported_environment,
        provenance,
        resolved,
    )


@contextmanager
def temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Apply benchmark-specific environment values for one in-process run."""

    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


__all__ = [
    "BENCH_RUN_SCHEMA",
    "COLLECT_ONLY_FLAGS",
    "BenchRunPlan",
    "load_bench_run_plan",
    "temporary_environment",
]
