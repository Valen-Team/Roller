"""Diagnostics for ``roller doctor``."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import shutil
import subprocess
import sys

from roller.docker_runtime import DockerRuntime
from roller.version import __version__

_OPTIONAL_EXTRAS: tuple[
    tuple[str, tuple[int, int] | None, tuple[tuple[str, str], ...]], ...
] = (
    ("swebench-pro", None, (("pyarrow", ">=14"),)),
    ("vials", None, (("pyarrow", ">=14"),)),
    ("zerobench", None, (("datasets", ">=2.14"), ("Pillow", ">=10"))),
    ("pdf", (3, 11), (("pypdfium2", ">=4.30"), ("Pillow", ">=13"))),
    ("e2b", (3, 11), (("e2b", "==2.66.0"),)),
    ("stirrup", (3, 12), (("stirrup", "==0.2.2"),)),
)


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for segment in raw.split("."):
        match = re.match(r"(\d+)", segment)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def _satisfies(installed: str, requirement: str) -> bool:
    if requirement.startswith("=="):
        return installed == requirement[2:]
    if requirement.startswith(">="):
        actual = _version_tuple(installed)
        minimum = _version_tuple(requirement[2:])
        width = max(len(actual), len(minimum))
        return actual + (0,) * (width - len(actual)) >= minimum + (0,) * (
            width - len(minimum)
        )
    return True


def _check_extras() -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    python_xy = sys.version_info[:2]
    for extra, min_python, requirements in _OPTIONAL_EXTRAS:
        if min_python is not None and python_xy < min_python:
            min_label = ".".join(str(part) for part in min_python)
            results.append((extra, f"unsupported (requires Python >= {min_label})"))
            continue
        versions = {pkg: _dist_version(pkg) for pkg, _ in requirements}
        missing = [pkg for pkg, ver in versions.items() if ver is None]
        if missing:
            results.append((extra, f"not installed (missing: {', '.join(missing)})"))
            continue
        incompatible = [
            f"{pkg} {versions[pkg]} (requires {req})"
            for pkg, req in requirements
            if not _satisfies(versions[pkg] or "", req)
        ]
        if incompatible:
            results.append((extra, f"incompatible ({', '.join(incompatible)})"))
            continue
        detail = ", ".join(f"{pkg} {ver}" for pkg, ver in versions.items())
        results.append((extra, f"installed ({detail})"))
    return results


def _sandbox_providers(*, e2b_installed: bool) -> list[str]:
    eps = list(importlib.metadata.entry_points(group="roller.sandbox_providers"))
    providers = [f"{ep.name} ({ep.value})" for ep in eps]
    if e2b_installed and not any(ep.name == "e2b" for ep in eps):
        providers.append("e2b (roller.sandbox_e2b:E2BProvider, built-in)")
    return sorted(providers)


def _check_docker(binary: str) -> tuple[bool, str]:
    resolved = shutil.which(binary)
    if resolved is None:
        return (
            False,
            f"not found ({binary}); import, view, and remote sandboxes still work",
        )
    runtime = DockerRuntime(binary)
    try:
        result = runtime.version(timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return (
            False,
            f"unavailable ({exc}); import, view, and remote sandboxes still work",
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "daemon not running"
        return (
            False,
            f"daemon unavailable ({detail}); import, view, and remote sandboxes still work",
        )
    first_line = (
        (result.stdout or "").strip().splitlines()[0] if result.stdout else "available"
    )
    return True, f"available ({first_line})"


def _doctor(args: argparse.Namespace) -> int:
    python_ver = sys.version.split()[0]
    print(f"Python: {python_ver}")
    print(f"roller-agent: {__version__}")

    docker_ok, docker_msg = _check_docker(getattr(args, "docker_binary", "docker"))
    print(f"Docker: {docker_msg}")
    if sys.platform == "darwin":
        net_note = (
            "check Docker Desktop -> Settings -> Resources -> Network -> Enable host networking"
            if docker_ok
            else "skipped (Docker unavailable)"
        )
        print(f"macOS host networking: {net_note}")

    print("Optional extras:")
    extras = _check_extras()
    for extra, status in extras:
        print(f"  {extra}: {status}")
    installed_names = {
        extra for extra, status in extras if status.startswith("installed")
    }
    if "pdf" in installed_names and "stirrup" in installed_names:
        print("  Warning: pdf and stirrup extras have conflicting Pillow requirements")

    providers = _sandbox_providers(e2b_installed="e2b" in installed_names)
    if providers:
        print("Sandbox providers:")
        for provider in providers:
            print(f"  {provider}")
    else:
        print("Sandbox providers: none discovered")

    return 0
