"""Diagnostics for ``roller doctor``."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys

from roller.docker_runtime import DockerRuntime
from roller.version import __version__

_OPTIONAL_EXTRAS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("swebench-pro", ("pyarrow",)),
    ("vials", ("pyarrow",)),
    ("zerobench", ("datasets", "Pillow")),
    ("pdf", ("pypdfium2", "Pillow")),
    ("e2b", ("e2b",)),
    ("stirrup", ("stirrup",)),
)


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_extras() -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for extra, packages in _OPTIONAL_EXTRAS:
        versions = {pkg: _dist_version(pkg) for pkg in packages}
        if all(v is not None for v in versions.values()):
            detail = ", ".join(f"{pkg} {ver}" for pkg, ver in versions.items())
            results.append((extra, f"installed ({detail})"))
        else:
            missing = [pkg for pkg, ver in versions.items() if ver is None]
            results.append((extra, f"not installed (missing: {', '.join(missing)})"))
    return results


def _sandbox_providers() -> list[str]:
    eps = importlib.metadata.entry_points(group="roller.sandbox_providers")
    return sorted(f"{ep.name} ({ep.value})" for ep in eps)


def _check_docker(binary: str) -> tuple[bool, str]:
    resolved = shutil.which(binary)
    if resolved is None:
        return False, f"not found ({binary}); import, view, and remote sandboxes still work"
    runtime = DockerRuntime(binary)
    try:
        result = runtime.version(timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"unavailable ({exc}); import, view, and remote sandboxes still work"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "daemon not running"
        return False, f"daemon unavailable ({detail}); import, view, and remote sandboxes still work"
    first_line = (result.stdout or "").strip().splitlines()[0] if result.stdout else "available"
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
    installed_names = {extra for extra, status in extras if status.startswith("installed")}
    if "pdf" in installed_names and "stirrup" in installed_names:
        print("  Warning: pdf and stirrup extras have conflicting Pillow requirements")

    providers = _sandbox_providers()
    if providers:
        print("Sandbox providers:")
        for provider in providers:
            print(f"  {provider}")
    else:
        print("Sandbox providers: none discovered")

    return 0
