"""The execution contract for optional remote sandbox providers."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from subprocess import CompletedProcess
from typing import Protocol

from roller.storage import atomic_write_json


class SandboxProcess(Protocol):
    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class Sandbox(Protocol):
    version: str

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float,
    ) -> CompletedProcess[str]: ...

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        log_path: Path,
    ) -> SandboxProcess: ...

    def upload(
        self, source: Path, target: str, *, contents: bool = False, timeout: float
    ) -> None: ...

    def download(
        self, source: str, target: Path, *, contents: bool = False, timeout: float
    ) -> None: ...

    def close(self) -> None: ...


class SandboxProvider(Protocol):
    def resolve(self, template: str) -> str:
        """Resolve the template to its stable environment version."""
        ...

    def create(
        self,
        template: str,
        version: str,
        *,
        name: str,
        lifetime: float,
        timeout: float,
        allocated: Callable[[str], None],
    ) -> Sandbox:
        """Report the owned ID through allocated before waiting for readiness."""
        ...

    def remove(self, instance_id: str) -> None:
        """Remove a recorded instance, tolerating one already gone."""
        ...


def load_sandbox_provider(name: str) -> SandboxProvider:
    if name == "e2b":
        from roller.sandbox_e2b import E2BProvider

        return E2BProvider()
    plugins = entry_points(group="roller.sandbox_providers", name=name)
    if len(plugins) != 1:
        raise ValueError(
            f"expected one installed sandbox provider named {name!r}; found {len(plugins)}"
        )
    return next(iter(plugins)).load()()


def route_sandbox_http(
    box: Sandbox, base: str, key: str, timeout: float
) -> tuple[str, str]:
    """Use a provider's collector-side inference route when one is available."""
    route = getattr(box, "route_http", None)
    if route is None:
        return base, key
    return route(base, key, timeout), "sandbox-relay"


def route_judge_environment(box: Sandbox, env: dict[str, str], timeout: float) -> None:
    for service in ("OPENROUTER", "OPENAI"):
        base, key = f"{service}_BASE_URL", f"{service}_API_KEY"
        if env.get(base) and key in env:
            env[base], env[key] = route_sandbox_http(box, env[base], env[key], timeout)


@dataclass(frozen=True)
class SandboxTarget:
    name: str
    template: str
    version: str

    def provenance(self) -> dict[str, str]:
        return {
            "sandbox_provider": self.name,
            "sandbox_template": self.template,
            "sandbox_version": self.version,
        }


class SandboxRun:
    """Track this run's allocations so a resumed run can release leftovers."""

    def __init__(self, run_dir: Path, target: SandboxTarget, provider: SandboxProvider):
        self.target = target
        self.provider = provider
        self._resources = run_dir / ".sandbox-resources"
        self._lock = threading.Lock()
        self._stopped = False

    def create(self, name: str, lifetime: float, timeout: float) -> Sandbox:
        with self._lock:
            if self._stopped:
                raise RuntimeError("the run is stopping; no sandbox was started")
        receipt = self._resources / f"{name}.json"

        def allocated(instance_id: str) -> None:
            with self._lock:
                atomic_write_json(
                    receipt, {"provider": self.target.name, "instance_id": instance_id}
                )
                if not self._stopped:
                    return
            self._remove(receipt)
            raise RuntimeError("the run stopped while allocating a sandbox")

        try:
            box = self.provider.create(
                self.target.template,
                self.target.version,
                name=name,
                lifetime=lifetime,
                timeout=timeout,
                allocated=allocated,
            )
        except Exception:
            self._remove(receipt)
            raise
        with self._lock:
            stopped = self._stopped
        if stopped:
            self.close(name, box)
            raise RuntimeError("the run stopped while starting a sandbox")
        return box

    def close(self, name: str, box: Sandbox) -> None:
        box.close()
        (self._resources / f"{name}.json").unlink(missing_ok=True)

    def _remove(self, receipt: Path) -> None:
        try:
            record = json.loads(receipt.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        if record["provider"] != self.target.name:
            raise ValueError("stored sandbox receipt uses a different provider")
        self.provider.remove(record["instance_id"])
        receipt.unlink(missing_ok=True)

    def cleanup(self) -> None:
        for receipt in sorted(self._resources.glob("*.json")):
            self._remove(receipt)

    def stop_launching(self) -> None:
        with self._lock:
            self._stopped = True

    def abort(self) -> None:
        self.stop_launching()
        self.cleanup()
