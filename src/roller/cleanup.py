"""Remove the containers and bundle directories that collectors left behind."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from roller.harnesses.docker import (
    HOST,
    HOST_LABEL,
    PID_LABEL,
    RUN_DIR_LABEL,
    DockerRuntime,
    stop_launches,
)
from roller.storage import atomic_write_json

logger = logging.getLogger(__name__)

BUNDLE_PREFIX = ".roller-bundle-"
OWNER_FILE = "owner.json"
# Releases that did not record the collector labelled their containers with these.
_OLDER_LABELS = ("ai.roller.harness", "ai.roller.role")
_PS_FORMAT = "\t".join(
    ["{{.ID}}", "{{.Names}}"]
    + [
        f'{{{{.Label "{label}"}}}}'
        for label in (*_OLDER_LABELS, PID_LABEL, HOST_LABEL, RUN_DIR_LABEL)
    ]
)


@dataclass(frozen=True)
class Container:
    id: str
    name: str
    pid: str
    host: str
    run_dir: str


def roller_containers(runtime: DockerRuntime, timeout: float = 30) -> list[Container]:
    """Every container Roller started on this Docker host, running or not."""

    result = runtime.ps(_PS_FORMAT, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"could not list containers: {detail}")
    containers = []
    for line in result.stdout.splitlines():
        fields = line.split("\t", 6)
        if len(fields) == 7 and any(fields[2:5]):
            containers.append(Container(fields[0], fields[1], *fields[4:]))
    return containers


def _remove(
    runtime: DockerRuntime, containers: list[Container], timeout: float = 60
) -> None:
    if not containers:
        return
    result = runtime.remove_all(
        [container.id for container in containers], timeout=timeout
    )
    if result.returncode != 0:
        # A rollout thread may have removed its own container in the meantime.
        logger.debug("docker rm: %s", (result.stderr or result.stdout).strip()[-500:])


def kept_because(pid: object, host: object) -> str | None:
    """Why a collector's leftovers must stay; ``None`` once that collector is gone."""

    if not host or not str(pid).isdigit():
        return "its collector is not recorded"
    if host != HOST:
        return f"its collector runs on {host}"
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # another user's process
    return f"collector {pid} is still running"


def remove_own_containers() -> int:
    """Remove every container of this process, which starts none afterwards."""

    removed = 0
    for binary in stop_launches():
        runtime = DockerRuntime(binary)
        try:
            own = [
                container
                for container in roller_containers(runtime, timeout=10)
                if container.pid == str(os.getpid()) and container.host == HOST
            ]
            _remove(runtime, own, timeout=20)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            logger.warning("could not remove the containers of this run: %s", exc)
            continue
        removed += len(own)
    return removed


def remove_leftover_containers(runtime: DockerRuntime, run_dir: Path) -> int:
    """Remove the containers that dead collectors left for ``run_dir``."""

    leftover = [
        container
        for container in roller_containers(runtime)
        if container.run_dir == str(run_dir)
        and kept_because(container.pid, container.host) is None
    ]
    _remove(runtime, leftover)
    return len(leftover)


def record_owner(directory: Path, run_dir: Path) -> None:
    """Note in a bundle directory which collector it belongs to."""

    atomic_write_json(
        directory / OWNER_FILE,
        {"pid": os.getpid(), "host": HOST, "run_dir": str(run_dir)},
    )


def _owner(directory: Path) -> dict[str, object]:
    try:
        owner = json.loads((directory / OWNER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return owner if isinstance(owner, dict) else {}


def clean(
    runtime: DockerRuntime,
    path: Path | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[str]:
    """Remove what dead collectors left behind and describe it line by line.

    ``path`` limits the work to one run directory or output root; without it
    every Roller container is considered, and the bundle directories under
    ``runs``. ``force`` also removes what a collector may still be using.
    """

    lines: list[str] = []
    removed = "would remove" if dry_run else "removed"

    def decide(subject: str, pid: object, host: object) -> bool:
        reason = kept_because(pid, host)
        if reason is None or force:
            lines.append(
                f"{removed} {subject} ({reason or f'collector {pid} is gone'})"
            )
            return True
        lines.append(f"kept {subject}: {reason}; --force removes it")
        return False

    containers = [
        container
        for container in roller_containers(runtime)
        if path is None
        or Path(container.run_dir) == path
        or (container.run_dir and path in Path(container.run_dir).parents)
    ]
    stale = [
        container
        for container in containers
        if decide(f"container {container.name}", container.pid, container.host)
    ]
    if not dry_run:
        _remove(runtime, stale)

    roots = [Path("runs").resolve()] if path is None else [path, path.parent]
    for root in roots:
        for directory in sorted(root.glob(BUNDLE_PREFIX + "*")):
            owner = _owner(directory)
            if not directory.is_dir() or (
                root != path and path is not None and owner.get("run_dir") != str(path)
            ):
                continue
            if decide(str(directory), owner.get("pid"), owner.get("host")):
                if not dry_run:
                    shutil.rmtree(directory)
    return lines or ["Nothing to clean."]
