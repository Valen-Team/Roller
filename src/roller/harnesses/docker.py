from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

# Every container carries its collector, so `roller clean` and an interrupted
# run can tell which containers are theirs to remove.
RUN_DIR_LABEL = "ai.roller.run_dir"
PID_LABEL = "ai.roller.pid"
HOST_LABEL = "ai.roller.host"
HOST = socket.gethostname()


class _Launches:
    """The ``docker run`` calls of this process."""

    def __init__(self) -> None:
        self.changed = threading.Condition()
        self.run_dir: str | None = None
        self.closed = False
        self.in_flight = 0
        self.binaries: set[str] = set()


_launches = _Launches()


def begin_collection(run_dir: Path | None) -> None:
    """Label containers started from now on with ``run_dir``; ``None`` ends the run."""

    with _launches.changed:
        _launches.run_dir = None if run_dir is None else str(run_dir)
        _launches.closed = False


def stop_launches(timeout: float = 10.0) -> set[str]:
    """Refuse further ``docker run`` calls and wait for the ones in flight.

    Returns the Docker binaries this process has started containers with.
    """

    with _launches.changed:
        _launches.closed = True
        _launches.changed.wait_for(lambda: not _launches.in_flight, timeout)
        return set(_launches.binaries)


class DockerRuntime:
    """Small Docker subprocess boundary that is replaceable in tests."""

    def __init__(self, binary: str = "docker") -> None:
        self.binary = binary

    def run(
        self, args: Sequence[str], timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        if args and args[0] == "run":
            return self._start_container(args, timeout)
        return self._execute(args, timeout)

    def _execute(
        self, args: Sequence[str], timeout: float | None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Ctrl-C in the terminal goes to Roller alone, which decides what
            # happens to the rollouts; see Collector.collect().
            start_new_session=True,
        )

    def _start_container(
        self, args: Sequence[str], timeout: float | None
    ) -> subprocess.CompletedProcess[str]:
        with _launches.changed:
            if _launches.closed:
                raise RuntimeError("the run is stopping; no container was started")
            _launches.in_flight += 1
            _launches.binaries.add(self.binary)
            labels = {PID_LABEL: str(os.getpid()), HOST_LABEL: HOST}
            if _launches.run_dir is not None:
                labels[RUN_DIR_LABEL] = _launches.run_dir
        try:
            owner = [
                item
                for name, value in labels.items()
                for item in ("--label", f"{name}={value}")
            ]
            return self._execute(["run", *owner, *args[1:]], timeout)
        finally:
            with _launches.changed:
                _launches.in_flight -= 1
                _launches.changed.notify_all()

    def popen(
        self, args: Sequence[str], log_path: Path, append: bool = False
    ) -> subprocess.Popen[str]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a" if append else "w", encoding="utf-8")
        process = subprocess.Popen(
            [self.binary, *args],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        process._roller_log_file = log_file  # type: ignore[attr-defined]
        return process

    @staticmethod
    def close_log(process: subprocess.Popen[str] | None) -> None:
        if process is None:
            return
        log_file = getattr(process, "_roller_log_file", None)
        if log_file is not None and not log_file.closed:
            log_file.close()


def warn_if_image_changed(
    logger: logging.Logger,
    image: str,
    preflight_digest: str | None,
    rollout_digest: str | None,
) -> None:
    if preflight_digest and rollout_digest and preflight_digest != rollout_digest:
        logger.warning(
            "Docker image %s changed after preflight: expected %s, found %s; "
            "the rollout uses and records the current image",
            image,
            preflight_digest,
            rollout_digest,
        )


def remove_container(
    runtime: DockerRuntime,
    name: str,
    *,
    timeouts: Sequence[float] = (30, 60),
) -> subprocess.CompletedProcess[str]:
    """Remove a task container, retrying transient Docker/Podman failures once."""

    if not timeouts:
        raise ValueError("container removal requires at least one timeout")
    last_result: subprocess.CompletedProcess[str] | None = None
    last_error: subprocess.TimeoutExpired | None = None
    for timeout in timeouts:
        try:
            last_result = runtime.run(["rm", "-f", name], timeout=timeout)
            last_error = None
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            continue
        if last_result.returncode == 0:
            return last_result
    if last_error is not None:
        raise last_error
    assert last_result is not None
    return last_result
