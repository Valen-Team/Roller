from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
from collections.abc import Mapping, Sequence
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
    """Small Docker subprocess boundary that is replaceable in tests.

    The named methods below are the verbs Roller calls. They build Docker's own
    argument list, so nothing outside this module has to know the CLI. Their
    argv is exactly what callers wrote by hand before they existed, which is
    what keeps the recorded command lines — and the tests that assert on them —
    unchanged.
    """

    def __init__(self, binary: str = "docker") -> None:
        self.binary = binary

    def run(
        self, args: Sequence[str], timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        if args and args[0] == "run":
            return self._start_container(args, timeout)
        return self._execute(args, timeout)

    # -- images ---------------------------------------------------------

    def image_id(self, image: str, timeout: float | None = None) -> str | None:
        """Id the tag points at now, or ``None`` when the image is not local."""

        result = self.run(
            ["image", "inspect", image, "--format", "{{.Id}}"], timeout=timeout
        )
        return (result.stdout.strip() or None) if result.returncode == 0 else None

    def image_field(
        self, image: str, field: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        """``docker image inspect --format '{{.<field>}}'``, result as returned."""

        return self.run(
            ["image", "inspect", image, "--format", f"{{{{.{field}}}}}"],
            timeout=timeout,
        )

    def pull(
        self, image: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return self.run(["pull", image], timeout=timeout)

    def build(
        self,
        args: Sequence[str],
        timeout: float | None = None,
        *,
        engine: Sequence[str] = (),
    ) -> subprocess.CompletedProcess[str]:
        """``docker build`` with its options; ``args`` covers ``-f``..context.

        ``engine`` holds flags that belong before the subcommand rather than to
        it — rootless Podman needs ``--storage-opt`` there.
        """

        return self.run([*engine, "build", *args], timeout=timeout)

    def version(self, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        """Server version, and therefore whether a daemon answers at all."""

        return self.run(["version", "--format", "{{.Server.Version}}"], timeout=timeout)

    # -- containers -----------------------------------------------------

    def start(
        self,
        name: str,
        image: str,
        command: Sequence[str],
        *,
        labels: Mapping[str, str] | None = None,
        env: Sequence[tuple[str, str]] | None = None,
        volumes: Sequence[str] | None = None,
        network: str | None = None,
        user: str | None = None,
        detach: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Start a container and return once it is running.

        ``env`` keeps its order and repeats: callers rely on both (a later
        ``-e NAME=`` shadows an earlier one). ``volumes`` are ``source:target``
        with an optional ``:ro``. The run bookkeeping in ``_start_container``
        still sees ``run`` as the first word, so the owner labels stay put.
        """

        return self.run(
            self.start_args(
                name,
                image,
                command,
                labels=labels,
                env=env,
                volumes=volumes,
                network=network,
                user=user,
                detach=detach,
            ),
            timeout=timeout,
        )

    def start_args(
        self,
        name: str,
        image: str | None,
        command: Sequence[str] = (),
        *,
        labels: Mapping[str, str] | None = None,
        env: Sequence[tuple[str, str]] | None = None,
        volumes: Sequence[str] | None = None,
        network: str | None = None,
        user: str | None = None,
        detach: bool = True,
    ) -> list[str]:
        """The argv ``start`` builds, for callers that need it up front.

        ``image=None`` stops the argv after the container's options, which is
        what a caller that appends the image and command itself needs.
        """

        args = ["run"]
        if detach:
            args.append("-d")
        args += ["--name", name]
        for label, value in (labels or {}).items():
            args += ["--label", f"{label}={value}"]
        if user is not None:
            args += ["--user", user]
        if network:
            args += ["--network", network]
        for label, value in env or ():
            args += ["-e", f"{label}={value}"]
        for volume in volumes or ():
            args += ["-v", volume]
        if image is not None:
            args += [image, *command]
        return args

    def exec(
        self,
        container: str,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one command in a running container."""

        return self.run(
            self.exec_args(container, command, env=env, workdir=workdir),
            timeout=timeout,
        )

    def exec_args(
        self,
        container: str,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> list[str]:
        """The argv ``exec`` builds, for callers that need it up front."""

        args = ["exec"]
        for name, value in (env or {}).items():
            args += ["-e", f"{name}={value}"]
        if workdir is not None:
            args += ["-w", workdir]
        args.append(container)
        args.extend(command)
        return args

    def exec_logged(
        self,
        container: str,
        command: Sequence[str],
        log_path: Path,
        *,
        env: Mapping[str, str] | None = None,
        workdir: str | None = None,
        append: bool = False,
    ) -> subprocess.Popen[str]:
        """Run one command in a container with its output going to ``log_path``."""

        return self.popen(
            self.exec_args(container, command, env=env, workdir=workdir),
            log_path,
            append=append,
        )

    def kill(
        self, container: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return self.run(["kill", container], timeout=timeout)

    def remove(
        self, name: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return self.run(["rm", "-f", name], timeout=timeout)

    def remove_all(
        self, names: Sequence[str], timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return self.run(["rm", "-f", *names], timeout=timeout)

    def ps(
        self, format_spec: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return self.run(["ps", "-a", "--format", format_spec], timeout=timeout)

    # -- files ----------------------------------------------------------

    def copy_out(
        self,
        container: str,
        source: str,
        destination: Path | str,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Copy one path out of a container."""

        return self.run(
            ["cp", f"{container}:{source}", str(destination)], timeout=timeout
        )

    def copy_in(
        self,
        source: Path | str,
        container: str,
        destination: str,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Copy one path from the host into a container."""

        return self.run(
            ["cp", str(source), f"{container}:{destination}"], timeout=timeout
        )

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
            # ``run`` rather than ``remove``: callers pass test doubles that
            # only implement the process boundary.
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
