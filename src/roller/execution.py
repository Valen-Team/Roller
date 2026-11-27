"""Unified workspace execution backend abstracting Docker containers and remote sandboxes."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from roller.artifacts import ArtifactCollectionResult, collect_output_directory
from roller.docker_runtime import DockerRuntime, remove_container
from roller.models import RunRequest
from roller.sandbox import (
    Sandbox,
    SandboxProcess,
    SandboxRun,
    route_judge_environment,
    route_sandbox_http,
)

PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


class WorkspaceProcess(Protocol):
    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...

    def close_log(self) -> None: ...


class WorkspaceBackend(Protocol):
    is_remote: bool

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        options: Sequence[str] = (),
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        log_path: Path,
        append: bool = False,
    ) -> WorkspaceProcess: ...

    def upload(
        self,
        source: Path,
        target: str,
        *,
        contents: bool = False,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...

    def download(
        self,
        source: str,
        target: Path,
        *,
        contents: bool = False,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...

    def route_http(self, base: str, key: str, timeout: float) -> tuple[str, str]: ...

    def route_judge_env(self, env: dict[str, str], timeout: float) -> None: ...

    def collect_outputs(
        self,
        source: str,
        output_dir: Path,
        *,
        timeout: float,
        catch_errors: bool = True,
    ) -> tuple[ArtifactCollectionResult, str | None]: ...

    def close(self, *, retain: bool = False) -> None: ...


class _DockerProcess:
    def __init__(self, runtime: DockerRuntime, process: subprocess.Popen[str]) -> None:
        self._runtime = runtime
        self._process = process

    def wait(self, timeout: float | None = None) -> int:
        return self._process.wait(timeout=timeout)

    def kill(self) -> None:
        self._process.kill()
        self._process.wait(timeout=10)

    def close_log(self) -> None:
        self._runtime.close_log(self._process)


class _SandboxProcessAdapter:
    def __init__(self, process: SandboxProcess) -> None:
        self._process = process

    def wait(self, timeout: float | None = None) -> int:
        return self._process.wait(timeout=timeout)

    def kill(self) -> None:
        self._process.kill()

    def close_log(self) -> None:
        return None


class DockerWorkspace:
    """WorkspaceBackend backed by a running local Docker container."""

    is_remote = False

    def __init__(self, runtime: DockerRuntime, container: str) -> None:
        self.runtime = runtime
        self.container = container

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        options: Sequence[str] = (),
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del env
        return self.runtime.exec(
            self.container,
            argv,
            workdir=cwd,
            options=options,
            timeout=timeout,
        )

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        log_path: Path,
        append: bool = False,
    ) -> WorkspaceProcess:
        del env
        process = self.runtime.exec_logged(
            self.container,
            argv,
            log_path,
            workdir=cwd,
            append=append,
        )
        return _DockerProcess(self.runtime, process)

    def upload(
        self,
        source: Path,
        target: str,
        *,
        contents: bool = False,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        argument = f"{source}/." if contents else str(source)
        return self.runtime.copy_in(argument, self.container, target, timeout=timeout)

    def download(
        self,
        source: str,
        target: Path,
        *,
        contents: bool = False,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        remote = f"{source}/." if contents else source
        return self.runtime.copy_out(self.container, remote, target, timeout=timeout)

    def route_http(self, base: str, key: str, timeout: float) -> tuple[str, str]:
        del timeout
        return base, key

    def route_judge_env(self, env: dict[str, str], timeout: float) -> None:
        del env, timeout

    def collect_outputs(
        self,
        source: str,
        output_dir: Path,
        *,
        timeout: float,
        catch_errors: bool = True,
    ) -> tuple[ArtifactCollectionResult, str | None]:
        staging_root = output_dir / ".runtime-outputs"
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            copy_result = self.download(
                source, staging_root, contents=True, timeout=timeout
            )
            copy_error = (
                (
                    copy_result.stderr or copy_result.stdout or "docker cp failed"
                ).strip()[-2000:]
                if copy_result.returncode != 0
                else None
            )
            if copy_error is not None and not catch_errors:
                raise RuntimeError(copy_error)
            collected = collect_output_directory(staging_root, output_dir)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
        return collected, copy_error

    def close(self, *, retain: bool = False) -> None:
        if retain:
            return
        remove_container(self.runtime, self.container)


class SandboxWorkspace:
    """WorkspaceBackend backed by a remote :class:`~roller.sandbox.Sandbox`."""

    is_remote = True

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        name: str | None = None,
        sandbox_run: SandboxRun | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.name = name
        self.sandbox_run = sandbox_run

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        options: Sequence[str] = (),
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if env is None:
            merged = {
                name: os.environ[name] for name in PROXY_ENV_NAMES if name in os.environ
            }
            for option in options[1::2]:
                name, sep, value = option.partition("=")
                merged[name] = value if sep else os.environ[name]
            route_judge_environment(self.sandbox, merged, timeout)
        else:
            merged = dict(env)
        return self.sandbox.run(argv, cwd=cwd, env=merged, timeout=timeout)

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        log_path: Path,
        append: bool = False,
    ) -> WorkspaceProcess:
        del append
        process = self.sandbox.spawn(argv, cwd=cwd, env=env, log_path=log_path)
        return _SandboxProcessAdapter(process)

    def upload(
        self,
        source: Path,
        target: str,
        *,
        contents: bool = False,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        try:
            self.sandbox.upload(source, target, contents=contents, timeout=timeout)
        except Exception as exc:
            return subprocess.CompletedProcess([], 1, "", str(exc))
        return subprocess.CompletedProcess([], 0, "", "")

    def download(
        self,
        source: str,
        target: Path,
        *,
        contents: bool = False,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        try:
            self.sandbox.download(source, target, contents=contents, timeout=timeout)
        except Exception as exc:
            return subprocess.CompletedProcess([], 1, "", str(exc))
        return subprocess.CompletedProcess([], 0, "", "")

    def route_http(self, base: str, key: str, timeout: float) -> tuple[str, str]:
        return route_sandbox_http(self.sandbox, base, key, timeout)

    def route_judge_env(self, env: dict[str, str], timeout: float) -> None:
        route_judge_environment(self.sandbox, env, timeout)

    def collect_outputs(
        self,
        source: str,
        output_dir: Path,
        *,
        timeout: float,
        catch_errors: bool = True,
    ) -> tuple[ArtifactCollectionResult, str | None]:
        staging_root = (
            output_dir / ".runtime-outputs"
            if catch_errors
            else Path(tempfile.mkdtemp(dir=output_dir))
        )
        if catch_errors and staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            copy_error: str | None = None
            try:
                self.sandbox.download(
                    source, staging_root, contents=True, timeout=timeout
                )
            except Exception as exc:
                if not catch_errors:
                    raise
                copy_error = f"{type(exc).__name__}: {exc}"
            collected = collect_output_directory(staging_root, output_dir)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
        return collected, copy_error

    def close(self, *, retain: bool = False) -> None:
        del retain
        if self.sandbox_run is not None and self.name is not None:
            self.sandbox_run.close(self.name, self.sandbox)
        else:
            self.sandbox.close()


def workspace_backend(
    runtime: Any, container: str, sandbox: Sandbox | None = None
) -> WorkspaceBackend:
    """Wrap either a remote sandbox or a Docker container as a WorkspaceBackend."""

    if sandbox is not None:
        return SandboxWorkspace(sandbox)
    return DockerWorkspace(runtime, container)


def stage_sandbox_inputs(
    box: Sandbox,
    request: RunRequest,
    workspace: str,
    *,
    upload_timeout: float,
    run_checked: Callable[[list[str], str], Any],
    chmod_workspace: bool = True,
    chmod_legacy_workspace: bool = False,
) -> None:
    """Upload task bundle or host workspace inputs into a remote sandbox."""

    task, bundle = request.task, request.bundle
    if bundle is not None:
        if bundle.legacy_workspace or task.input_mode == "copy":
            box.upload(
                bundle.inputs_root, workspace, contents=True, timeout=upload_timeout
            )
            if chmod_workspace:
                run_checked(
                    ["chmod", "-R", "u+w", workspace], "sandbox workspace permissions"
                )
        box.upload(
            bundle.context_root, workspace, contents=True, timeout=upload_timeout
        )
    elif task.workspace is not None:
        box.upload(task.workspace, workspace, contents=True, timeout=upload_timeout)
        if chmod_legacy_workspace:
            run_checked(
                ["chmod", "-R", "u+w", workspace], "sandbox workspace permissions"
            )
