"""Pieces shared by every Harness that runs a task in a disposable Docker container."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from roller.artifacts import ArtifactCollectionResult, collect_output_directory
from roller.harnesses.docker import DockerRuntime, warn_if_image_changed
from roller.models import RunRequest, Task
from roller.version import __version__

logger = logging.getLogger(__name__)

ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
INPUTS_CONTAINER_PATH = "/roller/task/inputs"
CONTEXT_CONTAINER_PATH = "/roller/task/context"
OUTPUTS_CONTAINER_PATH = "/roller/task/outputs"
WORKSPACE_INPUT_CONTAINER_PATH = "/roller-input"


def error_text(result: subprocess.CompletedProcess[str], limit: int = 2000) -> str:
    return (result.stderr or result.stdout or "unknown error").strip()[-limit:]


def host_environment(task: Task, pass_env: Sequence[str]) -> dict[str, str]:
    """Host variables a task container inherits.

    ``pass_env`` names are optional; every variable the task names must be set.
    The collect preflight calls this for each task before the first rollout.
    """

    environment: dict[str, str] = {}
    for name in pass_env:
        if not ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid environment variable name: {name!r}")
        if os.environ.get(name):
            environment[name] = os.environ[name]
    for name in task.env_names:
        if not ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid task environment variable name: {name!r}")
        if not os.environ.get(name):
            raise ValueError(
                f"task {task.id!r} requires missing environment variable {name}"
            )
        environment[name] = os.environ[name]
    return environment


def host_api_key(label: str, name: str) -> str:
    """The API key a Harness reads from the host variable ``name``."""

    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{label} requires a non-empty API key in {name}")
    return value


def docker_run_args(
    harness: str,
    container: str,
    config: Any,
    environment: dict[str, str],
    request: RunRequest,
    *,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """``docker run`` options up to, but excluding, the image and its command.

    ``extra_env`` is passed to the container without being recorded as task
    environment in provenance.
    """

    args = ["run", "-d", "--name", container, "--label", f"ai.roller.harness={harness}"]
    if config.container_user is not None:
        args += ["--user", config.container_user]
    if config.network:
        args += ["--network", config.network]
    for name, value in sorted(environment.items()):
        args += ["-e", f"{name}={value}"]
    for name, value in (extra_env or {}).items():
        args += ["-e", f"{name}={value}"]
    bundle, task = request.bundle, request.task
    if bundle is not None:
        args += [
            "-v",
            f"{bundle.inputs_root}:{INPUTS_CONTAINER_PATH}:ro",
            "-v",
            f"{bundle.context_root}:{CONTEXT_CONTAINER_PATH}:ro",
        ]
        if not bundle.legacy_workspace and task.input_mode == "mount":
            for spec in task.inputs:
                source = bundle.inputs_root / spec.path
                if source.exists():
                    args += [
                        "-v",
                        f"{source}:{config.workspace_container_path}/{spec.path}:ro",
                    ]
    elif task.workspace is not None:
        if not task.workspace.is_dir():
            raise FileNotFoundError(
                f"task workspace is not a directory: {task.workspace}"
            )
        args += ["-v", f"{task.workspace}:{WORKSPACE_INPUT_CONTAINER_PATH}:ro"]
    return args


def collect_outputs(
    runtime: DockerRuntime,
    container: str,
    request: RunRequest,
    output_dir: Path,
    workspace: str,
    *,
    timeout: float | None = None,
    error_limit: int = 2000,
) -> tuple[ArtifactCollectionResult, str | None]:
    """Copy the agent output directory out of the container and vet it."""

    staging_root = output_dir / ".runtime-outputs"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    try:
        result = runtime.run(
            [
                "cp",
                f"{container}:{workspace}/{request.task.output_dir}/.",
                str(staging_root),
            ],
            timeout=timeout,
        )
        copy_error = None if result.returncode == 0 else error_text(result, error_limit)
        collected = collect_output_directory(staging_root, output_dir)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return collected, copy_error


def output_collection_metadata(
    collected: ArtifactCollectionResult | None, request: RunRequest, output_dir: Path
) -> dict[str, Any] | None:
    if collected is None:
        return None
    return {
        "mode": "all",
        "agent_output_dir": request.task.output_dir,
        "passed": collected.passed,
        "path": (
            collected.results_path.relative_to(output_dir).as_posix()
            if collected.results_path is not None
            else None
        ),
        "file_count": collected.file_count,
        "directory_count": collected.directory_count,
        "files": list(collected.files),
        "errors": list(collected.errors),
    }


def build_provenance(
    request: RunRequest,
    *,
    harness: str,
    docker_image: str | None,
    docker_image_digest: str | None,
    api_base_urls: list[str],
    environment: dict[str, str],
    attempt: int,
    started_at: str,
    finished_at: str,
    **versions: Any,
) -> dict[str, Any]:
    """``roller.provenance/v1``; ``versions`` holds the Harness-specific fields."""

    bundle = request.bundle
    return {
        "schema_version": "roller.provenance/v1",
        "roller_version": __version__,
        "task_id": request.task.id,
        "task_bundle_digest": bundle.manifest_digest if bundle else None,
        "input_tree_digest": bundle.input_tree_digest if bundle else None,
        "input_manifest_digest": bundle.input_manifest_digest if bundle else None,
        "context_manifest_digest": bundle.context_manifest_digest if bundle else None,
        "resume_safe": bundle.resume_safe if bundle else None,
        "harness": harness,
        **versions,
        "docker_image": docker_image,
        "docker_image_digest": docker_image_digest,
        "model": request.model,
        "api_base_urls": api_base_urls,
        "environment_names": sorted(environment),
        "attempt": attempt,
        "started_at": started_at,
        "finished_at": finished_at,
    }


class ContainerCommands:
    """Docker helpers for a Harness exposing ``config`` and ``runtime``."""

    error_limit = 2000
    # Monotonic end of the running session's task budget, set by its start().
    deadline: float | None = None

    @classmethod
    def _error_text(cls, result: subprocess.CompletedProcess[str]) -> str:
        return error_text(result, cls.error_limit)

    def _bounded(self, timeout: float | None, description: str) -> float | None:
        """Clip a step's own timeout to what is left of the task budget."""
        if self.deadline is None:
            return timeout
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{description}: task time budget exhausted")
        return remaining if timeout is None else min(timeout, remaining)

    def _checked(
        self,
        args: Sequence[str],
        description: str,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self.runtime.run(args, timeout=self._bounded(timeout, description))
        except subprocess.TimeoutExpired as exc:
            # A setup step hitting its own limit is not a task timeout. Drop the
            # exception too: its command text can carry container credentials.
            self._bounded(None, description)  # unless the budget was the limit
            raise RuntimeError(
                f"{description} timed out after {exc.timeout:g}s"
            ) from None
        if result.returncode != 0:
            raise RuntimeError(f"{description} failed: {self._error_text(result)}")
        return result

    def _collect_outputs(
        self, container: str, request: RunRequest, output_dir: Path
    ) -> tuple[ArtifactCollectionResult, str | None]:
        return collect_outputs(
            self.runtime,
            container,
            request,
            output_dir,
            self.config.workspace_container_path,
            error_limit=self.error_limit,
        )

    def _image_digest(self) -> str | None:
        """Id of the image a tag points at now; sessions start their container from it.

        Running by id keeps the recorded id equal to the image in use even if
        the tag moves between this inspection and container creation.
        """
        result = self.runtime.run(
            ["image", "inspect", self.config.docker_image, "--format", "{{.Id}}"]
        )
        digest = (result.stdout.strip() or None) if result.returncode == 0 else None
        warn_if_image_changed(
            logger, self.config.docker_image, self.config.preflight_image_digest, digest
        )
        return digest
