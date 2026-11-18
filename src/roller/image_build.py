"""Build task-specific benchmark environments with a Roller harness layer."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path

from roller.harnesses.docker import DockerRuntime
from roller.models import ContainerImageBuild, RunRequest

_OPENCODE_OVERLAY = """\
ARG BASE_IMAGE
ARG HARNESS_IMAGE
FROM ${HARNESS_IMAGE} AS roller_harness
FROM ${BASE_IMAGE}
USER root
RUN if ! command -v python3 >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends python3 \
      && rm -rf /var/lib/apt/lists/*; \
    fi
COPY --from=roller_harness /usr/local/bin/opencode /usr/local/bin/opencode
COPY --from=roller_harness /usr/local/bin/uv /usr/local/bin/uv
COPY --from=roller_harness /usr/local/bin/uvx /usr/local/bin/uvx
COPY --from=roller_harness /opt/roller-opencode-deps /opt/roller-opencode-deps
"""

_OVERLAYS = {"opencode": _OPENCODE_OVERLAY}


def _runtime(runtime: DockerRuntime | str) -> DockerRuntime:
    """Accept the Docker binary name this module has always been given."""

    return DockerRuntime(runtime) if isinstance(runtime, str) else runtime


@cache
def _build_command_prefix(runtime: DockerRuntime | str) -> list[str]:
    """Return the engine argv that precedes ``build`` for Docker and rootless Podman.

    Some upstream benchmark images contain ownership IDs outside the host's
    subordinate-ID range. Docker can unpack those layers directly; rootless
    Podman needs its documented single-UID fallback. The trailing ``build`` is
    included because callers assert on the whole prefix.
    """

    runtime = _runtime(runtime)
    result = runtime.version(timeout=30)
    if result.returncode == 0 and "podman" in result.stdout.lower():
        return [
            runtime.binary,
            "--storage-opt",
            "overlay.ignore_chown_errors=true",
            "build",
        ]
    return [runtime.binary, "build"]


def _inspect_image(runtime: DockerRuntime | str, image: str) -> str | None:
    # The verb itself retries nothing, so the three attempts stay here.
    runtime = _runtime(runtime)
    for attempt in range(3):
        try:
            image_id = runtime.image_id(image, timeout=120)
        except subprocess.TimeoutExpired:
            if attempt == 2:
                return None
            continue
        return image_id
    return None


def _build(
    runtime: DockerRuntime | str,
    *,
    image: str,
    context: Path,
    dockerfile_source: str,
    build_args: Iterable[tuple[str, str]],
    timeout_seconds: int,
) -> None:
    runtime = _runtime(runtime)
    build_args = tuple(build_args)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".Dockerfile", encoding="utf-8", delete=False
    ) as stream:
        stream.write(dockerfile_source.rstrip() + "\n")
        dockerfile = Path(stream.name)
    # ``_build_command_prefix`` is ``[engine, *flag, "build"]``; the runtime owns
    # the engine and the subcommand, so hand it only the flags between them.
    prefix = _build_command_prefix(runtime)
    engine = prefix[1:-1] if prefix[-1:] == ["build"] else prefix[1:]
    command = [
        "--file",
        str(dockerfile),
        "--tag",
        image,
    ]
    for name, value in build_args:
        command.extend(["--build-arg", f"{name}={value}"])
    # Docker treats these predefined proxy args specially: they are available
    # to RUN instructions but are omitted from image history unless a
    # Dockerfile explicitly redeclares them.
    declared = dict(build_args)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        lower = name.lower()
        value = (
            declared.get(name)
            or declared.get(lower)
            or os.environ.get(name)
            or os.environ.get(lower)
        )
        if not value:
            continue
        # curl commonly honors the uppercase forms while wget expects the
        # lowercase forms. Pass both predefined proxy args so upstream bench
        # Dockerfiles work consistently without persisting either in layers.
        for spelling in (name, lower):
            if spelling not in declared:
                command.extend(["--build-arg", f"{spelling}={value}"])
    command.append(str(context))
    result: subprocess.CompletedProcess | None = None
    try:
        for attempt in range(2):
            try:
                result = runtime.build(command, timeout=timeout_seconds, engine=engine)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Docker image build exceeded {timeout_seconds}s: {image}"
                ) from exc
            if result.returncode == 0 or attempt == 1:
                break
            # Registry and package mirrors occasionally reset long downloads.
            # A full build retry is cheap because successful layers are cached.
            time.sleep(2)
    finally:
        dockerfile.unlink(missing_ok=True)
    assert result is not None
    if result.returncode != 0:
        raise RuntimeError(
            f"Docker image build failed with code {result.returncode}: {image}"
        )


def prepare_task_images(
    runtime: DockerRuntime | str,
    requests: Iterable[RunRequest],
    harness_images: Mapping[str, str],
    *,
    cli_harness: str | None = None,
    rebuild: bool = False,
) -> dict[str, str]:
    """Build missing task environments and return task-id to image digest."""

    runtime = _runtime(runtime)
    specs: dict[str, tuple[ContainerImageBuild, str]] = {}
    task_images: dict[str, str] = {}
    for request in requests:
        spec = request.task.image_build
        image = request.task.docker_image
        if image is None:
            continue
        harness = cli_harness or request.task.harness
        if spec is not None:
            if harness != spec.harness:
                raise ValueError(
                    f"task {request.task.id!r} image requires the {spec.harness} "
                    f"harness, not {harness}; remove --harness or select {spec.harness}"
                )
            existing = specs.get(image)
            if existing is not None and existing[0] != spec:
                raise ValueError(f"conflicting image build definitions for {image}")
            specs[image] = (spec, harness)
        task_images[request.task.id] = image

    for image, (spec, harness) in specs.items():
        existing_image = _inspect_image(runtime, image)
        if not rebuild and existing_image:
            continue
        if rebuild or not _inspect_image(runtime, spec.base_image):
            _build(
                runtime,
                image=spec.base_image,
                context=spec.context,
                dockerfile_source=spec.dockerfile,
                build_args=spec.build_args,
                timeout_seconds=spec.timeout_seconds,
            )
        overlay = _OVERLAYS.get(harness)
        if overlay is None:
            raise ValueError(
                f"automatic task-environment layering is not implemented for {harness}"
            )
        harness_image = harness_images[harness]
        with tempfile.TemporaryDirectory(prefix="roller-image-overlay-") as temporary:
            _build(
                runtime,
                image=image,
                context=Path(temporary),
                dockerfile_source=overlay,
                build_args=(
                    ("BASE_IMAGE", spec.base_image),
                    ("HARNESS_IMAGE", harness_image),
                ),
                timeout_seconds=spec.timeout_seconds,
            )

    digests: dict[str, str] = {}
    for task_id, image in task_images.items():
        digest = _inspect_image(runtime, image)
        if not digest:
            raise RuntimeError(
                f"task-specific Docker image is unavailable after preparation: {image}"
            )
        digests[task_id] = digest
    return digests


__all__ = ["prepare_task_images"]
