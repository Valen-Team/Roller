"""Docker image composition for the SWE-Bench Pro harness."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

IMAGE_PULL_ATTEMPTS = 5
IMAGE_PULL_RETRY_SECONDS = 15
_IMAGE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]*$")
_IMAGE_LOCKS: dict[str, threading.Lock] = {}
_IMAGE_LOCKS_GUARD = threading.Lock()
_DERIVED_IMAGES: dict[tuple[str, str, str, str], str] = {}
_HARNESS_IMAGE_REVISIONS = {"openclaw": "isolated-node-v7"}
# Docker architecture -> (multiarch library directory, dynamic loader). The
# OpenClaw runtime carries its own glibc into instance images that ship an
# older one, so both names depend on the image architecture.
_GLIBC_LAYOUTS = {
    "amd64": ("x86_64-linux-gnu", "ld-linux-x86-64.so.2"),
    "arm64": ("aarch64-linux-gnu", "ld-linux-aarch64.so.1"),
}


def _image_lock(image: str) -> threading.Lock:
    """Serialize pulls and builds of one image without blocking the others."""

    with _IMAGE_LOCKS_GUARD:
        return _IMAGE_LOCKS.setdefault(image, threading.Lock())


def _image_id(runtime: Any, image: str) -> str:
    with _image_lock(image):
        return _pull_image_id(runtime, image)


def _image_architecture(runtime: Any, image: str) -> str:
    inspected = runtime.run(
        ["image", "inspect", image, "--format", "{{.Architecture}}"], timeout=60
    )
    if inspected.returncode != 0 or not inspected.stdout.strip():
        detail = (inspected.stderr or inspected.stdout or "").strip()[-1000:]
        raise RuntimeError(f"could not inspect Docker image {image}: {detail}")
    return inspected.stdout.strip()


def _pull_image_id(runtime: Any, image: str) -> str:
    inspected = runtime.run(
        ["image", "inspect", image, "--format", "{{.Id}}"], timeout=60
    )
    if inspected.returncode != 0:
        detail = ""
        for attempt in range(1, IMAGE_PULL_ATTEMPTS + 1):
            logger.info(
                "pulling SWE-Bench Pro runtime image %s (attempt %s/%s)",
                image,
                attempt,
                IMAGE_PULL_ATTEMPTS,
            )
            try:
                pulled = runtime.run(["pull", image], timeout=3600)
            except subprocess.TimeoutExpired as exc:
                detail = str(exc)
            else:
                detail = (pulled.stderr or pulled.stdout or "").strip()[-2000:]
            inspected = runtime.run(
                ["image", "inspect", image, "--format", "{{.Id}}"], timeout=60
            )
            if inspected.returncode == 0 and inspected.stdout.strip():
                break
            if attempt < IMAGE_PULL_ATTEMPTS:
                delay = min(
                    IMAGE_PULL_RETRY_SECONDS * (2 ** (attempt - 1)),
                    120,
                )
                logger.warning(
                    "Docker image pull failed for %s; retrying in %ss: %s",
                    image,
                    delay,
                    detail,
                )
                time.sleep(delay)
        else:
            raise RuntimeError(f"could not pull Docker image {image}: {detail}")
    if inspected.returncode != 0 or not inspected.stdout.strip():
        detail = (inspected.stderr or inspected.stdout or "").strip()[-1000:]
        raise RuntimeError(f"could not inspect Docker image {image}: {detail}")
    return inspected.stdout.strip()


def _openclaw_dockerfile(source_image: str, harness_image: str, architecture: str) -> str:
    library_dir, loader = _GLIBC_LAYOUTS[architecture]
    return f"""\
FROM {harness_image} AS harness
FROM {source_image}
USER root
COPY --from=harness /usr/local/bin/node /opt/roller-node/bin/node.real
COPY --from=harness /lib/{library_dir} /opt/roller-glibc
COPY --from=harness /app /opt/roller-openclaw
RUN if /opt/roller-node/bin/node.real --version >/dev/null 2>&1; then \\
        mv /opt/roller-node/bin/node.real /opt/roller-node/bin/node; \\
        rm -rf /opt/roller-glibc; \\
    else \\
        printf '%s\\n' '#!/bin/sh' \\
            'exec /opt/roller-glibc/{loader} --library-path /opt/roller-glibc /opt/roller-node/bin/node.real "$@"' \\
            > /opt/roller-node/bin/node; \\
        chmod 755 /opt/roller-node/bin/node; \\
    fi \\
    && printf '%s\\n' '#!/bin/sh' 'OPENCLAW_NO_RESPAWN=1 exec /opt/roller-node/bin/node /opt/roller-openclaw/openclaw.mjs "$@"' > /usr/local/bin/openclaw \\
    && chmod 755 /usr/local/bin/openclaw /opt/roller-node/bin/node \\
    && /opt/roller-node/bin/node --version
ENV PATH="/usr/local/bin:${{PATH}}"
ENTRYPOINT []
CMD ["/bin/bash"]
"""


def _repository_dockerfile(source_image: str, harness_image: str) -> str:
    return f"""\
FROM {source_image} AS instance
FROM {harness_image}
USER root
COPY --from=instance /app /app
ENTRYPOINT []
CMD ["/bin/bash"]
"""


def ensure_harness_image(
    runtime: Any,
    source_image: str,
    harness_image: str,
    harness_name: str,
) -> str:
    """Build an Agent image with one Harness and the official repository snapshot."""

    for image in (source_image, harness_image):
        if not _IMAGE_REF_RE.fullmatch(image):
            raise ValueError(f"unsafe Docker image reference: {image!r}")
    source_id = _image_id(runtime, source_image)
    harness_id = _image_id(runtime, harness_image)
    revision = _HARNESS_IMAGE_REVISIONS.get(harness_name, "v1")
    key = (source_id, harness_id, harness_name, revision)
    digest = hashlib.sha256("\0".join(key).encode()).hexdigest()[:20]
    tag = f"roller-swebench-pro-{harness_name}:{digest}"
    with _image_lock(tag):
        cached = _DERIVED_IMAGES.get(key)
        if cached is not None:
            return cached
        existing = runtime.run(
            ["image", "inspect", tag, "--format", "{{.Id}}"], timeout=60
        )
        if existing.returncode == 0:
            _DERIVED_IMAGES[key] = tag
            return tag
        # The derived image mixes files from both images, so they must target
        # the same architecture; the official instance images are linux/amd64.
        architecture = _image_architecture(runtime, source_image)
        harness_architecture = _image_architecture(runtime, harness_image)
        if harness_architecture != architecture:
            raise RuntimeError(
                f"SWE-Bench Pro image {source_image} is {architecture} but "
                f"{harness_name} image {harness_image} is {harness_architecture}; "
                f"use a linux/{architecture} {harness_name} image"
            )
        if harness_name == "openclaw":
            if architecture not in _GLIBC_LAYOUTS:
                supported = ", ".join(sorted(_GLIBC_LAYOUTS))
                raise RuntimeError(
                    f"SWE-Bench Pro with openclaw supports {supported} images; "
                    f"{source_image} is {architecture}"
                )
            dockerfile = _openclaw_dockerfile(source_image, harness_image, architecture)
        else:
            dockerfile = _repository_dockerfile(source_image, harness_image)
        with tempfile.TemporaryDirectory(prefix="roller-swebench-image-") as temporary:
            context = Path(temporary)
            (context / "Dockerfile").write_text(dockerfile, encoding="utf-8")
            built = runtime.run(["build", "-t", tag, str(context)], timeout=3600)
        if built.returncode != 0:
            detail = (built.stderr or built.stdout or "").strip()[-4000:]
            raise RuntimeError(
                f"could not combine {harness_name} with SWE-Bench Pro image "
                f"{source_image}: {detail}"
            )
        _DERIVED_IMAGES[key] = tag
        return tag
