"""Images a task sends to the model with its first message (``files.attach_images``)."""

from __future__ import annotations

import base64
from pathlib import Path, PurePosixPath
from typing import Any

from roller.models import RunRequest

IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def image_media_type(path: str) -> str | None:
    return IMAGE_MEDIA_TYPES.get(PurePosixPath(path).suffix.lower())


def attached_images(
    request: RunRequest, workspace: str, *, first_message: bool
) -> list[dict[str, str]]:
    """Workspace path and media type of each image this message carries."""

    if not first_message:
        return []
    return [
        {"path": f"{workspace.rstrip('/')}/{path}", "mime_type": image_media_type(path)}
        for path in request.task.attach_images
    ]


def image_source(request: RunRequest, path: str) -> Path:
    """Host file with the bytes of one attached image."""

    if request.bundle is not None:
        return request.bundle.inputs_root / path
    for item in request.task.inputs:
        if item.source is None:
            continue
        if item.path == path:
            return item.source
        if path.startswith(item.path + "/"):
            return item.source / path[len(item.path) + 1 :]
    raise ValueError(f"attached image has no source file: {path}")


def claude_user_message(request: RunRequest, prompt: str) -> dict[str, Any]:
    """The first ``--input-format=stream-json`` user message, images inline."""

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in request.task.attach_images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type(path),
                    "data": base64.b64encode(
                        image_source(request, path).read_bytes()
                    ).decode("ascii"),
                },
            }
        )
    return {"type": "user", "message": {"role": "user", "content": content}}


def openclaw_message(
    request: RunRequest, message: str, workspace: str, *, first_message: bool
) -> str:
    """Prefix the media markers OpenClaw resolves into native image blocks."""

    markers = [
        f"[media attached: {image['path']} ({image['mime_type']})]"
        for image in attached_images(request, workspace, first_message=first_message)
    ]
    return "\n".join([*markers, message])
