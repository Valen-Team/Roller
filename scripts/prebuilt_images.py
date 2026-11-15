"""Print the Harness images that CI builds and publishes, as a JSON list.

Each entry names the Docker build context, the local image name Roller uses
by default, the published reference, and a command that prints the pinned
version from inside the image.

Claude Code is absent on purpose: its package is not open source and its
licence does not grant redistribution, so that image is built locally.
"""

from __future__ import annotations

import json

from roller.harnesses.codex import DEFAULT_CODEX_IMAGE
from roller.harnesses.deepseek import DEFAULT_DEEPSEEK_IMAGE
from roller.harnesses.hermes import DEFAULT_HERMES_IMAGE
from roller.harnesses.opencode import DEFAULT_OPENCODE_IMAGE
from roller.harnesses.pi import DEFAULT_PI_IMAGE

REGISTRY = "ghcr.io/valen-team"
PI_PACKAGE = "/opt/roller-pi/node_modules/@earendil-works/pi-coding-agent/package.json"

# harness -> (default image, upstream licence, command that prints the version)
PUBLISHED = {
    "codex": (DEFAULT_CODEX_IMAGE, "Apache-2.0", "codex --version"),
    "hermes": (DEFAULT_HERMES_IMAGE, "MIT", "cat /opt/hermes/roller-revision"),
    "opencode": (DEFAULT_OPENCODE_IMAGE, "MIT", "opencode --version"),
    "deepseek": (
        DEFAULT_DEEPSEEK_IMAGE,
        "MIT",
        "/opt/roller-deepseek/bin/python -c "
        "'from importlib.metadata import version; print(version(\"deepseek-harness-sdk\"))'",
    ),
    "pi": (
        DEFAULT_PI_IMAGE,
        "MIT",
        f"node -e 'console.log(require(\"{PI_PACKAGE}\").version)'",
    ),
}


def prebuilt_images() -> list[dict[str, str]]:
    return [
        {
            "harness": harness,
            "context": f"docker/{harness}",
            "image": image,
            "ref": f"{REGISTRY}/{image}",
            "version": image.rsplit(":", 1)[1],
            "license": license_id,
            "check": check,
        }
        for harness, (image, license_id, check) in PUBLISHED.items()
    ]


if __name__ == "__main__":
    print(json.dumps(prebuilt_images()))
