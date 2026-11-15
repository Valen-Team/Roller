from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prebuilt_images", ROOT / "scripts" / "prebuilt_images.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrebuiltImagesTests(unittest.TestCase):
    def test_published_images_follow_the_default_image_names(self) -> None:
        images = module.prebuilt_images()
        self.assertEqual(
            [image["harness"] for image in images],
            ["codex", "hermes", "opencode", "deepseek", "pi"],
        )
        for image in images:
            with self.subTest(harness=image["harness"]):
                self.assertTrue((ROOT / image["context"] / "Dockerfile").is_file())
                self.assertEqual(image["ref"], f"ghcr.io/valen-team/{image['image']}")
                # A registry reference must be lowercase.
                self.assertEqual(image["ref"], image["ref"].lower())
                self.assertEqual(image["image"].rsplit(":", 1)[1], image["version"])

    def test_the_dockerfile_pins_the_version_in_the_image_tag(self) -> None:
        for image in module.prebuilt_images():
            dockerfile = (ROOT / image["context"] / "Dockerfile").read_text()
            pins = re.findall(r"^ARG \w*(?:VERSION|REVISION)=(\S+)$", dockerfile, re.M)
            with self.subTest(harness=image["harness"]):
                # Hermes tags the image with the short form of its revision.
                self.assertTrue(
                    any(pin.startswith(image["version"]) for pin in pins)
                    or image["version"] in dockerfile,
                    pins,
                )

    def test_claude_code_is_not_published(self) -> None:
        # Its licence does not grant redistribution.
        self.assertNotIn("claude-code", module.PUBLISHED)
