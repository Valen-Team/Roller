from __future__ import annotations

import base64
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from roller.bundles import BundleError, TaskBundleStore
from roller.cli import _build_parser, _collect
from roller.models import RunRequest
from roller.native_images import (
    attached_images,
    claude_user_message,
    image_source,
    openclaw_message,
)
from roller.tasks import TaskFormatError, load_tasks


class AttachImagesTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "chart.png").write_bytes(b"png-bytes")
        (self.root / "photos").mkdir()
        (self.root / "photos" / "cat.JPG").write_bytes(b"jpg-bytes")
        self.tasks = self.root / "tasks.json"

    def load(self, files: dict, **row):
        self.tasks.write_text(json.dumps([{"id": "inspect", "prompt": "Describe it", "files": files, **row}]))
        return load_tasks(self.tasks)[0]

    def files(self, **extra) -> dict:
        return {
            "inputs": [{"source": "chart.png", "target": "inputs/chart.png"}, "photos"],
            **extra,
        }

    def test_selected_inputs_keep_their_order_and_default_to_none(self) -> None:
        self.assertEqual(self.load(self.files()).attach_images, ())
        task = self.load(self.files(attach_images=["photos/cat.JPG", "inputs/chart.png"]))
        self.assertEqual(task.attach_images, ("photos/cat.JPG", "inputs/chart.png"))

    def test_rejects_paths_that_are_not_declared_input_images(self) -> None:
        for value, message in (
            ("inputs/chart.png", "must be a list"),
            (["../chart.png"], "parent traversal"),
            (["/inputs/chart.png"], "absolute"),
            (["inputs/chart.png", "inputs/chart.png"], "repeats"),
            (["chart.png"], "not declared in files.inputs"),
            (["inputs/chart.png.txt"], "supports"),
            (["photos"], "supports"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(TaskFormatError, message):
                self.load(self.files(attach_images=value))

    def test_selection_changes_the_bundle_identity_only_when_used(self) -> None:
        store = TaskBundleStore(self.root / "bundles")
        plain = self.load(self.files())
        selected = self.load(self.files(attach_images=["inputs/chart.png"]))
        reordered = self.load(self.files(attach_images=["photos/cat.JPG", "inputs/chart.png"]))
        digests = [store.prepare(task).manifest_digest for task in (plain, selected, reordered)]
        self.assertEqual(len(set(digests)), 3)
        # A task that attaches nothing keeps the digest it had before the field existed.
        self.assertEqual(store.prepare(replace(plain, attach_images=())).manifest_digest, digests[0])

        missing = self.load(self.files(attach_images=["photos/dog.png"]))
        with self.assertRaisesRegex(BundleError, "not an input file: photos/dog.png"):
            store.prepare(missing)

    def test_messages_carry_images_on_the_first_message_only(self) -> None:
        task = self.load(self.files(attach_images=["inputs/chart.png", "photos/cat.JPG"]))
        request = RunRequest(0, task, "model", 30)
        self.assertEqual(attached_images(request, "/workspace/", first_message=False), [])
        self.assertEqual(
            attached_images(request, "/workspace/", first_message=True),
            [
                {"path": "/workspace/inputs/chart.png", "mime_type": "image/png"},
                {"path": "/workspace/photos/cat.JPG", "mime_type": "image/jpeg"},
            ],
        )
        self.assertEqual(
            openclaw_message(request, "Describe it", "/workspace", first_message=True),
            "[media attached: /workspace/inputs/chart.png (image/png)]\n"
            "[media attached: /workspace/photos/cat.JPG (image/jpeg)]\n"
            "Describe it",
        )
        self.assertEqual(
            openclaw_message(request, "Next", "/workspace", first_message=False), "Next"
        )

    def test_claude_message_reads_the_snapshot_or_the_declared_source(self) -> None:
        task = self.load(self.files(attach_images=["inputs/chart.png", "photos/cat.JPG"]))
        request = RunRequest(0, task, "model", 30)
        self.assertEqual(image_source(request, "photos/cat.JPG").read_bytes(), b"jpg-bytes")
        self.assertEqual(image_source(request, "inputs/chart.png").read_bytes(), b"png-bytes")
        bundled = replace(request, bundle=TaskBundleStore(self.root / "bundles").prepare(task))
        (self.root / "chart.png").write_bytes(b"changed after the snapshot")
        message = claude_user_message(bundled, "Describe it")
        self.assertEqual(message["type"], "user")
        self.assertEqual(message["message"]["content"], [
            {"type": "text", "text": "Describe it"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(b"png-bytes").decode()}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(b"jpg-bytes").decode()}},
        ])

    def test_cli_rejects_harnesses_that_cannot_attach_images(self) -> None:
        self.load(self.files(attach_images=["inputs/chart.png"]))

        def dry_run(*extra: str) -> dict:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                _collect(_build_parser().parse_args([
                    "collect", "--tasks", str(self.tasks), "--model", "provider/model",
                    "--dry-run", *extra,
                ]))
            return json.loads(output.getvalue())

        for harness in ("openclaw", "codex", "claude-code", "opencode", "pi"):
            with self.subTest(harness=harness):
                plan = dry_run("--harness", harness)["plan"][0]
                self.assertEqual(plan["attach_images"], ["inputs/chart.png"])
        for extra, message in (
            (("--harness", "hermes"), "not supported by the Hermes Harness"),
            (("--harness", "deepseek"), "not supported by the DeepSeek Harness"),
            (("--harness", "opencode", "--no-opencode-vision"), "--opencode-vision"),
            (("--harness", "pi", "--pi-input-modalities", "text"), "--pi-input-modalities text,image"),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                dry_run(*extra)


if __name__ == "__main__":
    unittest.main()
