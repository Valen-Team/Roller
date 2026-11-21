from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from roller.blobs import load_blobs, store_blobs

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(256))).decode()


def image(data: str, mime: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


class BlobTests(unittest.TestCase):
    def test_each_image_is_stored_once_and_restored_exactly(self):
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "a"}, image(PNG)],
                },
                {
                    "role": "user",
                    "content": [image(PNG), image("R0lGOA==", "image/gif")],
                },
                # Not canonical base64: kept as sent.
                {"role": "user", "content": [image("R0lGOB==")]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "blobs"
            stored = store_blobs(body, directory)
            files = sorted(path.suffix for path in directory.iterdir())
            restored = load_blobs(stored, directory)
        self.assertEqual(files, [".gif", ".png"])
        urls = [
            part["image_url"]["url"]
            for message in stored["messages"]
            for part in message["content"]
            if part["type"] == "image_url"
        ]
        self.assertTrue(urls[0].startswith("data:image/png;base64,roller-blob:"))
        self.assertEqual(urls[0], urls[1])
        self.assertEqual(urls[3], "data:image/png;base64,R0lGOB==")
        self.assertNotIn(PNG, json.dumps(stored))
        self.assertEqual(json.dumps(restored), json.dumps(body))


if __name__ == "__main__":
    unittest.main()
