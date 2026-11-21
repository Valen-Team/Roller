from __future__ import annotations

import base64
import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from roller import cli, hf_dataset, xtuner_dataset
from roller.storage import atomic_write_json


def _make_png(width: int, height: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + b"\x00\x00\x00\x00"
    )


def _make_gif(width: int, height: int, version: bytes = b"GIF89a") -> bytes:
    return version + struct.pack("<HH", width, height) + b"\x00\x00\x00"


def _make_jpeg(
    width: int,
    height: int,
    *,
    orientation: int | None = None,
    big_endian_exif: bool = False,
    sof_marker: int = 0xC0,
) -> bytes:
    parts = [b"\xff\xd8"]
    if orientation is not None:
        endian = ">" if big_endian_exif else "<"
        order = b"MM" if big_endian_exif else b"II"
        tiff = (
            order
            + struct.pack(f"{endian}HI", 42, 8)
            + struct.pack(f"{endian}H", 1)
            + struct.pack(f"{endian}HHI", 0x0112, 3, 1)
            + struct.pack(f"{endian}HH", orientation, 0)
            + struct.pack(f"{endian}I", 0)
        )
        app1 = b"Exif\x00\x00" + tiff
        parts.append(b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1)
    sof = (
        struct.pack(">BHHB", 8, height, width, 3)
        + b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    )
    parts.append(bytes([0xFF, sof_marker]) + struct.pack(">H", len(sof) + 2) + sof)
    parts.append(b"\xff\xd9")
    return b"".join(parts)


def _make_webp_vp8x(width: int, height: int) -> bytes:
    payload = (
        b"\x00\x00\x00\x00"
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )
    chunk = b"VP8X" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk + b"\x00" * 4


def _make_webp_vp8l(width: int, height: int) -> bytes:
    bits = (width - 1) | ((height - 1) << 14)
    payload = b"\x2f" + struct.pack("<I", bits) + b"\x00" * 10
    chunk = b"VP8L" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def _make_webp_vp8(width: int, height: int) -> bytes:
    payload = (
        b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", width, height) + b"\x00" * 6
    )
    chunk = b"VP8 " + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def _data_uri(data: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_sample(
    run_dir: Path,
    name: str,
    *,
    task_id: str,
    request_index: int,
    score: float | None = 1.0,
    sft_record: dict | None = None,
    segments: list[dict] | None = None,
) -> Path:
    sample_dir = run_dir / "samples" / "glm" / name
    sample_dir.mkdir(parents=True)
    trajectory: dict = {"actf": "trace_ACTF.json", "sft": None}
    if segments is not None:
        seg_dir = sample_dir / "sft_segments"
        seg_dir.mkdir()
        seg_meta = []
        for idx, rec in enumerate(segments):
            rel = f"sft_segments/{idx + 1:04d}.json"
            atomic_write_json(sample_dir / rel, rec)
            seg_meta.append({"path": rel, "window_index": idx})
        trajectory["sft_segments"] = seg_meta
    elif sft_record is not None:
        atomic_write_json(sample_dir / "trace_SFT.json", sft_record)
        trajectory["sft"] = "trace_SFT.json"

    runtime: dict = {}
    if score is not None:
        runtime["bench"] = {
            "bench": "vials",
            "category": "Other",
            "scores": {"accuracy": score},
            "score_key": "accuracy",
        }
    atomic_write_json(
        sample_dir / "info.json",
        {
            "task_id": task_id,
            "model": "vllm/glm",
            "model_label": "glm",
            "harness": "stirrup",
            "status": "ok",
            "request_index": request_index,
            "sample_index": 0,
            "trajectory": trajectory,
            "runtime": runtime,
        },
    )
    return sample_dir


class ImageDimensionsTests(unittest.TestCase):
    def test_parses_png_gif_webp_and_jpeg_headers(self) -> None:
        self.assertEqual(
            xtuner_dataset.image_dimensions(_make_png(640, 360), "image/png"),
            (640, 360),
        )
        self.assertEqual(
            xtuner_dataset.image_dimensions(_make_gif(320, 240), "image/gif"),
            (320, 240),
        )
        self.assertEqual(
            xtuner_dataset.image_dimensions(_make_gif(128, 64, b"GIF87a"), "image/gif"),
            (128, 64),
        )
        self.assertEqual(
            xtuner_dataset.image_dimensions(_make_webp_vp8x(1024, 768), "image/webp"),
            (1024, 768),
        )
        self.assertEqual(
            xtuner_dataset.image_dimensions(_make_webp_vp8l(512, 256), "image/webp"),
            (512, 256),
        )
        self.assertEqual(
            xtuner_dataset.image_dimensions(_make_webp_vp8(800, 600), "image/webp"),
            (800, 600),
        )
        self.assertEqual(
            xtuner_dataset.image_dimensions(_make_jpeg(1920, 1080), "image/jpeg"),
            (1920, 1080),
        )
        self.assertEqual(
            xtuner_dataset.image_dimensions(
                _make_jpeg(1200, 800, sof_marker=0xC2), "image/jpeg"
            ),
            (1200, 800),
        )

    def test_jpeg_applies_exif_orientation_swap(self) -> None:
        # Orientation 1 keeps width x height; 6 and 8 swap to height x width.
        self.assertEqual(
            xtuner_dataset.image_dimensions(
                _make_jpeg(400, 300, orientation=1), "image/jpeg"
            ),
            (400, 300),
        )
        self.assertEqual(
            xtuner_dataset.image_dimensions(
                _make_jpeg(400, 300, orientation=6), "image/jpeg"
            ),
            (300, 400),
        )
        self.assertEqual(
            xtuner_dataset.image_dimensions(
                _make_jpeg(400, 300, orientation=8, big_endian_exif=True), "image/jpeg"
            ),
            (300, 400),
        )

    def test_invalid_headers_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            xtuner_dataset.image_dimensions(b"not-a-png", "image/png")
        with self.assertRaises(ValueError):
            xtuner_dataset.image_dimensions(b"not-a-jpeg", "image/jpeg")


class XtunerConversionTests(unittest.TestCase):
    def test_converts_openai_record_extracts_deduped_media_and_parses_tool_args(
        self,
    ) -> None:
        png = _make_png(640, 360)
        uri = _data_uri(png, "image/png")
        record = {
            "messages": [
                {"role": "system", "content": "System"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect <image> literal"},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                },
                {
                    "role": "assistant",
                    "reasoning_content": "Check pixels",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "view_image",
                                "arguments": '{"path":"a.png","zoom":2}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": [
                        {"type": "text", "text": "ok"},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                },
                {
                    "role": "assistant",
                    "reasoning_content": "Done",
                    "content": "42",
                    "tool_calls": [],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "view_image"}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            media_dir = Path(tmp) / "media"
            cache: dict[str, str] = {}
            converted = xtuner_dataset.sft_to_xtuner(
                record, media_dir=media_dir, written_media=cache
            )
            self.assertEqual(set(converted), {"messages", "tools"})
            self.assertEqual(len(cache), 1)
            files = list(media_dir.iterdir())
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_bytes(), png)

            user_img = converted["messages"][1]["content"][1]["image_url"]
            tool_img = converted["messages"][3]["content"][1]["image_url"]
            self.assertEqual(user_img["url"], f"media/{files[0].name}")
            self.assertEqual(user_img["image_wh"], [640, 360])
            self.assertEqual(tool_img, user_img)
            self.assertEqual(
                converted["messages"][1]["content"][0]["text"],
                "Inspect <image> literal",
            )
            self.assertEqual(
                converted["messages"][2]["tool_calls"][0]["function"]["arguments"],
                {"path": "a.png", "zoom": 2},
            )

    def test_inline_images_keeps_data_uri_with_image_wh(self) -> None:
        uri = _data_uri(_make_png(128, 96), "image/png")
        record = {
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": uri}}],
                },
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": "ok",
                    "tool_calls": [],
                },
            ],
            "tools": [],
        }
        converted = xtuner_dataset.sft_to_xtuner(record, media_dir=None)
        self.assertEqual(
            converted["messages"][1]["content"][0]["image_url"],
            {"url": uri, "image_wh": [128, 96]},
        )

    def test_translates_target_start_index_to_assistant_loss_false(self) -> None:
        record = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "q1"},
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": "a1",
                    "tool_calls": [],
                },
                {"role": "user", "content": "q2"},
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": "a2",
                    "tool_calls": [],
                },
                {"role": "user", "content": "q3"},
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": "a3",
                    "tool_calls": [],
                },
            ],
            "tools": [],
            "target_start_index": 4,
        }
        converted = xtuner_dataset.sft_to_xtuner(record)
        self.assertNotIn("target_start_index", converted)
        self.assertIs(converted["messages"][2]["loss"], False)
        self.assertNotIn("loss", converted["messages"][4])
        self.assertNotIn("loss", converted["messages"][6])
        self.assertNotIn("loss", converted["messages"][1])

    def test_upgrades_legacy_placeholder_record_without_synthetic_newline(self) -> None:
        uri = _data_uri(_make_png(300, 200), "image/png")
        legacy = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Here is the image\n<image>"},
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": "done",
                    "tool_calls": [],
                },
            ],
            "tools": [],
            "images": [uri],
        }
        converted = xtuner_dataset.sft_to_xtuner(legacy, media_dir=None)
        self.assertEqual(
            converted["messages"][1]["content"],
            [
                {"type": "text", "text": "Here is the image"},
                {
                    "type": "image_url",
                    "image_url": {"url": uri, "image_wh": [300, 200]},
                },
            ],
        )


class XtunerExportRunAndCliTests(unittest.TestCase):
    def test_cli_export_merges_linear_and_segments_with_min_score(self) -> None:
        uri = _data_uri(_make_png(200, 100), "image/png")
        linear = {
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look"},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                },
                {
                    "role": "assistant",
                    "reasoning_content": "r",
                    "content": "a",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "finish", "arguments": '{"ok":true}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "done"},
            ],
            "tools": [{"type": "function", "function": {"name": "finish"}}],
        }
        seg0 = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "part 1"},
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": "ans 1",
                    "tool_calls": [],
                },
            ],
            "tools": [],
            "target_start_index": 2,
        }
        seg1 = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "part 1"},
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": "ans 1",
                    "tool_calls": [],
                },
                {"role": "user", "content": "summarize"},
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": "summary",
                    "tool_calls": [],
                },
            ],
            "tools": [],
            "target_start_index": 4,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_sample(
                run_dir,
                "0000_pass_linear",
                task_id="t1",
                request_index=0,
                score=1.0,
                sft_record=linear,
            )
            _write_sample(
                run_dir,
                "0001_fail_linear",
                task_id="t2",
                request_index=1,
                score=0.0,
                sft_record=linear,
            )
            _write_sample(
                run_dir,
                "0002_pass_seg",
                task_id="t3",
                request_index=2,
                score=1.0,
                segments=[seg0, seg1],
            )

            out = Path(tmp) / "xtuner_out"
            buf = io.StringIO()
            with redirect_stdout(buf), self.assertRaises(SystemExit) as exited:
                cli.main(
                    [
                        "export",
                        str(run_dir),
                        "--format",
                        "xtuner",
                        "--min-score",
                        "1.0",
                        "--output",
                        str(out),
                    ]
                )
            self.assertEqual(exited.exception.code, 0)
            self.assertIn("Exported 3 records, 1 image in media/", buf.getvalue())

            rows = _read_jsonl(out / xtuner_dataset.XTUNER_FILE)
            idx = _read_jsonl(out / xtuner_dataset.XTUNER_INDEX_FILE)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(idx), 3)
            self.assertEqual([r["task_id"] for r in idx], ["t1", "t3", "t3"])
            self.assertEqual(idx[1]["segment"], "sft_segments/0001.json")
            self.assertEqual(idx[2]["segment"], "sft_segments/0002.json")

            # Linear record parsed args and extracted image.
            self.assertEqual(
                rows[0]["messages"][2]["tool_calls"][0]["function"]["arguments"],
                {"ok": True},
            )
            img_ref = rows[0]["messages"][1]["content"][1]["image_url"]
            self.assertTrue(img_ref["url"].startswith("media/"))
            self.assertEqual(img_ref["image_wh"], [200, 100])
            self.assertTrue((out / img_ref["url"]).is_file())

            # Segmented record translated target_start_index.
            self.assertIs(rows[2]["messages"][2]["loss"], False)
            self.assertNotIn("loss", rows[2]["messages"][4])

            # --inline-images keeps base64 URI and creates no media/ dir.
            out_inline = Path(tmp) / "xtuner_inline"
            with (
                redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as exited,
            ):
                cli.main(
                    [
                        "export",
                        str(run_dir),
                        "--format",
                        "xtuner",
                        "--min-score",
                        "1.0",
                        "--inline-images",
                        "--output",
                        str(out_inline),
                    ]
                )
            self.assertEqual(exited.exception.code, 0)
            self.assertFalse((out_inline / "media").exists())
            inline_rows = _read_jsonl(out_inline / xtuner_dataset.XTUNER_FILE)
            self.assertEqual(
                inline_rows[0]["messages"][1]["content"][1]["image_url"]["url"], uri
            )

    def test_inline_images_rejected_for_non_xtuner_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "samples").mkdir(parents=True)
            with self.assertRaises(SystemExit) as exited:
                cli.main(["export", str(run_dir), "--format", "sft", "--inline-images"])
            self.assertEqual(exited.exception.code, 2)

    def test_hf_export_include_runs_excludes_xtuner_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            record = {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "reasoning_content": "",
                        "content": "hello",
                        "tool_calls": [],
                    },
                ],
                "tools": [],
            }
            sample = _write_sample(
                run_dir,
                "00000000_t1",
                task_id="t1",
                request_index=0,
                score=1.0,
                sft_record=record,
            )
            (run_dir / "info.json").write_text(
                json.dumps({"state": "finished", "harness": "stirrup"}),
                encoding="utf-8",
            )
            (sample / "trace_ACTF.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ACTF_v1.0",
                        "system_prompt": "",
                        "tools": [],
                        "steps": [],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "xtuner").mkdir()
            (run_dir / "xtuner" / "xtuner.jsonl").write_text("{}\n", encoding="utf-8")

            output = Path(tmp) / "hf_out"
            hf_dataset.export_run(run_dir, output, environ={}, runs=True, machine="")
            exported_runs = list((output / "runs").iterdir())
            self.assertEqual(len(exported_runs), 1)
            self.assertFalse((exported_runs[0] / "xtuner").exists())


if __name__ == "__main__":
    unittest.main()
