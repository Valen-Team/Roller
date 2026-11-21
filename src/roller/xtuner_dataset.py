"""Export a Roller run as an XTuner multimodal SFT dataset.

``roller export RUN_DIR --format xtuner`` merges both unsegmented samples and
compaction segments into a single ``xtuner.jsonl`` (with a line-for-line
``xtuner_index.jsonl`` beside it):

- Messages keep standard OpenAI multimodal content lists, attaching
  ``image_wh: [width, height]`` (EXIF-oriented) to every ``image_url`` block.
- Images are extracted to ``<output>/media/`` (content-addressed, relative
  paths for XTuner's ``media_root``) unless ``inline_images=True`` keeps
  base64 data URIs inline.
- ``tool_calls[].function.arguments`` JSON strings are parsed into objects.
- ``target_start_index`` on compaction segments is translated into per-message
  ``"loss": false`` on earlier assistant turns.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roller.blobs import BLOBS_DIR
from roller.sft import normalize_sft_record, parse_image_data_uri
from roller.sft_dataset import (
    Sample,
    load_run_samples,
    sample_records,
    select_samples,
)
from roller.storage import write_jsonl

XTUNER_FILE = "xtuner.jsonl"
XTUNER_INDEX_FILE = "xtuner_index.jsonl"
MEDIA_DIR = "media"

_MEDIA_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}
_JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)


def _jpeg_exif_orientation(segment: bytes) -> int:
    """Return EXIF Orientation (1..8) from an APP1 payload, or 1 if absent."""
    if len(segment) < 14 or not segment.startswith(b"Exif\x00\x00"):
        return 1
    tiff = segment[6:]
    byte_order = tiff[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return 1
    if len(tiff) < 8 or struct.unpack(f"{endian}H", tiff[2:4])[0] != 42:
        return 1
    ifd_offset = struct.unpack(f"{endian}I", tiff[4:8])[0]
    if ifd_offset + 2 > len(tiff):
        return 1
    entry_count = struct.unpack(f"{endian}H", tiff[ifd_offset : ifd_offset + 2])[0]
    pos = ifd_offset + 2
    for _ in range(entry_count):
        if pos + 12 > len(tiff):
            break
        tag, field_type, count = struct.unpack(f"{endian}HHI", tiff[pos : pos + 8])
        if tag == 0x0112 and field_type == 3 and count >= 1:
            return struct.unpack(f"{endian}H", tiff[pos + 8 : pos + 10])[0]
        pos += 12
    return 1


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("invalid JPEG image header")
    pos = 2
    orientation = 1
    dims: tuple[int, int] | None = None
    while pos < len(data):
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker in (0xD9, 0xDA):  # EOI or Start of Scan
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if pos + 2 > len(data):
            break
        seg_len = struct.unpack(">H", data[pos : pos + 2])[0]
        if seg_len < 2 or pos + seg_len > len(data):
            break
        seg = data[pos + 2 : pos + seg_len]
        if marker == 0xE1 and orientation == 1:
            orientation = _jpeg_exif_orientation(seg)
        elif marker in _JPEG_SOF_MARKERS and len(seg) >= 5 and dims is None:
            height, width = struct.unpack(">HH", seg[1:5])
            dims = (width, height)
        pos += seg_len
    if dims is None:
        raise ValueError("JPEG image lacks a start-of-frame header")
    width, height = dims
    if orientation in {5, 6, 7, 8}:
        return height, width
    return width, height


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("invalid WebP image header")
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    if chunk == b"VP8L":
        if data[20] != 0x2F:
            raise ValueError("invalid lossless WebP header")
        bits = struct.unpack("<I", data[21:25])[0]
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            raise ValueError("invalid lossy WebP header")
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    raise ValueError(f"unsupported WebP chunk: {chunk!r}")


def image_dimensions(data: bytes, media_type: str) -> tuple[int, int]:
    """Return EXIF-oriented ``(width, height)`` for PNG, JPEG, WebP or GIF bytes."""
    if media_type == "image/png":
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
            raise ValueError("invalid PNG image header")
        width, height = struct.unpack(">II", data[16:24])
    elif media_type == "image/gif":
        if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
            raise ValueError("invalid GIF image header")
        width, height = struct.unpack("<HH", data[6:10])
    elif media_type == "image/jpeg":
        width, height = _jpeg_dimensions(data)
    elif media_type == "image/webp":
        width, height = _webp_dimensions(data)
    else:
        raise ValueError(f"unsupported image media type: {media_type!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"image has non-positive dimensions: {width}x{height}")
    return width, height


def _parse_tool_arguments(call: dict[str, Any], location: str) -> None:
    function = call.get("function")
    if not isinstance(function, dict):
        return
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location} has invalid JSON arguments") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"{location} arguments must decode to an object")
    function["arguments"] = arguments


def sft_to_xtuner(
    record: Mapping[str, Any],
    *,
    media_dir: Path | None = None,
    media_prefix: str = MEDIA_DIR,
    written_media: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert one Roller SFT record (new or legacy) to XTuner multimodal SFT format."""
    normalized = normalize_sft_record(record)
    target_start = normalized.pop("target_start_index", None)
    media_cache = written_media if written_media is not None else {}

    messages: list[dict[str, Any]] = normalized["messages"]
    for msg_index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            for part_index, part in enumerate(content):
                if part.get("type") != "image_url":
                    continue
                image_url = part["image_url"]
                uri = image_url["url"]
                media_type, raw = parse_image_data_uri(
                    uri, f"messages[{msg_index}].content[{part_index}].image_url.url"
                )
                width, height = image_dimensions(raw, media_type)
                image_url["image_wh"] = [width, height]
                if media_dir is not None:
                    digest = hashlib.sha256(raw).hexdigest()
                    relative = media_cache.get(digest)
                    if relative is None:
                        ext = _MEDIA_EXTENSIONS[media_type]
                        filename = f"{digest[:16]}.{ext}"
                        media_dir.mkdir(parents=True, exist_ok=True)
                        dest = media_dir / filename
                        if not dest.exists():
                            dest.write_bytes(raw)
                        relative = f"{media_prefix}/{filename}"
                        media_cache[digest] = relative
                    image_url["url"] = relative

        if message.get("role") == "assistant":
            if target_start is not None and msg_index < target_start:
                message["loss"] = False
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call_index, call in enumerate(calls):
                    if isinstance(call, dict):
                        _parse_tool_arguments(
                            call, f"messages[{msg_index}].tool_calls[{call_index}]"
                        )

    return {"messages": messages, "tools": normalized["tools"]}


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


@dataclass(frozen=True)
class Written:
    """What one XTuner export kept, wrote and left out."""

    records: int
    images: int
    inline_images: bool
    samples: int
    dropped: dict[str, int]
    score_range: tuple[float, float] | None

    def lines(self, output: Path, min_score: float | None = None) -> list[str]:
        media_note = ""
        if self.images and not self.inline_images:
            media_note = f", {_count(self.images, 'image')} in {MEDIA_DIR}/"
        lines = [
            f"Exported {_count(self.records, 'record')}{media_note} to "
            f"{output / XTUNER_FILE} (index: {XTUNER_INDEX_FILE})"
        ]
        dropped = sum(self.dropped.values())
        lines.append(
            f"Kept {self.samples - dropped} of {_count(self.samples, 'sample')}"
            + (f"; dropped {dropped}:" if dropped else "")
        )
        lines += [
            f"  {reason}: {count}" for reason, count in sorted(self.dropped.items())
        ]
        if min_score is not None:
            lines.append(
                f"--min-score {min_score:g} is compared with the score that "
                "`roller bench report` lists for each sample; "
                + (
                    f"in this run they range from {self.score_range[0]:g} "
                    f"to {self.score_range[1]:g}."
                    if self.score_range
                    else "no sample of this run has one."
                )
            )
        return lines


def write_xtuner_dataset(
    directory: Path,
    run_dir: Path,
    samples: Iterable[Sample],
    min_score: float | None = None,
    *,
    inline_images: bool = False,
) -> Written:
    """Write a merged XTuner JSONL dataset, its index, and optional ``media/`` files."""
    kept, dropped, score_range, total = select_samples(run_dir, samples, min_score)
    media_dir = None if inline_images else directory / MEDIA_DIR
    written_media: dict[str, str] = {}
    index: list[dict[str, Any]] = []

    def records() -> Iterator[dict[str, Any]]:
        for sample_dir, result, row in kept:
            for record, idx_row in sample_records(
                sample_dir, result, row, run_dir / BLOBS_DIR
            ):
                converted = sft_to_xtuner(
                    record, media_dir=media_dir, written_media=written_media
                )
                index.append(idx_row)
                yield converted

    index_path = directory / XTUNER_INDEX_FILE
    index_path.unlink(missing_ok=True)
    write_jsonl(directory / XTUNER_FILE, records())
    write_jsonl(index_path, index)
    return Written(
        records=len(index),
        images=len(written_media),
        inline_images=inline_images,
        samples=total,
        dropped=dropped,
        score_range=score_range,
    )


def export_run(
    run_dir: Path,
    output: Path,
    min_score: float | None = None,
    *,
    inline_images: bool = False,
) -> Written:
    """Export all kept linear and segmented records of a run for XTuner."""
    return write_xtuner_dataset(
        output,
        run_dir,
        load_run_samples(run_dir),
        min_score,
        inline_images=inline_images,
    )
