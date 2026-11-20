"""Render inline PDF documents for Messages backends that only accept images."""

from __future__ import annotations

from base64 import b64decode, b64encode
from contextlib import closing
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import Any

_pdfium_lock = Lock()


@lru_cache(maxsize=8)
def _pages(data: str) -> tuple[tuple[str, str], ...]:
    import pypdfium2

    pages = []
    # PDFium may only run in one thread per process.
    with _pdfium_lock, closing(pypdfium2.PdfDocument(b64decode(data))) as document:
        for page in document:
            with (
                closing(page),
                closing(page.get_textpage()) as text_page,
                closing(page.render(scale=2)) as bitmap,
                BytesIO() as png,
            ):
                bitmap.to_pil().save(png, format="PNG")
                pages.append(
                    (
                        text_page.get_text_bounded().strip(),
                        b64encode(png.getvalue()).decode(),
                    )
                )
    return tuple(pages)


def render_anthropic_pdfs(body: Any) -> bool:
    """Replace base64 PDFs with page text and images in place, preserving order."""
    changed = False

    def render(content: Any) -> Any:
        nonlocal changed
        if not isinstance(content, list):
            return content
        result = []
        for block in content:
            if not isinstance(block, dict):
                result.append(block)
                continue
            source = block.get("source") or {}
            if (
                block.get("type") == "document"
                and isinstance(source, dict)
                and source.get("type") == "base64"
                and source.get("media_type") == "application/pdf"
            ):
                for text, png in _pages(source["data"]):
                    if text:
                        result.append({"type": "text", "text": text})
                    result.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": png,
                            },
                        }
                    )
                if "cache_control" in block:
                    result[-1]["cache_control"] = block["cache_control"]
                changed = True
            elif block.get("type") == "tool_result" and "content" in block:
                result.append({**block, "content": render(block["content"])})
            else:
                result.append(block)
        return result

    if isinstance(body, dict):
        for message in body.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "user":
                message["content"] = render(message.get("content"))
    return changed
