"""Real Docker/service acceptance checks for image inputs. Credentials come only from an env var.

Each case gives the Agent one or two single-colour images and checks that the
answer is right and that SFT holds the images the model received, byte for
byte and in order.

Example: PYTHONPATH=src python scripts/test_multimodal_e2e.py \
  --harness pi --api-base URL --api-key-env OPENAI_API_KEY --model MODEL --output /tmp/mm-e2e
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path

from roller.cli import _build_parser, _collect
from roller.harness_registry import HARNESS_NAMES

READ = (
    "with your image-capable file viewing tool; do not inspect pixels with "
    "shell commands or scripts. "
)
# case -> (input images, images to attach to the first message, prompt, answer)
CASES = {
    "attach": (
        ["a.png"],
        ["a.png"],
        "An image is attached to this message. Reply with only the single colour "
        "that fills it, as one lowercase word. Do not use any tools.",
        "red",
    ),
    "tool-read": (
        ["a.png"],
        [],
        "There is an image file at media/a.png in your working directory. Open it "
        + READ
        + "Then reply with only the single colour that fills it, as one lowercase word.",
        "red",
    ),
    "two-images": (
        ["a.png", "b.png"],
        [],
        "Your working directory contains media/a.png and media/b.png. View media/a.png "
        "first and media/b.png second "
        + READ
        + "Each image is filled with one colour. "
        "Reply with exactly the two colours in that order as lowercase words separated "
        "by a comma and a space.",
        "red, blue",
    ),
}
COLOURS = {"a.png": (255, 0, 0), "b.png": (0, 0, 255)}
# These Harnesses reject files.attach_images before a container starts.
NO_ATTACH = {"hermes", "deepseek"}


def png(colour: tuple[int, int, int], size: int = 64) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + bytes(colour) * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def audit(run: Path, case: str, media: Path) -> dict:
    inputs, _, _, answer = CASES[case]
    sample = next((run / "samples").glob("*/*/info.json")).parent
    info = json.loads((sample / "info.json").read_text())
    assert info["status"] == "ok", info.get("error")
    sft = json.loads((sample / "trace_SFT.json").read_text())
    messages = sft["messages"]
    images = sft.get("images") or []
    placeholders = [
        message["role"]
        for message in messages
        for _ in range(str(message.get("content") or "").count("<image>"))
    ]
    assert len(images) == len(placeholders), (len(images), placeholders)
    expected = [
        hashlib.sha256((media / name).read_bytes()).hexdigest() for name in inputs
    ]
    received = [
        hashlib.sha256(base64.b64decode(uri.split(",", 1)[1])).hexdigest()
        for uri in images
    ]
    assert received == expected, "SFT images differ from the input files or their order"
    role = "user" if case == "attach" else "tool"
    assert set(placeholders) == {role}, placeholders
    final = next(
        str(message.get("content") or "").strip()
        for message in reversed(messages)
        if message["role"] == "assistant" and (message.get("content") or "").strip()
    )
    assert final.lower().rstrip(".") == answer, f"answered {final!r}"
    tools = [
        call.get("function", call)["name"]
        for message in messages
        for call in message.get("tool_calls") or []
    ]
    return {
        "case": case,
        "status": "passed",
        "sample": str(sample),
        "images": len(images),
        "image_roles": placeholders,
        "tools": tools,
        "usage": info.get("usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", choices=HARNESS_NAMES, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument(
        "--model", required=True, help="a model that accepts image input"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument(
        "--openclaw-api",
        choices=["openai-completions", "openai-responses"],
        default="openai-responses",
    )
    args = parser.parse_args()
    if not os.environ.get(args.api_key_env):
        parser.error("credential environment variable is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    media = args.output / "files" / "media"
    media.mkdir(parents=True, exist_ok=True)
    for name, colour in COLOURS.items():
        (media / name).write_bytes(png(colour))

    results = []
    for case in args.cases:
        inputs, attached, prompt, _ = CASES[case]
        if attached and args.harness in NO_ATTACH:
            result = {
                "case": case,
                "status": "not_applicable",
                "reason": "the Harness does not support files.attach_images",
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            continue
        task = {
            "id": case,
            "harness": args.harness,
            "model": "roller/" + args.model
            if args.harness == "openclaw"
            else args.model,
            "file_root": "files",
            "prompt": prompt,
            "files": {"inputs": [f"media/{name}" for name in inputs]},
        }
        if attached:
            task["files"]["attach_images"] = [f"media/{name}" for name in attached]
        tasks = args.output / f"{case}.tasks.json"
        tasks.write_text(json.dumps([task]))
        command = [
            "collect",
            "--tasks",
            str(tasks),
            "--timeout",
            "420",
            "--parallel",
            "1",
            "--retries",
            "0",
            "--output",
            str(args.output),
            "--run-name",
            case,
        ]
        if args.harness == "openclaw":
            config = {
                "providers": {
                    "roller": {
                        "baseUrl": args.api_base,
                        "apiKey": "${" + args.api_key_env + "}",
                        "api": args.openclaw_api,
                        "models": [
                            {
                                "id": args.model,
                                "name": args.model,
                                "input": ["text", "image"],
                                "reasoning": True,
                                "contextWindow": 128000,
                                "maxTokens": 8192,
                            }
                        ],
                    }
                }
            }
            config_path = args.output / "model-config.json"
            config_path.write_text(json.dumps(config))
            command += [
                "--model-config",
                str(config_path),
                "--pass-env",
                args.api_key_env,
            ]
        else:
            command += ["--api-base", args.api_base, "--api-key-env", args.api_key_env]
        if args.harness == "deepseek":
            # sdk-minimal has no image tool, and the model must declare image input.
            command += [
                "--deepseek-profile",
                "sdk",
                "--deepseek-input-modalities",
                "text,image",
            ]
        try:
            exit_code = _collect(_build_parser().parse_args(command))
            if exit_code != 0:
                sample_info = next(
                    (args.output / case / "samples").glob("*/*/info.json"), None
                )
                detail = (
                    json.loads(sample_info.read_text()).get("error", "")
                    if sample_info
                    else ""
                )
                raise AssertionError(f"collect exit={exit_code}: {detail}")
            result = audit(args.output / case, case, media)
        except Exception as exc:
            result = {"case": case, "status": "failed", "error": str(exc)}
        results.append(result)
        (args.output / "summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if all(r["status"] in {"passed", "not_applicable"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
