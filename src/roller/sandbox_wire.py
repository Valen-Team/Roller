"""Run Roller’s existing wire recorder inside a remote sandbox."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from roller.capture import CallRecorder
from roller.storage import atomic_write_json
from roller.token_ids import TOKEN_IDS_FILE
from roller.wire_trace import WireTraceProxy


def main(directory: Path) -> None:
    settings = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    recorder = CallRecorder(directory / "capture") if settings["calls"] else None
    proxy = WireTraceProxy(
        settings["api_base"],
        directory / "wire_trace.jsonl",
        mode="full",
        capture_recorder=recorder,
        capture_identity_header="x-roller-native-session-id" if recorder else None,
        token_ids_path=directory / TOKEN_IDS_FILE
        if settings.get("token_ids")
        else None,
    )
    proxy.start()
    try:
        atomic_write_json(directory / "ready.json", {"url": proxy.proxy_base_url})
        while not (directory / "stop").is_file():
            time.sleep(0.1)
    finally:
        proxy.stop()
        if recorder:
            recorder.seal()
        atomic_write_json(
            directory / "finished.json", {"error": recorder.error if recorder else None}
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
