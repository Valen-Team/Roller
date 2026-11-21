"""The token-level dataset of a run collected with ``--token-ids``.

``token_ids.jsonl`` in a sample holds one row per model call: the prompt
tokens the endpoint received after those of the call it continues, the tokens
it generated, and the logprob of each generated token. ``roller export
--format tokens`` joins the calls of a conversation into one sequence, where
every token appears once, and marks the positions the model generated.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from roller.capture import read_jsonl
from roller.sft_dataset import index_row
from roller.storage import write_jsonl
from roller.token_ids import TOKEN_IDS_FILE

TOKENS_FILE = "tokens.jsonl"


def sequences(
    rows: Iterable[Mapping[str, Any]], participants: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Join every call to the one it continues.

    A call that continues none, such as the first call of a subagent or the
    first after a context compaction, starts a new sequence.
    """

    joined: list[dict[str, Any]] = []
    # Each sequence that can still grow, by the call that ends it.
    open_sequences: dict[int, dict[str, Any]] = {}
    for row in rows:
        sequence = open_sequences.pop(row["continues"], None)
        if sequence is None:
            sequence = {
                "participant_id": participants.get(row["request_seq"]),
                "request_seqs": [],
                "finish_reasons": [],
                "token_ids": [],
                "generated": [],
                "logprobs": [],
            }
            joined.append(sequence)
        prompt, completion = row["prompt_token_ids"], row["completion_token_ids"]
        sequence["request_seqs"].append(row["request_seq"])
        sequence["finish_reasons"].append(row["finish_reason"])
        sequence["token_ids"] += [*prompt, *completion]
        sequence["generated"] += [0] * len(prompt) + [1] * len(completion)
        sequence["logprobs"] += [None] * len(prompt) + row["logprobs"]
        open_sequences[row["request_seq"]] = sequence
    return joined


def capture_participants(sample_dir: Path) -> dict[int, str]:
    """The participant of each call, from the capture of the last attempt."""

    attempts = sorted(sample_dir.glob("capture/*/calls.jsonl"))
    if not attempts:
        return {}
    return {
        call["request_seq"]: call["participant_id"] for call in read_jsonl(attempts[-1])
    }


def _participants(sample_dir: Path, result: Mapping[str, Any]) -> dict[int, str]:
    capture = result.get("runtime", {}).get("capture") or {}
    if "participant_ids" in capture:
        # --training-files-only removed the capture and kept this map.
        return {int(seq): pid for seq, pid in capture["participant_ids"].items()}
    return capture_participants(sample_dir)


def export_run(run_dir: Path, output: Path) -> tuple[int, int, int]:
    """Write ``tokens.jsonl``; return its samples, sequences and tokens."""

    info = json.loads((run_dir / "info.json").read_text(encoding="utf-8"))
    logprobs_modes = (info["configuration"].get("token_ids") or {}).get("logprobs", {})
    samples = [
        (path.parent, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(run_dir.glob("samples/*/*/info.json"))
        if (path.parent / TOKEN_IDS_FILE).is_file()
    ]
    samples.sort(key=lambda sample: sample[1].get("request_index") or 0)
    rows = []
    for sample_dir, result in samples:
        calls = read_jsonl(sample_dir / TOKEN_IDS_FILE)
        rows.append(
            {
                **index_row(run_dir, sample_dir, result),
                "status": result.get("status"),
                "logprobs_mode": logprobs_modes.get(result.get("model")),
                "request": calls[0]["request"] if calls else None,
                "sequences": sequences(calls, _participants(sample_dir, result)),
            }
        )
    write_jsonl(output / TOKENS_FILE, rows)
    joined = [sequence for row in rows for sequence in row["sequences"]]
    return len(rows), len(joined), sum(len(item["token_ids"]) for item in joined)
