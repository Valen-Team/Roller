"""The token-level dataset of a run collected with ``--claude-token-ids``.

``token_ids.jsonl`` in a sample holds one row per model call: the token ids
the endpoint received, the ones it generated, and the logprob of each
generated token. A later call resends the earlier ones, so its prompt starts
with the prompt and completion of the call before it. ``roller export --format
tokens`` joins such calls into one sequence, where every token appears once,
and marks the positions the model generated.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from roller.capture import read_jsonl
from roller.sft_dataset import index_row
from roller.storage import write_jsonl

TOKEN_IDS_FILE = "token_ids.jsonl"
TOKENS_FILE = "tokens.jsonl"


def sequences(
    rows: Iterable[Mapping[str, Any]], participants: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Join the calls of each participant whose prompts continue one another.

    A call that does not continue an earlier one, such as the first call of a
    subagent or the first after a context compaction, starts a new sequence.
    """

    joined: list[dict[str, Any]] = []
    for row in rows:
        prompt = row["prompt_token_ids"]
        participant = participants.get(row["request_seq"])
        sequence = next(
            (
                item
                for item in reversed(joined)
                if item["participant_id"] == participant
                and prompt[: len(item["token_ids"])] == item["token_ids"]
            ),
            None,
        )
        if sequence is None:
            sequence = {
                "participant_id": participant,
                "request_seqs": [],
                "finish_reasons": [],
                "token_ids": [],
                "generated": [],
                "logprobs": [],
            }
            joined.append(sequence)
        added = prompt[len(sequence["token_ids"]) :]
        completion = row["completion_token_ids"]
        sequence["request_seqs"].append(row["request_seq"])
        sequence["finish_reasons"].append(row["finish_reason"])
        sequence["token_ids"] += [*added, *completion]
        sequence["generated"] += [0] * len(added) + [1] * len(completion)
        sequence["logprobs"] += [None] * len(added) + row["logprobs"]
    return joined


def _participants(sample_dir: Path) -> dict[int, str]:
    """The participant of each call, from the capture of the last attempt."""

    attempts = sorted(sample_dir.glob("capture/*/calls.jsonl"))
    if not attempts:
        return {}
    return {
        call["request_seq"]: call["participant_id"] for call in read_jsonl(attempts[-1])
    }


def export_run(run_dir: Path, output: Path) -> tuple[int, int, int]:
    """Write ``tokens.jsonl``; return its samples, sequences and tokens."""

    info = json.loads((run_dir / "info.json").read_text(encoding="utf-8"))
    harness = info["configuration"].get("claude-code", {})
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
                "logprobs_mode": harness.get("token_logprobs", {}).get(
                    result.get("model")
                ),
                "request": calls[0]["request"] if calls else None,
                "sequences": sequences(calls, _participants(sample_dir)),
            }
        )
    write_jsonl(output / TOKENS_FILE, rows)
    joined = [sequence for row in rows for sequence in row["sequences"]]
    return len(rows), len(joined), sum(len(item["token_ids"]) for item in joined)
