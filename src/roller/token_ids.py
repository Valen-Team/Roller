"""Token ids and logprobs of the calls a Chat Completions endpoint serves.

An endpoint such as vLLM returns, on request, the token ids it received and
the ones it generated. With ``--token-ids`` the request recording proxy asks
for them on every Chat Completions call, whether the Harness made it or the
proxy translated it from another API, and takes them out of the response the
Harness reads. They go to ``token_ids.jsonl`` in the sample, one row per call.
"""

from __future__ import annotations

import json
import threading
from array import array
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

TOKEN_IDS_FILE = "token_ids.jsonl"


class TokenIds:
    """The token ids and sampled logprobs of one Chat Completions call."""

    def __init__(self, request: dict[str, Any]) -> None:
        """Ask for them in ``request``, which the caller goes on to send."""

        request.update(return_token_ids=True, logprobs=True)
        self.request = {
            key: value
            for key, value in request.items()
            if key not in ("messages", "tools")
        }
        self.prompt: list[int] = []
        self.completion: list[int] = []
        self.logprobs: list[float] = []
        self.finish_reason: str | None = None

    def add(self, payload: dict[str, Any]) -> None:
        """Read one stream chunk, or the whole body of an unstreamed response."""

        if payload.get("prompt_token_ids"):
            self.prompt = payload["prompt_token_ids"]
        for choice in payload.get("choices", []):
            self.completion += choice.get("token_ids") or []
            logprobs = (choice.get("logprobs") or {}).get("content") or []
            self.logprobs += [token["logprob"] for token in logprobs]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def take(self, payload: dict[str, Any]) -> bool:
        """``add``, then remove what the Harness did not ask for."""

        self.add(payload)
        payload.pop("prompt_token_ids", None)
        for choice in payload.get("choices", []):
            choice.pop("token_ids", None)
            choice.pop("logprobs", None)
        return True


class TokenLog:
    """``token_ids.jsonl`` of one sample.

    A later call of a conversation resends the earlier ones, so its prompt
    starts with the prompt and completion of the call before it. The row of
    such a call names that call in ``continues`` and holds only the prompt
    tokens that follow, which keeps the file linear in the length of the
    conversation. Any other call, such as the first of a subagent or the first
    after a context compaction, has ``continues`` null and its whole prompt.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        # The packed tokens of each conversation, by the call that ends it.
        self._conversations: dict[int, bytes] = {}

    def write(self, request_seq: int, tokens: TokenIds, complete: bool) -> None:
        prompt = array("i", tokens.prompt).tobytes()
        with self._lock:
            continues = max(
                (
                    seq
                    for seq, sent in self._conversations.items()
                    if prompt.startswith(sent)
                ),
                key=lambda seq: len(self._conversations[seq]),
                default=None,
            )
            sent = b"" if continues is None else self._conversations.pop(continues)
            self._conversations[request_seq] = (
                prompt + array("i", tokens.completion).tobytes()
            )
            row = {
                "request_seq": request_seq,
                "continues": continues,
                "prompt_token_ids": tokens.prompt[len(sent) // array("i").itemsize :],
                "completion_token_ids": tokens.completion,
                "logprobs": tokens.logprobs,
                "finish_reason": tokens.finish_reason,
                "complete": complete,
                "request": tokens.request,
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def logprobs_mode(url: str, model: str, api_key: str) -> str:
    """Whether the endpoint reports the logprobs of its sampling distribution.

    ``url`` is the Chat Completions URL. vLLM reports logprobs as they are
    before temperature and top-p unless it runs with ``--logprobs-mode
    processed_logprobs``. Between the two likeliest first tokens, the gap of
    the first kind does not move with temperature; the gap of the second kind
    doubles when the temperature halves.
    """

    def gap(temperature: float) -> float:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Name a colour."}],
            "max_tokens": 1,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 2,
            "return_token_ids": True,
        }
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urlopen(request, timeout=120) as response:
            payload = json.load(response)
        if not payload.get("prompt_token_ids"):
            raise RuntimeError(
                f"{model} at {url} returned no token ids; "
                "--token-ids needs an endpoint that honours return_token_ids"
            )
        first, second = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        return first["logprob"] - second["logprob"]

    return "processed" if gap(0.5) > 1.5 * gap(1.0) else "raw"
