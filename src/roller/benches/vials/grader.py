"""VIALS grader: the last ``ANSWER:`` line of answer.txt, judged against the reference.

It follows the upstream protocol (https://github.com/Handshake-AI-Research/VIALS,
vials/pipeline.py and judge.py): the judge sees the question, the reference
answer, the acceptable numeric range of a counting task and the extracted
answer, never the image, and is asked again with the images when its
reasoning starts with ``UNVERIFIABLE:``. It runs in the collecting process,
uses only the standard library, and reads the judge from ``VIALS_JUDGE_*`` or
``JUDGE_*``: ``BASE_URL`` and ``API_KEY``, and ``MODEL`` (default
gpt-6-luna), ``REASONING_EFFORT`` (default medium) and ``MAX_TOKENS``
(default 4096).
"""

import base64
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

JUDGE_ATTEMPTS = 3
JUDGE_TIMEOUT_SECONDS = 300
RETRYABLE_HTTP = (408, 429, 500, 502, 503, 504)
_UNVERIFIABLE_RULE = (
    "If evaluating equivalence would require seeing the image — for example, "
    "GTFA uses one labelling system (condition names, colour labels) and the "
    "candidate uses another (lane numbers, positional indices), and the "
    "mapping is not in the question text — return equivalent=false with "
    'your reasoning prefixed "UNVERIFIABLE:". Downstream tooling treats '
    "these as needing a multimodal regrade (or human review), not as "
    "confirmed model errors.\n\n"
)
JUDGE_SYSTEM = (
    "You are grading a scientific answer. Given the question, the "
    "ground-truth final answer (GTFA), and a candidate answer, decide "
    "whether the candidate expresses the same intended answer as the GTFA "
    "in the context of the question. You do NOT see the image.\n\n"
    "IMPORTANT: You are grading, not solving. Do NOT attempt to work out "
    "the correct answer to the question yourself and then compare the "
    "candidate to your own derivation. Treat the GTFA as the sole source "
    "of truth. Your only decision is whether the candidate expresses the "
    "same answer as the GTFA.\n\n"
    "Apply the standard of a competent domain scientist. ACCEPT variations "
    "that an expert would recognize as immaterial to the answer:\n"
    "  - Formatting: punctuation, whitespace, dash / hyphen style, "
    "thousands separators, and equivalent scientific notation for the "
    "same underlying value\n"
    "  - Capitalization: when the identity is otherwise unambiguous\n"
    "  - Units: omitted when unambiguous from the question or GTFA — "
    "but only when the numeric value itself matches\n"
    "  - Lead bounds: when an inclusive LeadBoundLower/LeadBoundUpper range "
    "is provided, a numeric candidate inside that range is equivalent "
    "even if it differs from the exact GTFA value\n"
    "  - Order and duplicates: irrelevant when the answer is a set or "
    "unordered list\n"
    "  - Terminology: synonymous scientific terms of the same specificity\n\n"
    "Numeric answers are STRICT. Do NOT invent tolerances, relative-error "
    'allowances, or "same order of magnitude" acceptance.\n'
    "  - The candidate must express the same numeric value as GTFA "
    "(after the formatting/unit rules above), unless LeadBounds apply\n"
    "  - If the question asks for a stated reporting precision, any "
    "difference at that precision is NOT equivalent\n"
    "  - Exact counts of discrete objects/events must match exactly; "
    "off-by-one is NEVER equivalent\n"
    "  - Different coefficients at the same power of ten are NOT "
    "equivalent\n\n"
    "REJECT when the candidate materially disagrees with GTFA on any of:\n"
    "  - Missing or extra items in a list\n"
    "  - The identity, count, direction, mechanism, or sign of a "
    "biological entity\n"
    "  - Any numeric mismatch under the strict numeric rules above "
    "(unless inside LeadBounds)\n"
    "  - A broader or narrower term when the question required a specific "
    "level of detail\n"
    "  - Extra conflicting numeric values or alternate counts that "
    "disagree with GTFA\n\n"
    'Refusals, meta-answers ("the provided image", "cannot determine"), '
    "and empty responses are ALWAYS not equivalent, regardless of GTFA.\n\n"
    + _UNVERIFIABLE_RULE
    + "Reply with the JSON schema. Keep the reasoning to ≤3 sentences and "
    "name the specific principle you applied."
)
# The second call sees the images, only to map one labelling system onto the other.
JUDGE_SYSTEM_WITH_IMAGE = JUDGE_SYSTEM.replace(
    "You do NOT see the image.",
    "You are also shown the task image. Use it ONLY to resolve "
    "labelling ambiguities between the candidate and the GTFA. "
    "Do NOT re-solve the question from the image.",
).replace(
    _UNVERIFIABLE_RULE,
    "With the image available, decide equivalence directly. Do NOT tag "
    "your reasoning as UNVERIFIABLE.\n\n",
)
JUDGE_QUESTION = "Are the candidate and ground-truth final answers equivalent?"
JUDGE_QUESTION_WITH_IMAGE = (
    "Using the image only to resolve labelling ambiguities, are the "
    "candidate and ground-truth final answers equivalent?"
)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "FreeFormJudgeResponse",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["reasoning", "equivalent"],
            "properties": {
                "reasoning": {"type": "string"},
                "equivalent": {"type": "boolean"},
            },
        },
    },
}
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}

# Line-anchored so prose such as "The answer: 5" is not taken for the final
# answer; bold and bullets allowed.
_ANCHORED_ANSWER = re.compile(
    r"(?mi)^[ \t]*(?:[-*+•·]|\d+[.)])?[ \t]*\**[ \t]*ANSWER\**[ \t]*:\**[ \t]*"
)
_LOOSE_ANSWER = re.compile(r"ANSWER[ \t]*:[ \t]*", re.IGNORECASE)


def extract_answer(text):
    """The text after the last ``ANSWER:`` marker, without LaTeX, fence or bold wrappers."""
    matches = list(_ANCHORED_ANSWER.finditer(text)) or list(
        _LOOSE_ANSWER.finditer(text)
    )
    if not matches:
        return None
    tail = text[matches[-1].end() :].strip()
    tail = re.sub(r"^\\?\[\s*", "", tail)
    tail = re.sub(r"\s*\\?\]\s*$", "", tail)
    tail = re.sub(r"^\$\$?\s*", "", tail)
    tail = re.sub(r"\s*\$\$?$", "", tail)
    tail = re.sub(r"^```(?:\w+)?\s*", "", tail)
    tail = re.sub(r"\s*```$", "", tail)
    tail = re.sub(r"^\*\*(.+)\*\*$", r"\1", tail.strip())
    return tail.strip() or None


def judge_text(question, reference, lower, upper, answer, closing=JUDGE_QUESTION):
    bounds = (
        ""
        if lower is None and upper is None
        else (
            "Lead-assigned acceptable numeric range (inclusive): "
            f"lower={'n/a' if lower is None else lower}, "
            f"upper={'n/a' if upper is None else upper}. "
            "If the candidate is a numeric answer inside this range, mark it "
            "equivalent to the GTFA.\n\n"
        )
    )
    return (
        f"Question:\n{question}\n\n"
        f"Ground-truth final answer:\n{reference}\n\n"
        f"{bounds}Candidate final answer:\n{answer}\n\n{closing}"
    )


def parse_verdict(raw):
    """Read the judge's JSON even when it is wrapped in code fences or prose."""
    candidates = [raw, re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", raw).strip()]
    found = re.search(r"\{.*\}", raw, re.DOTALL)
    if found:
        candidates.append(found.group(0))
    for text in candidates:
        try:
            value = json.loads(text)
        except ValueError:
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("reasoning"), str)
            and isinstance(value.get("equivalent"), bool)
        ):
            return {"reasoning": value["reasoning"], "equivalent": value["equivalent"]}
    return None


def _setting(name, default=""):
    return (
        os.environ.get("VIALS_JUDGE_" + name)
        or os.environ.get("JUDGE_" + name)
        or default
    )


def _retry_after(error):
    hint = error.headers.get("Retry-After") if error.headers else None
    if not hint:
        return 0
    try:
        return float(hint)
    except ValueError:
        try:
            return max(0, parsedate_to_datetime(hint).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return 0


def _send(payload):
    """One request: (content or None, error, whether asking again may help)."""
    base = _setting("BASE_URL").rstrip("/")
    url = (base if base.endswith("/v1") else base + "/v1") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if _setting("API_KEY"):
        headers["Authorization"] = "Bearer " + _setting("API_KEY")
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=JUDGE_TIMEOUT_SECONDS) as response:
            raw = json.load(response)
        choice = raw["choices"][0]
        content = choice["message"].get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Expected text in message.content")
        if choice["finish_reason"] not in ("stop", "length", "tool_calls"):
            raise ValueError(f"Unexpected finish_reason: {choice['finish_reason']}")
        return content, None, True, 0
    except urllib.error.HTTPError as exc:
        body = exc.read(2000).decode("utf-8", "replace")
        error = {"type": "HTTPError", "code": exc.code, "body": body}
        retry_after = _retry_after(exc) if exc.code == 429 else 0
        return None, error, exc.code in RETRYABLE_HTTP, retry_after
    except (
        OSError,
        http.client.HTTPException,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        return None, {"type": type(exc).__name__, "message": str(exc)[:300]}, True, 0


def _ask(system, content):
    """One parsed verdict, asking again when the reply is empty or unparsable."""
    model = _setting("MODEL", "gpt-6-luna")
    # OpenAI reasoning models take only the default temperature.
    reasoning = (
        model.lower()
        .rsplit("/", 1)[-1]
        .startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "response_format": RESPONSE_FORMAT,
        "max_completion_tokens" if reasoning else "max_tokens": int(
            _setting("MAX_TOKENS", "4096")
        ),
    }
    if _setting("REASONING_EFFORT", "medium"):
        payload["reasoning_effort"] = _setting("REASONING_EFFORT", "medium")
    if not reasoning:
        payload["temperature"] = 0
    errors = []
    for attempt in range(JUDGE_ATTEMPTS):
        content_text, error, retryable, retry_after = _send(payload)
        verdict = parse_verdict(content_text) if content_text is not None else None
        if verdict is not None:
            return dict(verdict, status="ok", attempts=attempt + 1)
        errors.append(error or {"content": (content_text or "")[:200]})
        if not retryable:
            break
        if attempt + 1 < JUDGE_ATTEMPTS:
            time.sleep(max(attempt + 1, retry_after))
    return {"status": "judge_error", "errors": errors}


def _image_part(path):
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    url = f"data:{MEDIA_TYPES[path.suffix]};base64,{data}"
    return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}


def judge(question, reference, lower, upper, answer, images):
    """Judge from text; show the images only when the labels cannot be mapped without them."""
    verdict = _ask(JUDGE_SYSTEM, judge_text(question, reference, lower, upper, answer))
    if verdict["status"] != "ok" or not verdict[
        "reasoning"
    ].lstrip().upper().startswith("UNVERIFIABLE:"):
        return verdict
    content = [
        *(_image_part(path) for path in images),
        {
            "type": "text",
            "text": judge_text(
                question, reference, lower, upper, answer, JUDGE_QUESTION_WITH_IMAGE
            ),
        },
    ]
    return dict(
        _ask(JUDGE_SYSTEM_WITH_IMAGE, content), multimodal=True, text_verdict=verdict
    )


def grade(
    workspace_path,
    question,
    reference,
    answer_lower_bound=None,
    answer_upper_bound=None,
    images=(),
):
    """Score answer.txt: 1 when the judge finds it equivalent, 0 otherwise.

    A missing file or a file without ``ANSWER:`` scores 0 without a judge
    call. A judge that fails leaves the sample unscored, to be graded again.
    """
    root = Path(workspace_path)
    path = root / "answer.txt"
    text = (
        path.read_bytes().decode("utf-8", "replace").strip() if path.is_file() else ""
    )
    answer = extract_answer(text) if text else None
    if answer is None:
        return {
            "overall_score": 0.0,
            "outcome": "no_answer",
            "answer_file": path.is_file(),
            "answer": None,
        }
    verdict = judge(
        question,
        reference,
        answer_lower_bound,
        answer_upper_bound,
        answer,
        [root / name for name in images],
    )
    if verdict["status"] != "ok":
        return {
            "overall_score": 0.0,
            "score_eligible": False,
            "error": "judge failed: " + json.dumps(verdict["errors"])[-500:],
            "answer": answer,
        }
    return {
        "overall_score": 1.0 if verdict["equivalent"] else 0.0,
        "outcome": "judged",
        "answer": answer,
        "verdict": verdict,
    }
