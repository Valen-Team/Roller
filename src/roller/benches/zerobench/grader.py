"""ZeroBench grader: the last ``{final answer}`` of the agent's last message.

It reproduces the official snippet (https://github.com/jonathan-roberts1/zerobench#code)
including its whitespace and brace semantics: the answer is the content of the
last pair of curly braces in the lower-cased final message, and it has to match
the reference exactly once stripped. No judge and no environment are involved,
so the score is reproducible from the transcript alone.
"""

import re


def _blocks(content):
    """The text of one message, thinking and tool calls left out."""

    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
            parts.append(item["text"])
    return parts


def final_text(transcript):
    """The last assistant message that carries visible text.

    Harnesses hand the grader their own journal shape, so an event is read
    either as a message itself or as one wrapped in ``message``. A reply whose
    last turn is a tool call leaves the preceding text standing, which is the
    answer the upstream protocol reads when a run hits its turn limit.
    """

    text = ""
    for event in transcript or []:
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            message = event
        if message.get("role") != "assistant":
            continue
        parts = [part for part in _blocks(message.get("content")) if part.strip()]
        if parts:
            text = "\n".join(parts)
    return text


def grade(transcript, workspace_path, reference):
    """Score the final message: 1 for the official strict match, 0 otherwise.

    A run with no visible final text scores 0 and says so rather than raising:
    the protocol counts a missing answer as wrong, and a grading error would
    instead leave the sample unscored and ask for a rerun.
    """

    content = final_text(transcript)
    if not content.strip():
        return {
            "overall_score": 0.0,
            "outcome": "no_final_message",
            "answer": None,
            "reference": reference,
        }
    matches = re.findall(r"\{(.*?)\}", content.strip().lower())
    answer = matches[-1] if matches else ""
    correct = answer[: len(reference)] == reference.strip().lower() and len(
        answer
    ) == len(reference.strip())
    return {
        "overall_score": 1.0 if correct else 0.0,
        "outcome": "graded" if matches else "no_braces",
        "answer": answer,
        "reference": reference,
    }
