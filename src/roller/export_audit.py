"""Credential-shaped redaction and value-free export review reports."""

from __future__ import annotations

import ipaddress
import json
import re
from collections import Counter
from pathlib import Path

from roller.redaction import REDACTED

# Boundaries also recognize JSON-escaped newlines immediately before a token.
_BOUNDARY = r"(?:(?<![A-Za-z0-9_-])|(?<=\\[nrt]))"
TOKEN_PATTERNS = tuple(
    re.compile(_BOUNDARY + pattern)
    for pattern in (
        r"sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])",
        r"gh[pousr]_[A-Za-z0-9]{36,}(?![A-Za-z0-9])",
        r"github_pat_[A-Za-z0-9_]{40,}(?![A-Za-z0-9_])",
        r"hf_[A-Za-z0-9]{30,}(?![A-Za-z0-9])",
        r"A[KS]IA[A-Z0-9]{16}(?![A-Za-z0-9])",
        r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    )
)
# Preserve the surrounding syntax so JSON remains parseable.
CONTEXT_PATTERNS = (
    (
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
        ),
        REDACTED,
    ),
    (
        re.compile(r"(?i:Bearer)(\s+)[A-Za-z0-9._~+/=-]{20,}"),
        lambda m: "Bearer" + m[1] + REDACTED,
    ),
    (
        re.compile(r":(?:(?:\\?/)){2}[^\s\"'<>/@:]+:[^\s\"'<>/@]+@"),
        "://" + REDACTED + "@",
    ),
    (
        re.compile(
            r"((?i:aws_secret_access_key|secretaccesskey)[\s\"'\\:=]{1,10})[A-Za-z0-9+/]{40}(?![A-Za-z0-9+/])"
        ),
        lambda m: m[1] + REDACTED,
    ),
)
_HOME = re.compile(
    r"/(?:home|Users)/(?!root(?:/|\b)|ubuntu(?:/|\b)|user(?:/|\b))[A-Za-z0-9][A-Za-z0-9._-]*"
)
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b"
)
_IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.]|\.\d)")


def replacements():
    return tuple((pattern, REDACTED) for pattern in TOKEN_PATTERNS) + CONTEXT_PATTERNS


def write_report(output: Path) -> Path:
    """Scan exported text; record locations and kinds without copying sensitive values."""
    findings = []
    counts: Counter[str] = Counter()
    skipped = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(output).as_posix()
        try:
            with path.open(encoding="utf-8") as source:
                for number, line in enumerate(source, 1):
                    kinds = []
                    if _EMAIL.search(line):
                        kinds.append("email")
                    if _HOME.search(line.replace("\\/", "/")):
                        kinds.append("home_path")
                    for match in _IP.finditer(line):
                        try:
                            ip = ipaddress.ip_address(match[0])
                        except ValueError:
                            continue
                        if (
                            ip.is_private
                            and not ip.is_loopback
                            and not ip.is_unspecified
                        ):
                            kinds.append("private_ip")
                            break
                    if any(pattern.search(line) for pattern in TOKEN_PATTERNS):
                        kinds.append("credential")
                    if "-----BEGIN" in line and "PRIVATE KEY" in line:
                        kinds.append("private_key")
                    for kind in kinds:
                        counts[kind] += 1
                        if len(findings) < 1000:
                            findings.append(
                                {"file": relative, "line": number, "kind": kind}
                            )
        except UnicodeDecodeError:
            skipped.append(relative)
    report = output.with_name(output.name + ".redaction-report.json")
    report.write_text(
        json.dumps(
            {
                "schema_version": "roller.redaction_report/v1",
                "counts": dict(counts),
                "findings": findings,
                "findings_truncated": sum(counts.values()) > len(findings),
                "unscanned_binary_files": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
