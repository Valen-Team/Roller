"""Package a Roller run as a Hugging Face dataset folder.

A run is written as one part per Harness and model, named ``<harness>-<model>``.
Each part holds the session of every sample for the trace viewer of the Hub and
one JSON Lines table with a row per sample. On request it also holds a Roller
run directory with the samples of that model, which ``roller view`` opens after
a download. Every part has its own paths, so further runs can be exported into
the same folder.

The export replaces what is private to the collection environment: the model
endpoints the run recorded, the hosts its ``NO_PROXY`` named, the name of the
machine, the host paths of the run, and the values of the credentials it passed
to its containers.
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from roller import export_audit
from roller.benches.report import SCORE_FILE, read_scores, sample_score
from roller.collector import CollectionConfig, Collector
from roller.redaction import REDACTED
from roller.sft import SFTConversionError, _image_data_uri
from roller.storage import safe_component

DATA_DIR = "data"
TRACES_DIR = "traces"
# The Hub splits traces with str.splitlines(), which treats these JSON string
# characters as line breaks even though they are part of a JSONL record.
_TRACE_LINE_BREAKS = re.compile("[\\u0085\\u2028\\u2029]")
# Native session schemas of ``trace.jsonl`` that the Hub reads as they are.
NATIVE_TRACES = {
    "openclaw-session-jsonl",
    "codex-session-jsonl",
    "claude-code-session-jsonl",
}
STATS_START = "<!-- roller:stats:start -->"
STATS_END = "<!-- roller:stats:end -->"

_CARD = """\
---
pretty_name: {name}
tags:
- roller
- agent
- trajectories
- tool-use
configs:
- config_name: traces
  data_files: {traces_dir}/**/*.jsonl
  default: true
- config_name: trajectories
  data_files: {data_dir}/*.jsonl
dataset_info:
- config_name: trajectories
  features:
  - {{name: run, dtype: string}}
  - {{name: sample, dtype: string}}
  - {{name: trace, dtype: string}}
  - {{name: task_id, dtype: string}}
  - {{name: upstream_task_id, dtype: string}}
  - {{name: run_index, dtype: int64}}
  - {{name: bench, dtype: string}}
  - {{name: category, dtype: string}}
  - {{name: model, dtype: string}}
  - {{name: harness, dtype: string}}
  - {{name: status, dtype: string}}
  - {{name: error, dtype: string}}
  - {{name: score, dtype: float64}}
  - {{name: grader_error, dtype: string}}
  - {{name: steps, dtype: int64}}
  - {{name: tool_calls, dtype: int64}}
  - {{name: input_tokens, dtype: int64}}
  - {{name: output_tokens, dtype: int64}}
  - {{name: total_tokens, dtype: int64}}
  - {{name: elapsed_seconds, dtype: float64}}
  - {{name: files, sequence: string}}
  - {{name: messages, dtype: string}}
  - {{name: tools, dtype: string}}
---

# {name}

Agent trajectories collected with [Roller](https://github.com/Valen-Team/Roller).
Each sample is one task run by one model in a disposable Docker container, with
its reasoning, tool calls, tool results, output files and run metadata.

{stats}

## Read the traces on the Hub

`{traces_dir}/` has the agent session of every sample. Open **Data Studio** on
this page and click a row to read the prompts, reasoning, tool calls and tool
results of a sample in the trace viewer.

## Load the table

`{data_dir}/` has one table per Harness and model, with one row per sample.
`messages` and `tools` are JSON strings in the format of an OpenAI chat request.

```python
import json

from datasets import load_dataset

rows = load_dataset("OWNER/NAME", "trajectories", split="train")
messages = json.loads(rows[0]["messages"])
```

Replace `OWNER/NAME` with the id of this dataset repository.
{browse}
## Files

| Path | Content |
| --- | --- |
| `{traces_dir}/<harness>-<model>/<index>_<task>.jsonl` | Agent session of one sample, which the Hub opens in its [trace viewer](https://huggingface.co/docs/hub/agent-traces) |
| `{data_dir}/<harness>-<model>.jsonl` | One row per sample: task, model, status, score, steps, tokens, `messages`, `tools` |
{run_files}"""

_CARD_BROWSE = """
## Browse the trajectories

`runs/{part}/` is a Roller run directory. Download it and open it in the Roller
Viewer to read every step of a sample:

```bash
hf download OWNER/NAME --repo-type dataset --include "runs/{part}/*" --local-dir {name}
uvx --from roller-agent roller view {name}/runs/{part}
```
"""

_CARD_RUN_FILES = """\
| `runs/<harness>-<model>/info.json` | Run configuration, progress and total usage |
| `runs/<harness>-<model>/sft.jsonl` | SFT records of the successful samples |
| `runs/<harness>-<model>/sft_index.jsonl` | Task, model and score for each SFT record |
| `runs/<harness>-<model>/samples/<model>/<index>_<task>/` | Native session, unified trajectory, SFT record, scores and output files of one sample |

The files of a sample are described in the
[trajectory outputs guide](https://valen-team.github.io/roller/traces/trajectory-outputs/).
"""

# The files of a sample that its session and its table row are built from.
_SAMPLE_FILES = (
    "info.json",
    SCORE_FILE,
    "trace.jsonl",
    "trace_ACTF.json",
    "trace_SFT.json",
)


@dataclass(frozen=True)
class Rules:
    """What to change, leave out and refuse while copying a run."""

    replace: tuple[
        tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...
    ] = ()
    exclude: tuple[str, ...] = ()
    forbid: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class Redaction:
    """What an export replaced on its own, and what it could not look for."""

    endpoints: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    credentials: tuple[str, ...] = ()
    # Credential variables without a value in the exporting environment.
    unset: tuple[str, ...] = ()
    # Hosts that ``NO_PROXY`` named.
    hosts: tuple[str, ...] = ()
    # The name of the exporting machine, when it was looked for.
    machine: str = ""
    # Imported sessions: the user names of their home directories, and the
    # number of account identifiers they recorded.
    users: tuple[str, ...] = ()
    identities: int = 0

    def lines(self) -> list[str]:
        lines = []
        replaced = [
            f"{len(self.endpoints)} endpoints" if self.endpoints else "",
            f"{len(self.hosts)} NO_PROXY hosts" if self.hosts else "",
            "the name of this machine" if self.machine else "",
            f"{len(self.paths)} host paths" if self.paths else "",
            f"{len(self.users)} home directories" if self.users else "",
            f"{self.identities} account identifiers" if self.identities else "",
            f"the values of {', '.join(self.credentials)}" if self.credentials else "",
        ]
        if any(replaced):
            lines.append("Redacted: " + ", ".join(text for text in replaced if text))
        if self.unset:
            lines.append(
                "Not set in this environment, so their values were not searched for: "
                + ", ".join(self.unset)
            )
        return lines


def load_rules(path: Path) -> Rules:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Rules(
        replace=tuple(
            (re.compile(rule["pattern"]), rule["with"])
            for rule in payload.get("replace", [])
        ),
        exclude=tuple(payload.get("exclude", [])),
        forbid=tuple(re.compile(pattern) for pattern in payload.get("forbid", [])),
    )


# A shorter value of a credential variable is a stand-in such as ``local``;
# replacing it would rewrite ordinary words of the trajectories.
_CREDENTIAL_LENGTH = 12
_CREDENTIAL_NAMES = ("key", "token", "secret", "password", "credential")
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
# Address blocks that every site uses; one in ``NO_PROXY`` names no site.
_COMMON_NETWORKS = {
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
}
# A machine name that is a default of an OS image, or a few letters, is an
# ordinary word in the trajectories.
_COMMON_HOSTNAMES = {"localhost", "ubuntu", "debian"}
_HOSTNAME_LENGTH = 6
# Paths of the run configuration: its key, whether it names a file, placeholder.
_CONFIGURED_PATHS = (
    ("bench", "root", False, "<bench-root>"),
    (None, "workspace", False, "<workspace>"),
    (None, "tasks", True, "<tasks-dir>"),
    (None, "thinking_capabilities_path", True, "<config-dir>"),
    ("*", "model_config_path", True, "<config-dir>"),
    ("*", "config_path", True, "<config-dir>"),
    ("*", "settings_path", True, "<config-dir>"),
)


# A shorter user name is an ordinary word, replaced only in its home paths;
# so is a shorter git user name, which is left alone.
_USER_LENGTH = 4
_COMMON_USERS = {"root", "ubuntu", "user"}


def _home_users(path: Any) -> list[str]:
    """The user name of a home directory that ``path`` lies in."""

    match = (
        re.match(r"/(?:Users|home)/([^/]+)", path) if isinstance(path, str) else None
    )
    return [match[1]] if match and match[1] not in _COMMON_USERS else []


def _git_identity() -> tuple[str, str]:
    """The global git user name and email of this machine."""

    def value(key: str) -> str:
        return subprocess.run(
            ["git", "config", "--global", key],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    return value("user.name"), value("user.email")


def _host_path(value: Any, is_file: bool) -> str | None:
    """The directory of a configured host path, when it is specific enough."""

    if not isinstance(value, str) or not value.startswith("/"):
        return None
    path = PurePosixPath(value).parent if is_file else PurePosixPath(value)
    return str(path) if len(path.parts) >= 3 else None


def _address(text: str) -> str:
    """A pattern for an IP address or block that is no part of a longer one."""

    return rf"(?<![\w.]){re.escape(text)}(?![\w]|\.\d)"


def _domain(text: str) -> str:
    """A pattern for a domain, alone or as the end of a longer host name."""

    return rf"(?<![\w-]){re.escape(text)}(?![\w-]|\.\w)"


def _name(text: str) -> str:
    """A pattern for a machine name that is no part of a longer name."""

    return rf"(?<![\w.-]){re.escape(text)}(?![\w-]|\.\w)"


def _no_proxy_hosts(value: str) -> dict[str, str]:
    """The entries of a ``NO_PROXY`` value that name a site, with their patterns.

    Loopback names, the address blocks every site uses and names of one label,
    such as ``.svc``, are left out.
    """

    hosts: dict[str, str] = {}
    for entry in value.split(","):
        entry = entry.strip().lstrip("*").lstrip(".")
        if not entry or entry in _LOOPBACK:
            continue
        try:
            network = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            host, _, port = entry.rpartition(":")
            entry = host if host and port.isdigit() else entry
            if "." in entry and entry not in _LOOPBACK:
                hosts[entry] = _domain(entry)
            continue
        if "/" in entry:
            if str(network) not in _COMMON_NETWORKS:
                hosts[entry] = _address(entry)
        elif not network.network_address.is_loopback:
            hosts[entry] = _address(entry)
    return hosts


def machine_name() -> str:
    """The name of this machine, fully qualified when the resolver knows it."""

    name, full = socket.gethostname(), socket.getfqdn()
    return full if full.startswith(name.split(".")[0] + ".") else name


def environment_redaction(
    run_dir: Path, environ: Mapping[str, str], machine: str = ""
) -> tuple[Rules, Redaction]:
    """Derive from a run what its export replaces without being told.

    The samples record the model endpoints and the names of the variables their
    containers received, and the run records the host paths of its configuration.
    An endpoint host becomes ``endpoint-<n>.example`` and a path its placeholder;
    the run directory becomes ``<run-dir>``. A credential variable that has a
    value in ``environ`` has that value replaced; one without a value is
    reported, because the export cannot look for it. The hosts that a passed
    ``NO_PROXY`` names become ``internal-<n>.example``.

    ``machine`` is the name of the machine the run was collected on. Its fully
    qualified name becomes ``collection-host.example``, the domain of that name
    ``collection-domain.example``, and the bare name ``collection-host``.
    """

    urls: set[str] = set()
    names: set[str] = set()
    users: list[str] = []
    identities: set[str] = set()
    imported = False
    for path in run_dir.glob("samples/*/*/info.json"):
        provenance = _json(path).get("provenance") or {}
        urls.update(url for url in provenance.get("api_base_urls") or [] if url)
        names.update(name for name in provenance.get("environment_names") or [] if name)
        source = provenance.get("import")
        if isinstance(source, dict):
            imported = True
            users += _home_users(source.get("cwd"))
            identities.update(value for value in source.get("identity") or [] if value)
    if imported:
        # An imported session names no variables; every credential of the
        # exporting environment may have reached it.
        names.update(
            name
            for name, value in environ.items()
            if value and name.lower().replace("_", "").endswith(_CREDENTIAL_NAMES)
        )
        users = list(dict.fromkeys(_home_users(str(Path.home())) + sorted(users)))

    literals: dict[str, str] = {}
    # The pattern of a literal that must not match inside a longer word.
    patterns: dict[str, str] = {}
    credentials: list[str] = []
    unset: list[str] = []
    no_proxy: dict[str, str] = {}
    no_proxy_values: set[str] = set()
    for name in sorted(names):
        value = environ.get(name, "")
        if value.startswith(("http://", "https://")):
            urls.add(value)
        elif name.lower() == "no_proxy":
            no_proxy.update(_no_proxy_hosts(value))
            if "," in value or "/" in value:
                no_proxy_values.add(value)
        elif name.lower().replace("_", "").endswith(_CREDENTIAL_NAMES):
            if not value:
                unset.append(name)
            elif len(value) >= _CREDENTIAL_LENGTH:
                literals[value] = REDACTED
                credentials.append(name)

    endpoints: list[str] = []
    for url in sorted(urls):
        parts = urlsplit(url)
        host = parts.hostname or ""
        netloc = parts.netloc.rpartition("@")[2]
        if not host or host in _LOOPBACK:
            continue
        placeholder = literals.get(netloc, f"endpoint-{len(endpoints) + 1}.example")
        if parts.username or parts.password:
            literals[url] = parts._replace(netloc=placeholder).geturl()
            literals[parts.netloc] = placeholder
        if netloc in literals:
            continue
        specific = "." in host or ":" in host
        if netloc != host or specific:
            literals[netloc] = placeholder
            endpoints.append(netloc)
        # A bare name such as ``inference`` is an ordinary word elsewhere.
        if specific:
            literals.setdefault(host, placeholder)

    hosts: list[str] = []
    for host, pattern in sorted(no_proxy.items()):
        if host not in literals:
            literals[host] = f"internal-{len(hosts) + 1}.example"
            patterns[host] = pattern
            hosts.append(host)
    for value in no_proxy_values:
        literals[value] = "localhost,127.0.0.1"

    short, _, domain = machine.partition(".")
    named = False
    if domain and short.lower() not in _LOOPBACK:
        literals.setdefault(machine, "collection-host.example")
        named = True
        if "." in domain and domain not in literals:
            literals[domain] = "collection-domain.example"
            patterns[domain] = _domain(domain)
    if (
        short.lower() not in _COMMON_HOSTNAMES
        and (len(short) >= _HOSTNAME_LENGTH or any(char.isdigit() for char in short))
        and short not in literals
    ):
        literals[short] = "collection-host"
        patterns[short] = _name(short)
        named = True

    configuration = _json(run_dir / "info.json").get("configuration") or {}
    paths: list[str] = []
    # The run directory as it was given and as it resolves, counted once.
    for location in (run_dir.resolve(), run_dir.absolute()):
        path = _host_path(str(location), False)
        if path and path not in literals:
            literals[path] = "<run-dir>"
            if not paths:
                paths.append(path)
    for section, key, is_file, placeholder in _CONFIGURED_PATHS:
        blocks = (
            [configuration]
            if section is None
            else [block for block in configuration.values() if isinstance(block, dict)]
            if section == "*"
            else [configuration.get(section) or {}]
        )
        for block in blocks:
            path = _host_path(block.get(key), is_file)
            if path and path not in literals:
                literals[path] = placeholder
                paths.append(path)

    for value in identities:
        literals[value] = REDACTED
    if imported:
        name, email = _git_identity()
        if len(name) >= _USER_LENGTH:
            literals.setdefault(name, REDACTED)
            patterns.setdefault(name, _name(name))
        if email:
            literals.setdefault(email, REDACTED)
    homes: list[tuple[re.Pattern[str], str]] = []
    for number, user in enumerate(users, 1):
        placeholder = "user" if number == 1 else f"user-{number}"
        escaped = re.escape(user)
        homes += [
            (re.compile(rf"/(Users|home)/{escaped}(?![\w.-])"), rf"/\1/{placeholder}"),
            (
                re.compile(rf"-(Users|home)-{escaped}(?![A-Za-z0-9])"),
                rf"-\1-{placeholder}",
            ),
        ]
        if len(user) >= _USER_LENGTH:
            homes.append((re.compile(f"(?i:{_name(user)})"), placeholder))

    if not literals:
        return Rules(replace=tuple(homes)), Redaction(
            unset=tuple(unset), users=tuple(users)
        )
    pattern = re.compile(
        "|".join(
            patterns.get(text) or re.escape(text)
            for text in sorted(literals, key=len, reverse=True)
        )
    )
    return (
        Rules(replace=((pattern, lambda match: literals[match.group(0)]), *homes)),
        Redaction(
            users=tuple(users),
            identities=len(identities),
            endpoints=tuple(endpoints),
            paths=tuple(paths),
            credentials=tuple(credentials),
            unset=tuple(unset),
            hosts=tuple(hosts),
            machine=machine if named else "",
        ),
    )


def _copy_file(source: Path, target: Path, rules: Rules, label: str) -> list[str]:
    """Copy one file and return where a forbidden pattern is still present."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if rules.replace or rules.forbid:
        found: list[str] = []
        private_key = False
        detect_keys = any(
            pattern is export_audit.TOKENS for pattern, _ in rules.replace
        )
        try:
            with (
                source.open(encoding="utf-8", newline="") as reader,
                target.open("w", encoding="utf-8", newline="") as writer,
            ):
                for number, line in enumerate(reader, 1):
                    if detect_keys:
                        if private_key:
                            end = re.search(
                                r"-----END [A-Z0-9 ]*PRIVATE KEY-----", line
                            )
                            if end is None:
                                continue
                            line = line[end.end() :]
                            private_key = False
                        start = re.search(
                            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", line
                        )
                        # A key spans lines in a text file, where its header
                        # ends a line; text that only names the header stays.
                        if start and not line[start.end() :].strip():
                            private_key = True
                            line = line[: start.start()] + REDACTED + "\n"
                    for pattern, text in rules.replace:
                        line = pattern.sub(text, line)
                    if any(pattern.search(line) for pattern in rules.forbid):
                        found.append(f"{label}:{number}")
                    writer.write(line)
            return found
        except UnicodeDecodeError:
            pass
    shutil.copyfile(source, target)
    return []


def _copy_files(
    run_dir: Path, files: Iterable[Path], destination: Path, rules: Rules
) -> list[str]:
    """Copy files of a run, keeping their paths, and return forbidden matches."""

    found: list[str] = []
    for source in files:
        relative = source.relative_to(run_dir).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in rules.exclude):
            continue
        found += _copy_file(
            source, destination / _export_path(relative, rules), rules, relative
        )
    return found


def _export_path(relative: str, rules: Rules) -> str:
    """Apply the same publication rules to paths generated from model aliases."""
    for pattern, replacement in rules.replace:
        relative = pattern.sub(replacement, relative)
    return relative


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _grade(
    sample_dir: Path, info: Mapping[str, Any]
) -> tuple[dict[str, Any], float | None, str | None]:
    """Return the bench record of a sample, its score and the grader error.

    The score is ``None`` when the sample was not graded or grading failed.
    """

    bench = read_scores(sample_dir) or (info.get("runtime") or {}).get("bench") or {}
    scores = bench.get("scores")
    if not isinstance(scores, dict):
        scores = bench
    error = bench.get("error") or scores.get("error")
    if not bench or error:
        return bench, None, error
    return (
        bench,
        sample_score(scores, bench.get("score_key") or "overall_score"),
        None,
    )


def _blocks(value: Any) -> list[dict[str, Any]]:
    """ACTF content as the text and image blocks of a Pi message."""

    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    blocks: list[dict[str, Any]] = []
    for block in value or []:
        if isinstance(block, str):
            blocks.append({"type": "text", "text": block})
        elif isinstance(block, Mapping) and block.get("type") == "image":
            try:
                header, _, data = _image_data_uri(block, "image").partition(",")
            except SFTConversionError:
                blocks.append({"type": "text", "text": "<image>"})
                continue
            media_type = header[len("data:") : -len(";base64")]
            blocks.append({"type": "image", "data": data, "mimeType": media_type})
        elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
            blocks.append({"type": "text", "text": block["text"]})
        else:
            blocks.append(
                {"type": "text", "text": json.dumps(block, ensure_ascii=False)}
            )
    return blocks


def _time(value: Any) -> datetime | None:
    """Parse an ACTF time (``YYYY-MM-DD HH:MM:SS.ffffff``, UTC)."""

    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    fields = function if isinstance(function, Mapping) else call
    arguments = fields.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    if not isinstance(arguments, Mapping):
        arguments = {"value": arguments}
    return {
        "type": "toolCall",
        "id": str(call.get("id", "")),
        "name": str(fields.get("name", "")),
        "arguments": dict(arguments),
    }


def _usage(metric: Mapping[str, Any]) -> dict[str, int]:
    def count(*names: str) -> int:
        return next((int(metric[name]) for name in names if metric.get(name)), 0)

    prompt, output = count("prompt_tokens_len"), count("completion_tokens_len")
    read = count("cache_read_tokens", "cache_read_tokens_len")
    write = count("cache_write_tokens", "cache_write_tokens_len")
    return {
        "input": max(prompt - read - write, 0),
        "output": output,
        "cacheRead": read,
        "cacheWrite": write,
        "totalTokens": prompt + output,
    }


def _tool_definitions(tools: Any) -> list[dict[str, Any]]:
    """The function tools of an ACTF trajectory as Pi tool declarations."""

    definitions: list[dict[str, Any]] = []
    for tool in tools if isinstance(tools, list) else []:
        function = tool.get("function", tool) if isinstance(tool, Mapping) else None
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            definitions.append(
                {
                    "name": function["name"],
                    "description": function.get("description") or "",
                    "parameters": function.get("parameters") or {},
                }
            )
    return definitions


def session_trace(
    trajectory: Mapping[str, Any], header: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Convert one ACTF trajectory to a session file the Hub reads as a trace.

    Used for the Harnesses whose native trace is not a session format the Hub
    knows. The lines follow Pi's session format, version 3, which the Hub
    accepts from any harness that names itself in the header. ``header`` gives
    the session ``id`` and ``harness`` and may give ``name``, ``model`` and
    further metadata. ``cwd`` is empty, because ACTF does not record the working
    directory; the Hub reads a session as Pi's format only when the header has
    the field. The system prompt and the tools become a ``system``
    message. Every ACTF step becomes a ``user`` message, when it has user input,
    an ``assistant`` message with thinking, text and tool-call blocks, and one
    ``toolResult`` message per tool result.
    """

    steps = trajectory.get("steps") or []
    clock = next(
        (time for step in steps if (time := _time(step.get("started_at")))),
        datetime.fromtimestamp(0, timezone.utc),
    )

    def stamp() -> str:
        return clock.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    lines: list[dict[str, Any]] = [
        {"type": "session", "version": 3, "timestamp": stamp(), "cwd": "", **header}
    ]

    def entry(kind: str, **fields: Any) -> None:
        parent = lines[-1]["id"] if len(lines) > 1 else None
        lines.append(
            {
                "type": kind,
                "id": f"{len(lines):08x}",
                "parentId": parent,
                "timestamp": stamp(),
                **fields,
            }
        )

    def message(**fields: Any) -> None:
        entry("message", message={**fields, "timestamp": int(clock.timestamp() * 1000)})

    model = str(header.get("model") or "")
    provider, _, model_id = model.rpartition("/")
    if model:
        entry("model_change", provider=provider, modelId=model_id)
    if header.get("name"):
        entry("session_info", name=header["name"])
    tools = _tool_definitions(trajectory.get("tools"))
    if trajectory.get("system_prompt") or tools:
        system = "\n".join(
            block["text"] for block in _blocks(trajectory.get("system_prompt") or "")
        )
        message(
            role="system", content=system, **({"toolsAdded": tools} if tools else {})
        )

    names: dict[str, str] = {}
    for step in steps:
        clock = _time(step.get("started_at")) or clock
        user = _blocks(step.get("user_content", ""))
        if user:
            message(role="user", content=user)
        assistant = step.get("assistant_content") or {}
        content: list[dict[str, Any]] = []
        if assistant.get("reasoning_content"):
            content.append(
                {"type": "thinking", "thinking": str(assistant["reasoning_content"])}
            )
        content += _blocks(assistant.get("content", ""))
        calls = [_tool_call(call) for call in assistant.get("tool_calls") or []]
        names.update((call["id"], call["name"]) for call in calls)
        reply: dict[str, Any] = {
            "role": "assistant",
            "content": content + calls,
            "provider": provider,
            "model": model_id,
        }
        metric = step.get("metric") or {}
        if metric:
            reply["usage"] = _usage(metric)
        if metric.get("stop_reason"):
            reply["stopReason"] = metric["stop_reason"]
        message(**reply)
        for result in step.get("observation") or []:
            call_id = str(result.get("toolCallId", ""))
            message(
                role="toolResult",
                toolCallId=call_id,
                toolName=str(result.get("toolName") or names.get(call_id, "")),
                content=_blocks(result.get("content", "")),
                isError=bool(result.get("isError")),
            )
    return lines


def _trace_line(line: str) -> str:
    return _TRACE_LINE_BREAKS.sub(lambda match: f"\\u{ord(match[0]):04x}", line)


def _write_trace(
    path: Path,
    sample_dir: Path,
    info: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    header: Mapping[str, Any],
) -> bool:
    """Write the session trace of one sample and tell whether it has one.

    Preserve native OpenClaw, Codex and Claude Code records, escaping Unicode
    line separators for the Hub. An OpenClaw session header also gets
    ``header``, which names the harness and gives the session an id that is
    unique in the run. Convert other samples from ACTF. A DeepSeek sample
    without an ACTF keeps its native journal as evidence of the partial run.
    """

    raw = sample_dir / "trace.jsonl"
    schema = (info.get("trajectory") or {}).get("raw_schema")
    preserve_native = schema in NATIVE_TRACES or (
        schema == "roller.deepseek-events/v1" and not trajectory.get("steps")
    )
    if preserve_native and raw.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        sessions = 0
        with (
            raw.open(encoding="utf-8", newline="") as reader,
            path.open("w", encoding="utf-8", newline="") as writer,
        ):
            for line in reader:
                if schema == "openclaw-session-jsonl" and '"session"' in line:
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("type") == "session":
                        # Roller starts every OpenClaw session under the same id.
                        sessions += 1
                        suffix = f"#{sessions}" if sessions > 1 else ""
                        row = {**row, **header, "id": header["id"] + suffix}
                        line = json.dumps(row, ensure_ascii=False) + "\n"
                writer.write(_trace_line(line))
        return True
    if not trajectory.get("steps"):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for row in session_trace(trajectory, header):
            writer.write(_trace_line(json.dumps(row, ensure_ascii=False)) + "\n")
    return True


def sample_row(
    run_name: str,
    run_root: Path,
    sample_dir: Path,
    trace_path: Path | None = None,
    root: Path | None = None,
    results: Path | None = None,
) -> dict[str, Any]:
    """Describe one sample as a table row.

    With ``trace_path``, also write the session of the sample there and record
    the path, relative to ``root``, in the row. ``results`` is the directory
    whose file names fill ``files``, the ``results`` of the sample by default.
    """

    info = _json(sample_dir / "info.json")
    bench, score, grader_error = _grade(sample_dir, info)
    metadata = bench.get("metadata") or {}
    trajectory = _json(sample_dir / "trace_ACTF.json")
    steps = trajectory.get("steps") or []
    sft = _json(sample_dir / "trace_SFT.json")
    usage = info.get("usage") or {}
    task_id = str(info.get("task_id") or sample_dir.name)
    model = info.get("model") or sample_dir.parent.name
    sample = sample_dir.relative_to(run_root).as_posix()
    results = results or sample_dir / "results"
    trace = None
    if trace_path is not None and root is not None:
        header = {
            "harness": info.get("harness"),
            "id": f"{run_name}/{sample_dir.name}",
            "name": f"{task_id} · {model}",
            "model": model,
            "roller": {
                "run": run_name,
                "sample": sample,
                "task_id": task_id,
                "status": info.get("status") or "partial",
                "score": score,
            },
        }
        if _write_trace(trace_path, sample_dir, info, trajectory, header):
            trace = trace_path.relative_to(root).as_posix()
    return {
        "run": run_name,
        "sample": sample,
        "trace": trace,
        "task_id": task_id,
        "upstream_task_id": str(metadata.get("upstream_task_id") or task_id),
        "run_index": int(metadata.get("run_index") or 1),
        "bench": bench.get("bench"),
        "category": bench.get("category"),
        "model": model,
        "harness": info.get("harness"),
        "status": info.get("status") or "partial",
        "error": info.get("error"),
        "score": score,
        "grader_error": grader_error,
        "steps": len(steps),
        "tool_calls": sum(
            len((step.get("assistant_content") or {}).get("tool_calls") or [])
            for step in steps
        ),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "elapsed_seconds": float(
            (info.get("timing") or {}).get("elapsed_seconds") or 0.0
        ),
        "files": sorted(
            path.relative_to(results).as_posix()
            for path in results.rglob("*")
            if path.is_file()
        ),
        "messages": json.dumps(sft["messages"], ensure_ascii=False) if sft else None,
        "tools": json.dumps(sft["tools"], ensure_ascii=False) if sft else None,
    }


def _parts(run_dir: Path) -> dict[str, list[Path]]:
    """Group the sample directories of a run by Harness and model.

    A part is named ``<harness>-<model>``, with the model id without its
    provider: ``openclaw-glm-5`` for ``vllm/glm-5`` on OpenClaw. Parts of a run
    that share that name are named after the whole model reference.
    """

    groups: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(run_dir.glob("samples/*/*/info.json")):
        info = _json(path)
        harness = str(info.get("harness") or "agent")
        model = str(info.get("model") or path.parent.parent.name)
        groups.setdefault((harness, model), []).append(path.parent)
    names = {
        key: safe_component(f"{key[0]}-{key[1].rsplit('/', 1)[-1]}") for key in groups
    }
    shared = Counter(names.values())
    return {
        names[key]
        if shared[names[key]] == 1
        else safe_component(f"{key[0]}-{key[1]}"): samples
        for key, samples in sorted(groups.items())
    }


def _rewrite_run_files(run_root: Path, had_report: bool) -> None:
    """Make the run-level files of a part describe the samples it holds.

    ``info.json``, ``sft.jsonl``, ``sft_index.jsonl`` and ``sft_segments.jsonl`` are written by the
    code that writes them during collection, and ``bench-report.json`` by the
    code of ``roller bench report``.
    """

    results = sorted(
        (_json(path) for path in run_root.glob("samples/*/*/info.json")),
        key=lambda result: result.get("request_index") or 0,
    )
    before = _json(run_root / "info.json")
    collector = Collector(CollectionConfig(run_dir=run_root), lambda slot: None)
    collector._write_run_info(
        results, total_requests=len(results), state=before.get("state") or "finished"
    )
    info = _json(run_root / "info.json")
    for key in ("updated_at", "finished_at"):
        if key in before:
            info[key] = before[key]
        else:
            info.pop(key, None)
    (run_root / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    collector._write_sft_dataset(results)
    report = run_root / "bench-report.json"
    report.unlink(missing_ok=True)
    if had_report and any((r.get("runtime") or {}).get("bench") for r in results):
        from roller.cli import _bench_report  # imported here: cli imports this module

        with contextlib.redirect_stdout(io.StringIO()):
            _bench_report(SimpleNamespace(run_dir=str(run_root), output=str(report)))


def stats_table(rows: Iterable[Mapping[str, Any]]) -> str:
    """Summarize the samples of each part, a Harness and a model, as Markdown.

    The mean score divides by every sample of the model, so a sample without a
    score counts as zero, as it does in ``roller bench report``.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["run"], []).append(row)
    graded = any(
        row["score"] is not None or row["grader_error"]
        for group in groups.values()
        for row in group
    )
    # An endpoint that reports no usage leaves every count at zero.
    metered = any(row["total_tokens"] for group in groups.values() for row in group)
    header = ["Trajectories", "Harness", "Model", "Samples", "Completed"]
    if graded:
        header += ["Scored", "Mean score"]
    header += ["Mean steps"] + ["Total tokens"] * metered
    lines = [
        "| " + " | ".join(header) + " |",
        "| --- | --- | --- |" + " ---: |" * (len(header) - 3),
    ]
    for run, group in sorted(groups.items()):
        scores = [row["score"] for row in group if row["score"] is not None]
        cells = [
            f"`{run}`",
            str(group[0]["harness"] or ""),
            f"`{str(group[0]['model']).rsplit('/', 1)[-1]}`",
            str(len(group)),
            str(sum(row["status"] == "ok" for row in group)),
        ]
        if graded:
            cells += [str(len(scores)), f"{sum(scores) / len(group):.3f}"]
        cells.append(f"{sum(row['steps'] for row in group) / len(group):.1f}")
        if metered:
            cells.append(f"{sum(row['total_tokens'] for row in group):,}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tables(output: Path) -> Iterator[dict[str, Any]]:
    """Read the rows of every part exported into a dataset folder."""

    for table in sorted((output / DATA_DIR).glob("*.jsonl")):
        with table.open(encoding="utf-8") as reader:
            for line in reader:
                yield json.loads(line)


def _write_card(output: Path, name: str, part: str, table: str, runs: bool) -> None:
    """Create the dataset card, or refresh the statistics of an existing one."""

    card = output / "README.md"
    stats = f"{STATS_START}\n{table}\n{STATS_END}"
    if not card.is_file():
        text = _CARD.format(
            name=name,
            data_dir=DATA_DIR,
            traces_dir=TRACES_DIR,
            stats=stats,
            browse=_CARD_BROWSE.format(name=name, part=part) if runs else "",
            run_files=_CARD_RUN_FILES if runs else "",
        )
    else:
        text = card.read_text(encoding="utf-8")
        start, end = text.find(STATS_START), text.find(STATS_END)
        if start < 0 or end < start:
            return
        text = text[:start] + stats + text[end + len(STATS_END) :]
    card.write_text(text, encoding="utf-8")


def export_run(
    run_dir: Path,
    output: Path,
    rules: Rules | None = None,
    environ: Mapping[str, str] | None = None,
    redact: bool = True,
    runs: bool = False,
    machine: str | None = None,
    preserve_task_keys: bool = False,
) -> tuple[list[dict[str, Any]], Redaction]:
    """Write a run into a dataset folder; return its table rows and what was redacted.

    The run is split into one part per Harness and model. A part named
    ``<harness>-<model>`` gets ``traces/<part>/`` and ``data/<part>.jsonl``, so
    the folder can hold the parts of further runs; exporting a part again
    replaces them. With ``runs``, a part also gets ``runs/<part>/``, a Roller run
    directory with every file of the samples of that model. The statistics of
    the card cover every part in the folder.

    With ``redact``, the exported files have the endpoints, hosts, host paths
    and credential values of the collection replaced, see
    ``environment_redaction``; ``environ`` supplies the credential values,
    ``os.environ`` by default, and ``machine`` the name of the collection
    machine, the name of this machine by default. ``rules`` are applied in
    either case.

    ``preserve_task_keys`` skips credential-shape replacements for benchmarks
    with deliberate key fixtures. Known environment credentials and collection
    paths are still replaced when ``redact`` is enabled.

    Raises ``ValueError`` when a forbidden pattern is left in an exported file.
    """

    derived, redaction = (
        environment_redaction(
            run_dir,
            os.environ if environ is None else environ,
            machine_name() if machine is None else machine,
        )
        if redact
        else (Rules(), Redaction())
    )
    run_dir, output = run_dir.resolve(), output.resolve()
    extra = rules or Rules()
    rules = Rules(
        derived.replace
        + (export_audit.replacements() if redact and not preserve_task_keys else ())
        + extra.replace,
        extra.exclude,
        extra.forbid,
    )
    parts: dict[str, list[Path]] = {}
    source_samples: dict[str, Path] = {}
    for name, samples in _parts(run_dir).items():
        for pattern, replacement in rules.replace:
            name = pattern.sub(replacement, name)
        name = safe_component(name)
        parts.setdefault(name, []).extend(samples)
        source_samples.update(
            {
                _export_path(sample.relative_to(run_dir).as_posix(), rules): sample
                for sample in samples
            }
        )
    shared = [
        path
        for path in sorted(run_dir.rglob("*"))
        if runs
        and path.is_file()
        and output != path
        and output not in path.parents
        and (run_dir / "samples") not in path.parents
        and path.relative_to(run_dir).parts[0] not in {"atif", "hf", "sft"}
        and not path.name.endswith(".redaction-report.json")
    ]
    with contextlib.ExitStack() as stack:
        # The sessions and the tables are built from redacted copies. Without
        # ``runs`` the copies hold only the files these need and are not kept.
        copies = (
            output / "runs"
            if runs
            else Path(stack.enter_context(tempfile.TemporaryDirectory()))
        )
        found: list[str] = []
        for name, samples in parts.items():
            for stale in (output / "runs" / name, output / TRACES_DIR / name):
                if stale.exists():
                    shutil.rmtree(stale)
            files = shared + [
                path
                for sample in samples
                for path in (
                    sorted(sample.rglob("*"))
                    if runs
                    else [sample / file for file in _SAMPLE_FILES]
                )
                if path.is_file()
            ]
            found += _copy_files(run_dir, files, copies / name, rules)
        if found:
            shown = ", ".join(sorted(set(found))[:20])
            raise ValueError(
                f"{len(found)} lines still match a forbidden pattern: {shown}"
            )
        rows: list[dict[str, Any]] = []
        for name in parts:
            destination = copies / name
            if runs:
                _rewrite_run_files(
                    destination, (run_dir / "bench-report.json").is_file()
                )
            part = [
                sample_row(
                    name,
                    destination,
                    path.parent,
                    output / TRACES_DIR / name / f"{path.parent.name}.jsonl",
                    output,
                    None
                    if runs
                    else source_samples[path.parent.relative_to(destination).as_posix()]
                    / "results",
                )
                for path in sorted(destination.glob("samples/*/*/info.json"))
            ]
            table = output / DATA_DIR / f"{name}.jsonl"
            table.parent.mkdir(parents=True, exist_ok=True)
            with table.open("w", encoding="utf-8") as writer:
                for row in part:
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += part
    if not runs:
        # An earlier export with ``runs`` may have left the directory empty.
        with contextlib.suppress(OSError):
            (output / "runs").rmdir()
    if parts:
        _write_card(
            output, run_dir.name, next(iter(parts)), stats_table(_tables(output)), runs
        )
    if output.is_dir():
        export_audit.write_report(output)
    return rows, redaction
