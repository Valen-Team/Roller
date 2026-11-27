"""Environment and rule-based redaction for Hugging Face dataset exports."""

from __future__ import annotations

import fnmatch
import ipaddress
import json
import re
import shutil
import socket
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import roller.export_audit as export_audit
from roller.harnesses.opencode import LOCAL_PROXY_BYPASS, _ensure_local_proxy_bypass
from roller.redaction import REDACTED


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _export_path(relative: str, rules: Rules) -> str:
    """Apply the same publication rules to paths generated from model aliases."""
    for pattern, replacement in rules.replace:
        relative = pattern.sub(replacement, relative)
    return relative


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
    opencode_proxy = False
    for path in run_dir.glob("samples/*/*/info.json"):
        info = _json(path)
        provenance = info.get("provenance") or {}
        urls.update(url for url in provenance.get("api_base_urls") or [] if url)
        environment_names = provenance.get("environment_names") or []
        names.update(name for name in environment_names if name)
        opencode_proxy |= info.get("harness") == "opencode" and any(
            name.lower() == "no_proxy" for name in environment_names if name
        )
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
    contextual: list[tuple[re.Pattern[str], str]] = []
    if opencode_proxy:
        proxy = {
            name: environ[name] for name in ("NO_PROXY", "no_proxy") if name in environ
        }
        _ensure_local_proxy_bypass(proxy)
        values = {proxy["NO_PROXY"], ",".join(LOCAL_PROXY_BYPASS)}
        pattern = "|".join(
            re.escape(value) for value in sorted(values, key=len, reverse=True)
        )
        contextual.append(
            (
                re.compile(rf"(?i)(\bno_proxy\s*=\s*)(?:{pattern})(?![\w.,:/-])"),
                r"\1localhost,127.0.0.1",
            )
        )
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
            # Also inside names such as ``pytest-of-<user>``, ``_Users_<user>_``
            # and ``<user>.github.io``, after a JSON escape such as ``\n`` and
            # after a URL escape such as ``%2F``.
            after = r"(?:(?<![a-z0-9])|(?<=\\[nrt])|(?<=%[0-9a-f]{2}))"
            homes.append(
                (re.compile(rf"(?i){after}{escaped}(?![a-z0-9])"), placeholder)
            )

    if not literals:
        return Rules(replace=(*contextual, *homes)), Redaction(
            unset=tuple(unset), users=tuple(users)
        )
    pattern = re.compile(
        "|".join(
            patterns.get(text) or re.escape(text)
            for text in sorted(literals, key=len, reverse=True)
        )
    )
    return (
        Rules(
            replace=(
                *contextual,
                (pattern, lambda match: literals[match.group(0)]),
                *homes,
            )
        ),
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
