"""Flatten an image through a filtered export; never retain its sensitive layers."""

import argparse
import io
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path
from urllib.parse import urlsplit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default="wildclawbench-ubuntu:v1.3")
parser.add_argument("--target", default="roller-wildclaw-clean:1.3")
parser.add_argument("--output", type=Path, default=Path("wildclaw-image-build"))
args = parser.parse_args()
ROOT = args.output.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "images").mkdir(exist_ok=True)
(ROOT / "audit").mkdir(exist_ok=True)
SOURCE = args.source
TARGET = args.target
report = {
    "started_at": time.time(),
    "source": SOURCE,
    "target": TARGET,
    "removed": [],
    "rewritten": [],
    "credential_locations": [],
}
config = json.loads(subprocess.check_output(["docker", "image", "inspect", SOURCE]))[0]
report["source_image_id"] = config["Id"]
secrets = set()
for item in config["Config"].get("Env", []):
    name, _, value = item.partition("=")
    if "proxy" in name.lower():
        try:
            parts = urlsplit(value)
        except ValueError:
            continue
        if parts.username or parts.password:
            report["credential_locations"].append(
                {"location": "image.Config.Env." + name, "kind": "proxy_user_password"}
            )
            for s in [value, parts.netloc, parts.password]:
                if s and len(s) >= 8:
                    secrets.add(s.encode())
container = subprocess.check_output(
    ["docker", "create", SOURCE, "/bin/true"], text=True
).strip()
try:
    # State files are copied only into process memory for value-free auditing.
    for folder in [
        "/root/.openclaw/identity",
        "/root/.openclaw/devices",
        "/root/.config/moltbook",
    ]:
        p = subprocess.run(
            ["docker", "cp", container + ":" + folder, "-"], capture_output=True
        )
        if p.returncode:
            continue
        with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:
            for m in t:
                if not m.isfile() or m.size > 2_000_000:
                    continue
                raw = t.extractfile(m).read()
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                keys = []

                def walk(x, path="", keys=keys):
                    if isinstance(x, dict):
                        for k, v in x.items():
                            here = path + "." + k
                            if (
                                isinstance(v, str)
                                and re.search(
                                    r"(?i)private.?key|token|api.?key|password|secret",
                                    k,
                                )
                                and len(v) >= 8
                            ):
                                keys.append({"field": here})
                                secrets.add(v.encode())
                            elif isinstance(v, dict | list):
                                walk(v, here)
                    elif isinstance(x, list):
                        for i, v in enumerate(x):
                            walk(v, path + f"[{i}]")

                walk(d)
                if keys:
                    report["credential_locations"].append(
                        {"location": folder + "/" + Path(m.name).name, "fields": keys}
                    )
    # Account-bearing URLs anywhere in startup files are recognized without reuse.
    for file in [
        "/root/.bashrc",
        "/root/.profile",
        "/root/.zshrc",
        "/etc/environment",
        "/root/.openclaw/workspace/skills/aisa-twitter-api/SKILL.md",
    ]:
        p = subprocess.run(
            ["docker", "cp", container + ":" + file, "-"], capture_output=True
        )
        if p.returncode:
            continue
        with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:
            for m in t:
                if not m.isfile():
                    continue
                raw = t.extractfile(m).read()
                urls = re.findall(rb"https?://[^\s\"\x27<>]+", raw)
                count = 0
                for url in urls:
                    try:
                        v = urlsplit(url.decode())
                    except Exception:
                        continue
                    if (
                        v.username
                        and v.password
                        and not (v.hostname or "").endswith(
                            (".example", ".example.com", ".example.org")
                        )
                    ):
                        count += 1
                        for s in [url, v.password.encode()]:
                            if len(s) >= 8:
                                secrets.add(s)
                if count:
                    report["credential_locations"].append(
                        {
                            "location": file,
                            "kind": "proxy_user_password",
                            "count": count,
                        }
                    )
    excluded = [
        "root/.config/moltbook",
        "root/.ssh",
        "root/.aws",
        "root/.azure",
        "root/.kube",
        "root/.docker",
        "root/.npm",
        "root/.local/share/keyrings",
        "root/.local/share/moltbook",
    ]
    excluded_files = {
        "root/.bash_history",
        "root/.zsh_history",
        "root/.python_history",
        "root/.git-credentials",
        "root/.netrc",
        "root/.npmrc",
        "root/.pypirc",
        "root/.condarc",
    }

    def banned(name):
        personal_openclaw = name.startswith("root/.openclaw/") and not (
            name.startswith("root/.openclaw/workspace/skills/")
            or name
            in {
                "root/.openclaw/workspace/skills",
                "root/.openclaw/workspace",
                "root/.openclaw",
            }
        )
        return (
            personal_openclaw
            or name in excluded_files
            or any(name == x or name.startswith(x + "/") for x in excluded)
            or (
                name.startswith("root/.cache/")
                and not name.startswith("root/.cache/ms-playwright/")
            )
        )

    source = subprocess.Popen(["docker", "export", container], stdout=subprocess.PIPE)
    outfile = ROOT / "images/clean-rootfs.tar"
    report["files_kept"] = 0
    with (
        tarfile.open(fileobj=source.stdout, mode="r|") as src,
        tarfile.open(outfile, "w") as dst,
    ):
        for m in src:
            name = m.name.lstrip("./")
            if banned(name):
                if len(report["removed"]) < 100:
                    report["removed"].append(name)
                continue
            if not m.isfile():
                dst.addfile(m)
                continue
            f = src.extractfile(m)
            if m.size <= 4_000_000:
                raw = f.read()
                original = raw
                # Only personal image config and installed account-bearing skills are rewritten.
                # Task fixtures and benchmark code are never modified by this sanitizer.
                if name.startswith(
                    (
                        "root/.openclaw/workspace/skills/",
                        "root/.bashrc",
                        "root/.profile",
                        "root/.zshrc",
                        "etc/environment",
                        "etc/profile.d/",
                    )
                ):
                    for secret in sorted(secrets, key=len, reverse=True):
                        raw = raw.replace(secret, b"[REMOVED_IMAGE_CREDENTIAL]")
                if name in ["root/.bashrc", "root/.profile", "root/.zshrc"]:
                    # Retain language initialization but remove all inherited proxy exports.
                    lines = raw.splitlines(keepends=True)
                    raw = b"".join(
                        line
                        for line in lines
                        if not re.search(
                            rb"(?i)(?:https?|all|no)_proxy|proxy_on|proxy_off|MY_HTTP_PROXY",
                            line,
                        )
                    )
                if raw != original:
                    report["rewritten"].append(name)
                    m.size = len(raw)
                dst.addfile(m, io.BytesIO(raw))
            else:
                dst.addfile(m, f)
            report["files_kept"] += 1
    if source.wait() != 0:
        raise RuntimeError("docker export failed")
    report["rootfs_bytes"] = outfile.stat().st_size
    # Independent rescan of the flattened output for known credential bytes/state.
    hits = []
    with tarfile.open(outfile) as check:
        for m in check:
            n = m.name.lstrip("./")
            if banned(n):
                hits.append({"file": n, "kind": "forbidden_state"})
            if not m.isfile():
                continue
            stream = check.extractfile(m)
            pending = b""
            overlap = max([len(v) for v in secrets] or [1]) - 1
            while data := stream.read(1024 * 1024):
                pending += data
                if any(v in pending for v in secrets):
                    hits.append({"file": n, "kind": "known_credential"})
                    break
                pending = pending[-overlap:] if overlap else b""
    report["residual_hits"] = hits
    assert not hits, "Credential remnants found in sanitized rootfs"
    path = next(
        (
            x.split("=", 1)[1]
            for x in config["Config"].get("Env", [])
            if x.startswith("PATH=")
        ),
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    )
    assert not any(s in path.encode() for s in secrets)
    cmd = [
        "docker",
        "import",
        "--change",
        "ENV PATH=" + path,
        "--change",
        "WORKDIR /tmp_workspace",
        "--change",
        'CMD ["/bin/bash"]',
        str(outfile),
        TARGET,
    ]
    new = subprocess.check_output(cmd, text=True).strip()
    report["clean_image_id"] = new
    clean = json.loads(subprocess.check_output(["docker", "image", "inspect", TARGET]))[
        0
    ]
    report["clean_layers"] = clean["RootFS"]["Layers"]
    report["clean_env_names"] = [
        v.split("=", 1)[0] for v in clean["Config"].get("Env", [])
    ]
    assert len(report["clean_layers"]) == 1
    report.update(state="sanitized_and_verified", finished_at=time.time())
    (ROOT / "audit/image-sanitization.json").write_text(json.dumps(report, indent=2))
    os.chmod(ROOT / "audit/image-sanitization.json", 0o600)
    print(
        f"Created {TARGET}: personal builder state removed; benchmark workspaces are staged separately and unchanged.",
        flush=True,
    )
finally:
    subprocess.run(["docker", "rm", container], stdout=subprocess.DEVNULL)
