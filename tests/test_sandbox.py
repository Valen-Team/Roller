from __future__ import annotations

import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock

import pytest

from roller.cli import _build_parser, _collect
from roller.sandbox import SandboxRun, SandboxTarget, route_sandbox_http
from roller.sandbox_files import unpack_archive


class FailingProvider:
    def __init__(self):
        self.removed = []

    def resolve(self, template):
        assert template == "bench-runtime"
        return "runtime-2026"

    def create(self, template, version, *, name, lifetime, timeout, allocated):
        assert template == "bench-runtime" and version == "runtime-2026"
        allocated("owned-instance")
        raise RuntimeError("startup did not complete")

    def remove(self, instance_id):
        self.removed.append(instance_id)


def test_failed_sandbox_start_releases_its_allocation(tmp_path: Path):
    provider = FailingProvider()
    target = SandboxTarget("test", "bench-runtime", "runtime-2026")
    run = SandboxRun(tmp_path, target, provider)
    with pytest.raises(RuntimeError, match="startup did not complete"):
        run.create("first-attempt", 120, 20)
    assert provider.removed == ["owned-instance"]
    assert not list((tmp_path / ".sandbox-resources").glob("*.json"))


def test_remote_collect_is_docker_free_and_records_environment_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    bench = tmp_path / "benchmark"
    bench.mkdir()
    workspace = bench / "workspace" / "sample" / "exec"
    workspace.mkdir(parents=True)
    (workspace / "fixture.bin").write_bytes(b"\x00\xff")
    (bench / "sample.md").write_text(
        "## Prompt\nAnswer briefly.\n\n## Workspace Path\n\nworkspace/sample\n"
    )
    output = tmp_path / "runs"
    argv = [
        "collect",
        "--bench",
        "wildclawbench",
        "--bench-root",
        str(bench),
        "--harness",
        "deepseek",
        "--model",
        "mock-model",
        "--output",
        str(output),
        "--run-name",
        "remote",
        "--sandbox-provider",
        "test",
        "--sandbox-template",
        "bench-runtime",
        "--retries",
        "0",
    ]
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")

    def unexpected(*args, **kwargs):
        pytest.fail(f"unexpected preflight: {args}, {kwargs}")

    monkeypatch.setattr("roller.cli.collect._docker_preflight", unexpected)
    monkeypatch.setattr("roller.cli.collect.prepare_task_images", unexpected)
    monkeypatch.setattr("roller.cli.collect.load_sandbox_provider", unexpected)
    parser = _build_parser()
    assert _collect(parser.parse_args([*argv, "--dry-run"])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sandbox_provider"] == "test"
    assert payload["sandbox_template"] == "bench-runtime"
    assert payload["docker_image"] is None

    provider = FailingProvider()
    monkeypatch.setattr(
        "roller.cli.collect.load_sandbox_provider", lambda name: provider
    )
    assert _collect(parser.parse_args(argv)) == 1
    assert provider.removed == ["owned-instance"]
    configuration = json.loads((output / "remote" / "info.json").read_text())[
        "configuration"
    ]
    assert configuration["deepseek"]["sandbox_version"] == "runtime-2026"

    monkeypatch.setattr(provider, "resolve", lambda template: "different-runtime")
    monkeypatch.setattr(provider, "create", unexpected)
    with pytest.raises(FileExistsError, match="sandbox_version"):
        _collect(parser.parse_args([*argv, "--resume"]))


def test_sandbox_archive_refuses_to_traverse_a_link(tmp_path: Path):
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as stream:
        entry = tarfile.TarInfo("outside")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../escaped"
        stream.addfile(entry)
    with pytest.raises(ValueError, match="not a file or directory"):
        unpack_archive(archive, tmp_path / "target")
    assert not (tmp_path / "escaped").exists()


def test_sandbox_wire_records_token_ids_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from roller import sandbox_wire

    made = []

    class Proxy:
        proxy_base_url = "http://127.0.0.1:1"

        def __init__(self, *args, **kwargs):
            made.append(kwargs)
            (tmp_path / "stop").touch()

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(sandbox_wire, "WireTraceProxy", Proxy)
    for token_ids in (False, True):
        (tmp_path / "stop").unlink(missing_ok=True)
        (tmp_path / "config.json").write_text(
            json.dumps(
                {"api_base": "http://model", "calls": False, "token_ids": token_ids}
            )
        )
        sandbox_wire.main(tmp_path)
    assert [kwargs["token_ids_path"] for kwargs in made] == [
        None,
        tmp_path / "token_ids.jsonl",
    ]


class MemoryProcess:
    """The ``SandboxProcess`` half of the documented contract."""

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        pass


class MemoryBox:
    """The minimal ``Sandbox`` from CONTRIBUTING.md, backed by a dict of files."""

    version = "memory-1"

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.closed = False

    def run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float,
    ) -> CompletedProcess[str]:
        return CompletedProcess(argv, 0, " ".join(argv), "")

    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        log_path: Path,
    ) -> MemoryProcess:
        Path(log_path).write_text(" ".join(argv), encoding="utf-8")
        return MemoryProcess()

    def upload(
        self,
        source: Path,
        target: str,
        *,
        contents: bool = False,
        timeout: float,
    ) -> None:
        self.files[target] = Path(source).read_bytes()

    def download(
        self,
        source: str,
        target: Path,
        *,
        contents: bool = False,
        timeout: float,
    ) -> None:
        Path(target).write_bytes(self.files[source])

    def close(self) -> None:
        self.closed = True


class MemoryProvider:
    """The minimal ``SandboxProvider`` from CONTRIBUTING.md."""

    def __init__(self) -> None:
        self.removed: list[str] = []

    def resolve(self, template: str) -> str:
        return "memory-1"

    def create(
        self,
        template: str,
        version: str,
        *,
        name: str,
        lifetime: float,
        timeout: float,
        allocated: Callable[[str], None],
    ) -> MemoryBox:
        assert version == self.resolve(template)
        allocated(f"{template}/{name}")
        return MemoryBox()

    def remove(self, instance_id: str) -> None:
        self.removed.append(instance_id)


def test_documented_provider_contract_runs_without_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    provider = MemoryProvider()
    template = "bench-runtime"
    target = SandboxTarget("memory", template, provider.resolve(template))
    run = SandboxRun(tmp_path, target, provider)

    box = run.create("attempt-1", 60, 10)
    assert box.run(["true"], timeout=5).returncode == 0
    assert box.spawn(["sleep", "1"], log_path=tmp_path / "spawn.log").wait() == 0

    base, key = "https://model.example/v1", "fixture-key"
    relay = "https://relay.example/v1"
    assert route_sandbox_http(box, base, key, 5) == (base, key)
    route = Mock(return_value=relay)
    monkeypatch.setattr(box, "route_http", route, raising=False)
    assert route_sandbox_http(box, base, key, 5) == (relay, "sandbox-relay")
    route.assert_called_once_with(base, key, 5)

    upload = tmp_path / "input.txt"
    upload.write_text("payload", encoding="utf-8")
    box.upload(upload, "/roller/input.txt", timeout=5)
    copied = tmp_path / "copy.txt"
    box.download("/roller/input.txt", copied, timeout=5)
    assert copied.read_text(encoding="utf-8") == "payload"

    receipt = tmp_path / ".sandbox-resources" / "attempt-1.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["instance_id"] == (
        "bench-runtime/attempt-1"
    )
    run.close("attempt-1", box)
    assert box.closed
    assert not receipt.exists()

    run.create("attempt-2", 60, 10)
    run.cleanup()
    assert provider.removed == ["bench-runtime/attempt-2"]
