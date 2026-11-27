from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from roller.cli import _build_parser, _collect
from roller.sandbox import SandboxRun, SandboxTarget
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
