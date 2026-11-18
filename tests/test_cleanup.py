from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from roller.cleanup import HOST, clean, kept_because
from roller.harnesses.docker import DockerRuntime, begin_collection


def test_cleanup_keeps_live_unknown_and_foreign_collectors():
    with patch("roller.cleanup.os.kill"):
        assert "still running" in kept_because("123", HOST)
    assert "not recorded" in kept_because("", "")
    assert "other-host" in kept_because("123", "other-host")
    with patch("roller.cleanup.os.kill", side_effect=ProcessLookupError):
        assert kept_because("123", HOST) is None


def test_cleanup_dry_run_does_not_remove_dead_owners_files_or_containers(tmp_path):
    bundle = tmp_path / ".roller-bundle-test"
    bundle.mkdir()
    (bundle / "owner.json").write_text(
        json.dumps({"pid": 123, "host": HOST, "run_dir": str(tmp_path / "run")})
    )

    class ListRuntime(DockerRuntime):
        """Answers the container listing; every other argv would be a removal."""

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[list[str]] = []

        def run(self, args, timeout=None):
            self.calls.append(list(args))
            return subprocess.CompletedProcess(
                list(args),
                0,
                f"abc\ttest\tcodex\t\t123\t{HOST}\t{tmp_path / 'run'}\n",
                "",
            )

    runtime = ListRuntime()
    with patch("roller.cleanup.os.kill", side_effect=ProcessLookupError):
        lines = clean(runtime, tmp_path, dry_run=True)
        assert len(lines) == 2
        assert all("would remove" in line for line in lines)
        assert bundle.exists()
        # One listing, and no removal while the run is a dry run.
        assert len(runtime.calls) == 1
        clean(runtime, tmp_path)
    assert not bundle.exists()
    assert any(call[:2] == ["rm", "-f"] for call in runtime.calls)


def test_launches_carry_owner_labels_and_separate_process_group(tmp_path):
    runtime = DockerRuntime()
    begin_collection(tmp_path)
    try:
        with patch("roller.harnesses.docker.subprocess.run") as run:
            runtime.run(["run", "-d", "image"])
        args = run.call_args.args[0]
        assert f"ai.roller.run_dir={tmp_path}" in args
        assert any(item.startswith("ai.roller.pid=") for item in args)
        assert run.call_args.kwargs["start_new_session"] is True
    finally:
        begin_collection(None)
