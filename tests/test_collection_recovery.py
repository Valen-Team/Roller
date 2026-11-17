from __future__ import annotations

import json
import os
import signal
import threading
import time
from unittest.mock import patch

import pytest
from test_collector import FakeHarness, make_requests

from roller.cli import _build_parser, _collect
from roller.collector import CollectionConfig, CollectionStopped, Collector
from roller.models import HarnessResult


def test_missing_credentials_fail_before_docker_even_on_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"id": "hello", "prompt": "hello"}) + "\n")
    for flags in ([], ["--dry-run"]):
        args = _build_parser().parse_args(
            [
                "collect",
                "--tasks",
                str(tasks),
                "--harness",
                "codex",
                "--model",
                "m",
                "--output",
                str(tmp_path / "runs"),
                *flags,
            ]
        )
        with patch("roller.cli._docker_preflight") as docker:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                _collect(args)
            docker.assert_not_called()
    assert not (tmp_path / "runs").exists()


def test_circuit_breaker_stops_dispatch_and_keeps_checkpoint(tmp_path):
    calls = []

    class Broken:
        name = "fake"

        def run(self, request, output_dir, attempt):
            calls.append(request.index)
            time.sleep(0.01)
            return HarnessResult.error_result("service unavailable")

    collector = Collector(
        CollectionConfig(
            run_dir=tmp_path, parallel=1, launch_interval=0.02, max_consecutive_errors=2
        ),
        lambda slot: Broken(),
    )
    with pytest.raises(CollectionStopped, match="2 rollouts in a row"):
        collector.collect(make_requests())
    info = json.loads((tmp_path / "info.json").read_text())
    assert info["state"] == "interrupted"
    assert info["summary"]["processed"] == len(calls)
    assert len(calls) < 4


def test_first_sigint_harvests_running_results(tmp_path):
    state = {
        "lock": threading.Lock(),
        "calls": {},
        "active": 0,
        "max_active": 0,
        "delay": 0.1,
    }
    collector = Collector(
        CollectionConfig(run_dir=tmp_path, parallel=1),
        lambda slot: FakeHarness(state, slot),
    )

    def interrupt():
        while not state["active"]:
            time.sleep(0.001)
        time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGINT)

    sender = threading.Thread(target=interrupt)
    sender.start()
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        collector.collect(make_requests())
    sender.join()
    info = json.loads((tmp_path / "info.json").read_text())
    assert info["state"] == "interrupted"
    assert info["summary"]["ok"] >= 1
    assert info["summary"]["processed"] < 4
    assert (
        len((tmp_path / "sft.jsonl").read_text().splitlines()) == info["summary"]["ok"]
    )
    assert signal.getsignal(signal.SIGINT) == before


def test_termination_cleans_containers_and_checkpoints(tmp_path):
    active = threading.Event()
    release = threading.Event()

    class Waiting:
        name = "fake"

        def run(self, request, output_dir, attempt):
            active.set()
            release.wait(2)
            return HarnessResult.error_result("container removed")

    def terminate():
        active.wait(2)
        os.kill(os.getpid(), signal.SIGTERM)

    sender = threading.Thread(target=terminate)
    sender.start()
    collector = Collector(
        CollectionConfig(run_dir=tmp_path, parallel=1), lambda slot: Waiting()
    )
    try:
        with patch(
            "roller.collector.remove_own_containers",
            side_effect=lambda: release.set() or 1,
        ) as clean:
            with pytest.raises(CollectionStopped, match="SIGTERM"):
                collector.collect(make_requests())
            clean.assert_called_once()
        assert (
            json.loads((tmp_path / "info.json").read_text())["state"] == "interrupted"
        )
    finally:
        release.set()
        sender.join()
