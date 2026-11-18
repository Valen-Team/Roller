from __future__ import annotations

import logging
import subprocess
import tempfile
import unittest
from pathlib import Path

from roller.harnesses.docker import (
    DockerRuntime,
    remove_container,
    warn_if_image_changed,
)


class FakeRuntime:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def run(self, args, timeout=None):
        self.calls.append((list(args), timeout))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RecordingRuntime(DockerRuntime):
    """A ``DockerRuntime`` whose process boundary records instead of executing.

    The named verbs build argv on top of ``run``, so recording here pins the
    exact command line every verb emits.
    """

    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        super().__init__("docker")
        self.calls: list[list[str]] = []
        self.processes: list[list[str]] = []
        self.stdout = stdout
        self.returncode = returncode

    def run(self, args, timeout=None):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(list(args), self.returncode, self.stdout, "")

    def popen(self, args, log_path, append=False):
        self.processes.append(list(args))
        return None


class RuntimeVerbTests(unittest.TestCase):
    """Each verb must emit the argv callers wrote by hand before it existed."""

    def test_start_args_match_the_hand_written_run_line(self):
        runtime = RecordingRuntime()

        args = runtime.start_args(
            "roller-task",
            "image:tag",
            ["tail", "-f", "/dev/null"],
            labels={"ai.roller.harness": "codex"},
            env=[("B", "2"), ("A", "1")],
            volumes=["/host:/guest:ro"],
            network="host",
            user="0",
        )

        self.assertEqual(
            args,
            [
                "run",
                "-d",
                "--name",
                "roller-task",
                "--label",
                "ai.roller.harness=codex",
                "--user",
                "0",
                "--network",
                "host",
                "-e",
                "B=2",
                "-e",
                "A=1",
                "-v",
                "/host:/guest:ro",
                "image:tag",
                "tail",
                "-f",
                "/dev/null",
            ],
        )

    def test_start_args_without_an_image_stops_before_it(self):
        runtime = RecordingRuntime()

        self.assertEqual(
            runtime.start_args("box", None, labels={"a": "b"}),
            ["run", "-d", "--name", "box", "--label", "a=b"],
        )

    def test_exec_args_put_env_then_workdir_then_container(self):
        runtime = RecordingRuntime()

        args = runtime.exec_args(
            "box",
            ["python3", "-c", "pass"],
            env={"A": "1", "B": "2"},
            workdir="/workspace",
        )

        self.assertEqual(
            args,
            [
                "exec",
                "-e",
                "A=1",
                "-e",
                "B=2",
                "-w",
                "/workspace",
                "box",
                "python3",
                "-c",
                "pass",
            ],
        )

    def test_exec_goes_through_run(self):
        runtime = RecordingRuntime()

        runtime.exec("box", ["true"], timeout=5)

        self.assertEqual(runtime.calls, [["exec", "box", "true"]])

    def test_copy_verbs_name_the_container_side(self):
        runtime = RecordingRuntime()

        runtime.copy_out("box", "/state/trace.jsonl", Path("/tmp/trace.jsonl"))
        runtime.copy_in(Path("/tmp/in.json"), "box", "/tmp/out.json")

        self.assertEqual(
            runtime.calls,
            [
                ["cp", "box:/state/trace.jsonl", "/tmp/trace.jsonl"],
                ["cp", "/tmp/in.json", "box:/tmp/out.json"],
            ],
        )

    def test_image_id_returns_none_when_the_image_is_absent(self):
        self.assertIsNone(RecordingRuntime(returncode=1).image_id("missing:tag"))
        self.assertEqual(
            RecordingRuntime(stdout="sha256:abc\n").image_id("present:tag"),
            "sha256:abc",
        )

    def test_image_field_names_the_template_field(self):
        runtime = RecordingRuntime(stdout="amd64")

        runtime.image_field("image:tag", "Architecture", timeout=60)

        self.assertEqual(
            runtime.calls,
            [["image", "inspect", "image:tag", "--format", "{{.Architecture}}"]],
        )

    def test_listing_and_bulk_removal(self):
        runtime = RecordingRuntime()

        runtime.ps("{{.ID}}")
        runtime.remove_all(["a", "b"], timeout=60)

        self.assertEqual(
            runtime.calls,
            [["ps", "-a", "--format", "{{.ID}}"], ["rm", "-f", "a", "b"]],
        )

    def test_exec_logged_streams_into_a_log_file(self):
        runtime = RecordingRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "rollout.log"
            runtime.exec_logged("box", ["tail", "-f", "/dev/null"], log, append=True)

        self.assertEqual(
            runtime.processes, [["exec", "box", "tail", "-f", "/dev/null"]]
        )


class RemoveContainerTests(unittest.TestCase):
    def test_warns_when_rollout_image_differs_from_preflight(self):
        logger = logging.getLogger("roller.tests.image-digest")

        with self.assertLogs(logger, level="WARNING") as captured:
            warn_if_image_changed(
                logger,
                "roller:test",
                "sha256:preflight",
                "sha256:rollout",
            )

        self.assertIn("changed after preflight", captured.output[0])
        self.assertIn("sha256:preflight", captured.output[0])
        self.assertIn("sha256:rollout", captured.output[0])

        with self.assertNoLogs(logger, level="WARNING"):
            warn_if_image_changed(
                logger,
                "roller:test",
                "sha256:same",
                "sha256:same",
            )
            warn_if_image_changed(logger, "roller:test", None, "sha256:rollout")

    def test_retries_timeout_with_longer_budget(self):
        timeout = subprocess.TimeoutExpired(["docker", "rm"], 30)
        success = subprocess.CompletedProcess([], 0, "", "")
        runtime = FakeRuntime([timeout, success])

        result = remove_container(runtime, "roller-test")

        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            runtime.calls,
            [(["rm", "-f", "roller-test"], 30), (["rm", "-f", "roller-test"], 60)],
        )

    def test_retries_nonzero_result(self):
        failure = subprocess.CompletedProcess([], 1, "", "busy")
        success = subprocess.CompletedProcess([], 0, "", "")
        runtime = FakeRuntime([failure, success])

        result = remove_container(runtime, "roller-test")

        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(runtime.calls), 2)

    def test_raises_last_timeout(self):
        first = subprocess.TimeoutExpired(["docker", "rm"], 30)
        second = subprocess.TimeoutExpired(["docker", "rm"], 60)
        runtime = FakeRuntime([first, second])

        with self.assertRaises(subprocess.TimeoutExpired) as raised:
            remove_container(runtime, "roller-test")

        self.assertIs(raised.exception, second)


if __name__ == "__main__":
    unittest.main()
