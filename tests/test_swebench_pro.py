from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from roller.benches import get_bench
from roller.benches.base import BenchBinding, BenchFormatError, GradeSpec
from roller.benches.grading import GradedHarness, GradeOutcome
from roller.benches.swebench_pro import (
    _DERIVED_IMAGES,
    _GRADER_SOURCE,
    SWEBenchProAdapter,
    SWEBenchProGradedHarness,
    _image_id,
    _test_fixture_patch,
    ensure_harness_image,
)
from roller.cli import main
from roller.harnesses.hermes import HermesConfig, HermesHarness
from roller.harnesses.openclaw import OpenClawConfig, OpenClawHarness
from roller.models import HarnessResult, RunRequest, Task


def _record(instance_id: str = "instance_owner__repo-deadbeef-v1") -> dict[str, str]:
    return {
        "repo": "owner/repo",
        "instance_id": instance_id,
        "base_commit": "a" * 40,
        "patch": "GOLD SOURCE PATCH MUST NOT LEAK",
        "test_patch": (
            "diff --git a/tests/hidden.txt b/tests/hidden.txt\n"
            "index 8d1a494..a6a0371 100644\n"
            "--- a/tests/hidden.txt\n"
            "+++ b/tests/hidden.txt\n"
            "@@ -1 +1 @@\n"
            "-BASE\n"
            "+HIDDEN\n"
        ),
        "problem_statement": '"Fix the implementation.\\nKeep compatibility."',
        "requirements": '"- Return the expected value."',
        "interface": '"Type: Function\\nName: solve"',
        "repo_language": "python",
        "fail_to_pass": "['required_test']",
        "pass_to_pass": "['regression_test']",
        "issue_specificity": '["major_bug"]',
        "issue_categories": '["back_end_knowledge"]',
        "before_repo_set_cmd": (
            f"git reset --hard {'a' * 40}\n"
            f"git checkout {'a' * 40}\n"
            "git checkout deadbeef -- tests/hidden.txt"
        ),
        "selected_test_files_to_run": '["tests/hidden.txt"]',
        "dockerhub_tag": "owner.repo-owner__repo-deadbeef-v1",
    }


def _write_fixture(root: Path, record: dict[str, str] | None = None) -> dict[str, str]:
    row = record or _record()
    (root / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    scripts = root / "run_scripts" / row["instance_id"]
    scripts.mkdir(parents=True)
    (scripts / "run_script.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (scripts / "parser.py").write_text("pass\n", encoding="utf-8")
    return row


class SWEBenchProAdapterTests(unittest.TestCase):
    def test_package_facade_preserves_the_original_import_surface(self) -> None:
        import roller.benches.swebench_pro as facade
        from roller.benches.swebench_pro.adapter import SWEBenchProAdapter as Adapter
        from roller.benches.swebench_pro.grader_runtime import (
            SWEBenchProGradedHarness as GradedHarnessAdapter,
        )

        assert facade.__file__ is not None
        self.assertEqual(Path(facade.__file__).name, "__init__.py")
        self.assertIs(facade.SWEBenchProAdapter, Adapter)
        self.assertIs(facade.SWEBenchProGradedHarness, GradedHarnessAdapter)

    def test_packaged_grader_source_matches_protocol_revision(self) -> None:
        self.assertEqual(
            hashlib.sha256(_GRADER_SOURCE.encode()).hexdigest(),
            "ce712519edd15de60da0b4cc73c996055af7445a85e1438a833dfc6ea3c09a14",
        )
        compile(_GRADER_SOURCE, "swebench_pro_grader.py", "exec")

    def test_load_exports_prompt_and_hides_gold_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _write_fixture(root)
            row = SWEBenchProAdapter(cache_root=root / "cache").load(root)[0]

            self.assertEqual(row["id"], record["instance_id"])
            self.assertIn("Requirements:\n- Return the expected value.", row["prompt"])
            self.assertNotIn("GOLD SOURCE PATCH MUST NOT LEAK", json.dumps(row))
            self.assertIn("rm -rf .git", row["runtime"]["setup"][0])
            binding = BenchBinding.from_dict(row["bench"])
            assert binding.grade is not None
            self.assertEqual(binding.grade.workspace_path, "/app")
            self.assertFalse(binding.grade.needs_transcript)
            self.assertEqual(len(binding.grade.stage), 3)

    def test_load_honors_current_bench_selection_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            adapter = SWEBenchProAdapter(cache_root=root / "cache")

            row = adapter.load(root, timeout_multiplier=0.5)[0]

            self.assertEqual(row["timeout_seconds"], 1800)
            with self.assertRaisesRegex(BenchFormatError, "has no variants"):
                adapter.load(root, variant="unsupported")

    def test_fixture_repair_only_stages_non_executable_test_data(self) -> None:
        fixture = (
            "diff --git a/internal/ext/testdata/sorted.json b/internal/ext/testdata/sorted.json\n"
            "new file mode 100644\n--- /dev/null\n"
            "+++ b/internal/ext/testdata/sorted.json\n@@ -0,0 +1 @@\n"
            '+{"expected": true}\n'
        )
        excluded = (
            "diff --git a/internal/ext/export.go b/internal/ext/export.go\n"
            "--- a/internal/ext/export.go\n+++ b/internal/ext/export.go\n"
            "@@ -1 +1 @@\n-BASE\n+GOLD SOURCE PATCH MUST NOT LEAK\n"
            "diff --git a/testdata/run.py b/testdata/run.py\n"
            "new file mode 100644\n--- /dev/null\n+++ b/testdata/run.py\n"
            "@@ -0,0 +1 @@\n+GOLD EXECUTABLE MUST NOT LEAK\n"
            "diff --git a/testdata/link.json b/testdata/link.json\n"
            "new file mode 120000\n--- /dev/null\n+++ b/testdata/link.json\n"
            "@@ -0,0 +1 @@\n+../../production.json\n"
        )
        self.assertEqual(
            _test_fixture_patch(fixture + excluded),
            (fixture, ["internal/ext/testdata/sorted.json"]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record()
            record["patch"] = fixture + excluded
            _write_fixture(root, record)
            row = SWEBenchProAdapter(
                cache_root=root / "cache",
                repair_test_fixtures=True,
                exclude_agent_tests=True,
            ).load(root)[0]
            self.assertNotIn('{"expected": true}', row["prompt"])
            self.assertNotIn("GOLD SOURCE PATCH MUST NOT LEAK", json.dumps(row))
            spec = BenchBinding.from_dict(row["bench"]).grade
            assert spec is not None
            self.assertEqual(spec.stage[-1][1], ".roller-grade/test-fixtures.patch")
            self.assertEqual(Path(spec.stage[-1][0]).read_text(), fixture)
            self.assertTrue(spec.kwargs["exclude_agent_tests"])

    def test_grading_options_reach_the_registered_adapter_from_the_environment(
        self,
    ) -> None:
        options = {
            "SWEBENCH_PRO_REPAIR_TEST_FIXTURES": "1",
            "SWEBENCH_PRO_EXCLUDE_AGENT_TESTS": "true",
            "SWEBENCH_PRO_GO_PROXY": "https://goproxy.example,direct",
        }
        fixture = (
            "diff --git a/testdata/sorted.json b/testdata/sorted.json\n"
            "new file mode 100644\n--- /dev/null\n"
            "+++ b/testdata/sorted.json\n@@ -0,0 +1 @@\n"
            '+{"expected": true}\n'
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record()
            record["patch"] = fixture
            _write_fixture(root, record)
            cache = {"SWEBENCH_PRO_CACHE_DIR": str(root / "cache")}
            output = root / "tasks.jsonl"

            with patch.dict(os.environ, {**options, **cache}):
                with self.assertRaises(SystemExit) as exited:
                    main(
                        [
                            "bench",
                            "export",
                            "swebench-pro",
                            str(root),
                            "-o",
                            str(output),
                        ]
                    )
            self.assertEqual(exited.exception.code, 0)
            row = json.loads(output.read_text())
            binding = BenchBinding.from_dict(row["bench"])
            assert binding.grade is not None
            self.assertEqual(
                binding.metadata["evaluation_protocol"]["repair_test_fixtures"], True
            )
            self.assertEqual(
                binding.metadata["evaluation_protocol"]["exclude_agent_tests"], True
            )
            self.assertEqual(
                binding.metadata["go_proxy"], options["SWEBENCH_PRO_GO_PROXY"]
            )
            self.assertTrue(binding.grade.kwargs["exclude_agent_tests"])
            self.assertTrue(binding.grade.kwargs["prepare_go_dependencies"])
            self.assertEqual(
                binding.grade.kwargs["supplemental_test_fixtures"],
                ["testdata/sorted.json"],
            )
            self.assertIn(
                "GOPROXY=https://goproxy.example,direct", row["runtime"]["setup"][0]
            )

            unset = {name: "" for name in options}
            with patch.dict(os.environ, {**unset, **cache}):
                row = get_bench("swebench-pro").load(root)[0]
            binding = BenchBinding.from_dict(row["bench"])
            assert binding.grade is not None
            self.assertEqual(
                binding.metadata["evaluation_protocol"]["repair_test_fixtures"], False
            )
            self.assertIsNone(binding.metadata["go_proxy"])
            self.assertFalse(binding.grade.kwargs["exclude_agent_tests"])
            self.assertEqual(len(binding.grade.stage), 3)

            with patch.dict(
                os.environ, {**cache, "SWEBENCH_PRO_EXCLUDE_AGENT_TESTS": "maybe"}
            ):
                with self.assertRaisesRegex(
                    BenchFormatError,
                    "SWEBENCH_PRO_EXCLUDE_AGENT_TESTS must be a boolean",
                ):
                    get_bench("swebench-pro").load(root)

    def test_load_rejects_unknown_and_unsafe_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            adapter = SWEBenchProAdapter(cache_root=root / "cache")
            with self.assertRaisesRegex(
                BenchFormatError, "unknown SWE-Bench Pro task ids"
            ):
                adapter.load(root, task_ids=["missing"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root, _record("../escape"))
            with self.assertRaisesRegex(BenchFormatError, "unsafe instance_id"):
                SWEBenchProAdapter(cache_root=root / "cache").load(root)


class _FakeDockerRuntime:
    def __init__(self, harness_image: str, architectures=None) -> None:
        self.images = {harness_image: "sha256:harness"}
        self.architectures = architectures or {}
        self.dockerfile = ""

    def run(self, args, timeout=None):
        del timeout
        arguments = list(args)
        if arguments[:2] == ["image", "inspect"] and "{{.Architecture}}" in arguments:
            architecture = self.architectures.get(arguments[2], "amd64")
            return subprocess.CompletedProcess(arguments, 0, f"{architecture}\n", "")
        if arguments[:2] == ["image", "inspect"]:
            image_id = self.images.get(arguments[2])
            return subprocess.CompletedProcess(
                arguments, 0 if image_id else 1, f"{image_id}\n" if image_id else "", ""
            )
        if arguments[:1] == ["pull"]:
            self.images[arguments[1]] = "sha256:source"
            return subprocess.CompletedProcess(arguments, 0, "pulled\n", "")
        if arguments[:1] == ["build"]:
            self.dockerfile = (Path(arguments[-1]) / "Dockerfile").read_text()
            return subprocess.CompletedProcess(arguments, 0, "built\n", "")
        raise AssertionError(arguments)


class _RetryingFakeDockerRuntime:
    def __init__(self) -> None:
        self.image_id = None
        self.pull_attempts = 0

    def run(self, args, timeout=None):
        del timeout
        arguments = list(args)
        if arguments[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                arguments,
                0 if self.image_id else 1,
                f"{self.image_id}\n" if self.image_id else "",
                "",
            )
        if arguments[:1] == ["pull"]:
            self.pull_attempts += 1
            if self.pull_attempts == 3:
                self.image_id = "sha256:source"
                return subprocess.CompletedProcess(arguments, 0, "pulled\n", "")
            return subprocess.CompletedProcess(arguments, 1, "", "unexpected EOF")
        raise AssertionError(arguments)


class SWEBenchProImageTests(unittest.TestCase):
    @patch("roller.benches.swebench_pro.images.time.sleep")
    def test_image_pull_retries_transient_failures(self, sleep) -> None:
        runtime = _RetryingFakeDockerRuntime()

        self.assertEqual(
            _image_id(runtime, "registry.example/source:latest"),
            "sha256:source",
        )
        self.assertEqual(runtime.pull_attempts, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [15, 30])

    def test_wrap_openclaw_builds_instance_image(self) -> None:
        suffix = uuid.uuid4().hex
        source = f"registry.example/source:{suffix}"
        image = f"registry.example/openclaw:{suffix}"
        runtime = _FakeDockerRuntime(image)
        harness = OpenClawHarness(OpenClawConfig(docker_image=image), 3, runtime)
        binding = BenchBinding(
            name="swebench-pro",
            metadata={"source_image": source, "base_commit": "a" * 40},
            grade=GradeSpec(code="def grade(workspace_path): return {}"),
        )
        _DERIVED_IMAGES.clear()

        semaphore = threading.Semaphore(1)
        wrapped = SWEBenchProAdapter().wrap_harness(
            harness, binding, grading_semaphore=semaphore
        )

        self.assertIsInstance(wrapped, GradedHarness)
        self.assertIs(wrapped.grading_semaphore, semaphore)
        self.assertEqual(wrapped.inner.config.workspace_container_path, "/app")
        self.assertIn(f"FROM {source}", runtime.dockerfile)
        self.assertIn(f"FROM {image} AS harness", runtime.dockerfile)
        self.assertIn(
            "COPY --from=harness /usr/local/bin/node /opt/roller-node/bin/node.real",
            runtime.dockerfile,
        )
        self.assertIn(
            "exec /opt/roller-glibc/ld-linux-x86-64.so.2 --library-path "
            "/opt/roller-glibc /opt/roller-node/bin/node.real",
            runtime.dockerfile,
        )
        self.assertNotIn(
            "COPY --from=harness /usr/local/bin/node /usr/local/bin/node",
            runtime.dockerfile,
        )
        self.assertNotIn(
            "cp -a /opt/roller-glibc/. /lib/x86_64-linux-gnu/",
            runtime.dockerfile,
        )
        self.assertIn(
            "exec /opt/roller-node/bin/node /opt/roller-openclaw/openclaw.mjs",
            runtime.dockerfile,
        )
        self.assertIn("OPENCLAW_NO_RESPAWN=1 exec", runtime.dockerfile)
        self.assertIn("then \\\n", runtime.dockerfile)
        self.assertIn("printf '%s\\n'", runtime.dockerfile)
        self.assertIn("ENTRYPOINT []", runtime.dockerfile)

    def test_wrap_keeps_gateway_port_capture_and_runtime_of_the_harness(self) -> None:
        suffix = uuid.uuid4().hex
        source = f"registry.example/source:{suffix}"
        image = f"registry.example/openclaw:{suffix}"
        runtime = _FakeDockerRuntime(image)
        config = OpenClawConfig(
            docker_image=image,
            model_config=Path("models.json"),
            wire_trace_mode="full",
            capture_calls=True,
            gateway_port_start=20000,
            container_user=None,
            preflight_image_digest="sha256:harness",
        )
        harness = OpenClawHarness(config, 3, runtime)
        binding = BenchBinding(
            name="swebench-pro",
            metadata={"source_image": source, "base_commit": "a" * 40},
            grade=GradeSpec(code="def grade(workspace_path): return {}"),
        )

        _DERIVED_IMAGES.clear()

        inner = get_bench("swebench-pro").wrap_harness(harness, binding).inner

        self.assertIsInstance(inner, OpenClawHarness)
        self.assertIs(inner.runtime, runtime)
        self.assertEqual(inner.worker_slot, 3)
        self.assertEqual(inner.gateway_port, harness.gateway_port)
        self.assertEqual(inner.gateway_port, 20030)
        self.assertTrue(inner.config.capture_calls)
        self.assertEqual(inner.config.wire_trace_mode, "full")
        self.assertIsNone(inner.capture)
        self.assertTrue(
            inner.config.docker_image.startswith("roller-swebench-pro-openclaw:")
        )
        self.assertEqual(inner.config.container_user, "0")
        # The stock image digest must not be compared with the derived image.
        self.assertIsNone(inner.config.preflight_image_digest)

    def test_derived_image_follows_the_image_architecture(self) -> None:
        suffix = uuid.uuid4().hex
        source = f"registry.example/source:{suffix}"
        image = f"registry.example/openclaw:{suffix}"
        runtime = _FakeDockerRuntime(image, {source: "arm64", image: "arm64"})
        _DERIVED_IMAGES.clear()

        ensure_harness_image(runtime, source, image, "openclaw")

        self.assertIn(
            "COPY --from=harness /lib/aarch64-linux-gnu /opt/roller-glibc",
            runtime.dockerfile,
        )
        self.assertIn(
            "exec /opt/roller-glibc/ld-linux-aarch64.so.1 --library-path",
            runtime.dockerfile,
        )
        self.assertNotIn("x86", runtime.dockerfile)

    def test_mismatched_or_unknown_architecture_fails_before_building(self) -> None:
        for architectures, harness_name, message in (
            ({"harness": "arm64"}, "hermes", "is amd64 but hermes image .* is arm64"),
            (
                {"harness": "s390x", "source": "s390x"},
                "openclaw",
                "supports amd64, arm64",
            ),
        ):
            with self.subTest(harness=harness_name):
                suffix = uuid.uuid4().hex
                source = f"registry.example/source:{suffix}"
                image = f"registry.example/harness:{suffix}"
                runtime = _FakeDockerRuntime(
                    image,
                    {
                        {"harness": image, "source": source}[name]: value
                        for name, value in architectures.items()
                    },
                )
                _DERIVED_IMAGES.clear()
                with self.assertRaisesRegex(RuntimeError, message):
                    ensure_harness_image(runtime, source, image, harness_name)
                self.assertEqual(runtime.dockerfile, "")

    def test_slow_pull_of_one_image_does_not_block_another(self) -> None:
        suffix = uuid.uuid4().hex
        slow = f"registry.example/slow:{suffix}"
        fast = f"registry.example/fast:{suffix}"
        pulling = threading.Event()
        release = threading.Event()

        class Runtime:
            def __init__(self) -> None:
                self.pulled: set[str] = set()

            def run(self, args, timeout=None):
                del timeout
                arguments = list(args)
                if arguments[:1] == ["pull"]:
                    if arguments[1] == slow:
                        pulling.set()
                        release.wait(30)
                    self.pulled.add(arguments[1])
                    return subprocess.CompletedProcess(arguments, 0, "", "")
                present = arguments[2] in self.pulled
                return subprocess.CompletedProcess(
                    arguments, 0 if present else 1, "sha256:id\n" if present else "", ""
                )

        runtime = Runtime()
        worker = threading.Thread(target=_image_id, args=(runtime, slow))
        worker.start()
        try:
            self.assertTrue(pulling.wait(30))
            finished = threading.Event()
            other = threading.Thread(
                target=lambda: (_image_id(runtime, fast), finished.set())
            )
            other.start()
            self.assertTrue(finished.wait(30))
            other.join()
        finally:
            release.set()
            worker.join()

    def test_wrap_non_openclaw_copies_repo_into_harness_image(self) -> None:
        suffix = uuid.uuid4().hex
        source = f"registry.example/source:{suffix}"
        image = f"registry.example/hermes:{suffix}"
        runtime = _FakeDockerRuntime(image)
        harness = HermesHarness(HermesConfig(docker_image=image), 2, runtime)
        binding = BenchBinding(
            name="swebench-pro",
            metadata={"source_image": source, "base_commit": "a" * 40},
            grade=GradeSpec(code="def grade(workspace_path): return {}"),
        )
        _DERIVED_IMAGES.clear()

        wrapped = SWEBenchProAdapter().wrap_harness(harness, binding)

        self.assertIsInstance(wrapped.inner, HermesHarness)
        self.assertEqual(wrapped.inner.config.workspace_container_path, "/app")
        self.assertIn(f"FROM {source} AS instance", runtime.dockerfile)
        self.assertIn(f"FROM {image}", runtime.dockerfile)
        self.assertIn("COPY --from=instance /app /app", runtime.dockerfile)


class SWEBenchProGraderTests(unittest.TestCase):
    def _git(self, workspace: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=workspace, capture_output=True, text=True, check=True
        ).stdout

    def test_image_precheck_failure_becomes_grade_error(self) -> None:
        for failure in (
            RuntimeError("could not pull Docker image"),
            subprocess.TimeoutExpired(["image", "inspect"], 60),
        ):
            with self.subTest(failure=type(failure).__name__):
                runtime = Mock()
                runtime.run.return_value = subprocess.CompletedProcess([], 0, "", "")
                inner = HermesHarness(HermesConfig(), 0, runtime)
                binding = BenchBinding(
                    "swebench-pro",
                    metadata={"source_image": "source:test", "base_commit": "a" * 40},
                    grade=GradeSpec("def grade(workspace_path): return {}"),
                )
                semaphore = threading.BoundedSemaphore(1)
                harness = SWEBenchProGradedHarness(
                    inner, binding, grading_semaphore=semaphore
                )
                session = SimpleNamespace(
                    container_started=True, container="agent-container"
                )
                request = RunRequest(0, Task("demo", ("fix it",), 0), "model", 30)
                with (
                    patch.object(harness, "_extract_patch", return_value=("", None)),
                    patch(
                        "roller.benches.swebench_pro.grader_runtime._image_id",
                        side_effect=failure,
                    ),
                ):
                    outcome = harness._grade(session, request, "ok")

                self.assertIsNotNone(outcome)
                self.assertEqual(outcome.error, str(failure))
                self.assertEqual(outcome.scores["error"], str(failure))
                runtime.run.assert_not_called()
                self.assertTrue(semaphore.acquire(blocking=False))
                semaphore.release()

    def test_grading_failures_preserve_agent_artifacts_and_cleanup(self) -> None:
        for failure_stage in ("precheck", "extraction", "startup", "unexpected"):
            with (
                self.subTest(stage=failure_stage),
                tempfile.TemporaryDirectory() as tmp,
            ):
                output_dir = Path(tmp)
                runtime = Mock()
                runtime.run.return_value = subprocess.CompletedProcess([], 0, "", "")
                inner = HermesHarness(HermesConfig(), 0, runtime)
                binding = BenchBinding(
                    "swebench-pro",
                    metadata={"source_image": "source:test", "base_commit": "a" * 40},
                    grade=GradeSpec("def grade(workspace_path): return {}"),
                )
                harness = SWEBenchProGradedHarness(inner, binding)
                session = Mock(container_started=True, container="agent-container")
                trajectory = output_dir / "trace.json"

                def finish(status, error):
                    trajectory.write_text('{"saved": true}\n', encoding="utf-8")
                    runtime.run(["rm", "-f", session.container], timeout=30)
                    return HarnessResult(
                        status, 0.1, error, trajectory_path=str(trajectory)
                    )

                session.finish.side_effect = finish
                failure = RuntimeError(f"{failure_stage} failed")
                request = RunRequest(0, Task("demo", ("fix it",), 0), "model", 30)
                with (
                    patch.object(inner, "create_session", return_value=session),
                    patch.object(
                        harness, "_extract_patch", return_value=("", None)
                    ) as extract,
                    patch(
                        "roller.benches.swebench_pro.grader_runtime._image_id",
                        return_value="sha256:source",
                    ) as image,
                    patch.object(harness, "_grade", wraps=harness._grade) as grade,
                ):
                    if failure_stage == "precheck":
                        image.side_effect = failure
                    elif failure_stage == "extraction":
                        extract.side_effect = failure
                    elif failure_stage == "startup":

                        def run(args, timeout=None):
                            del timeout
                            if args[0] == "run":
                                raise failure
                            return subprocess.CompletedProcess(args, 0, "", "")

                        runtime.run.side_effect = run
                    else:
                        grade.side_effect = failure
                    result = harness.run(request, output_dir, 1)

                session.finish.assert_called_once_with("ok", None)
                self.assertEqual(result.status, "ok")
                self.assertIsNone(result.error)
                self.assertEqual(result.trajectory_path, str(trajectory))
                self.assertEqual(json.loads(trajectory.read_text()), {"saved": True})
                runtime.run.assert_any_call(["rm", "-f", "agent-container"], timeout=30)
                score = json.loads((output_dir / "score.json").read_text())
                self.assertEqual(score["error"], str(failure))
                self.assertEqual(result.metadata["bench"]["error"], str(failure))
                if failure_stage == "startup":
                    calls = [c.args[0] for c in runtime.run.call_args_list]
                    launch = next(c for c in calls if c[0] == "run")
                    grader_container = launch[launch.index("--name") + 1]
                    self.assertIn(["rm", "-f", grader_container], calls)
                    self.assertLess(
                        calls.index(["rm", "-f", grader_container]),
                        calls.index(["rm", "-f", "agent-container"]),
                    )

    def test_grader_extracts_committed_patch_and_restores_hidden_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._git(workspace, "init", "-q")
            self._git(workspace, "config", "user.name", "Roller Test")
            self._git(workspace, "config", "user.email", "roller@test.invalid")
            (workspace / "app.py").write_text("value = 'base'\n")
            tests = workspace / "tests"
            tests.mkdir()
            (tests / "hidden.txt").write_text("BASE\n")
            self._git(workspace, "add", ".")
            self._git(workspace, "commit", "-qm", "baseline")
            self._git(workspace, "update-ref", "refs/roller/baseline", "HEAD")
            with (workspace / ".git/info/exclude").open("a") as handle:
                handle.write(".roller-grade/\n.roller_outputs/\n")
            (tests / "hidden.txt").write_text("HIDDEN\n")
            hidden_patch = self._git(workspace, "diff", "--", "tests/hidden.txt")
            self._git(workspace, "checkout", "--", "tests/hidden.txt")
            (workspace / "app.py").write_text("value = 'model-fixed'\n")
            (tests / "hidden.txt").write_text("AGENT-CHEAT\n")
            self._git(workspace, "add", ".")
            self._git(workspace, "commit", "-qm", "agent commit")

            grade_dir = workspace / ".roller-grade"
            grade_dir.mkdir()
            (grade_dir / "test.patch").write_text(hidden_patch)
            (grade_dir / "run_script.sh").write_text(
                "#!/bin/bash\n"
                "if grep -q model-fixed app.py && grep -q HIDDEN tests/hidden.txt; then\n"
                " echo 'required_test PASSED'; echo 'regression_test PASSED'\n"
                "else echo 'required_test FAILED'; fi\n"
            )
            (grade_dir / "parser.py").write_text(
                "import json,pathlib,sys\n"
                "tests=[]\n"
                "for line in pathlib.Path(sys.argv[1]).read_text().splitlines():\n"
                " name,status=line.split(); tests.append({'name':name,'status':status})\n"
                "pathlib.Path(sys.argv[3]).write_text(json.dumps({'tests':tests}))\n"
            )
            namespace = {}
            exec(_GRADER_SOURCE, namespace)  # noqa: S102
            scores = namespace["roller_grade"](
                str(workspace),
                ["required_test"],
                ["regression_test"],
                ["tests/hidden.txt"],
                ["tests/hidden.txt"],
                30,
            )

            self.assertEqual(scores["overall_score"], 1.0)
            patch = (workspace / ".roller_outputs/model.patch").read_text()
            self.assertIn("model-fixed", patch)
            self.assertEqual((tests / "hidden.txt").read_text(), "HIDDEN\n")

    def test_fresh_grader_cleanup_failure_does_not_mask_score(self) -> None:
        class RecordingRuntime:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def run(self, args, timeout=None):
                del timeout
                arguments = list(args)
                self.calls.append(arguments)
                if arguments[:2] == ["rm", "-f"]:
                    raise RuntimeError("docker cleanup unavailable")
                stdout = (
                    "sha256:source\n" if arguments[:2] == ["image", "inspect"] else ""
                )
                return subprocess.CompletedProcess(arguments, 0, stdout, "")

        runtime = RecordingRuntime()
        inner = HermesHarness(
            HermesConfig(docker_image="registry.example/hermes:test"),
            1,
            runtime,
        )
        binding = BenchBinding(
            name="swebench-pro",
            metadata={
                "source_image": "registry.example/source:test",
                "base_commit": "a" * 40,
            },
            grade=GradeSpec(code="def roller_grade(workspace_path): return {}"),
        )
        harness = SWEBenchProGradedHarness(inner, binding)
        session = SimpleNamespace(container_started=True, container="agent-container")
        request = SimpleNamespace(task=SimpleNamespace(id="instance-test"))
        expected = GradeOutcome({"overall_score": 1.0, "resolved": 1}, 0.1)

        with patch.object(
            harness, "_extract_patch", return_value=("diff --git a/a b/a\n", None)
        ):
            with patch(
                "roller.benches.swebench_pro.ContainerGrader.grade",
                return_value=expected,
            ) as grade:
                outcome = harness._grade(session, request, "ok")

        self.assertIs(outcome, expected)
        run_call = next(call for call in runtime.calls if call[:2] == ["run", "-d"])
        grader_container = run_call[run_call.index("--name") + 1]
        self.assertIn("ai.roller.role=swebench-pro-grader", run_call)
        self.assertEqual(
            run_call[-3:], ["registry.example/source:test", "-c", "tail -f /dev/null"]
        )
        grade.assert_called_once()
        self.assertEqual(grade.call_args.args[1], grader_container)
        self.assertIn(["rm", "-f", grader_container], runtime.calls)

    def test_dropped_binary_changes_are_recorded_in_the_score(self) -> None:
        class Runtime:
            def __init__(self) -> None:
                self.patch = None

            def run(self, args, timeout=None):
                del timeout
                arguments = list(args)
                if arguments[0] == "cp" and arguments[2].endswith(":/tmp/model.patch"):
                    self.patch = Path(arguments[1]).read_text()
                stdout = (
                    "sha256:source\n" if arguments[:2] == ["image", "inspect"] else ""
                )
                return subprocess.CompletedProcess(arguments, 0, stdout, "")

        text = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n"
        binary = (
            "diff --git a/assets/logo.png b/assets/logo.png\n"
            "index 1111111..2222222 100644\nGIT binary patch\nliteral 1\nIcmZPo000310RR91\n\n"
            "diff --git a/bin/tool b/bin/tool\nnew file mode 100755\n"
            "Binary files /dev/null and b/bin/tool differ\n"
        )
        runtime = Runtime()
        binding = BenchBinding(
            name="swebench-pro",
            metadata={
                "source_image": "registry.example/source:test",
                "base_commit": "a" * 40,
            },
            grade=GradeSpec(code="def roller_grade(workspace_path): return {}"),
        )
        harness = SWEBenchProGradedHarness(
            HermesHarness(HermesConfig(), 0, runtime), binding
        )
        session = SimpleNamespace(container_started=True, container="agent-container")
        request = SimpleNamespace(task=SimpleNamespace(id="instance-test"))

        with (
            patch.object(harness, "_extract_patch", return_value=(text + binary, None)),
            patch(
                "roller.benches.swebench_pro.ContainerGrader.grade",
                return_value=GradeOutcome(
                    {"overall_score": 0.0, "dropped_binary_files": []}, 0.1
                ),
            ),
            self.assertLogs(
                "roller.benches.swebench_pro.grader_runtime", "WARNING"
            ) as logs,
        ):
            outcome = harness._grade(session, request, "ok")

        assert outcome is not None
        self.assertEqual(runtime.patch, text)
        self.assertEqual(
            outcome.scores["dropped_binary_files"], ["assets/logo.png", "bin/tool"]
        )
        self.assertIn("assets/logo.png, bin/tool", logs.output[0])

    def test_failed_artifact_copy_keeps_the_computed_grade(self) -> None:
        class Runtime:
            def run(self, args, timeout=None):
                del timeout
                arguments = list(args)
                stdout = (
                    "sha256:source\n" if arguments[:2] == ["image", "inspect"] else ""
                )
                return subprocess.CompletedProcess(arguments, 0, stdout, "")

        binding = BenchBinding(
            name="swebench-pro",
            metadata={
                "source_image": "registry.example/source:test",
                "base_commit": "a" * 40,
            },
            grade=GradeSpec(code="def roller_grade(workspace_path): return {}"),
        )
        harness = SWEBenchProGradedHarness(
            HermesHarness(HermesConfig(), 0, Runtime()), binding
        )
        session = SimpleNamespace(container_started=True, container="agent-container")
        request = SimpleNamespace(task=SimpleNamespace(id="instance-test"))

        with (
            patch.object(harness, "_extract_patch", return_value=("", None)),
            patch(
                "roller.benches.swebench_pro.ContainerGrader.grade",
                return_value=GradeOutcome({"overall_score": 1.0, "resolved": 1}, 0.1),
            ),
            patch.object(
                harness,
                "_copy_grade_outputs",
                side_effect=subprocess.TimeoutExpired("docker cp", 120),
            ),
            self.assertLogs(
                "roller.benches.swebench_pro.grader_runtime", "WARNING"
            ) as logs,
        ):
            outcome = harness._grade(session, request, "ok")

        assert outcome is not None
        self.assertEqual(outcome.scores["overall_score"], 1.0)
        self.assertIsNone(outcome.error)
        self.assertIn("grade outputs were not copied", logs.output[0])

    def test_opt_in_grading_preserves_patch_and_excludes_test_sources(self) -> None:
        for exclude_tests, with_fixtures, expected in [
            (False, True, 0.0),
            (True, False, 0.0),
            (True, True, 1.0),
        ]:
            with self.subTest(exclude_tests=exclude_tests, with_fixtures=with_fixtures):
                with tempfile.TemporaryDirectory() as temporary:
                    workspace = Path(temporary)
                    self._git(workspace, "init", "-q")
                    self._git(workspace, "config", "user.name", "roller")
                    self._git(workspace, "config", "user.email", "roller@test.invalid")
                    (workspace / "app.go").write_text("BASE\n")
                    self._git(workspace, "add", ".")
                    self._git(workspace, "commit", "-qm", "baseline")
                    self._git(workspace, "update-ref", "refs/roller/baseline", "HEAD")
                    with (workspace / ".git/info/exclude").open("a") as handle:
                        handle.write(".roller-grade/\n.roller_outputs/\n")
                    (workspace / "hidden_test.go").write_text("OFFICIAL\n")
                    self._git(workspace, "add", "-N", ".")
                    hidden_patch = self._git(workspace, "diff")
                    self._git(workspace, "reset", "--hard", "HEAD")
                    (workspace / "testdata").mkdir()
                    (workspace / "testdata/expected.json").write_text('{"ok":true}\n')
                    self._git(workspace, "add", "-N", ".")
                    fixture_patch = self._git(workspace, "diff")
                    self._git(workspace, "reset", "--hard", "HEAD")
                    (workspace / "app.go").write_text("MODEL_FIXED\n")
                    (workspace / "self_test.go").write_text("DUPLICATE TestHosts\n")
                    (workspace / "tests").mkdir()
                    (workspace / "tests/helper.js").write_text("MODEL TEST\n")
                    grade_dir = workspace / ".roller-grade"
                    grade_dir.mkdir()
                    (grade_dir / "test.patch").write_text(hidden_patch)
                    (grade_dir / "test-fixtures.patch").write_text(fixture_patch)
                    (grade_dir / "run_script.sh").write_text(
                        "#!/bin/bash\n"
                        "if grep -q MODEL_FIXED app.go && test -f hidden_test.go "
                        "&& test -f testdata/expected.json && ! test -f self_test.go "
                        "&& ! test -f tests/helper.js; then\n"
                        " echo 'required PASSED'\nelse echo 'required FAILED'; fi\n"
                    )
                    (grade_dir / "parser.py").write_text(
                        "import json,pathlib,sys\n"
                        "name,status=pathlib.Path(sys.argv[1]).read_text().split()\n"
                        "pathlib.Path(sys.argv[3]).write_text(json.dumps("
                        "{'tests':[{'name':name,'status':status}]}))\n"
                    )
                    namespace = {}
                    exec(_GRADER_SOURCE, namespace)  # noqa: S102
                    result = namespace["roller_grade"](
                        str(workspace),
                        ["required"],
                        [],
                        ["hidden_test.go"],
                        ["hidden_test.go"],
                        30,
                        exclude_agent_tests=exclude_tests,
                        supplemental_test_fixtures=(
                            ["testdata/expected.json"] if with_fixtures else []
                        ),
                    )
                    self.assertEqual(result["overall_score"], expected)
                    original = (workspace / ".roller_outputs/model.patch").read_text()
                    graded = (workspace / ".roller_outputs/graded.patch").read_text()
                    self.assertIn("MODEL_FIXED", graded)
                    self.assertIn("self_test.go", original)
                    self.assertIn("tests/helper.js", original)
                    self.assertEqual("self_test.go" not in graded, exclude_tests)
                    self.assertEqual("tests/helper.js" not in graded, exclude_tests)
                    self.assertNotIn('{"ok":true}', original)

    def test_go_dependency_repair_detects_stdout_and_preserves_first_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            self._git(workspace, "init", "-q")
            self._git(workspace, "config", "user.name", "roller")
            self._git(workspace, "config", "user.email", "roller@test.invalid")
            (workspace / "go.mod").write_text("module example.test/repaired\n")
            (workspace / "hidden_test.go").write_text("BASE\n")
            self._git(workspace, "add", ".")
            self._git(workspace, "commit", "-qm", "baseline")
            self._git(workspace, "update-ref", "refs/roller/baseline", "HEAD")
            with (workspace / ".git/info/exclude").open("a") as handle:
                handle.write(".roller-grade/\n.roller_outputs/\n")
            (workspace / "hidden_test.go").write_text("OFFICIAL\n")
            hidden_patch = self._git(workspace, "diff", "--", "hidden_test.go")
            self._git(workspace, "reset", "--hard", "HEAD")
            grade_dir = workspace / ".roller-grade"
            grade_dir.mkdir()
            (grade_dir / "test.patch").write_text(hidden_patch)
            (grade_dir / "run_script.sh").write_text(
                "#!/bin/bash\n"
                "if test -f go.sum; then echo 'required PASSED'; "
                "else echo 'missing go.sum entry'; fi\n"
            )
            (grade_dir / "parser.py").write_text(
                "import json,pathlib,sys\n"
                "text=pathlib.Path(sys.argv[1]).read_text()\n"
                "status='PASSED' if 'required PASSED' in text else 'FAILED'\n"
                "pathlib.Path(sys.argv[3]).write_text(json.dumps("
                "{'tests':[{'name':'required','status':status}]}))\n"
            )
            tool_dir = root / "bin"
            tool_dir.mkdir()
            fake_go = tool_dir / "go"
            fake_go.write_text("#!/bin/sh\nprintf 'module-sum\\n' > go.sum\n")
            fake_go.chmod(0o755)
            namespace = {}
            exec(_GRADER_SOURCE, namespace)  # noqa: S102
            with patch.dict(os.environ, {"PATH": f"{tool_dir}:{os.environ['PATH']}"}):
                result = namespace["roller_grade"](
                    str(workspace),
                    ["required"],
                    [],
                    ["hidden_test.go"],
                    ["hidden_test.go"],
                    30,
                    prepare_go_dependencies=True,
                )
            self.assertEqual(result["resolved"], 1)
            self.assertEqual(result["dependency_preparation"]["returncode"], 0)
            self.assertGreater(result["dependency_preparation"]["patch_bytes"], 0)
            self.assertIn(
                "missing go.sum entry",
                (
                    workspace / ".roller_outputs/before-dependency-stdout.log"
                ).read_text(),
            )


if __name__ == "__main__":
    unittest.main()
