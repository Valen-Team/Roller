from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from roller.harness_registry import HARNESS_NAMES
from roller.models import RunRequest, Task, TaskBundle
from roller.run_lock import (
    LEGACY_MIGRATION_FILENAME,
    build_run_lock,
    complete_legacy_adoption,
    ensure_run_lock,
)


def _request(root: Path, digest: str = "bundle-a") -> RunRequest:
    bundle = TaskBundle(
        inputs_root=root,
        context_root=root,
        manifest_digest=digest,
        input_tree_digest="input-tree",
        input_manifest_digest="input-manifest",
        context_manifest_digest="context-manifest",
    )
    return RunRequest(
        0,
        Task(id="task", prompts=("hello",), source_index=0),
        "proxy/model",
        30,
        bundle,
    )


def _requests(
    root: Path,
    source_indices,
    models=("proxy/model",),
    samples: int = 1,
    timeout: int = 30,
) -> list[RunRequest]:
    """Requests in the order of ``build_requests``: task, then model, then sample."""

    requests: list[RunRequest] = []
    for source_index in source_indices:
        task = Task(
            id=f"task-{source_index}", prompts=("hello",), source_index=source_index
        )
        for model in models:
            for sample_index in range(samples):
                requests.append(
                    RunRequest(
                        len(requests),
                        task,
                        model,
                        timeout,
                        _request(root).bundle,
                        sample_index=sample_index,
                        sample_count=samples,
                    )
                )
    return requests


class RunLockGrowthTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "lock.json"

    def lock(self, source_indices, version="0.4.0", execution=None, **options):
        return build_run_lock(
            _requests(self.root, source_indices, **options),
            roller_version=version,
            execution={"thinking": "high"} if execution is None else execution,
            model_config=None,
        )

    def record(self, lock) -> dict:
        self.path.unlink(missing_ok=True)
        ensure_run_lock(self.path, lock, resume=False, existing_run=False)
        return json.loads(self.path.read_text())

    def resume(self, lock, **options):
        return ensure_run_lock(
            self.path, lock, resume=True, existing_run=True, **options
        )

    def assert_mismatch(self, lock, message: str) -> None:
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError) as caught:
            self.resume(lock)
        self.assertIn("lock mismatch", str(caught.exception))
        self.assertIn(message, str(caught.exception))
        self.assertEqual(self.path.read_bytes(), before)

    def test_more_tasks_and_models_grow_the_recorded_request_list(self) -> None:
        recorded = self.record(self.lock(range(2)))
        grown = self.lock(range(3), models=("proxy/model", "other/model"))
        # The second recorded request moves from position 1 to position 2.
        self.assertEqual(
            [
                (row["request_index"], row["task_id"], row["model"])
                for row in grown["requests"][:3]
            ],
            [
                (0, "task-0", "proxy/model"),
                (1, "task-0", "other/model"),
                (2, "task-1", "proxy/model"),
            ],
        )

        with self.assertLogs("roller.run_lock", "INFO") as logs:
            decision = self.resume(grown)

        self.assertTrue(decision.reuse_existing_samples)
        self.assertFalse(decision.created)
        self.assertEqual(
            logs.output,
            ["INFO:roller.run_lock:lock.json now lists 6 requests (2 before)"],
        )
        self.assertEqual(
            json.loads(self.path.read_text()),
            {**recorded, "requests": grown["requests"]},
        )
        saved = self.path.read_bytes(), self.path.stat().st_mtime_ns
        with self.assertNoLogs("roller.run_lock", "INFO"):
            self.resume(grown)
        self.assertEqual((self.path.read_bytes(), self.path.stat().st_mtime_ns), saved)
        # The grown list is now the recorded one.
        self.assert_mismatch(
            self.lock(range(2)),
            "4 recorded requests are not in the current command; "
            "first: task-0 | other/model (task index 0)",
        )

    def test_a_recorded_request_must_stay_in_the_command(self) -> None:
        self.record(self.lock(range(3)))
        self.assert_mismatch(
            self.lock(range(1, 4)),
            "1 recorded request is not in the current command; "
            "first: task-0 | proxy/model (task index 0)",
        )
        self.assert_mismatch(
            self.lock(range(1)),
            "2 recorded requests are not in the current command; "
            "first: task-1 | proxy/model (task index 1)",
        )
        self.assert_mismatch(
            self.lock(range(3), models=("other/model",)),
            "3 recorded requests are not in the current command; "
            "first: task-0 | proxy/model (task index 0)",
        )
        # A task that moved in the task list has another sample directory.
        moved = self.lock(range(3))
        for row in moved["requests"]:
            row["source_index"] += 1
        self.assert_mismatch(
            moved,
            "3 recorded requests are not in the current command; "
            "first: task-0 | proxy/model (task index 0)",
        )

    def test_the_same_requests_in_another_order_are_accepted(self) -> None:
        models = ("proxy/model", "other/model")
        self.record(self.lock(range(2), models=models))
        reordered = self.lock(range(2), models=models[::-1])

        self.assertTrue(self.resume(reordered).reuse_existing_samples)

        self.assertEqual(
            json.loads(self.path.read_text())["requests"], reordered["requests"]
        )

    def test_a_recorded_request_must_stay_equal(self) -> None:
        self.record(self.lock(range(2)))
        self.assert_mismatch(
            self.lock(range(4), timeout=60),
            "2 recorded requests differ from the current command; "
            "first: task-0 | proxy/model: timeout_seconds: recorded=30, current=60",
        )
        # --samples changes sample_count, and with it the sample directory names.
        self.assert_mismatch(
            self.lock(range(2), samples=2),
            "2 recorded requests differ from the current command; "
            "first: task-0 | proxy/model: sample_count: recorded=1, current=2",
        )
        changed = self.lock(range(3))
        changed["requests"][1]["task_bundle"]["manifest_digest"] = "bundle-b"
        self.assert_mismatch(
            changed,
            "1 recorded request differs from the current command; "
            "first: task-1 | proxy/model: task_bundle.manifest_digest: "
            'recorded="bundle-a", current="bundle-b"',
        )

        self.record(self.lock(range(1), samples=2))
        self.assert_mismatch(
            self.lock(range(1), samples=3),
            "2 recorded requests differ from the current command; "
            "first: task-0 | proxy/model | sample 1 of 2: "
            "sample_count: recorded=2, current=3",
        )

    def test_settings_are_still_compared_exactly_when_the_run_grows(self) -> None:
        self.record(self.lock(range(2)))
        self.assert_mismatch(
            self.lock(range(4), execution={"thinking": "low"}),
            '$.execution.thinking: recorded="high", current="low"',
        )

    def test_resumes_under_another_patch_release_of_the_minor_series(self) -> None:
        self.record(self.lock(range(2)))
        saved = self.path.read_bytes(), self.path.stat().st_mtime_ns
        with self.assertNoLogs("roller.run_lock", "INFO"):
            self.resume(self.lock(range(2)))
        for version in ("0.4.3", "0.4.10", "0.4.1.dev2", "0.4.0rc1", "0.4.2+local"):
            with (
                self.subTest(version=version),
                self.assertLogs("roller.run_lock", "INFO") as logs,
            ):
                decision = self.resume(self.lock(range(2), version=version))
                self.assertTrue(decision.reuse_existing_samples)
                self.assertEqual(
                    logs.output,
                    [
                        "INFO:roller.run_lock:run was created by Roller 0.4.0; "
                        f"resuming with Roller {version}"
                    ],
                )
                # The lock keeps the version that created the run.
                self.assertEqual(
                    (self.path.read_bytes(), self.path.stat().st_mtime_ns), saved
                )
        for version in ("0.5.0", "0.5.0rc1", "0.3.9", "0.40.0", "1.4.0"):
            with self.subTest(version=version):
                self.assert_mismatch(
                    self.lock(range(2), version=version),
                    f"the run was created by Roller 0.4.0 and this is Roller {version}; "
                    "a run resumes only within its minor series (0.4.x)",
                )

    def test_dev_and_unparsed_versions(self) -> None:
        self.record(self.lock(range(1), version="0.5.0.dev1"))
        self.resume(self.lock(range(1), version="0.5.0"))
        self.resume(self.lock(range(1), version="0.5.2rc1"))
        self.assert_mismatch(
            self.lock(range(1), version="0.6.0.dev1"),
            "a run resumes only within its minor series (0.5.x)",
        )
        # A version without major.minor only matches itself.
        self.record(self.lock(range(1), version="local"))
        self.resume(self.lock(range(1), version="local"))
        self.assert_mismatch(
            self.lock(range(1), version="0.4.0"),
            '$.roller_version: recorded="local", current="0.4.0"',
        )

    def test_a_grown_run_under_a_new_patch_release_keeps_its_version(self) -> None:
        self.record(self.lock(range(1)))
        with self.assertLogs("roller.run_lock", "INFO") as logs:
            self.resume(self.lock(range(2), version="0.4.3"))
        self.assertEqual(len(logs.output), 2)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["roller_version"], "0.4.0")
        self.assertEqual(len(saved["requests"]), 2)

    def test_task_images_grow_with_the_requests(self) -> None:
        def execution(**digests):
            return {
                "thinking": "high",
                "task_images": {
                    task: {"image": f"{task}:1", "digest": digest}
                    for task, digest in digests.items()
                },
            }

        self.record(self.lock(range(1), execution=execution(**{"task-0": "sha-0"})))
        grown = self.lock(
            range(2), execution=execution(**{"task-0": "sha-0", "task-1": "sha-1"})
        )
        self.assert_mismatch(
            self.lock(
                range(2),
                execution=execution(**{"task-0": "rebuilt", "task-1": "sha-1"}),
            ),
            '$.execution.task_images.task-0.digest: recorded="sha-0", current="rebuilt"',
        )
        self.assert_mismatch(
            self.lock(range(2), execution=execution(**{"task-1": "sha-1"})),
            "$.execution.task_images.task-0: missing from current lock",
        )

        self.resume(grown)

        self.assertEqual(json.loads(self.path.read_text()), grown)

    def test_growth_keeps_the_recorded_encoding_of_the_settings(self) -> None:
        def execution(port: int, **claude):
            return {
                "openclaw": {"network": "host", "gateway_port_start": port},
                "claude-code": claude,
            }

        current = self.lock(range(2), execution=execution(18829, config_sha256="new"))
        historical = self.lock(
            range(2), execution=execution(18829, settings_fingerprint="old")
        )
        recorded = self.lock(
            range(1), execution=execution(18789, settings_fingerprint="old")
        )
        recorded["execution"]["claude-code"]["external_thinking"] = "off"
        self.path.write_text(json.dumps(recorded), encoding="utf-8")
        self.assert_mismatch(current, "$.execution.claude-code.config_sha256")

        self.resume(current, compatible_expected=[historical])

        self.assertEqual(
            json.loads(self.path.read_text()),
            {**recorded, "requests": current["requests"]},
        )


class RunLockTests(unittest.TestCase):
    def test_disabled_retired_options_are_compared_without_rewriting_old_lock(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            expected = {
                "execution": {name: {"thinking": "high"} for name in HARNESS_NAMES}
            }
            recorded = deepcopy(expected)
            for settings in recorded["execution"].values():
                settings["external_thinking"] = "off"
            recorded["execution"]["claude-code"].update(
                signature_recovery="off",
                signature_recovery_attempts=3,
                signature_recovery_max_tokens=12000,
            )
            path.write_text(json.dumps(recorded))
            before = path.read_bytes(), path.stat().st_mtime_ns
            expected_before = deepcopy(expected)

            decision = ensure_run_lock(path, expected, resume=True, existing_run=True)

            self.assertTrue(decision.reuse_existing_samples)
            self.assertFalse(decision.created)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
            self.assertEqual(expected, expected_before)

    def test_retired_compatibility_does_not_hide_active_unknown_or_other_changes(
        self,
    ) -> None:
        cases = (
            [
                ("codex", {"external_thinking": mode})
                for mode in ("external", "external_always", "unknown", None, False)
            ]
            + [
                ("claude-code", {"signature_recovery": mode})
                for mode in ("fill", "prefer", "unknown", None, False)
            ]
            + [
                ("claude-code", {"signature_recovery_attempts": 1}),
                ("claude-code", {"signature_recovery_max_tokens": 8192}),
                ("codex", {"external_thinking": "off", "thinking": "low"}),
                ("unknown-harness", {"external_thinking": "off"}),
            ]
        )
        for name, changes in cases:
            with (
                self.subTest(name=name, changes=changes),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = Path(temporary) / "lock.json"
                expected = {"execution": {name: {"thinking": "high"}}}
                recorded = deepcopy(expected)
                recorded["execution"][name].update(changes)
                path.write_text(json.dumps(recorded))
                before = path.read_bytes()
                with self.assertRaisesRegex(FileExistsError, "lock mismatch"):
                    ensure_run_lock(path, expected, resume=True, existing_run=True)
                self.assertEqual(path.read_bytes(), before)

    def test_builds_secret_safe_structural_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = build_run_lock(
                [_request(Path(temporary))],
                roller_version="1.2.3",
                execution={"thinking": "high"},
                model_config={
                    "providers": {
                        "proxy": {
                            "baseUrl": (
                                "https://user:password@example.test/v1"
                                "?api-version=2026-08-01&token=secret#fragment"
                            ),
                            "apiKey": "secret-key",
                        }
                    }
                },
            )

        self.assertEqual(
            lock["requests"][0]["task_bundle"]["manifest_digest"], "bundle-a"
        )
        self.assertEqual(lock["requests"][0]["model_label"], "proxy/model")
        provider = lock["model_config"]["providers"]["proxy"]
        self.assertEqual(
            provider["baseUrl"],
            "https://example.test/v1?api-version=2026-08-01&token=%3Credacted%3E",
        )
        self.assertEqual(provider["apiKey"], "<redacted>")
        self.assertNotIn("secret-key", json.dumps(lock))
        self.assertNotIn("password", json.dumps(lock))
        self.assertNotIn("token=secret", json.dumps(lock))

    def test_model_label_is_locked_separately_from_served_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary))
            request = RunRequest(
                request.index,
                request.task,
                request.model,
                request.timeout_seconds,
                request.bundle,
                model_label="display-model",
            )
            lock = build_run_lock(
                [request], roller_version="1.2.3", execution={}, model_config=None
            )

        self.assertEqual(lock["requests"][0]["model"], "proxy/model")
        self.assertEqual(lock["requests"][0]["model_label"], "display-model")

    def test_reuses_only_an_equal_existing_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "lock.json"
            lock = build_run_lock(
                [_request(root)],
                roller_version="1.2.3",
                execution={"thinking": "high"},
                model_config=None,
            )
            created = ensure_run_lock(path, lock, resume=False, existing_run=False)
            reused = ensure_run_lock(path, lock, resume=True, existing_run=True)

            self.assertTrue(created.created)
            self.assertFalse(created.reuse_existing_samples)
            self.assertFalse(reused.created)
            self.assertTrue(reused.reuse_existing_samples)

            changed = {**lock, "execution": {"thinking": "low"}}
            with self.assertRaisesRegex(FileExistsError, r"execution.thinking"):
                ensure_run_lock(path, changed, resume=True, existing_run=True)

    def test_accepts_an_explicit_historical_lock_without_rewriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            expected = {
                "schema_version": "roller.run-lock/v1",
                "execution": {"claude-code": {"config_sha256": "new-sha256"}},
            }
            historical = {
                "schema_version": "roller.run-lock/v1",
                "execution": {"claude-code": {"settings_fingerprint": "old-blake2b"}},
            }
            path.write_text(json.dumps(historical), encoding="utf-8")
            before = path.read_bytes(), path.stat().st_mtime_ns

            decision = ensure_run_lock(
                path,
                expected,
                resume=True,
                existing_run=True,
                compatible_expected=[historical],
            )

            self.assertTrue(decision.reuse_existing_samples)
            self.assertFalse(decision.created)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_adopts_legacy_run_without_reusing_unlocked_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "info.json").write_text("{}", encoding="utf-8")
            lock = build_run_lock(
                [_request(root)],
                roller_version="1.2.3",
                execution={},
                model_config=None,
            )

            decision = ensure_run_lock(
                root / "lock.json", lock, resume=True, existing_run=True
            )

            self.assertTrue(decision.legacy_run)
            self.assertFalse(decision.reuse_existing_samples)
            self.assertTrue((root / "lock.json").is_file())
            self.assertTrue((root / LEGACY_MIGRATION_FILENAME).is_file())

            interrupted = ensure_run_lock(
                root / "lock.json", lock, resume=True, existing_run=True
            )
            self.assertTrue(interrupted.legacy_run)
            self.assertFalse(interrupted.reuse_existing_samples)

            complete_legacy_adoption(root / "lock.json")
            trusted = ensure_run_lock(
                root / "lock.json", lock, resume=True, existing_run=True
            )
            self.assertFalse(trusted.legacy_run)
            self.assertTrue(trusted.reuse_existing_samples)

    def test_refuses_to_overwrite_invalid_existing_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            path.write_text("not json", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                ensure_run_lock(
                    path,
                    {"schema_version": "roller.run-lock/v1"},
                    resume=True,
                    existing_run=True,
                )


if __name__ == "__main__":
    unittest.main()
