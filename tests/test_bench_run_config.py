from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from roller.benches import list_benches
from roller.benches.run_config import (
    BENCH_RUN_SCHEMA,
    COLLECT_ONLY_FLAGS,
    load_bench_run_plan,
    temporary_environment,
)
from roller.cli import _bench, _build_parser
from roller.harnesses import HARNESS_CLI_OPTIONS
from roller.option_specs import BENCH_STRUCTURED_FLAGS, EXECUTION_OPTIONS


class BenchRunConfigTests(unittest.TestCase):
    def _write(self, root: Path, name: str, payload: dict) -> Path:
        path = root / name
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _base(self) -> dict:
        return {
            "schema_version": BENCH_RUN_SCHEMA,
            "model": {
                "id": "example/model",
                "base_url": "${MODEL_BASE_URL}",
                "api_key_env": "MODEL_API_KEY",
                "thinking": "high",
            },
            "harness": {"id": "codex"},
            "execution": {
                "parallel": 4,
                "grader_parallel": 2,
                "samples": 1,
                "wire_trace": "actf",
            },
        }

    def test_every_collect_option_is_mapped_or_excluded(self) -> None:
        subparsers = next(
            action
            for action in _build_parser()._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        collect = {
            action.option_strings[0]
            for action in subparsers.choices["collect"]._actions
            if action.option_strings
        } - {"-h"} | {"--help"}
        mapped = set(BENCH_STRUCTURED_FLAGS)
        mapped.update(opt.flag for opt in EXECUTION_OPTIONS)
        for options in HARNESS_CLI_OPTIONS.values():
            mapped.update(opt.flag for opt in options if opt.yaml_key is not None)
        for adapter in list_benches():
            mapped.update(
                opt.flag for opt in adapter.cli_options if opt.yaml_key is not None
            )

        self.assertEqual(collect - mapped - COLLECT_ONLY_FLAGS, set())
        self.assertEqual(mapped & COLLECT_ONLY_FLAGS, set())
        self.assertEqual((mapped | COLLECT_ONLY_FLAGS) - collect, set())

    def test_maps_capture_launch_and_config_set_options(self) -> None:
        cases = {
            "openclaw": (
                {"config_set": ["diagnostics.stuckSessionAbortMs=2700000", "a.b=1"]},
                {
                    "openclaw_config_set": [
                        ("diagnostics.stuckSessionAbortMs", "2700000"),
                        ("a.b", "1"),
                    ]
                },
            ),
            "codex": (
                {"config_set": ['model_verbosity="low"'], "image_detail": "high"},
                {
                    "codex_config_set": [("model_verbosity", '"low"')],
                    "codex_image_detail": "high",
                },
            ),
            "claude-code": (
                {"capture_calls": True, "unique_tool_ids": True},
                {"claude_capture_calls": True, "claude_unique_tool_ids": True},
            ),
            "pi": ({"input_modalities": "text"}, {"pi_input_modalities": "text"}),
            "hermes": ({"toolsets": ["native"]}, {"hermes_toolset": ["native"]}),
            "opencode": (
                {"native_tools": True, "deny_tools": ["webfetch"]},
                {"opencode_native_tools": True, "opencode_deny_tool": ["webfetch"]},
            ),
            "deepseek": (
                {"profile": "sdk", "permission_mode": "workspace-write"},
                {
                    "deepseek_profile": "sdk",
                    "deepseek_permission_mode": "workspace-write",
                },
            ),
        }
        for harness, (extra, expected) in cases.items():
            with (
                self.subTest(harness=harness),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                payload = self._base()
                payload["benchmark"] = {"id": "pinchbench", "root": "/bench/pinch"}
                payload["harness"] = {"id": harness, "extra": extra}
                payload["execution"] = {
                    "launch_interval": 2.5,
                    "retries": 2,
                    "retry_backoff": 30,
                    "capture_debug": "full",
                    "capture_calls": True,
                    "thinking_capabilities": "capabilities.json",
                }
                plan = load_bench_run_plan(
                    self._write(root, "run.yaml", payload),
                    environ={
                        "MODEL_BASE_URL": "https://api.example.com/v1",
                        "MODEL_API_KEY": "secret",
                    },
                    temporary_root=root,
                )
                args = plan.collect_args
                for attr, value in expected.items():
                    self.assertEqual(getattr(args, attr), value)
                self.assertEqual(args.launch_interval, 2.5)
                self.assertEqual(args.retries, 2)
                self.assertEqual(args.retry_backoff, 30.0)
                self.assertEqual(args.capture_debug, "full")
                self.assertEqual(args.thinking_capabilities, "capabilities.json")
                self.assertTrue(args.capture_calls)

    def test_layers_config_and_maps_common_model_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self._write(root, "base.yaml", self._base())
            extra = self._write(
                root,
                "bench.yaml",
                {
                    "benchmark": {
                        "id": "wildclawbench",
                        "root": "/bench/wildclaw",
                        "categories": ["03_Social_Interaction"],
                        "timeout_multiplier": 2,
                    },
                    "execution": {"parallel": 8, "run_name": "smoke"},
                },
            )
            plan = load_bench_run_plan(
                base,
                [str(extra)],
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )

        args = plan.collect_args
        self.assertEqual(args.parallel, 8)
        self.assertEqual(args.grader_parallel, 2)
        self.assertEqual(args.api_base, "https://model.example/v1")
        self.assertEqual(args.api_key_env, "MODEL_API_KEY")
        self.assertEqual(args.thinking, "high")
        self.assertEqual(args.bench_category, ["03_Social_Interaction"])
        self.assertNotIn("secret-value", repr(vars(args)))
        self.assertEqual(len(plan.provenance["sources"]), 2)
        effective = plan.provenance["effective"]
        self.assertEqual(effective["model"]["thinking"], "high")
        self.assertEqual(effective["model"]["base_url"], "https://model.example/v1")
        self.assertEqual(effective["model"]["api_key_env"], "MODEL_API_KEY")
        self.assertEqual(effective["harness"]["id"], "codex")
        self.assertEqual(effective["execution"]["parallel"], 8)

    def test_openclaw_uses_common_endpoint_to_generate_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "pinchbench", "root": "/bench/pinch"}
            config["harness"] = {
                "id": "openclaw",
                "extra": {"model_request_timeout": 900, "cache_trace": False},
            }
            config["model"]["params"] = {
                "contextWindow": 131072,
                "maxTokens": 32768,
            }
            path = self._write(root, "run.yaml", config)
            plan = load_bench_run_plan(
                path,
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )
            args = plan.collect_args
            model_config = Path(args.model_config)
            payload = json.loads(model_config.read_text(encoding="utf-8"))

            provider = payload["providers"]["example"]
            self.assertEqual(provider["baseUrl"], "https://model.example/v1")
            self.assertEqual(provider["apiKey"], "${MODEL_API_KEY}")
            self.assertEqual(provider["models"][0]["id"], "model")
            self.assertEqual(provider["models"][0]["maxTokens"], 32768)
            self.assertFalse(args.cache_trace)
            self.assertNotIn("secret-value", model_config.read_text(encoding="utf-8"))

    def test_openclaw_provider_keeps_upstream_model_id_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "pinchbench", "root": "/bench/pinch"}
            config["harness"] = {"id": "openclaw"}
            config["model"]["id"] = "organization/model"
            config["model"]["provider"] = "roller"
            path = self._write(root, "run.yaml", config)
            plan = load_bench_run_plan(
                path,
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )
            args = plan.collect_args
            model_config = Path(args.model_config)
            payload = json.loads(model_config.read_text(encoding="utf-8"))

            self.assertEqual(args.model, ["roller/organization/model"])
            self.assertEqual(
                payload["providers"]["roller"]["models"][0]["id"],
                "organization/model",
            )
            self.assertEqual(
                plan.provenance["effective"]["model"]["provider"], "roller"
            )

    def test_openclaw_defaults_provider_for_plain_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "pinchbench", "root": "/bench/pinch"}
            config["harness"] = {"id": "openclaw"}
            config["model"]["id"] = "plain-model"
            path = self._write(root, "run.yaml", config)
            plan = load_bench_run_plan(
                path,
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )
            args = plan.collect_args
            model_config = Path(args.model_config)
            payload = json.loads(model_config.read_text(encoding="utf-8"))

            self.assertEqual(args.model, ["roller/plain-model"])
            self.assertEqual(
                payload["providers"]["roller"]["models"][0]["id"], "plain-model"
            )

    def test_non_openclaw_keeps_model_id_when_provider_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "skillsbench", "root": "/bench/skills"}
            config["harness"] = {"id": "opencode"}
            config["model"]["id"] = "organization/model"
            config["model"]["provider"] = "roller"
            path = self._write(root, "run.yaml", config)
            plan = load_bench_run_plan(
                path,
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )
            args = plan.collect_args

            self.assertEqual(args.model, ["organization/model"])

    def test_qwenclawbench_maps_benchmark_specific_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {
                "id": "qwenclawbench",
                "root": "/bench/qwenclaw",
                "extra": {"runs_per_task": 3},
            }
            config["harness"] = {"id": "openclaw"}
            path = self._write(root, "run.yaml", config)
            plan = load_bench_run_plan(
                path,
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )

        self.assertEqual(plan.collect_args.bench_runs_per_task, 3)
        self.assertEqual(
            plan.provenance["effective"]["benchmark"]["extra"],
            {"runs_per_task": 3},
        )

    def test_rejects_benchmark_extra_for_the_wrong_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {
                "id": "pinchbench",
                "root": "/bench/pinch",
                "extra": {"runs_per_task": 3},
            }
            path = self._write(root, "run.yaml", config)

            with self.assertRaisesRegex(
                ValueError, "benchmark.extra has unknown field.*runs_per_task"
            ):
                load_bench_run_plan(
                    path,
                    environ={
                        "MODEL_BASE_URL": "https://model.example/v1",
                        "MODEL_API_KEY": "secret-value",
                    },
                    temporary_root=root,
                )

    def test_openclaw_custom_model_config_can_clear_common_connection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_config = self._base()
            base_config["benchmark"] = {"id": "pinchbench", "root": "/bench"}
            base_config["harness"] = {"id": "openclaw"}
            base = self._write(root, "base.yaml", base_config)
            model_config = root / "models.json"
            model_config.write_text("{}", encoding="utf-8")
            override = self._write(
                root,
                "custom.yaml",
                {
                    "model": {"base_url": None, "api_key_env": None},
                    "harness": {"extra": {"model_config": str(model_config)}},
                },
            )

            plan = load_bench_run_plan(
                base,
                [str(override)],
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )

        args = plan.collect_args
        self.assertEqual(args.model_config, str(model_config))
        self.assertIsNone(args.api_base)
        self.assertIsNone(args.api_key_env)

    def test_inline_extra_config_replaces_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {
                "id": "skillsbench",
                "root": "/bench/skills",
                "task_ids": ["first", "second"],
            }
            config["harness"] = {"id": "opencode"}
            path = self._write(root, "run.yaml", config)
            plan = load_bench_run_plan(
                path,
                [
                    json.dumps(
                        {
                            "benchmark": {
                                "variant": "no-skill",
                                "task_ids": ["only"],
                            },
                            "harness": {
                                "extra": {
                                    "max_steps": 100,
                                    "output_tokens": 32768,
                                    "vision": False,
                                }
                            },
                        }
                    )
                ],
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )

        args = plan.collect_args
        self.assertEqual(args.bench_task_id, ["only"])
        self.assertFalse(args.opencode_vision)
        self.assertEqual(args.opencode_max_steps, 100)
        self.assertEqual(args.opencode_output_tokens, 32768)

    def test_null_clears_harness_specific_settings_when_switching_harness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "wildclawbench", "root": "/bench"}
            config["harness"] = {
                "id": "openclaw",
                "image": "wildclawbench-openclaw:latest",
                "extra": {"max_tokens": 32768},
            }
            base = self._write(root, "base.yaml", config)
            switch = self._write(
                root,
                "codex.yaml",
                {
                    "harness": {
                        "id": "codex",
                        "image": "wildclawbench-codex:latest",
                        "extra": None,
                    }
                },
            )
            plan = load_bench_run_plan(
                base,
                [str(switch)],
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )

        args = plan.collect_args
        self.assertEqual(args.harness, "codex")
        self.assertEqual(args.docker_image, "wildclawbench-codex:latest")
        self.assertIsNone(args.openclaw_max_tokens)
        self.assertEqual(
            plan.provenance["effective"]["model"]["api_protocol"],
            "openai-responses",
        )

    def test_thinking_alias_is_canonical_in_arguments_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "pinchbench", "root": "/bench"}
            config["model"]["thinking"] = "none"
            path = self._write(root, "run.yaml", config)
            plan = load_bench_run_plan(
                path,
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "secret-value",
                },
                temporary_root=root,
            )

        self.assertEqual(plan.collect_args.thinking, "off")
        self.assertEqual(plan.provenance["effective"]["model"]["thinking"], "off")

    def test_environment_references_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "pinchbench", "root": "${ROOT}/pinch"}
            path = self._write(root, "run.yaml", config)
            with self.assertRaisesRegex(ValueError, "complete value"):
                load_bench_run_plan(
                    path,
                    environ={
                        "ROOT": "/bench",
                        "MODEL_BASE_URL": "https://model.example/v1",
                        "MODEL_API_KEY": "secret-value",
                    },
                    temporary_root=root,
                )

            config["benchmark"]["root"] = "${MISSING_ROOT}"
            path = self._write(root, "run.yaml", config)
            with self.assertRaisesRegex(ValueError, "MISSING_ROOT"):
                load_bench_run_plan(
                    path,
                    environ={
                        "MODEL_BASE_URL": "https://model.example/v1",
                        "MODEL_API_KEY": "secret-value",
                    },
                    temporary_root=root,
                )

    def test_unknown_harness_extra_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "skillsbench", "root": "/bench"}
            config["harness"] = {"id": "opencode", "extra": {"typo": 1}}
            path = self._write(root, "run.yaml", config)
            with self.assertRaisesRegex(ValueError, "harness.extra has unknown"):
                load_bench_run_plan(
                    path,
                    environ={
                        "MODEL_BASE_URL": "https://model.example/v1",
                        "MODEL_API_KEY": "secret-value",
                    },
                    temporary_root=root,
                )

    def test_protocol_must_match_non_openclaw_harness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "wildclawbench", "root": "/bench"}
            config["harness"] = {"id": "codex"}
            config["model"]["api_protocol"] = "openai-completions"
            path = self._write(root, "run.yaml", config)
            with self.assertRaisesRegex(ValueError, "openai-responses"):
                load_bench_run_plan(
                    path,
                    environ={
                        "MODEL_BASE_URL": "https://model.example/v1",
                        "MODEL_API_KEY": "secret-value",
                    },
                    temporary_root=root,
                )

    def test_common_model_connection_maps_to_every_non_openclaw_harness(self) -> None:
        protocols = {
            "codex": "openai-responses",
            "claude-code": "anthropic-messages",
            "hermes": "openai-completions",
            "opencode": "openai-completions",
            "deepseek": "openai-completions",
            "pi": "openai-completions",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for harness, protocol in protocols.items():
                with self.subTest(harness=harness):
                    config = self._base()
                    config["benchmark"] = {
                        "id": "wildclawbench",
                        "root": "/bench",
                    }
                    config["harness"] = {"id": harness}
                    config["model"]["api_protocol"] = protocol
                    path = self._write(root, f"{harness}.yaml", config)
                    plan = load_bench_run_plan(
                        path,
                        environ={
                            "MODEL_BASE_URL": "https://model.example/v1",
                            "MODEL_API_KEY": "secret-value",
                        },
                        temporary_root=root,
                    )

                    args = plan.collect_args
                    self.assertEqual(args.harness, harness)
                    self.assertEqual(args.api_base, "https://model.example/v1")
                    self.assertEqual(args.api_key_env, "MODEL_API_KEY")
                    self.assertEqual(
                        plan.provenance["effective"]["model"]["api_protocol"],
                        protocol,
                    )

    def test_benchmark_environment_is_scoped_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {
                "id": "pinchbench",
                "root": "/bench",
                "environment": {"PINCHBENCH_JUDGE_API_KEY": "${JUDGE_KEY}"},
            }
            path = self._write(root, "run.yaml", config)
            plan = load_bench_run_plan(
                path,
                environ={
                    "MODEL_BASE_URL": "https://model.example/v1",
                    "MODEL_API_KEY": "model-secret",
                    "JUDGE_KEY": "judge-secret",
                },
                temporary_root=root,
            )

        self.assertEqual(plan.environment["PINCHBENCH_JUDGE_API_KEY"], "judge-secret")
        self.assertNotIn("judge-secret", json.dumps(plan.provenance))
        old = os.environ.get("PINCHBENCH_JUDGE_API_KEY")
        with temporary_environment(plan.environment):
            self.assertEqual(os.environ["PINCHBENCH_JUDGE_API_KEY"], "judge-secret")
        self.assertEqual(os.environ.get("PINCHBENCH_JUDGE_API_KEY"), old)

    def test_endpoint_rejects_credentials_query_and_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, endpoint in enumerate(
                (
                    "https://user:password@model.example:8443/v1",
                    "https://model.example/v1?token=secret",
                    "https://model.example/v1#fragment",
                )
            ):
                with self.subTest(endpoint=endpoint):
                    config = self._base()
                    config["benchmark"] = {"id": "pinchbench", "root": "/bench"}
                    config["model"]["base_url"] = endpoint
                    path = self._write(root, f"run-{index}.yaml", config)
                    with self.assertRaisesRegex(
                        ValueError, "credentials|query or fragment"
                    ):
                        load_bench_run_plan(
                            path,
                            environ={"MODEL_API_KEY": "model-secret"},
                            temporary_root=root,
                        )

    def test_effective_endpoint_preserves_ipv6_literal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._base()
            config["benchmark"] = {"id": "pinchbench", "root": "/bench"}
            config["model"]["base_url"] = "http://[::1]:8080/v1"
            path = self._write(root, "run.yaml", config)
            plan = load_bench_run_plan(
                path,
                environ={"MODEL_API_KEY": "model-secret"},
                temporary_root=root,
            )

        self.assertEqual(
            plan.provenance["effective"]["model"]["base_url"],
            "http://[::1]:8080/v1",
        )


class BenchRunCliTests(unittest.TestCase):
    def _pinch_root(self, root: Path) -> Path:
        bench = root / "pinch"
        tasks = bench / "tasks"
        tasks.mkdir(parents=True)
        (bench / "assets").mkdir()
        (tasks / "manifest.yaml").write_text(
            "run_first: [task_sanity]\ncategories:\n  setup: [task_sanity]\n",
            encoding="utf-8",
        )
        (tasks / "task_sanity.md").write_text(
            """---
id: task_sanity
name: Sanity
category: setup
grading_type: automated
timeout_seconds: 45
---

## Prompt

Say hello.

## Automated Checks

```python
def grade(transcript, workspace_path):
    return {"ok": 1.0}
```
""",
            encoding="utf-8",
        )
        return bench

    def test_bench_run_dry_run_reaches_existing_collect_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = self._pinch_root(root)
            config = {
                "schema_version": BENCH_RUN_SCHEMA,
                "benchmark": {
                    "id": "pinchbench",
                    "root": str(bench),
                    "variant": "automated-only",
                    "task_ids": ["task_sanity"],
                },
                "model": {
                    "id": "example/model",
                    "base_url": "${MODEL_BASE_URL}",
                    "api_key_env": "MODEL_API_KEY",
                    "thinking": "off",
                },
                "harness": {"id": "openclaw"},
                "execution": {"parallel": 2, "samples": 1},
            }
            path = root / "run.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            args = _build_parser().parse_args(
                ["bench", "run", "--config", str(path), "--dry-run"]
            )
            output = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        "MODEL_BASE_URL": "https://model.example/v1",
                        "MODEL_API_KEY": "secret-value",
                    },
                    clear=False,
                ),
                contextlib.redirect_stdout(output),
            ):
                result = _bench(args)

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["bench"]["name"], "pinchbench")
        self.assertEqual(payload["tasks"], 1)
        self.assertEqual(payload["trajectories"], 1)
        self.assertEqual(payload["parallel"], 2)
        self.assertEqual(payload["plan"][0]["timeout_seconds"], 45)
        self.assertEqual(
            payload["bench_run_config"]["schema_version"], BENCH_RUN_SCHEMA
        )
        self.assertNotIn("secret-value", output.getvalue())


if __name__ == "__main__":
    unittest.main()
