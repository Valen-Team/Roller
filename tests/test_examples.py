from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import yaml
from jsonschema import Draft202012Validator

from roller.benches import BENCHES
from roller.benches.run_config import load_bench_run_plan
from roller.bundles import TaskBundleStore
from roller.harness_registry import HARNESS_NAMES
from roller.harnesses.openclaw import load_model_config
from roller.tasks import load_tasks


class ExampleFilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.examples_root = cls.project_root / "examples"

    def test_four_example_tasks_and_their_input_bundle_are_valid(self) -> None:
        tasks = load_tasks(self.examples_root / "tasks" / "tasks.jsonl")

        self.assertEqual(
            [task.id for task in tasks],
            [
                "01_text_only",
                "02_local_markdown",
                "03_user_memory_context",
                "04_multi_turn",
            ],
        )
        self.assertEqual([len(task.prompts) for task in tasks], [1, 1, 1, 2])
        self.assertEqual(len(tasks[0].inputs), 0)
        self.assertEqual(tasks[1].inputs[0].path, "inputs/product-brief.md")
        self.assertEqual(
            tasks[1].inputs[0].source,
            self.examples_root
            / "tasks/files/02_local_markdown/inputs/product-brief.md",
        )
        self.assertEqual(
            [item.role for item in tasks[2].context_files], ["user", "memory"]
        )
        self.assertEqual(tasks[1].output_dir, "outputs")

        with tempfile.TemporaryDirectory() as temporary:
            bundles = TaskBundleStore(Path(temporary)).prepare_all(tasks)
            markdown_bundle = bundles["02_local_markdown"]
            persona_bundle = bundles["03_user_memory_context"]
            self.assertTrue(
                (markdown_bundle.inputs_root / "inputs/product-brief.md").is_file()
            )
            self.assertTrue((persona_bundle.context_root / "USER.md").is_file())
            self.assertTrue((persona_bundle.context_root / "MEMORY.md").is_file())

    def test_custom_api_example_expands_environment_placeholders(self) -> None:
        config = load_model_config(
            self.examples_root / "config" / "openclaw" / "models.json",
            {
                "MY_PROXY_BASE_URL": "https://proxy.example/v1",
                "MY_PROXY_API_KEY": "secret",
            },
        )

        provider = config["providers"]["my-openai-proxy"]
        self.assertEqual(provider["baseUrl"], "https://proxy.example/v1")
        self.assertEqual(provider["apiKey"], "secret")
        self.assertEqual(provider["api"], "openai-responses")
        self.assertEqual(
            provider["models"],
            [{"id": "my-model", "name": "My model"}],
        )

        multimodal = load_model_config(
            self.examples_root / "config" / "openclaw" / "models-multimodal.json",
            {
                "MY_PROXY_BASE_URL": "https://proxy.example/v1",
                "MY_PROXY_API_KEY": "secret",
            },
        )
        multimodal_model = multimodal["providers"]["my-openai-proxy"]["models"][0]
        self.assertEqual(multimodal_model["id"], "my-multimodal-model")
        self.assertEqual(multimodal_model["input"], ["text", "image"])

    def test_native_harness_config_examples_are_safe_and_parseable(self) -> None:
        codex_path = self.examples_root / "config" / "codex" / "config.toml.example"
        codex_config = tomllib.loads(codex_path.read_text(encoding="utf-8"))
        self.assertEqual(codex_config, {"model_verbosity": "medium"})

        claude_path = (
            self.examples_root / "config" / "claude-code" / "settings.json.example"
        )
        claude_settings = json.loads(claude_path.read_text(encoding="utf-8"))
        self.assertFalse(claude_settings["includeCoAuthoredBy"])

        combined = codex_path.read_text(encoding="utf-8") + claude_path.read_text(
            encoding="utf-8"
        )
        self.assertNotIn("apiKey", combined)
        self.assertNotIn("API_KEY", combined)
        self.assertNotIn('"hooks"', combined)

    def test_openclaw_reasoning_example_uses_native_model_profiles(self) -> None:
        config = load_model_config(
            self.examples_root / "config/openclaw/models-reasoning.json",
            {
                "MY_PROXY_BASE_URL": "https://proxy.example/v1",
                "MY_PROXY_API_KEY": "secret",
                "MY_CHAT_BASE_URL": "https://chat.example/v1",
                "MY_CHAT_API_KEY": "secret",
            },
        )

        gpt = config["providers"]["openai"]
        self.assertEqual(gpt["api"], "openai-responses")
        self.assertEqual(gpt["models"][0]["id"], "gpt-5.6-sol")
        self.assertEqual(gpt["models"][0]["agentRuntime"], {"id": "openclaw"})

        chat = config["providers"]["my-chat-proxy"]["models"][0]
        self.assertEqual(
            chat["compat"]["supportedReasoningEfforts"],
            ["low", "medium", "high"],
        )
        self.assertNotIn("thinkingLevelMap", chat)

    def test_user_simulator_example_is_an_independent_dynamic_task(self) -> None:
        tasks = load_tasks(self.examples_root / "tasks" / "user-simulator.jsonl")
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.prompts, ())
        self.assertIsNotNone(task.user_simulator)
        assert task.user_simulator is not None
        self.assertEqual(task.user_simulator.harness, "openclaw")
        self.assertEqual(task.user_simulator.max_turns, 6)
        self.assertIn("HR", task.user_simulator.instruction)

    def test_environment_example_includes_model_and_search_variables(self) -> None:
        content = (self.examples_root / "env" / "api.env.example").read_text()
        self.assertIn("MY_PROXY_BASE_URL=https://api.example.com/v1", content)
        self.assertIn("MY_PROXY_API_KEY=replace-with-your-api-key", content)
        self.assertIn("BRAVE_API_KEY=replace-with-your-brave-api-key", content)

    def test_benchmark_config_examples_are_layerable_and_secret_free(self) -> None:
        config_root = self.examples_root / "benchmarks"
        environment = {
            "ROLLER_MODEL_ID": "example/model",
            "ROLLER_MODEL_BASE_URL": "https://model.example/v1",
            "ROLLER_MODEL_API_KEY": "model-secret",
            "ROLLER_BENCH_OUTPUT": "/tmp/roller-benchmark-results",
            "ROLLER_JUDGE_BASE_URL": "https://judge.example/v1",
            "ROLLER_JUDGE_API_KEY": "judge-secret",
            "ROLLER_JUDGE_MODEL": "example-judge",
            "ROLLER_SEARCH_API_KEY": "search-secret",
            "WILDCLAWBENCH_ROOT": "/tmp/wildclawbench",
            "PINCHBENCH_ROOT": "/tmp/pinchbench",
            "QWENCLAWBENCH_ROOT": "/tmp/qwenclawbench",
            "SKILLSBENCH_ROOT": "/tmp/skillsbench",
        }
        cases = (
            ("wildclawbench.yaml", (), "wildclawbench", "openclaw"),
            ("pinchbench.yaml", (), "pinchbench", "openclaw"),
            ("qwenclawbench.yaml", (), "qwenclawbench", "openclaw"),
            ("skillsbench.yaml", (), "skillsbench", "opencode"),
            (
                "skillsbench.yaml",
                ("skillsbench-no-skill.yaml",),
                "skillsbench",
                "opencode",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for primary, additional, benchmark, harness in cases:
                with self.subTest(primary=primary, additional=additional):
                    overlays = [str(config_root / primary)] + [
                        str(config_root / name) for name in additional
                    ]
                    plan = load_bench_run_plan(
                        config_root / "base.yaml",
                        overlays,
                        environ=environment,
                        temporary_root=Path(temporary),
                    )
                    argv = list(plan.collect_argv)
                    self.assertEqual(argv[argv.index("--bench") + 1], benchmark)
                    self.assertEqual(argv[argv.index("--harness") + 1], harness)
                    if benchmark == "qwenclawbench":
                        self.assertEqual(
                            argv[argv.index("--bench-runs-per-task") + 1], "3"
                        )
                        self.assertEqual(argv[argv.index("--samples") + 1], "1")
                    self.assertNotIn("model-secret", " ".join(argv))
                    self.assertNotIn("judge-secret", json.dumps(plan.provenance))
                    self.assertNotIn("search-secret", json.dumps(plan.provenance))

        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in config_root.glob("*.yaml")
        )
        self.assertNotIn("sk-", combined)
        self.assertNotIn("joyrouter", combined.lower())

    def test_benchmark_run_schema_tracks_registered_harnesses(self) -> None:
        schema = json.loads(
            (self.project_root / "schemas" / "bench-run.schema.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            "roller.bench-run/v1",
        )
        self.assertEqual(
            set(schema["properties"]["harness"]["properties"]["id"]["enum"]),
            set(HARNESS_NAMES),
        )

    def _task_schema(self) -> dict:
        return json.loads(
            (self.project_root / "schemas" / "task.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_task_schema_tracks_registered_harnesses(self) -> None:
        schema = self._task_schema()
        self.assertEqual(schema["$defs"]["harness"]["enum"], list(HARNESS_NAMES))

        def harness_enums(node, path=""):
            if isinstance(node, dict):
                if path != "/$defs/harness" and "openclaw" in node.get("enum", ()):
                    yield path
                for key, value in node.items():
                    yield from harness_enums(value, f"{path}/{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    yield from harness_enums(value, f"{path}/{index}")

        # Every Harness field refers to the shared definition.
        self.assertEqual(list(harness_enums(schema)), [])

    def test_task_schema_accepts_the_tasks_the_loader_accepts(self) -> None:
        validator = Draft202012Validator(self._task_schema())
        rows = [
            json.loads(line)
            for name in ("tasks.jsonl", "user-simulator.jsonl")
            for line in (self.examples_root / "tasks" / name)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        rows += json.loads(
            (self.examples_root / "coverage" / "all_tasks.json").read_text(
                encoding="utf-8"
            )
        )["tasks"]
        base = {"schema_version": "roller.task/v1", "id": "t", "prompt": "p"}
        rows += [{**base, "harness": name} for name in HARNESS_NAMES]
        rows += [
            {
                **base,
                "files": {
                    "inputs": [
                        {"source": "source.txt", "target": "renamed.txt"},
                        {"path": "nested/config.yaml", "content": "a: 1\n"},
                    ],
                    "input_mode": "copy",
                },
            },
            {
                "schema_version": "roller.task/v1",
                "id": "sessions",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "second", "new_session": True},
                ],
            },
            {
                **base,
                "runtime": {
                    "image": "task:1",
                    "workspace_path": "/app",
                    "image_build": {
                        "base_image": "python:3.12",
                        "context": "build",
                        "dockerfile": "FROM python:3.12\n",
                        "harness": "pi",
                        "args": {"VERSION": 3},
                    },
                },
            },
        ]
        for row in rows:
            errors = [error.message for error in validator.iter_errors(row)]
            self.assertEqual(errors, [], row.get("id"))

        rejected = [
            {**base, "harness": "unknown"},
            {**base, "files": {"inputs": [{"source": "a.txt", "content": "x"}]}},
            {**base, "files": {"inputs": [{"content": "x"}]}},
            {**base, "files": {"input_mode": "link"}},
            {
                **base,
                "runtime": {
                    "image_build": {
                        "base_image": "b",
                        "context": "c",
                        "dockerfile": "d",
                        "harness": "pi",
                    }
                },
            },
            {**base, "runtime": {"workspace_path": "relative"}},
        ]
        for row in rejected:
            with self.subTest(row=row):
                self.assertFalse(validator.is_valid(row))

    def test_every_benchmark_example_names_a_registered_bench(self) -> None:
        named = set()
        for path in sorted((self.examples_root / "benchmarks").glob("*.yaml")):
            benchmark = yaml.safe_load(path.read_text(encoding="utf-8")).get(
                "benchmark", {}
            )
            if "id" not in benchmark:
                continue
            with self.subTest(config=path.name):
                self.assertIn(benchmark["id"], BENCHES)
                self.assertEqual(path.stem, benchmark["id"])
                script = self.project_root / "scripts" / f"prepare-{path.stem}.sh"
                self.assertTrue(script.is_file())
                self.assertTrue(script.stat().st_mode & 0o111)
            named.add(benchmark["id"])
        self.assertTrue(named)

    def test_benchmark_examples_ship_with_the_package(self) -> None:
        build = tomllib.loads(
            (self.project_root / "pyproject.toml").read_text(encoding="utf-8")
        )["tool"]["hatch"]["build"]["targets"]
        self.assertEqual(
            build["wheel"]["force-include"]["examples/benchmarks"],
            "roller/examples/benchmarks",
        )
        self.assertIn("/examples/benchmarks", build["sdist"]["include"])


if __name__ == "__main__":
    unittest.main()
