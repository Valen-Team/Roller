# Contributing to Roller

Defects and feature requests go to [Issues](https://github.com/Valen-Team/Roller/issues). New Harnesses, new benches, fixes and documentation improvements are welcome as [pull requests](https://github.com/Valen-Team/Roller/pulls).

## Development setup

Roller needs Python 3.10 or later. Docker is needed only to run real rollouts, not for the test suite.

```bash
git clone https://github.com/Valen-Team/Roller.git
cd Roller
python -m pip install -e ".[test]"
```

With uv, `uv sync --extra test` creates the environment and `uv run pytest` runs the tests.

## Checks

CI runs the following on every pull request. Run them locally before you push:

```bash
python -m pytest        # unit tests, about two minutes
ruff check              # lint; install with: python -m pip install ruff
ruff format --check     # formatting; drop --check to apply it
```

`pre-commit install` sets up a hook that runs both ruff commands on every commit, using [.pre-commit-config.yaml](.pre-commit-config.yaml).

The tests run on Linux and macOS without Docker or network access. Tests that execute real containers are skipped unless their environment variables are set.

To check a change against a real Harness, build its image as described in the Harness guide and run a small task list with `roller collect`, or one of the acceptance scripts: `scripts/test_multimodal_e2e.py` for image inputs and `scripts/test_subagents_e2e.py` for subagents. Both need a model endpoint and take the API key from an environment variable.

## Where things live

| Area | Location |
| --- | --- |
| Command line & option specs | `src/roller/cli/` (`parser.py`, `collect.py`, `bench.py`, `harness_setup.py`, `preflight.py`, `commands.py`) and `src/roller/option_specs.py` |
| Execution backends | `src/roller/docker_runtime.py`, `src/roller/execution.py`, `src/roller/sandbox.py`, `src/roller/sandbox_e2b.py` |
| Wire proxy & capture | `src/roller/wire_trace.py`, `src/roller/wire_transforms.py`, `src/roller/capture.py`, `src/roller/messages_bridge.py`, `src/roller/responses_bridge.py` |
| Harness adapters | `src/roller/harnesses/` |
| Code that runs inside the containers | `src/roller/assets/` |
| Bench adapters | `src/roller/benches/` |
| ACTF trace converters | `src/roller/actf/` |
| Session import | `src/roller/importer.py` |
| SFT & XTuner export | `src/roller/sft.py`, `src/roller/compaction_sft.py`, `src/roller/compaction_parsers.py`, `src/roller/sft_dataset.py`, `src/roller/xtuner_dataset.py` |
| Hugging Face export & redaction | `src/roller/hf_dataset.py`, `src/roller/hf_redaction.py`, `src/roller/redaction.py` |
| Viewer | `src/roller/viewer/` |
| Documentation | `roller/` and its Chinese counterpart in the [website repository](https://github.com/Valen-Team/valen-team.github.io) |
| Machine-readable schemas | `schemas/` |

## Architecture layering

Modules under `src/roller/` follow a strict seven-layer dependency order (each layer may import only from itself and earlier layers):

1. **Data, lock, storage & registries**: `models`, `tasks`, `bundles`, `storage`, `artifacts`, `run_lock`, `redaction`, `option_specs`, `sample_metadata`, `thinking`, `harness_registry`, `usage`, `version`
2. **Execution backends**: `docker_runtime`, `execution`, `cleanup`, `image_build`, `sandbox`, `sandbox_e2b`, `sandbox_files`
3. **Wire proxy & protocol capture**: `wire_trace`, `wire_transforms`, `sandbox_wire`, `capture*`, `*_capture`, `messages_bridge`, `responses_bridge`, `token_ids`, `blobs`, `model_protocols`
4. **Harnesses**: `harnesses/*`, `stirrup_agent` (declaring their own `CLI_OPTIONS`, invoking ACTF/usage converters through `harness_registry`)
5. **Benches**: `benches/*` (translating YAML configs directly into structured `OptionSpec` namespaces)
6. **Traces, exports, import & Viewer**: `actf/*`, `sft*`, `compaction_*`, `xtuner_dataset`, `atif`, `hf_dataset`, `hf_redaction`, `integrity`, `importer`, `run_outputs`, `token_dataset`, `viewer/*`
7. **Orchestration & CLI**: `collector`, `sample_lifecycle`, `conversation`, `cli/*`, `roller/__init__`

## Adding a Bench adapter

An adapter translates: it reads a checked-out benchmark under `--bench-root` and returns ordinary `roller.task/v1` rows, each carrying its scorer in a `bench` block. The collect path is unchanged, and the adapter never starts a container.

1. Subclass `BenchAdapter` from `src/roller/benches/base.py` in a module under `src/roller/benches/`.
2. Add an instance to the tuple that builds `BENCHES` in `src/roller/benches/__init__.py`, and export the class from `__all__` there.
3. Test `load` against a small synthetic root, as the existing adapters do, so the suite needs neither Docker nor network access.

| Member | Needed | Role |
| --- | --- | --- |
| `name` | yes | The `--bench` value, and the `bench.name` of every row; `BENCHES` is keyed by it |
| `description` | no | The one line `roller bench list` prints |
| `variants`, `default_variant` | no | The `--bench-variant` values `roller bench list` prints; `load` rejects a variant it does not offer |
| `cli_options` | no | `OptionSpec`s the bench adds to `roller collect` and to bench run configs |
| `load(root, *, categories, task_ids, variant, timeout_multiplier)` | yes | Return the selected rows; the selection is the adapter's to apply |
| `categories(root)` | no | Defaults to the `bench.category` values of the loaded rows |
| `required_env(binding)` | no | Groups of alternative variable names, one of which must be set before any container starts; defaults to `bench.grade.env` |
| `aggregate(samples)` | no | The report summary; defaults to the mean score over every sample |

The smallest complete adapter:

```python
import json
import math

from roller.benches.base import BenchAdapter, BenchBinding, BenchFormatError, GradeSpec

# Runs inside the task's container after the agent exits, and returns a dict.
GRADER = """\
import os


def grade(transcript, workspace_path):
    answer = os.path.join(workspace_path, "answer.txt")
    return {"overall_score": 1.0 if os.path.exists(answer) else 0.0}
"""


class MyBenchAdapter(BenchAdapter):
    name = "mybench"
    description = "One JSON task per file under tasks/"

    def load(
        self,
        root,
        *,
        categories=None,
        task_ids=None,
        variant=None,
        timeout_multiplier=1.0,
    ):
        if variant is not None:
            raise BenchFormatError(f"{self.name}: this bench has no variants")
        rows = []
        for path in sorted(root.glob("tasks/*.json")):
            task = json.loads(path.read_text(encoding="utf-8"))
            if task_ids and task["id"] not in task_ids:
                continue
            if categories and task.get("category") not in categories:
                continue
            binding = BenchBinding(
                name=self.name,
                category=task.get("category"),
                grade=GradeSpec(code=GRADER, env_names=("MYBENCH_JUDGE_KEY",)),
            )
            rows.append(
                {
                    "schema_version": "roller.task/v1",
                    "id": task["id"],
                    "prompt": task["prompt"],
                    "timeout_seconds": max(
                        1, math.ceil(task["timeout_seconds"] * timeout_multiplier)
                    ),
                    "bench": binding.as_dict(),
                }
            )
        return rows
```

The grader runs before teardown, so it can read the workspace the agent just wrote. A grader that must run on the collector instead copies its inputs out with `GradeSpec.collector_files`, and an answer key is staged from the host with `GradeSpec.stage` rather than shipped in the task's own seed files.

## Adding a sandbox provider

A sandbox provider runs a Harness in a remote environment instead of a local Docker container. `e2b` ships in `src/roller/sandbox_e2b.py` behind the `e2b` extra; any other provider is a separate package that implements the protocols in `src/roller/sandbox.py` and registers a factory under the `roller.sandbox_providers` entry point, so Roller itself needs no change.

```toml
[project.entry-points."roller.sandbox_providers"]
myprovider = "my_provider:MyProvider"
```

Roller loads that entry point and calls it with no arguments. `--sandbox-provider myprovider --sandbox-template NAME` select the provider and the template it runs in, `roller doctor` lists the providers installed next to Roller, and the template installs the same Roller version because `roller.sandbox_wire` runs inside it.

| Interface | Member | Contract |
| --- | --- | --- |
| `SandboxProvider` | `resolve(template)` | The template's stable version. Roller records it as provenance and hands it back to `create` |
| | `create(template, version, *, name, lifetime, timeout, allocated)` | Report the instance ID through `allocated` before waiting for readiness, so a run that is cancelled or crashes can still release it, then return the box. `e2b` resolves the template again before and after starting it, so a rebuilt template cannot serve a run that preflighted the old one |
| | `remove(instance_id)` | Release a recorded instance and tolerate one that is already gone; a resumed run releases its leftovers through it, and so does one that is stopped |
| `Sandbox` | `version` | The version the box was created from; a box that reports another one makes the Harness refuse it |
| | `run(argv, *, cwd, env, timeout)` | Run a command to completion and return a `subprocess.CompletedProcess` |
| | `spawn(argv, *, cwd, env, log_path)` | Start a background process and return an object with `wait(timeout)` and `kill()` |
| | `upload(source, target, *, contents, timeout)` | Copy a host file or directory in; `contents` copies a directory's contents rather than the directory itself |
| | `download(source, target, *, contents, timeout)` | Copy a sandbox file or directory out to the collector |
| | `close()` | Stop the box and everything it started |
| | `route_http(base, key, timeout)` | Optional. Return the relay's sandbox-accessible base URL as a string. Roller pairs it with the key `sandbox-relay`. Without this method, Roller uses the original base URL and key |

## Pull requests

- Keep a pull request to one change. Describe the problem, the change and how you verified it.
- Keep module imports flowing downward through the seven architecture layers above.
- Add or update tests for behaviour you change.
- When behaviour changes, update the handbook in the [website repository](https://github.com/Valen-Team/valen-team.github.io), in `roller/` and `i18n/zh-Hans/docusaurus-plugin-content-docs-roller/current/`, and the two READMEs when they are affected. The English and Chinese pages mirror each other section by section.
- A new `roller collect` option is declared as an `OptionSpec` (in `src/roller/option_specs.py`, the Harness module's `CLI_OPTIONS`, or the Bench adapter's `cli_options`) and mapped to a YAML key or listed in `COLLECT_ONLY_FLAGS`. A test fails otherwise.
- A setting that changes what a sample contains belongs in the run lock, so that `--resume` refuses to mix samples collected with different settings. A new option must leave the lock of existing runs unchanged when it is not used.
- A run resumes under every patch release of the minor series that created it, so one run can hold samples collected by several patch releases. A change to the SFT projection fits a patch release, because Resume rewrites `trace_SFT.json` from the trace. A change after which new samples no longer belong with those an earlier release of the series collected waits for the next minor release.
- Commit subjects follow the `type(scope): summary` form used in the history, for example `fix(openclaw): turn off heartbeat polls during a rollout`.
- Add a line to the Unreleased section of [CHANGELOG.md](CHANGELOG.md) for a change users will notice.

## Documentation

The handbook is published at [valen-team.github.io/roller](https://valen-team.github.io/roller/) and is built with the website. Its source is in the [website repository](https://github.com/Valen-Team/valen-team.github.io).

## Upgrading pinned Harness images

`scripts/prebuilt_images.py` lists the Harness images published to `ghcr.io/valen-team`. The Images workflow builds them on every release tag and can be started by hand. When upgrading a pinned Harness version:

1. Check upstream CLI/SDK compatibility against the adapter in `src/roller/harnesses/` and any container-side bridge in `src/roller/assets/`.
2. Update the `ARG` or package pin in `docker/<harness>/Dockerfile`, the default image constant in `src/roller/harnesses/`, the pinned version in `src/roller/assets/` (if the Harness uses a bridge), matching scripts (`scripts/prepare-*.sh`), benchmark templates (`examples/benchmarks/*.yaml`), `pyproject.toml`/`uv.lock` (for Python SDK extras), and unit tests. `tests/test_prebuilt_images.py` checks that the Dockerfile pin matches the default image tag.
3. Build each changed non-GHCR Dockerfile on `linux/amd64` and run the version check from `scripts/prebuilt_images.py` plus the container-side bridge `--check` probe (where applicable).

## Releasing

1. Smoke-test both non-conflicting optional extra sets in clean virtual environments:
   - `.[test,swebench-pro,vials,zerobench,pdf,e2b]`
   - `.[test,stirrup]` (Python 3.12+)
2. In the release PR, sync `README.md`, `README_zh-CN.md`, and cookbook READMEs (Harness/Benchmark counts, pinned image tags, capability matrix, and removal of any pre-release `git+https://` install notes; PyPI renders `README.md` from the release tag). Prepare the matching English/Chinese handbook updates in `valen-team.github.io` locally and verify with `npm run build`, but hold off on pushing the website until the release tag has finished publishing.
3. Set the version in `src/roller/version.py` and refresh `uv.lock` with `uv lock`.
4. In `CHANGELOG.md`, rename the Unreleased section to `## <version> - <date>` and start a new empty Unreleased section. A test fails while the changelog has no section for the package version.
5. Run `pytest`, `ruff check`, and `ruff format --check`, merge the release PR, then push the tag `v<version>`. The release workflow builds the package, publishes it to PyPI, creates the GitHub release from that changelog section, and runs the Images workflow to publish the updated `ghcr.io/valen-team/roller-*` images.
6. Once the PyPI release and GHCR images are live, push the prepared handbook updates to `valen-team.github.io` so every install and `docker pull` command in the live documentation works immediately.

## Code of conduct

Participation in this project is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities as described in [SECURITY.md](SECURITY.md), not in a public issue.
