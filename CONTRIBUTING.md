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
| Command line | `src/roller/cli.py` |
| Harness adapters | `src/roller/harnesses/`, with trace converters in `src/roller/*_actf.py` |
| Code that runs inside the containers | `src/roller/assets/` |
| Bench adapters | `src/roller/benches/` |
| SFT export | `src/roller/sft.py`, `src/roller/compaction_sft.py`, `src/roller/sft_dataset.py` |
| Viewer | `src/roller/viewer/` |
| Documentation | `roller/` and its Chinese counterpart in the [website repository](https://github.com/Valen-Team/valen-team.github.io) |
| Machine-readable schemas | `schemas/` |

## Pull requests

- Keep a pull request to one change. Describe the problem, the change and how you verified it.
- Add or update tests for behaviour you change.
- When behaviour changes, update the handbook in the [website repository](https://github.com/Valen-Team/valen-team.github.io), in `roller/` and `i18n/zh-Hans/docusaurus-plugin-content-docs-roller/current/`, and the two READMEs when they are affected. The English and Chinese pages mirror each other section by section.
- A new `roller collect` option also needs an entry in `src/roller/benches/run_config.py`, either mapped to a YAML key or listed as collect-only. A test fails otherwise.
- A setting that changes what a sample contains belongs in the run lock, so that `--resume` refuses to mix samples collected with different settings. A new option must leave the lock of existing runs unchanged when it is not used.
- A run resumes under every patch release of the minor series that created it, so one run can hold samples collected by several patch releases. A change to the SFT projection fits a patch release, because Resume rewrites `trace_SFT.json` from the trace. A change after which new samples no longer belong with those an earlier release of the series collected waits for the next minor release.
- Commit subjects follow the `type(scope): summary` form used in the history, for example `fix(openclaw): turn off heartbeat polls during a rollout`.
- Add a line to the Unreleased section of [CHANGELOG.md](CHANGELOG.md) for a change users will notice.

## Documentation

The handbook is published at [valen-team.github.io/roller](https://valen-team.github.io/roller/) and is built with the website. Its source is in the [website repository](https://github.com/Valen-Team/valen-team.github.io).

## Harness images

`scripts/prebuilt_images.py` lists the Harness images published to `ghcr.io/valen-team`. The Images workflow builds them on every release tag and can be started by hand. Changing a pinned Harness version means changing the default image name in `src/roller/harnesses/` and the `ARG` in the Dockerfile together; a test compares the two.

## Releasing

1. Set the version in `src/roller/version.py`.
2. In `CHANGELOG.md`, rename the Unreleased section to `## <version> - <date>` and start a new empty Unreleased section. A test fails while the changelog has no section for the package version.
3. Merge, then push the tag `v<version>`. The release workflow builds the package, publishes it to PyPI and creates the GitHub release from that changelog section.

## Code of conduct

Participation in this project is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities as described in [SECURITY.md](SECURITY.md), not in a public issue.
