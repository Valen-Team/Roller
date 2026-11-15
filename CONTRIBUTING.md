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
ruff check              # pyflakes rules; install with: python -m pip install ruff
```

The tests run on Linux and macOS without Docker or network access. Tests that execute real containers are skipped unless their environment variables are set.

To check a change against a real Harness, build its image as described in the Harness guide and run `scripts/test_subagents_e2e.py` or a small task list with `roller collect`.

## Where things live

| Area | Location |
| --- | --- |
| Command line | `src/roller/cli.py` |
| Harness adapters | `src/roller/harnesses/`, with trace converters in `src/roller/*_actf.py` |
| Code that runs inside the containers | `src/roller/assets/` |
| Bench adapters | `src/roller/benches/` |
| SFT export | `src/roller/sft.py`, `src/roller/compaction_sft.py` |
| Viewer | `src/roller/viewer/` |
| Documentation | `docs/en/` and `docs/zh/` |
| Machine-readable schemas | `schemas/` |

## Pull requests

- Keep a pull request to one change. Describe the problem, the change and how you verified it.
- Add or update tests for behaviour you change.
- Update the documentation in both `docs/en` and `docs/zh`, and the two READMEs when they are affected. The pages mirror each other section by section.
- A new `roller collect` option also needs an entry in `src/roller/benches/run_config.py`, either mapped to a YAML key or listed as collect-only. A test fails otherwise.
- A setting that changes what a sample contains belongs in the run lock, so that `--resume` refuses to mix samples collected with different settings. A new option must leave the lock of existing runs unchanged when it is not used.
- Commit subjects follow the `type(scope): summary` form used in the history, for example `fix(openclaw): turn off heartbeat polls during a rollout`.
- Add a line to the Unreleased section of [CHANGELOG.md](CHANGELOG.md) for a change users will notice.

## Building the documentation

```bash
python -m pip install -r docs/requirements-docs.txt
cd docs && make html
```

## Code of conduct

Participation in this project is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities as described in [SECURITY.md](SECURITY.md), not in a public issue.
