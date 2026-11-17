# Changelog

Notable changes to Roller. Versions follow the tags of this repository and the `roller-agent` releases on PyPI.

## Unreleased

### Added

- `--codex-tool-images-as-user` keeps tool text and attaches the original tool-image bytes as a following user message for compatible Responses backends.
- `--preserve-task-keys` preserves deliberate benchmark key fixtures during HF export while redacting known collection credentials.
- `docker/wildclawbench` provides cleaned base/Codex image recipes, license notices and collection examples.

- Collection credential preflight, detailed failure summaries, progress, ETA, 60-second heartbeats and `collect.log`; `--max-consecutive-errors` stops dispatch after 10 consecutive errors by default, and `--retry-backoff` spaces sample retries.
- Grader retries in the live container and `--resume-unscored` to rerun agents whose grading failed. Grading failures now cause a nonzero exit.
- SFT record indexes and `roller export --format sft --min-score` for score-filtered training data.
- HF credential-shape redaction and a residual review report beside the exported dataset.
- English/Chinese Viewer controls, benchmark scores, sorting/filtering and inline user-message images.
- Collector ownership labels, signal cleanup and `roller clean` for leftovers of dead collectors.

### Fixed

- Grader proxy bypass expands configured CIDR ranges to matching endpoint hosts.
- HF export rules also apply to generated model group names and sample paths; configured proxy bypass values and URL credentials are redacted.
- Export review avoids repeated email-pattern backtracking on long image payloads.

- Benchmark reports aggregate each model separately, include failed samples in denominators and use consistent category denominators.
- Resume accepts task/model supersets and compatible patch upgrades, and rebuilds derived SFT artifacts without rerunning valid samples.
- HF exports exclude previous export directories and review reports from shared run files.

## 0.4.0 - 2026-11-16

Roller 0.4.0 exports a run for publishing. `roller export` writes ATIF, the trajectory format of Harbor, and with `--format hf` a Hugging Face dataset folder that the Hub opens in its trace viewer, with the endpoints, host names, host paths and credentials of the collection environment replaced. Five Harness images are prebuilt at `ghcr.io/valen-team`.

```bash
uv tool install roller-agent    # or: python -m pip install roller-agent
```

One change affects existing setups: OpenClaw runs in `/root/.openclaw/workspace` instead of `/tmp_workspace`. It is described under Changed.

### Added

- Prebuilt `linux/amd64` images of the Codex, Hermes, OpenCode, DeepSeek and Pi Harnesses at `ghcr.io/valen-team`. Claude Code is built locally because its licence does not allow redistribution.
- `roller export RUN_DIR` converts the trajectories of a run to ATIF-v1.7, the trajectory format of [Harbor](https://github.com/harbor-framework/harbor). Each sample becomes a `trajectory.json` with its images stored as files.
- `roller export RUN_DIR --format hf` packages a run as a Hugging Face dataset folder: a dataset card with the statistics of each model, the agent session of every sample, which the Hub opens in its trace viewer, and a table with one row per sample. The folder holds what is published, and `--include-runs` adds a Roller run directory with every file of the samples, which `roller view` opens. A run is written as one part per Harness and model, named `<harness>-<model>`, so further runs can be exported into the same folder. The exported files are redacted by default: the model endpoints, the hosts that `NO_PROXY` names, the name and domain of the machine, the run directory and the host paths of the configuration, and the credential values of the run are replaced with placeholders, and `--no-redact` turns this off. `--redact-rules FILE` adds rules that replace text, leave out files, or stop the export on a forbidden pattern. The statistics table of the card has no token column for a run whose endpoint reported no usage.

### Changed

- OpenClaw runs in `/root/.openclaw/workspace`, the workspace path of an OpenClaw install, which Roller links to the task workspace. A session records this path as its working directory, so tools that recognize OpenClaw sessions by it, such as the trace viewer of the Hugging Face Hub, label them as OpenClaw. It was `/tmp_workspace` before. The task workspace and its files stay at the same paths. WildClawBench upstream runs OpenClaw in the same path.
- The project status is Beta, in the package classifiers and the documentation.
- The handbook moved to [valen-team.github.io/roller](https://valen-team.github.io/roller/) and is maintained in the website repository. `docs/` and the Sphinx build are removed from this repository.

### Fixed

- ACTF `metric.prompt_tokens_len` of an OpenClaw step counts the whole prompt, including cache reads and writes, as it does on the other six Harnesses. It left the cached tokens out before. OpenClaw, Hermes, OpenCode, DeepSeek and Pi steps also record `cache_read_tokens` and `cache_write_tokens`. Trajectories of existing runs keep the values they were written with.

## 0.3.0 - 2026-11-15

Roller 0.3.0 adds the SWE-Bench Pro bench adapter, keeps images in the trajectories of all seven Harnesses, and lets a benchmark run be described in a YAML file. It also fixes defects found by running every Harness against a real model.

```bash
uv tool install roller-agent    # or: python -m pip install roller-agent
```

Four changes affect existing setups: `capture=full` Context is now stored in the run directory, an OpenClaw model must be declared in `--model-config`, WildClawBench scores samples whose agent timed out, and OpenClaw runs with heartbeat polls and transcript redaction turned off. Each is described under Changed.

### Added

- SWE-Bench Pro bench adapter, scored by the hidden tests in the official instance images (#54).
- `roller bench run --config FILE` describes a benchmark run in layered YAML. The example configurations ship in the package and are written by `roller init` (#67).
- `files.attach_images` sends input images with the first user message on OpenClaw, Codex, Claude Code, OpenCode and Pi. User-message images are kept in ACTF and SFT (#63, #76).
- `--pi-input-modalities` and `--deepseek-input-modalities` declare whether the model accepts images. The Viewer shows images returned by tools (#80, #109).
- Native tool selection for Hermes (`--hermes-toolset`), OpenCode (`--opencode-native-tools`, `--opencode-deny-tool`) and DeepSeek (`--deepseek-profile`, `--deepseek-permission-mode`). Defaults are unchanged (#78).
- `--retry-backoff`, `--claude-unique-tool-ids` and `--codex-image-detail` (#76).
- `capture=full` Context is kept in the run directory under `context/`, addressed by content and readable by the owner only (#73).
- A dry run lists `task_bundle_digest`, `input_tree_digest` and `context_manifest_digest` for every planned rollout (#106).
- `--resume-verify-outputs` rereads recorded output files on Resume and reruns a sample whose content no longer matches its SHA-256.
- Python 3.13 is tested and declared, and the package ships a `py.typed` marker.
- `scripts/test_multimodal_e2e.py` checks image inputs against a real Harness and model.

### Changed

- A dry run validates inputs without copying them or writing any file (#106).
- With `--model-config`, an OpenClaw model must be listed by its provider in the file; a mistyped model stops the run before a container starts (#115).
- Containers start from the image id that was inspected, and a tag that changed since preflight is reported (#74, #115).
- OpenClaw runs with transcript redaction and heartbeat polls turned off, so task text is recorded verbatim and no model request is made outside the task (#96, #108).
- DeepSeek profiles other than `sdk-minimal` run commands unconfined inside the task container by default, and their delegation tools are not offered to the model (#109).
- WildClawBench grades a sample whose agent timed out or failed on the workspace it left, instead of leaving it unscored (#67).
- Output files are hashed while they are copied, and the Claude Code settings digest is SHA-256. Locks with the earlier digest still resume (#74).

### Fixed

- OpenClaw subagent samples collected over Chat Completions were excluded from SFT (#110).
- An OpenClaw reply over Chat Completions whose text was interrupted by reasoning was stored with a line break inserted at the interruption.
- The task JSON Schema listed three of the seven Harnesses and rejected several fields the loader accepts (#107).
- The test suite failed on macOS (#111).

## 0.2.1 - 2026-11-15

First release on PyPI.

### Added

- `roller init [DIRECTORY]` writes the example tasks, model configuration templates and `api.env.example`, so the quick start runs from a plain package install.
- The `roller-agent` package, published through Trusted Publishing on every `v*` tag.
- A `test` extra, and CI on Python 3.10 and 3.12 that also runs the quick start from the built wheel.

### Changed

- The build backend is hatchling and the version is defined once in `src/roller/version.py`.
- The installation guides lead with `uv` and `pip`.

## 0.2.0 - 2026-11-11

First tagged release: seven Harnesses (OpenClaw, Codex, Claude Code, Hermes, OpenCode, DeepSeek Harness, Pi Agent), four bench adapters (WildClawBench, SkillsBench, PinchBench, QwenClawBench), batch collection with Resume, User Simulator tasks, ACTF and SFT export, and the Viewer.
