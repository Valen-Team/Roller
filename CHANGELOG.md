# Changelog

Notable changes to Roller. Versions follow the tags of this repository and the `roller-agent` releases on PyPI.

## Unreleased

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
