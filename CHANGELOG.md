# Changelog

Notable changes to Roller. Versions follow the tags of this repository and the `roller-agent` releases on PyPI.

## Unreleased

### Added

- `roller doctor` reports Python, `roller-agent`, Docker availability, optional extras, and sandbox providers.
- `roller bench report --json` and `--format json` print the benchmark report JSON directly to stdout without summary formatting.

## 0.7.0 - 2026-11-27

Roller 0.7.0 adds the Stirrup Harness, Claude Code remote sandbox execution, the VIALS and ZeroBench bench adapters, OpenAI multimodal SFT records with XTuner export, `--training-files-only` compact runs, upgraded pinned Harness versions, and a decoupled layered module architecture.

```bash
uv tool install roller-agent    # or: python -m pip install roller-agent
```

### Added

- `roller export --format xtuner` exports a run as a single merged `xtuner.jsonl` (combining unsegmented rollouts and compaction segments, with a line-for-line `xtuner_index.jsonl`, supporting `--min-score`). Images are extracted to `<output>/media/` with EXIF-oriented `image_wh` metadata (`--inline-images` keeps base64 data URIs inline), `tool_calls[].function.arguments` JSON strings are parsed into objects, and `target_start_index` is translated into per-message `"loss": false` on earlier assistant turns.
- Stirrup Harness (`--harness stirrup`, the `stirrup` extra, Python 3.12 or later). Stirrup 0.2.2's agent loop runs in the Roller process with `code_exec`, `view_image` and `finish`; the sample's Docker container or remote sandbox executes the commands and holds the files, and gets no model key. Model requests go from Roller through the local request recording proxy, so `--wire-trace full`, `--capture-calls` and `--token-ids` work in Docker and remote sandboxes alike. The request fields are set by `--stirrup-*` options and `--thinking`, in a fixed order. A history summary splits SFT into windows from captured Calls, and the summary request is a window of its own.
- SFT windows built from captured Chat Completions Calls keep the images of user and tool messages, as the linear export does.
- Claude Code Harness in a remote sandbox (`--harness claude-code --sandbox-provider NAME --sandbox-template NAME`). The CLI itself runs in the sandbox, started from the release the template mounts (`--claude-sandbox-binary` for a path other than `/opt/claude-code/bin/claude`), and the sandbox holds the task's files. A sandbox created without egress cannot reach the request recording proxy, so the provider's optional `route_http` carries the CLI's requests out through the sandbox API while the proxy stays on the collecting host: `--wire-trace full`, `--capture-calls` and `--token-ids` write the same wire trace, Calls, blobs and token ids as under Docker, and the native session trace, its ACTF conversion, usage and output collection are read from the sandbox the same way. `--network` and `--keep-container-on-error` apply to Docker only.
- `--claude-tool-images-inline` keeps an image a tool returned in the tool message itself instead of the user message that follows it, for endpoints and chat templates that render it there. Off by default, which is how vLLM's own Messages endpoint arranges such a turn.
- `--claude-context-window` tells Claude Code the served context length, so it compacts its history before the endpoint would refuse the request and end the sample. In a remote sandbox the Harness also fixes the CLI settings an agentic benchmark run depends on: no attribution header, which varies per conversation and would defeat the served model's prefix cache; no experimental betas, telemetry or error reporting; and a reply timeout the relay's whole-response delivery fits inside.
- ZeroBench bench (`--bench zerobench`, the `zerobench` extra): the 100 main questions (`--bench-variant zerobench`) or the 334 sub-questions (`--bench-variant zerobench_subquestions`), read from Hugging Face at a pinned revision (`ZEROBENCH_REVISION` to change it). Each task carries its images as files under `/task` for the agent to crop and view, with the official question and answer-format instruction, and the official strict `{final answer}` rule scores the final message with no judge and no environment variables. `bench report` adds avg@k and pass@k over k samples per question. `examples/cookbook/zerobench-claude-code` runs the protocol with Claude Code in a remote sandbox.
- SFT windows rebuilt from captured Anthropic Calls carry the images a tool returned, in the tool message itself or in the user message after it, whichever the collection served (`--claude-tool-images-inline`). A trajectory whose tools answer with images -- a vision agent cropping and viewing a picture -- could not be converted at all before.
- Text blocks of one message are concatenated in those windows rather than joined with a newline, as the endpoint concatenates them. The separator added a character the model was never served; it kept the window's length and changed one token id, so only a token-level comparison against the recorded ids showed it.
- VIALS bench (`--bench vials`, the `vials` extra): the 161 questions of the `tasks` test split, read from Hugging Face at a pinned revision (`VIALS_REVISION` to change it) and cached under `--bench-root`. Each task follows the upstream tool-assisted agent adapter: its prompt, the images as files under `/task`, and the answer in `/task/answer.txt`, judged against the reference with the upstream judge prompt. `bench report` adds pass^k, avg@k and pass@k over k samples per question. `docker/vials` builds the tools image and `examples/cookbook/vials-stirrup` runs the protocol with Stirrup.
- `roller collect --training-files-only` (`training_files_only` in a bench run configuration) keeps only the training files of a sample once it has finished and been graded: `trace_SFT.json` or `sft_segments/`, `token_ids.jsonl`, `info.json` and `score.json`, with each image of the SFT records stored once for the run, in the run directory's `blobs/`. The trace, ACTF, Calls and wire trace are removed after the context windows have been cut from the Calls, and the run-level `sft.jsonl` is not written. `roller export --format sft`, `xtuner` and `tokens` give the same output as for the run collected without the option, `--resume` takes these samples as finished and `bench report` reads their scores. The Viewer, the Hugging Face and ATIF exports and rebuilding SFT need the removed files. Samples that failed, or whose grading failed, keep all their files.
- `bench.grade.collector_files` grades a copy of those workspace files in the collecting process, for containers without network; the judge's key stays out of the container.
- `files.output_dir: null` gives a Stirrup task no output directory: the workspace holds only the task's files and what the agent writes, and nothing is collected from it. VIALS tasks use it.

### Changed

- Upgraded all eight pinned Harness versions, Dockerfiles, and optional SDK extras: OpenClaw `2026.11.10`, Claude Code `2.1.332`, Codex `0.172.0`, DeepSeek Harness `0.2.2rc2`, Hermes `8fc037444731` (`v0.23.4`, `python:3.14-slim-bookworm`, `uv==0.13.10`), OpenCode `1.19.6`, Pi Agent `1.6.1`, `stirrup==0.2.2`, and `e2b==2.66.0` (#167).
- Decoupled Roller's internal architecture into a strictly layered dependency graph and split oversized modules (`cli/`, `actf/`, `execution.py`, `docker_runtime.py`, `option_specs.py`, `run_outputs.py`, `sample_lifecycle.py`, `sample_metadata.py`, `hf_redaction.py`, `wire_transforms.py`, `compaction_parsers.py`, and per-Harness config/session modules) while preserving the CLI, YAML run configs, on-disk run/lock formats, public package exports, and `roller.sandbox_providers` entry point (#166).
- Task prompts reach the model as written; Roller no longer strips leading and trailing whitespace from `prompt`, `question` and `messages[].content`. A blank prompt is still rejected. Tasks whose prompt had such whitespace get a new bundle digest.
- Stirrup passes no proxy variables into the container, since its model requests start in the Roller process.
- Multimodal SFT records (`trace_SFT.json`, `sft.jsonl`, `sft_segments.jsonl`) now use standard OpenAI content lists (`{"type": "text", "text": ...}` and `{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}`) instead of flattening blocks with `<image>` placeholders and a top-level `images` list. Text-only messages keep plain string `content`, literal `<image>` in source text no longer blocks SFT export, and no extra newline is inserted before images. `roller export --format sft` and `--format xtuner` still upgrade legacy `<image>` + `images` runs on disk.

- `wire_trace.jsonl` and the captured Calls store each image of the request bodies once per sample, in `blobs/<sha256>.<subtype>`, and keep `data:<mime>;base64,roller-blob:<sha256>.<subtype>` in place of the data URI. Every request resends the images of the conversation so far, so both files grew with the square of the number of turns: a Stirrup sample with 18 requests and 7 images takes 14 MB instead of 94 MB. Roller restores the original data URIs, character for character, when it reads the files; runs collected earlier are read as they are.
- The run-level `calls.jsonl` lists the exportable Calls without their `context` and `output` (`schema_version: roller.call-index/v1`). It held a second copy of every captured Call; the request and response stay in the sample's `capture/`.
- `--token-ids` works in a remote sandbox (`--sandbox-provider`). The wire recorder in the sandbox writes `token_ids.jsonl` and Roller copies it into the sample. The sandbox template needs this Roller version.
- `roller export --format hf` writes the trace of a Hermes sample as a Hermes session, the format of `hermes sessions export`, so the Hub lists it as a Hermes trace. It was a Pi session converted from ACTF.
- The `pdf` extra installs on Python 3.11 and later, with Pillow 13 or later. Pillow 13 has the fixes for the vulnerabilities reported in earlier versions and needs Python 3.11. On Python 3.10, `--claude-pdf-as-images` reports that the extra needs Python 3.11.

### Fixed

- The Hugging Face export of imported sessions also replaces the local user name inside longer names, such as `pytest-of-<user>`, `_Users_<user>_` and `<user>.github.io`, and right after an escaped line break or a URL escape such as `%2F`. A user name still left, such as one joined to a word, is listed in the redaction report as `user_name`.
- `roller export --format hf --include-runs` leaves out the `tokens/` folder that `roller export --format tokens` writes into the run, as it already did for the other export folders.
- SFT windows built from captured Calls stay open when the Harness replays a tool call with schema defaults filled in or with the argument keys in another order, as Claude Code does. Each such call started a new SFT record that repeated the messages of the record before it.

## 0.6.0

### Added

- Token-in, token-out collection from a Chat Completions endpoint that returns token ids, such as vLLM. With `--capture-calls`, `--token-ids` saves the prompt and completion token ids of every model call with the logprob of each sampled token in `token_ids.jsonl`. A call that continues an earlier one holds only the prompt tokens that follow it, so the file grows with the length of the conversation. The run records whether the endpoint reports logprobs before or after temperature and top-p. `roller export --format tokens` joins the calls of a conversation into one sequence in which every token appears once, with the positions the model generated marked.
- `--token-ids` works with every Harness in Docker. Hermes, OpenCode, DeepSeek Harness, Pi Agent and OpenClaw call Chat Completions themselves. Claude Code and Codex do through `--claude-chat-completions` and `--codex-chat-completions`, which translate their Messages and Responses requests to Chat Completions at the request recording proxy, with the same prompt vLLM renders for its own Messages and Responses endpoints.
- The WildClawBench cookbook includes an OpenCode collection script with full request and call capture.

### Changed

- Roller now calls Docker through the named verbs of `DockerRuntime` (`exec`, `copy_in`, `copy_out`, `kill`, `remove`, `ps`, `image_id`, …) instead of writing `docker` argument lists at each call site. The argv and recorded provenance are unchanged, and the Docker command line is spelled in one module.

### Fixed

- OpenCode uses the configured output budget for model requests. ACTF keeps observed turns and tool calls from attempts that hit a context overflow or end with a running tool. Those attempts are unavailable for SFT.
- Hugging Face export preserves the OpenCode native journal when ACTF is unavailable and redacts OpenCode's effective proxy bypass assignment.

## 0.5.0

### Added

- Single-agent WildClawBench collection with DeepSeek Harness can use a remote sandbox. `roller collect` and `roller bench run` accept an execution provider and template. Docker remains the default. The `e2b` extra enables E2B and the WildClawBench cookbook includes the template recipe.
- The WildClawBench cookbook includes a DeepSeek collection script and an assets-only preparation mode for collectors without a Docker daemon.
- `roller import` imports the local Claude Code and Codex sessions, compressed Codex sessions included, as a run that the Viewer and every export read. Imported samples record a title, a project and where their tools and system prompt came from (`tools_source`, `system_prompt_source`), which `sft_index.jsonl` carries.
- The Hugging Face export of imported sessions replaces their home directories, user names and account identifiers, the global git name and email, and the values of credential variables in the exporting environment.
- SFT records keep `tool_reference` blocks of tool results as JSON text.
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

- DeepSeek exports retain the native journal when an attempt has no ACTF trajectory. Runtime and capture artifacts remain available when output collection fails.
- Benchmark graders read JSONL journals one record at a time, including journals with large image payloads.
- HF export no longer drops the rest of a file after text that names a private key header without its key, which broke the JSON of the trajectory.
- HF export matches the credential shapes in one pass per line, which halves its time on large runs.
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
