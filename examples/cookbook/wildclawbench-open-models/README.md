# WildClawBench on OpenClaw and Codex

[English tutorial](https://valen-team.github.io/roller/cookbook/wildclawbench-open-models/) · [中文教程](https://valen-team.github.io/zh-Hans/roller/cookbook/wildclawbench-open-models/) · [Trajectories](https://huggingface.co/datasets/Valen-Team/Roller-WildClawBench-Trajectories)

Collect, grade and browse 28 WildClawBench tasks with DeepSeek-V4.1-Flash-1017,
GLM-5.3-Flash, Kimi K3 and Kimi K3.1 Pro. Each model runs each task three times:
84 attempts per collection. The dataset has 336 OpenClaw attempts and 168
Codex attempts from DeepSeek and GLM. The published batches retain successful,
timed-out and failed attempts. Current results and browsable sessions are on HF.

## Setup

Use a Git checkout for the options in this example (PyPI 0.4.0 predates them).
Install with `pip install -e .` from the checkout, start Docker, and download [WildClawBench](https://github.com/InternLM/WildClawBench)
with its task assets. The image build instructions are in
[`docker/wildclawbench`](../../../docker/wildclawbench).

```bash
export WILDCLAWBENCH_ROOT=~/WildClawBench
export OPENROUTER_BASE_URL=http://localhost:8001/v1
export OPENROUTER_API_KEY=not-required
export JUDGE_MODEL=glm-5.3-flash
```

Use your own model and judge endpoints. For internal services reached through a
proxy, put their hostnames in `NO_PROXY`. Both agents use the same task list and
judge; agents and graders have no Roller retries in this example.

## Codex

The endpoint must support the Responses API, including tools and images. Run
one model per invocation; separate invocations can run simultaneously.

```bash
export MODEL=glm-5.3-flash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=not-required
export MODEL_CONTEXT_WINDOW=98304 # match your serving configuration
export RUN_NAME=wildclawbench-codex-glm
./collect-codex.sh --dry-run
./collect-codex.sh
roller view runs/wildclawbench-codex-glm
```

For a backend that drops images from tool outputs, set `TOOL_IMAGES_AS_USER=1`.
This attaches the same image bytes as a following user image message and keeps
the associated tool text. Kimi K3.1 Pro backends that lose tool images
need this compatibility setting; other backends should leave it off.

For a one-task check before a full collection:

```bash
printf '%s\n' 05_Creative_Synthesis_task_10_social_poster_multi_crop > one-task.txt
TASK_IDS_FILE="$PWD/one-task.txt" RUNS_PER_TASK=1 PARALLEL=1 \
  RUN_NAME=wildclawbench-codex-check ./collect-codex.sh
```

## OpenClaw

`models.json` describes the four model IDs behind an OpenAI-compatible Chat
Completions endpoint. Change the model IDs and context windows to match your
server, then run:

```bash
export ROLLER_INFER_BASE_URL=http://localhost:8000/v1
export ROLLER_INFER_API_KEY=not-required
./collect.sh --dry-run
./collect.sh
```

## Export and append a batch

```bash
roller export runs/wildclawbench-codex-glm --format hf --output dataset \
  --preserve-task-keys --redact-rules redact.json
hf upload <owner>/<dataset> dataset . --repo-type dataset
```

Export other models to the same `dataset` folder to add their parts. Each part
uses `traces/<harness>-<model>/` and `data/<harness>-<model>.jsonl`. When collecting
another batch of the same Harness and model, give it a separate folder/name
before uploading so an earlier batch stays available. Keep a short batch note
with the date, model settings, task selection and any compatibility settings.

The HF exporter replaces collection endpoints, host paths and known environment
credentials. `--preserve-task-keys` keeps deliberately planted benchmark keys;
review the resulting files and add environment-specific replacements to
`redact.json`. Raw runs stay unchanged. The exported table keeps execution status,
score, grader error and output filenames.

The two Harness batches are an observed comparison under their documented
settings. Their prompts, tools, context management, software versions and
backend compatibility behavior differ.
