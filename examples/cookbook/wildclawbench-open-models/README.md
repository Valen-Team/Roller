# WildClawBench on OpenClaw, Codex, Claude Code and DeepSeek

[English tutorial](https://valen-team.github.io/roller/cookbook/wildclawbench-open-models/) · [中文教程](https://valen-team.github.io/zh-Hans/roller/cookbook/wildclawbench-open-models/) · [Trajectories](https://huggingface.co/datasets/Valen-Team/Roller-WildClawBench-Trajectories)

Collect, grade and browse 28 WildClawBench tasks with DeepSeek-V4.1-Flash-1017,
GLM-5.3-Flash, Kimi K3 and Kimi K3.1 Pro. Each model runs each task three times:
84 attempts per model and harness, or 336 per harness. Hugging Face hosts the
OpenClaw, Codex, Claude Code and DeepSeek Harness collections, including
successful, timed-out and failed attempts.

## Setup

Use a Git checkout for the options in this example. Install with `pip install -e .`
from the root of the checkout and download
[WildClawBench](https://github.com/InternLM/WildClawBench) with its task assets.
Start Docker for local execution, or follow the E2B template setup for DeepSeek.
Run the commands below from `examples/cookbook/wildclawbench-open-models`.
The image build instructions are in
[`docker/wildclawbench`](../../../docker/wildclawbench).

```bash
export WILDCLAWBENCH_ROOT=~/WildClawBench
export OPENROUTER_BASE_URL=http://localhost:8001/v1
export OPENROUTER_API_KEY=not-required
export JUDGE_MODEL=glm-5.3-flash
```

Use your own model and judge endpoints. For internal services reached through a
proxy, put their hostnames in `NO_PROXY`. The harnesses use the same task
list and judge. The scripts disable retries for agents and graders.

## Claude Code

Use an Anthropic Messages endpoint that streams text and tool calls and accepts
images, including images returned by tools. The task image uses Claude Code
2.1.222. It downloads the pinned CLI from the npm registry when first invoked
in a new container.

```bash
docker build --platform linux/amd64 -f ../../../docker/wildclawbench/Dockerfile.claude-code \
  --build-arg INSTALL_CLAUDE_CODE=1 \
  -t roller-wildclaw-claude-code:1.3-claude2.1.222 ../../../docker/wildclawbench
export CLAUDE_CODE_IMAGE=roller-wildclaw-claude-code:1.3-claude2.1.222
export MODEL=glm-5.3-flash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-required
./collect-claude-code.sh --dry-run
./collect-claude-code.sh
roller view runs/wildclawbench-claude-code-glm-5.3-flash
```

The script runs one model at a time. The collection used a local build with
the same CLI version installed before the tasks started. See the
[task image notes](../../../docker/wildclawbench) for the registry tag.
Set `TASK_IDS_FILE`, `RUNS_PER_TASK`, and `PARALLEL` for a small trial.

If your Messages gateway rejects PDF `document` blocks, install the optional PDF
dependencies with `pip install -e '.[pdf]'` from the Roller checkout and run
`./collect-claude-code.sh --claude-pdf-as-images`. Roller forwards extracted
text and a PNG for each page. The wire trace records what the endpoint receives.
For an endpoint with a smaller context window, set
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` to its configured token limit.

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

## DeepSeek

Use a Chat Completions endpoint that streams tool calls. From the Roller
checkout, build the [DeepSeek task image or E2B template](../../../docker/wildclawbench).
The collection script uses one model per invocation.

```bash
export MODEL=glm-5.3-flash
export MODEL_BASE=http://localhost:8000/v1
export DEEPSEEK_API_KEY=not-required
export DEEPSEEK_IMAGE=roller-wildclaw-deepseek:1.3-deepseek0.1.5rc1
./collect-deepseek.sh --dry-run
./collect-deepseek.sh
```

For E2B, set `SANDBOX_PROVIDER=e2b` and `SANDBOX_TEMPLATE` to your built
template ID. Set `MODEL_BASE` and `OPENROUTER_BASE_URL` to addresses reachable
from the template. Pass `--deepseek-input-modalities text,image` for models
that accept images and use the context and output limits of your endpoint.
Pass `--deepseek-max-steps` to set a per-turn step limit for the batch.

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

The four Harness batches are an observed comparison under their documented
settings. Their prompts, tools, context management, software versions and
backend compatibility behavior differ.
