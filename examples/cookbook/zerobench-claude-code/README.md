# ZeroBench with Claude Code

Collect and grade [ZeroBench](https://huggingface.co/datasets/jonathan-roberts1/zerobench)
with tools. The agent is the Claude Code CLI, running inside a remote sandbox
created without egress. Each question's images are files under `/task`, which
the agent crops, enhances and views as it works; the answer is the last
`{...}` group of its final message, matched against the reference by the
official strict rule, with no judge.

`--bench-variant zerobench` runs the 100 main questions and
`--bench-variant zerobench_subquestions` the 334 visual sub-questions the main
questions decompose into. The report gives avg@k and pass@k over the samples
per question; five is the number the benchmark reports.

## Setup

The `zerobench` extra reads the dataset:

```bash
pip install 'roller-agent[zerobench]'
```

You need a sandbox provider registered under the `roller.sandbox_providers`
entry point, and a template whose instances mount a pinned Claude Code release
read-only. The default path is `/opt/claude-code/bin/claude`; pass
`--claude-sandbox-binary` for another. The template also supplies the Python
tools the prompt names (numpy, scipy, sympy, pillow, imageio,
opencv-python-headless, scikit-image, matplotlib, pandas, networkx,
pytesseract), because the sandbox cannot install them.

Roller reads the dataset through `datasets`, at a pinned revision recorded in
each task; set `ZEROBENCH_REVISION` to use other data. It writes each
question's images under `--bench-root`. For an offline collector, populate the
Hugging Face cache first and export `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`
with `HF_HOME` pointing at it — the adapter leaves the dataset cache to those
variables rather than moving it under `--bench-root`, so a prepared cache is
used as it stands.

## How the model requests get out

The CLI speaks the Messages API to a proxy, but its sandbox has no egress, so
it cannot open a connection to the collecting host. The provider's optional
`route_http` gives it a base URL inside the sandbox and carries each request
out through the sandbox API; the request recording proxy stays on the host,
behind that relay. So `--wire-trace full`, `--capture-calls` and `--token-ids`
write the same wire trace, Calls, blobs and token ids they write under Docker,
and the run is usable as training data.

`--claude-tool-images-inline` keeps the image a `Read` returns in the tool
message itself rather than in a user message after it. Use it when the served
endpoint and the chat template you will train with both render it there — then
the prompt the model answered and the prompt a trainer tokenizes are the same.
Leave it off otherwise; that is how vLLM's own Messages endpoint arranges the
turn.

## If you will train on the run

Both sides have to cap an image at the same number of visual placeholders, or
a trainer tokenizes a different prompt than the model answered. The cap is one
processor field, and the serving and training sides set it from different
places: the served processor uses `image_processor.max_image_tokens` from the
model's `processor_config.json`, while a trainer usually overrides the same
field. ZeroBench ships large images — the main split's median is 2006×1656 and
its largest is 5559×4608, and 40 of its 100 questions carry an image that
needs more placeholders than a typical served cap allows — so this is the
common case here, not an edge case. Read the served value and set the
trainer's to match before you tokenize.

Images the agent writes itself count too: a crop it saves and views with
`Read` goes through the same cap.

## Run

```bash
export MODEL=glm-5.3-flash
export MODEL_BASE=http://localhost:8000/v1
export SANDBOX_PROVIDER=my-cluster
export SANDBOX_TEMPLATE=zerobench-claude-code
./collect.sh --dry-run
./collect.sh
```

```bash
roller bench report runs/zerobench-claude-code-glm-5.3-flash
roller export runs/zerobench-claude-code-glm-5.3-flash --format xtuner \
  --output zerobench-sft
```

The main questions are the benchmark's own held-out set. Decide before you
train whether a run of them is evaluation or training data, and say which in
the run's notes: a set used for both reports a score that cannot be read.
