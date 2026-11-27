# VIALS with Stirrup

Collect and grade [VIALS](https://huggingface.co/datasets/handshake-ai-research/VIALS)
with tools. The agent is Stirrup 0.2.2 with `code_exec`, `view_image` and
`finish`. The task follows the tool-assisted agent adapter of VIALS: its prompt,
the question's images as files under `/task`, and the answer read from the last
`ANSWER:` line of `/task/answer.txt`. Each of the 161 questions runs three
times, and the report gives pass^3, avg@3 and pass@3.

## Setup

The `stirrup` extra needs Python 3.12 or later; the `vials` extra reads the
dataset:

```bash
pip install 'roller-agent[stirrup,vials]'
```

Build the tools image: Python 3.13 with the eleven Python packages the prompt
names and the Tesseract OCR engine. Run the commands below from
`examples/cookbook/vials-stirrup`.

```bash
docker build -t roller-vials ../../../docker/vials
```

Roller downloads the `test` split of the `tasks` configuration into
`vials-data/` (`VIALS_DATA`) at a pinned revision, and records the revision in
each task. Set `VIALS_REVISION` to another branch, tag or commit to use other
data. The containers run without network; the agent loop, the model requests
and the judge run in the Roller process.

## Run

```bash
export MODEL=glm-5.3-flash
export MODEL_BASE=http://localhost:8000/v1
export JUDGE_BASE_URL=https://api.openai.com/v1
export JUDGE_API_KEY=...
./collect.sh --dry-run
./collect.sh
roller bench report runs/vials-stirrup-glm-5.3-flash
```

`collect.sh` sends every request with reasoning effort `max`, temperature 1,
top_p 0.95, `max_tokens` 65536 and a seed derived from 42, the question and the
sample; the agent has 100 turns, a 98,304-token context (history is summarized
at 70% of it) and 300 seconds per command. Set `PARALLEL` for the number of
containers at a time.

## Grading

The judge sees the question, the reference answer, the acceptable numeric range
of the five counting questions and the extracted answer, and replies whether
they are equivalent. When its reasoning starts with `UNVERIFIABLE:`, it is asked
again with the images. A missing `answer.txt` or one without `ANSWER:` scores 0
without a judge call. The judge defaults to `gpt-6-luna` with reasoning effort
`medium` and 4,096 output tokens; set `JUDGE_MODEL`, `JUDGE_REASONING_EFFORT`
and `JUDGE_MAX_TOKENS` to change them, or the same names with a `VIALS_` prefix
to set them for VIALS only. A sample whose judge call fails stays unscored, and
`roller collect --resume` grades it again.

pass^3 is the share of questions answered correctly in all three samples, avg@3
the share of correct samples, and pass@3 the share of questions answered
correctly at least once. The report gives them once every sample is scored.
