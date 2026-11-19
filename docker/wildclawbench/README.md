# WildClawBench task images

Pull the Linux amd64 base and Codex task images:

```bash
docker pull ghcr.io/valen-team/roller-wildclaw-base:1.3-clean.1
docker pull ghcr.io/valen-team/roller-wildclaw-codex:1.3-codex0.149.0
```

The Claude Code task image uses the registry tag
`ghcr.io/valen-team/roller-wildclaw-claude-code:1.3-claude2.1.222`. The build
command below works from the base image.

The base keeps WildClawBench's installed task environment and removes personal
builder state, including OpenClaw device identity, saved account credentials,
shell history and proxy credentials. It is exported into a fresh image so the
removed data is absent from the published layers as well. The Codex derivative
adds Codex CLI 0.149.0, Node.js and ripgrep. The Claude Code image installs
the pinned CLI from the npm registry on first use in a new container.

Task workspaces, deliberately planted test keys and graders come from the
benchmark checkout when Roller starts each task. The image cleaner does not
edit those inputs. Images do not include a model endpoint or API credentials.

To build from the original image, download and load `wildclawbench-ubuntu:v1.3`
from [the upstream dataset](https://huggingface.co/datasets/internlm/WildClawBench),
then run from the Roller repository:

```bash
python docker/wildclawbench/clean_image.py
docker build -f docker/wildclawbench/Dockerfile.base \
  -t roller-wildclaw-base:1.3-clean.1 docker/wildclawbench

docker build --build-arg WILDCLAW_BASE=roller-wildclaw-base:1.3-clean.1 \
  -t roller-wildclaw-codex:1.3-codex0.149.0 docker/wildclawbench

docker build -f docker/wildclawbench/Dockerfile.claude-code \
  --build-arg WILDCLAW_BASE=roller-wildclaw-base:1.3-clean.1 \
  --build-arg INSTALL_CLAUDE_CODE=1 \
  -t roller-wildclaw-claude-code:1.3-claude2.1.222 docker/wildclawbench
```

Build files and the local inspection report are written to `wildclaw-image-build/`.
The local Claude Code build installs the CLI once during the build. Point the
collection script at this image with `CLAUDE_CODE_IMAGE`.
The script leaves the source image untouched. Benchmark dependencies needed by
a task are installed by that task's setup commands; internet-free environments
must make the corresponding packages and browser downloads available locally.

Build the DeepSeek task image from the same local base using the SDK and native
runtime pinned by Roller.

```bash
docker build --build-arg BASE_IMAGE=roller-wildclaw-base:1.3-clean.1 \
  -t roller-wildclaw-deepseek:1.3-deepseek0.1.5rc1 docker/deepseek
```

In `roller collect`, select the DeepSeek harness and the task image with
`--harness deepseek` and `--deepseek-image roller-wildclaw-deepseek:1.3-deepseek0.1.5rc1`.

### Remote execution

Roller can run single-agent WildClawBench tasks with DeepSeek through an
installed sandbox provider. Install the same Roller checkout in the remote
template so it contains `roller.sandbox_wire`. The template must also have
the task environment and the pinned DeepSeek SDK and runtime.

```bash
python -m pip install 'git+https://github.com/Valen-Team/Roller.git'
roller collect --bench wildclawbench --bench-root "$WILDCLAWBENCH_ROOT" \
  --bench-task-id "$TASK_ID" --harness deepseek --model "$MODEL" \
  --api-base "$MODEL_BASE" --wire-trace full --capture-calls \
  --pass-env JUDGE_MODEL \
  --sandbox-provider "$SANDBOX_PROVIDER" --sandbox-template "$SANDBOX_TEMPLATE"
```

Bench run configuration uses `execution.sandbox_provider` and
`execution.sandbox_template`. Third-party providers register a factory in the
`roller.sandbox_providers` entry point group and implement the contract in
`roller.sandbox`.

#### E2B template

Use Python 3.11 or later on the collector. From the Roller checkout, install
the E2B extra and prepare the task assets without starting Docker. Install the
Hugging Face CLI, yt-dlp, ffmpeg and ModelScope for the asset preparation script.
Set `E2B_API_KEY` on the collector and use model and judge endpoints reachable
from E2B.

```bash
python -m pip install -e '.[e2b]'
export WILDCLAWBENCH_ROOT="$PWD/.benchmarks/wildclawbench"
WILDCLAW_ASSETS_ONLY=1 scripts/prepare-wildclawbench.sh "$WILDCLAWBENCH_ROOT"

build_dir=$(mktemp -d)
python -m pip wheel --no-deps . --wheel-dir "$build_dir"
export ROLLER_TEMPLATE_WHEELS="$build_dir"
export WILDCLAW_BASE=ghcr.io/valen-team/roller-wildclaw-base:1.3-clean.1
export E2B_TEMPLATE_NAME=roller-wildclaw-deepseek
python - <<'PY'
import os
from pathlib import Path
from e2b import Template, default_build_logger

wheels = Path(os.environ["ROLLER_TEMPLATE_WHEELS"])
wheel = next(wheels.glob("roller_agent-*.whl"))
dockerfile = Path("docker/deepseek/Dockerfile").read_text().replace(
    "FROM ${BASE_IMAGE}", "FROM " + os.environ["WILDCLAW_BASE"], 1
)
template = (
    Template(file_context_path=wheels)
    .from_dockerfile(dockerfile)
    .copy(wheel.name, f"/tmp/{wheel.name}")
    .run_cmd(f"/opt/roller-deepseek/bin/pip install /tmp/{wheel.name}", user="root")
    .set_envs({"PATH": "/opt/roller-deepseek/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)
build = Template.build(
    template, name=os.environ["E2B_TEMPLATE_NAME"], on_build_logs=default_build_logger()
)
print(build.template_id, build.build_id)
PY
```

Use the template ID printed by the build for `SANDBOX_TEMPLATE`. Roller records
the E2B build ID for the selected template tag in the run lock. In the
collection command above, set `SANDBOX_PROVIDER=e2b`.

For a first run, set `TASK_ID=01_Productivity_Flow_task_6_calendar_scheduling`.
This task installs its Python warmup dependencies from PyPI and uses a local
grader. If your collector uses a loopback HTTP proxy, unset its proxy
variables for this command so the sandbox can reach its own package and
model endpoints. The three declared files are `scheduled.ics`,
`unscheduled.json` and `decision_log.md`.

The same single-task trial can be run with `roller bench run --config remote.yaml`.
Save the following as `remote.yaml` and set the referenced environment variables,
including `DEEPSEEK_API_KEY`.

```yaml
schema_version: roller.bench-run/v1
benchmark:
  id: wildclawbench
  root: ${WILDCLAWBENCH_ROOT}
  task_ids: [ "${TASK_ID}" ]
model:
  id: ${MODEL}
  base_url: ${MODEL_BASE}
  api_key_env: DEEPSEEK_API_KEY
harness:
  id: deepseek
execution:
  sandbox_provider: ${SANDBOX_PROVIDER}
  sandbox_template: ${SANDBOX_TEMPLATE}
  parallel: 1
  grader_parallel: 1
  retries: 0
  grader_retries: 0
  wire_trace: full
  capture_calls: true
  pass_env: [JUDGE_MODEL]
```

See [the collection example](../../examples/cookbook/wildclawbench-open-models)
for the task list, model and judge configuration, and export commands.
Third-party licenses remain applicable; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)
and the license files within each image.
