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

See [the collection example](../../examples/cookbook/wildclawbench-open-models)
for the task list, model and judge configuration, and export commands.
Third-party licenses remain applicable; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)
and the license files within each image.
