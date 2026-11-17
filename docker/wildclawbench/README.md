# WildClawBench task images

Two Linux amd64 images are available:

```bash
docker pull ghcr.io/valen-team/roller-wildclaw-base:1.3-clean.1
docker pull ghcr.io/valen-team/roller-wildclaw-codex:1.3-codex0.149.0
```

The base keeps WildClawBench's installed task environment and removes personal
builder state, including OpenClaw device identity, saved account credentials,
shell history and proxy credentials. It is exported into a fresh image so the
removed data is absent from the published layers as well. The derivative adds
Codex CLI 0.149.0, Node.js and ripgrep.

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
```

Build files and the local inspection report are written to `wildclaw-image-build/`.
The script leaves the source image untouched. Benchmark dependencies needed by
a task are installed by that task's setup commands; internet-free environments
must make the corresponding packages and browser downloads available locally.

See [the collection example](../../examples/cookbook/wildclawbench-open-models)
for the task list, model and judge configuration, and export commands.
Third-party licenses remain applicable; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)
and the license files within each image.
