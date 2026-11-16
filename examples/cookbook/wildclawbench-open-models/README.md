# 实战案例：用两个自部署模型采集 WildClawBench

[教程](https://valen-team.github.io/zh-Hans/roller/cookbook/wildclawbench-open-models/) · [数据集](https://huggingface.co/datasets/Valen-Team/Roller-WildClawBench-Trajectories) · [Examples](../../README.md)

本目录是一次完整 Roller 运行的配置和命令：GLM-5.3-Flash 和 DeepSeek-V4.1-Flash 在
OpenClaw Harness 中各运行 28 个 WildClawBench 任务，每个任务三次，共 168 条轨迹；评分在任务容器内完成，
run 导出后发布为 Hugging Face 数据集。数据集目前包含前 8 个任务的 44 条轨迹：

| 模型 | Sample 数 | 完成 | 已评分 | 平均分 | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4.1-flash-1017` | 22 | 18 | 22 | 0.306 | 70.5 |
| `glm-5.3-flash` | 22 | 12 | 22 | 0.277 | 35.5 |

## 文件

| 文件 | 内容 |
| --- | --- |
| [`models.json`](models.json) | 两个模型的 OpenClaw 模型配置，地址和 key 由环境变量提供 |
| [`task-ids.txt`](task-ids.txt) | 选中的 28 个 WildClawBench Task ID，每行一个 |
| [`collect.sh`](collect.sh) | 设置 judge 环境变量并执行 `roller collect` |
| [`redact.json`](redact.json) | 导出时追加的脱敏规则，按自己的环境修改 |

## 运行

准备 Roller、Docker、WildClawBench 检出目录及其镜像，见
[WildClawBench 指南](https://valen-team.github.io/zh-Hans/roller/benches/wildclawbench/)。
然后设置模型服务的地址和 key，两个模型由同一个接口提供：

```bash
export WILDCLAWBENCH_ROOT=~/WildClawBench
export ROLLER_INFER_BASE_URL=http://<host>:<port>/v1
export ROLLER_INFER_API_KEY=<key>
```

```bash
# 检查运行计划，不启动容器，也不调用模型
./collect.sh --dry-run

# 采集并评分；中断后用 ./collect.sh --resume 继续
./collect.sh

# 查看结果
roller view runs/wildclawbench-open-models
```

换用其他模型时，修改 `models.json` 中的 provider，以及 `collect.sh` 中的两个 `--model`
和 `JUDGE_MODEL`。

## 发布为数据集

```bash
roller export runs/wildclawbench-open-models \
  --format hf --output dataset --redact-rules redact.json

hf upload <owner>/<name> dataset . --repo-type dataset --exclude "runs/*"
```

导出目录的 `traces/` 中是每个 sample 的 OpenClaw 原生会话，Hub 会把它们识别为 agent traces，在数据集
页面上直接用轨迹查看器展示。导出的文件是脱敏后的：模型端点、主机路径和凭据取值会被替换为占位符。
`redact.json` 追加环境特有的替换，把其中的主机名和权重目录换成自己环境的写法；`forbid`
中的模式在替换后仍有匹配时，导出会停下并指出所在的文件和行。请在采集所用的环境中执行导出，凭据变量在那里有值。
`--exclude` 让包含全部运行文件的 `runs/` 留在本地。导出目录的结构和脱敏的内容见
[Hugging Face 导出](https://valen-team.github.io/zh-Hans/roller/traces/trajectory-outputs/#hugging-face-export)。
