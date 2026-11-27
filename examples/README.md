# Roller Examples

[项目首页](../README_zh-CN.md) · [快速上手](https://valen-team.github.io/zh-Hans/roller/start/getting-started/) · [Task 格式](https://valen-team.github.io/zh-Hans/roller/tasks/task-format/) · [产物说明](https://valen-team.github.io/zh-Hans/roller/traces/trajectory-outputs/)

本目录包含配置模板、task 和运行命令。安装步骤、task 字段和轨迹格式见对应指南。

| 使用目的 | 入口 |
| --- | --- |
| 验证模型服务、Harness 和轨迹采集能否正常运行 | [基础 Task 清单](#基础-task-清单)：4 条文本、文件、Context 和多轮任务 |
| 检查不同输入形式和 Harness 组合的功能 | [Coverage 功能测试](#coverage-功能测试)：按场景选择任务，附离线校验和结果验收 |
| 通过一条命令导出、运行并汇总 benchmark | [Benchmark 一键运行](#benchmark-一键运行)：QwenClawBench 脚本与 YAML 运行配置 |
| 参照一次完整的真实运行 | [实战案例](cookbook/wildclawbench-open-models/README.md)：两个自部署模型在 WildClawBench 上采集、评分，并发布为 Hugging Face 数据集 |

以下命令均在 Roller 项目根目录下执行。

## 目录内容

```text
examples/
├── coverage/                       功能覆盖测试套件
│   ├── README.md / README.en.md     中英文使用说明
│   ├── COVERAGE.md / manifest.json  场景索引和机器可读预期
│   ├── all_tasks.json              合并任务模板
│   ├── config/                     测试用模型与环境配置模板
│   ├── cases/                      按场景与 Harness 分组的清单
│   ├── fixtures/                   已包含的输入小文件
│   ├── negative/                   预期拒绝的输入
│   └── suite.py                    列表、校验、分组运行与验收
├── cookbook/                       实战案例
│   ├── wildclawbench-open-models/  两个自部署模型采集 WildClawBench 的配置和命令
│   ├── vials-stirrup/              Stirrup 采集并评分 VIALS 的配置和命令
│   └── zerobench-claude-code/      Claude Code 在远程 sandbox 采集 ZeroBench 的配置和命令
├── benchmarks/                     `roller bench run` 的 YAML 运行配置
│   ├── base.yaml                   模型连接与执行设置
│   └── <bench>.yaml                各 benchmark 的任务选择、评分环境和 Harness
├── env/
│   └── api.env.example
├── config/
│   ├── openclaw/
│   │   ├── models.json
│   │   ├── models-reasoning.json
│   │   └── models-multimodal.json
│   ├── codex/
│   │   └── config.toml.example
│   └── claude-code/
│       └── settings.json.example
└── tasks/
    ├── tasks.jsonl
    ├── user-simulator.jsonl
    └── files/
        ├── 02_local_markdown/
        │   └── inputs/product-brief.md
        └── 03_user_memory_context/
            └── context/
                ├── USER.md
                └── MEMORY.md
```

## Benchmark 一键运行

项目根目录的脚本依次执行 `bench export`、`collect` 和
`bench report`，退出时自动删除临时 task 清单，并将报告写入
`<OUTPUT_ROOT>/<RUN_NAME>/report.json`：

```bash
# QwenClawBench：纯 automated task 的最小 smoke。
QWENCLAW_ROOT=/path/to/QwenClawBench \
MODEL=local-vllm/qwen3-30b-a3b-local \
MODEL_CONFIG=examples/config/openclaw/models-local-vllm.json \
LOCAL_VLLM_API_KEY=local \
QWENCLAW_REQUIRE_JUDGE=0 \
TASK_IDS=task_00036_find_largest_file_in_downloads_directory \
RUNS_PER_TASK=1 \
./scripts/run-qwenclawbench.sh
```

`HARNESS` 默认为 `openclaw`。使用其他 Harness 时，设置 `API_BASE` 和
`API_KEY_ENV`；后者为保存 Key 的环境变量名，而非 Key 本身。其他通用变量包括
`PARALLEL`、`RETRIES`、`OUTPUT_ROOT`、`RUN_NAME`、`TASK_IDS`、
`CATEGORIES` 以及可选的 `MAX_STEPS`。QwenClawBench 的 `RUNS_PER_TASK`
默认为 3；运行 hybrid task 前必须设置 `QWENCLAW_JUDGE_BASE_URL`、
`QWENCLAW_JUDGE_API_KEY` 和 `QWENCLAW_JUDGE_MODEL`。脚本之后的参数将原样传递
给 `roller collect`，例如 `./scripts/run-qwenclawbench.sh --resume`。

完整的运行与评分约束见
[QwenClawBench 文档](https://valen-team.github.io/zh-Hans/roller/benches/qwenclawbench/)。

### 用 YAML 配置运行

`benchmarks/` 中的 YAML 文件描述同样的 benchmark 运行：`base.yaml` 包含模型连接和执行设置，
其余文件各对应一个 benchmark。`roller bench run` 读取这些文件并启动运行：

```bash
roller bench run --config examples/benchmarks/base.yaml \
  --extra-config examples/benchmarks/qwenclawbench.yaml --dry-run
```

字段和分层规则见 [Benchmark 运行配置](https://valen-team.github.io/zh-Hans/roller/benches/benchmark-standard/)。

## Coverage 功能测试

[完整使用说明](coverage/README.md) · [English quickstart](coverage/README.en.md) · [场景清单](coverage/COVERAGE.md)

Coverage 用于检查清单解析、文件和 Context 输入、Runtime 配置、产物回收、评分，以及多种 Harness 的执行与 Resume 行为。共 1624 条正常任务，分为 406 组，每组 4 条，另有 20 条预期拒绝的输入。

正常任务中，1428 条覆盖 51 种场景在 OpenClaw、Codex、Claude Code、Hermes、OpenCode、DeepSeek、Pi 上的执行；196 条覆盖全部 49 种 Task Agent × User Simulator 配对。

任务以加法、读取小文件、写入短答案为主，对话最多两轮。输入文件已随仓库提供，获取仓库后无需另行准备 `Roller_Data`。

### 先做离线校验

离线校验需要 Python 3.10+，无需 Docker、API Key 或额外 Python 依赖：

```bash
python examples/coverage/suite.py list
python examples/coverage/suite.py validate
python examples/coverage/validate_merged.py

# 检查一组 4 条任务的 CLI 参数、模型路由和文件打包，不请求模型。
python examples/coverage/suite.py run --case formats/jsonl/codex --dry-run
```

将 `--case formats/jsonl/codex` 改为 `--case '*'` 即可遍历全部 406 组。错误输入由校验脚本逐条验证，不作为正常任务提交给模型。

### 配置后运行一组任务

按 [Getting Started](https://valen-team.github.io/zh-Hans/roller/start/getting-started/) 准备 Roller 和所选 Harness 的镜像，然后复制 Coverage 的 [环境配置模板](coverage/config/.env.example)。以下以 Codex 的 4 条加法任务为例：

```bash
cp examples/coverage/config/.env.example examples/coverage/config/.env
# 编辑 .env，填写实际的 ROLLER_TEST_CODEX_MODEL、服务地址和 API Key。
set -a
source examples/coverage/config/.env
set +a

python examples/coverage/suite.py run --case formats/jsonl/codex \
  -- --codex-api-base "$ROLLER_CODEX_BASE_URL" \
     --codex-api-key-env ROLLER_CODEX_API_KEY

python examples/coverage/suite.py check --case formats/jsonl/codex
```

默认并发数为 2，单次超时为 180 秒，重试次数为 0；结果写入 `runs/coverage/formats__jsonl__codex/`。本地 `.env` 和 `runs/` 已被 Git 忽略。其他 Harness、Simulator 配对及结果检查方式见 [Coverage 使用说明](coverage/README.md)。

### 选择任务入口

| 需要验证的内容 | 使用方式 |
| --- | --- |
| 某个输入场景或 Harness | 从 [场景清单](coverage/COVERAGE.md) 选择组名，使用 `suite.py run --case ...` 运行，使用 `suite.py check` 验收；此方式保留原始格式、自定义 key 和特殊 ID 的测试 |
| 一次运行多个 Harness | 使用 [merge.py](coverage/merge.py) 选择场景并绑定实际模型，生成单个 JSON 后传给 `roller collect`；完整模板见 [all_tasks.json](coverage/all_tasks.json) |
| 检查合并运行的结果 | 使用 [report.py](coverage/report.py) 检查采集状态、答案、产物和轨迹；`--fail-on-check` 可用于 CI |
| 检查已有样本能否 Resume | 使用 [audit_resume.py](coverage/audit_resume.py) 执行只读审计；实际恢复时，仍在原采集命令中添加 `--resume` |

总 JSON 中的模型名为占位值，合并时将统一原始清单格式和 ID；本地模型配置与任务副本的生成命令见 [混合 Harness 运行说明](coverage/README.md#4-使用一个总-json-运行混合-harness)。其中 28 条可选视觉任务需要模型、服务与图像工具支持；Hermes、OpenCode、DeepSeek 和 Pi 各 4 条视觉任务用于记录暂未支持的视觉输入；二进制文件读取测试仅验证文件访问能力。

## 基础样例的采集准备

先按照 [Getting Started](https://valen-team.github.io/zh-Hans/roller/start/getting-started/#1-准备运行环境) 安装 Roller，并准备至少一个 Harness 镜像。

复制并编辑本地 API 环境文件：

```bash
cp examples/env/api.env.example examples/env/api.env

set -a
source examples/env/api.env
set +a
```

`examples/env/api.env` 已被 Git 忽略。请勿将真实 Key 写入 JSON、task 或命令行。

## Harness 配置模板

| Harness | 文件 | 用途 |
| --- | --- | --- |
| OpenClaw | `config/openclaw/models.json` | 最小文本模型；model ref 为 `my-openai-proxy/my-model` |
| OpenClaw | `config/openclaw/models-reasoning.json` | GPT Responses 与 Chat Completions reasoning 模型 |
| OpenClaw | `config/openclaw/models-multimodal.json` | 声明 `text` + `image` 输入能力 |
| Codex | `config/codex/config.toml.example` | 可选的原生 `config.toml`；展示 verbosity 和自定义 MCP 写法 |
| Claude Code | `config/claude-code/settings.json.example` | 可选的原生 `settings.json`，不包含 Key、权限放宽或 hook |

运行前，按模型服务的要求填写模板中的 base URL、环境变量和参数值：

- Responses API：`"api": "openai-responses"`；
- Chat Completions：`"api": "openai-completions"`；
- OpenClaw 的 `--model` 必须与 `providers.<name>` 和 `models[].id` 完全对应；
- Codex 和 Claude Code 不读取 OpenClaw models JSON，分别直接使用 Responses 和 Anthropic
  Messages API；二者的 model、endpoint、Key 环境变量和 effort 均通过 CLI 设置；
- `--codex-config` 和 `--claude-settings` 是可选的 Harness 原生配置入口，不是模型注册表。

完整规则见 [OpenClaw Harness 指南](https://valen-team.github.io/zh-Hans/roller/harnesses/openclaw-configuration/#31-模型配置)。

## 基础 Task 清单

[`tasks/tasks.jsonl`](tasks/tasks.jsonl) 每行包含一条 task：

| ID | 输入 | 验证点 |
| --- | --- | --- |
| `01_text_only` | 单轮文本 | 最小 prompt 与文本回复 |
| `02_local_markdown` | `inputs/product-brief.md` | 读取文件并生成 `results/name.txt` |
| `03_user_memory_context` | `USER.md` + `MEMORY.md` | Context 注入和 capture |
| `04_multi_turn` | 两条 `messages` | 同一 Agent session 的预定义多轮 |

这四条 task 可由七种 Harness 运行，可用于检查运行环境。

## OpenClaw：基础采集

### 1. Dry-run

```bash
roller collect \
  --tasks examples/tasks/tasks.jsonl \
  --model my-openai-proxy/my-model \
  --model-config examples/config/openclaw/models.json \
  --parallel 4 \
  --dry-run
```

计划中的 turn 数应依次为 `1、1、1、2`。

### 2. 正式运行

```bash
roller collect \
  --tasks examples/tasks/tasks.jsonl \
  --model my-openai-proxy/my-model \
  --model-config examples/config/openclaw/models.json \
  --parallel 4 \
  --timeout 600 \
  --output runs \
  --run-name examples-openclaw
```

## Codex：运行相同 Task

服务必须支持 Responses 协议，且本地已构建 `roller-codex:0.172.0`：

```bash
roller collect \
  --harness codex \
  --tasks examples/tasks/tasks.jsonl \
  --model gpt-5.6-sol \
  --api-base "$ROLLER_CODEX_BASE_URL" \
  --api-key-env ROLLER_CODEX_API_KEY \
  --codex-config examples/config/codex/config.toml.example \
  --thinking high \
  --reasoning-summary detailed \
  --parallel 2 \
  --timeout 900 \
  --output runs \
  --run-name examples-codex
```

`--codex-config` 可省略；示例文件仅设置 Roller 未覆盖的 text verbosity，并保留一个已注释的
MCP 模板。第四条 task 通过 `codex exec resume --last` 延续 session。镜像、API 和 reasoning
的详细说明见 [Codex Harness 指南](https://valen-team.github.io/zh-Hans/roller/harnesses/codex-configuration/)。

## OpenCode：原生会话与文件工具

按 [OpenCode 指南](https://valen-team.github.io/zh-Hans/roller/harnesses/opencode-configuration/) 构建固定版本镜像，并配置流式 Chat Completions 服务：

```bash
python examples/coverage/suite.py run --case data/txt/opencode \
  -- --opencode-api-base "$ROLLER_OPENCODE_BASE_URL" \
     --opencode-api-key-env ROLLER_OPENCODE_API_KEY
python examples/coverage/suite.py check --case data/txt/opencode
```

支持 Agent、Simulator、原生 session 续接、ACTF/SFT 和 Resume；暂不支持视觉验收。

## DeepSeek Harness：原生 SDK 与持久化 Bash

按 [DeepSeek 指南](https://valen-team.github.io/zh-Hans/roller/harnesses/deepseek-configuration/) 构建镜像，并配置 DeepSeek 兼容服务：

```bash
python examples/coverage/suite.py run --case data/txt/deepseek \
  --deepseek-model deepseek-v4-flash \
  -- --deepseek-api-base "$ROLLER_DEEPSEEK_BASE_URL" \
     --deepseek-api-key-env ROLLER_DEEPSEEK_API_KEY --thinking off
python examples/coverage/suite.py check --case data/txt/deepseek
```

使用 `--case '*/deepseek' --exclude-case 'vision/*'` 选择 200 条非视觉输入任务。涉及 DeepSeek 的 13 种 Simulator 配对共 52 条，可用 `--case 'simulators/pair/deepseek--*' --case 'simulators/pair/*--deepseek'` 选择。

## Pi Agent Harness：原生会话与文件工具

按 [Pi 指南](https://valen-team.github.io/zh-Hans/roller/harnesses/pi-configuration/) 构建镜像，并配置支持工具调用的流式 Chat Completions 服务：

```bash
python examples/coverage/suite.py run --case data/txt/pi \
  --pi-model "$ROLLER_TEST_PI_MODEL" \
  -- --pi-api-base "$ROLLER_PI_BASE_URL" \
     --pi-api-key-env ROLLER_PI_API_KEY --thinking off
python examples/coverage/suite.py check --case data/txt/pi
```

使用 `--case '*/pi' --exclude-case 'vision/*'` 选择 200 条非视觉输入任务。涉及 Pi 的 13 种 Simulator 配对共 52 条，可用 `--case 'simulators/pair/pi--*' --case 'simulators/pair/*--pi'` 选择。多轮对话中保留会话历史和文件；Bash 每次调用均启动独立 Shell。

## Hermes：文本、文件与多轮任务

按 [Hermes 指南](https://valen-team.github.io/zh-Hans/roller/harnesses/hermes-configuration/) 构建镜像并配置 Chat Completions API，然后执行：

```bash
python examples/coverage/suite.py run --case data/txt/hermes \
  --hermes-model your-model-id \
  -- --hermes-api-base "$OPENAI_BASE_URL" --hermes-api-key-env OPENAI_API_KEY
python examples/coverage/suite.py check --case data/txt/hermes
```

Hermes 也可作为 User Simulator。当前固定启用容器内 `terminal`、`file` 工具，支持原生历史恢复、ACTF/SFT 和 Resume；暂不支持视觉工具。

## Claude Code：运行相同 Task

服务必须支持 Anthropic `/v1/messages`，且本地已构建 `roller-claude-code:2.1.332`：

```bash
roller collect \
  --harness claude-code \
  --tasks examples/tasks/tasks.jsonl \
  --model claude-sonnet-4-6-thinking \
  --api-base "$ROLLER_ANTHROPIC_BASE_URL" \
  --api-key-env ROLLER_ANTHROPIC_API_KEY \
  --claude-settings examples/config/claude-code/settings.json.example \
  --thinking high \
  --parallel 2 \
  --timeout 900 \
  --output runs \
  --run-name examples-claude-code
```

`--claude-settings` 可省略；示例文件仅关闭 commit co-author 标记，不改变 permission mode、thinking
或工具行为。第四条 task 通过 `claude --continue` 延续 session。镜像、endpoint 和 thinking
配置见 [Claude Code Harness 指南](https://valen-team.github.io/zh-Hans/roller/harnesses/claude-code-configuration/)。

## Stirrup：agent 在采集进程，容器执行代码

Stirrup 需要 Python 3.12 或更高版本，并安装 `stirrup` extra：

```bash
pip install 'roller-agent[stirrup]'
```

与其他 Harness 不同，agent loop 在采集进程内运行，容器只执行它发出的命令并保存 task 文件，
因此容器拿不到模型 Key，请求同样经过本地录制代理，`--wire-trace full`、`--capture-calls` 和
`--token-ids` 在 Docker 与远程 sandbox 中的行为一致。工具有 `code_exec`、`view_image` 和
`finish`，一个 task 只运行一条 prompt。镜像只需 bash、GNU coreutils 和 task 用到的软件，默认
`python:3.13-slim`，可用 `--stirrup-image` 替换；远程 sandbox 通过
`--sandbox-provider NAME --sandbox-template NAME` 选择。

请求字段由 `--stirrup-*` 选项设置：`--stirrup-max-steps`（默认 30，含 finish 轮）、
`--stirrup-context-length`（默认 128000，历史在 70% 处汇总）、`--stirrup-output-tokens`
（`max_tokens`，默认 8192）、`--stirrup-command-timeout`（默认 300 秒）、`--stirrup-temperature`、
`--stirrup-top-p`、`--stirrup-seed`（每条样本的 seed 由它、task ID 和样本序号推导），以及可重复的
`--stirrup-request-set KEY=JSON`；`--thinking` 写入 `chat_template_kwargs.reasoning_effort`。

完整的一次真实运行见 [VIALS × Stirrup 实战案例](cookbook/vials-stirrup/README.md)：161 道科学
图像问答，每题 3 条样本，容器无网络，报告给出 pass^3、avg@3 和 pass@3。

```bash
cd examples/cookbook/vials-stirrup
export MODEL=glm-5.3-flash
export MODEL_BASE=http://localhost:8000/v1
export JUDGE_BASE_URL=https://api.openai.com/v1
export JUDGE_API_KEY=...

./collect.sh --dry-run
./collect.sh
```

## 进阶案例

### 动态 User Simulator

[`tasks/user-simulator.jsonl`](tasks/user-simulator.jsonl) 中，另一个 OpenClaw Harness 扮演 HR，根据 Task Agent 的当前草稿逐轮补充招聘需求：

```bash
roller collect \
  --tasks examples/tasks/user-simulator.jsonl \
  --model-config examples/config/openclaw/models.json \
  --parallel 1 \
  --timeout 900 \
  --output runs \
  --run-name examples-user-simulator
```

Task 已内置 `model: my-openai-proxy/my-model`。动态对话额外生成 `conversation.json` 和 `user_simulator/` 轨迹；Task Agent 与 Simulator 可在各自字段中选择 OpenClaw、Codex、Claude Code、Hermes、OpenCode、DeepSeek 或 Pi。详见 [User Simulator](https://valen-team.github.io/zh-Hans/roller/tasks/user-simulator/)。

### 多模态输入

使用 [`config/openclaw/models-multimodal.json`](config/openclaw/models-multimodal.json)，并在 task 的 `files.inputs` 中声明本地图像：

```json
{
  "schema_version": "roller.task/v1",
  "id": "inspect_image",
  "file_root": "files/inspect_image",
  "prompt": "查看 inputs/example.png，把描述保存到 outputs/description.md。",
  "files": {
    "inputs": ["inputs/example.png"],
    "output_dir": "outputs"
  }
}
```

```bash
roller collect \
  --tasks your-image-tasks.jsonl \
  --model my-openai-proxy/my-multimodal-model \
  --model-config examples/config/openclaw/models-multimodal.json \
  --parallel 2 \
  --output runs \
  --run-name examples-vision
```

模型配置用于声明图像能力；服务与模型也必须支持对应协议下的视觉输入。

## 实战案例

| 案例 | 内容 |
| --- | --- |
| [wildclawbench-open-models](cookbook/wildclawbench-open-models/README.md) | 两个自部署模型在 WildClawBench 上采集、评分，并发布为 Hugging Face 数据集 |
| [vials-stirrup](cookbook/vials-stirrup/README.md) | Stirrup 用 `code_exec`、`view_image` 和 `finish` 回答 161 道 VIALS 科学图像问答，报告 pass^3、avg@3 和 pass@3 |
| [zerobench-claude-code](cookbook/zerobench-claude-code/README.md) | Claude Code 在无出网 sandbox 中回答 ZeroBench 问题，报告 avg@k 和 pass@k |

每个案例目录都提供 `README.md` 和可直接执行的 `collect.sh`。运行条件、数据下载和评分规则见各自的 README。

## 验收与查看

以 OpenClaw run 为例：

```bash
roller view runs/examples-openclaw
```

重点检查：

- `02_local_markdown` sample 是否包含 `results/name.txt`；
- `04_multi_turn` 是否包含两条 user message；
- 每个成功 sample 是否有 `trace.jsonl`、`trace_ACTF.json`、`trace_SFT.json` 和 `info.json`；
- run 根目录 `sft.jsonl` 是否按 request 顺序汇总成功样本。

编写 task 时参见 [Task](https://valen-team.github.io/zh-Hans/roller/tasks/task-format/)；如需选择训练、调试与审计所需的文件，参见 [轨迹产出文件指南](https://valen-team.github.io/zh-Hans/roller/traces/trajectory-outputs/)。
