# SWE-Core

**一个以 Shell 为执行接口、以可读控制流为核心的轻量软件工程 Agent。**

SWE-Core 将编程任务组织成一个直接的循环：模型读取任务和消息历史，生成 Shell 命令，执行环境返回结果，模型据此继续修改和验证，直到结束任务。它可以用于本地代码修改、问题排查，也适合作为研究 Agent 行为、提示词和模型适配的基础实现。

项目提供命令行交互、Python 组件接口、Docker 执行环境，以及可逐步浏览的 JSON 运行轨迹。核心代码位于 [`src/swe_core`](src/swe_core)，要求 **Python 3.10+**。

## 设计理念

### 用 Shell 连接模型与工程工具

读取文件、搜索代码、修改内容和运行测试，都通过同一个 `bash` 工具完成。模型可以组合环境中已有的命令，框架负责解析动作、执行命令和反馈结果。

当前模型适配器基于 LiteLLM 的 **Chat Completions 与 tool calling**：每次有效响应需要包含至少一个 `bash` tool call。因此，接入的模型与服务需要支持这一调用方式。

### 让控制流保持清晰

`DefaultAgent` 围绕 `query → execute_actions → observation` 展开。消息按执行顺序追加到历史中，下一轮模型调用继续使用这些上下文；内部附加的统计和调试字段在发送 API 前移除。

这种结构便于检查模型看到了什么、执行了什么，以及执行结果如何影响下一步决策。格式错误会作为反馈进入后续上下文，并受连续错误次数限制。

### 分离决策、模型适配与执行环境

| 组件 | 职责 | 当前实现 |
| --- | --- | --- |
| Agent | 控制循环、消息历史、限制检查与轨迹保存 | `DefaultAgent`、`InteractiveAgent` |
| Model | 请求模型、解析动作、格式化执行反馈 | `LitellmModel`，以及用于离线验证的确定性模型 |
| Environment | 执行命令、返回输出与退出码、识别任务结束标记 | `LocalEnvironment`、`DockerEnvironment` |

三类组件通过 [`Protocol` 接口](src/swe_core/__init__.py)约定行为，工厂支持短名称与完整 Python 导入路径。更换模型适配器或执行环境时，可以复用 Agent 控制循环。

### 命令独立执行，文件修改持续保留

每个动作启动新的 Shell 进程；Docker 模式则在同一个持续运行的容器中启动新的命令进程。文件修改会保留，但 `cd` 和临时 `export` 不会跨动作继承。需要指定目录时，应配置 `environment.cwd`，或在同一条命令中使用 `cd /path && ...`。

### 将运行过程作为可检查的产物

配置输出路径后，Agent 在每步结束时保存轨迹，覆盖正常执行、任务结束、限制退出和循环内异常路径。文件通过同目录临时文件与原子替换写入，便于排查失败原因、比较运行过程和复盘模型行为。

## 快速开始

在仓库根目录安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

以下命令以 Linux、macOS 或 WSL 的 Shell 为例。配置模型名称及对应服务的凭据：

```bash
export OPENAI_API_KEY='<your-api-key>'
export SWE_CORE_MODEL='openai/<model-name>'
```

将占位符替换为实际凭据和支持 tool calling 的模型。进入需要处理的项目目录后运行：

```bash
core --task '检查当前项目，修复失败测试，并运行相关测试验证修改' \
  --output ./run.traj.json
```

默认使用本地执行环境和 `confirm` 模式，在执行模型命令前请求确认。也可以不传任务，在终端交互输入。

以下入口等价：

```bash
core --help
SWE-Core --help
python -m swe_core --help
```

无人值守运行时，显式关闭命令确认与结束确认：

```bash
core --task '创建 hello.txt，写入 hello 并验证内容' \
  --output ./hello.traj.json \
  --yolo --exit-immediately
```

这里的“提交结果”表示 Agent 结束任务。内置提示词使用 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 作为结束标记，该标记本身不会执行 Git commit 或 push。

## 交互模式

`InteractiveAgent` 在基础循环之上增加终端交互，可以在输入提示处切换模式：

| 模式 | 行为 | 切换命令 |
| --- | --- | --- |
| `confirm` | 执行模型命令前确认；可配置正则白名单 | `/c` |
| `yolo` | 自动执行模型命令 | `/y` |
| `human` | 直接执行用户输入的命令 | `/u` |

命令确认和任务结束确认分别控制：`--yolo` 关闭前者，`--exit-immediately` 关闭后者。

## 配置

默认配置位于 [`src/swe_core/config/core.yaml`](src/swe_core/config/core.yaml)，包含 Jinja2 提示词模板、反馈格式、交互模式和运行限制。

配置支持 YAML 文件、内置配置名称和点分隔的 `key=value`：

```bash
core -c core.yaml \
  -c agent.step_limit=30 \
  -c agent.wall_time_limit_seconds=600 \
  -c environment.timeout=60 \
  --cost-limit 1.0 \
  --task '检查并修复失败测试'
```

**指定 `-c` 后，默认配置不会自动加载。** 需要继承默认提示词时，应显式保留 `-c core.yaml`。多个配置从左到右递归合并，显式 CLI 参数最后覆盖。

| 配置项 | 默认值 | 含义 |
| --- | --- | --- |
| `agent.step_limit` | `0` | 模型查询次数上限，`0` 表示禁用 |
| `agent.cost_limit` | `3.0` | 按 LiteLLM 估算的美元成本上限，`0` 表示禁用 |
| `agent.wall_time_limit_seconds` | `0` | 总运行时间限制，`0` 表示禁用 |
| `agent.max_consecutive_format_errors` | `3` | 连续格式错误上限 |
| `environment.timeout` | `30` | 单条命令的等待超时，单位为秒 |

Agent 的步数、费用和总时限在模型查询前检查；费用可能因最后一次请求而超过设定值，总时限也不会立即中断正在进行的模型请求或命令。

### 自定义模型服务

例如，将以下内容保存为 `model.yaml`，为 OpenAI-compatible 服务指定 API 地址：

```yaml
model:
  model_class: litellm
  model_name: openai/<model-name>
  model_kwargs:
    api_base: https://your-provider.example/v1
    max_tokens: 4096
```

```bash
core -c core.yaml -c model.yaml --task '分析当前项目的入口与模块结构'
```

凭据通过环境变量提供。若服务缺少 LiteLLM 所需的计价信息，可设置 `model.cost_tracking=ignore_errors`；此时计价失败会记录为 `0`，费用限制不能可靠反映实际支出。

### 项目环境变量

| 环境变量 | 用途 |
| --- | --- |
| `SWE_CORE_MODEL` | YAML 和 CLI 均未指定模型时使用的模型名称 |
| `SWE_CORE_DEFAULT_CONFIG_PATH` | 替换默认配置文件 |
| `SWE_CORE_CONFIG_DIR` | 配置文件的附加搜索目录 |
| `SWE_CORE_GLOBAL_CONFIG_DIR` | 用户级配置、交互历史和默认轨迹的存放目录 |
| `SWE_CORE_DOCKER_EXECUTABLE` | Docker 兼容命令的可执行文件路径 |
| `SWE_CORE_INSPECTOR_STYLE_PATH` | 轨迹浏览器的自定义样式文件 |

## Docker 执行环境

模型请求仍由宿主机上的 Python 进程发出，Agent 命令在容器内执行。将以下内容保存为 `docker.yaml`，并将挂载源替换为项目的绝对路径：

```yaml
environment:
  environment_class: docker
  image: python:3.12-slim
  cwd: /workspace
  timeout: 60
  run_args:
    - --rm
    - --network=none
    - --volume
    - /absolute/path/to/project:/workspace
```

```bash
core -c core.yaml -c docker.yaml \
  --task '检查 /workspace 中的代码并运行已有测试' \
  --output ./docker-run.traj.json
```

需要可用的 Docker 服务，镜像内需具备 Bash 和任务依赖。示例关闭容器网络，因此测试依赖应预先装入镜像；挂载目录中的修改会同步到宿主机。

## Python 接口

可以直接组合组件，在脚本中运行任务。以下示例复用内置提示词，并为基础 Agent 设置次数上限和输出路径：

```python
import os

from swe_core.agents.default import DefaultAgent
from swe_core.config import get_config_from_spec
from swe_core.environments.local import LocalEnvironment
from swe_core.models.litellm_model import LitellmModel

config = get_config_from_spec("core.yaml")
model = LitellmModel(
    **(config["model"] | {"model_name": os.environ["SWE_CORE_MODEL"]})
)
env = LocalEnvironment(**config["environment"])
agent = DefaultAgent(
    model,
    env,
    **(config["agent"] | {
        "step_limit": 20,
        "output_path": "./python-run.traj.json",
    }),
)

result = agent.run("创建 hello.txt，写入 hello 并验证内容")
print(result["exit_status"])
```

`DefaultAgent` 自动执行命令；终端确认、模式切换等能力由 `InteractiveAgent` 提供。

## 轨迹记录与浏览

CLI 默认将最近一次运行保存到系统用户配置目录下的 `SWE-Core/last_core_run.traj.json`。使用 `--output` 为不同任务分别指定文件，可以保留多次运行记录。

轨迹采用 UTF-8 JSON，格式标识为 `SWE-Core-1.1`，包含配置、模型调用统计、消息历史、退出状态和提交内容。模型消息还可包含原始响应、成本和时间戳等元数据。

在真实终端中打开单个文件或整个轨迹目录：

```bash
core-extra inspect ./run.traj.json
core-e inspect ./trajectories/
```

浏览器基于 Textual，支持消息与工具输出浏览、运行概览、逐消息元数据，以及已有 reasoning 内容的显示。目录模式递归查找 `*.traj.json`。

| 按键 | 操作 |
| --- | --- |
| `h` / `l` 或左右方向键 | 上一步 / 下一步 |
| `0` / `$` | 第一步 / 最后一步 |
| `H` / `L` | 上一份 / 下一份轨迹 |
| `j` / `k` 或上下方向键 | 滚动 |
| `r` | 切换已有 reasoning 内容的显示 |
| `m` | 切换概览和元数据的显示 |
| `R` | 从磁盘重新加载 |
| `e` / `E` | 使用外部 jless 查看当前步骤 / 完整 JSON |
| `q` | 退出 |

常规浏览无需 jless；仅 `e` / `E` 需要另行安装该工具。未记录的统计字段会显示为“未记录”。

## 源码导航

| 路径 | 内容 |
| --- | --- |
| [`agents/`](src/swe_core/agents) | 基础控制循环、交互模式与用户输入 |
| [`models/`](src/swe_core/models) | LiteLLM 适配、动作解析、重试和确定性测试模型 |
| [`environments/`](src/swe_core/environments) | 本地及 Docker 命令执行 |
| [`config/`](src/swe_core/config) | 配置加载、提示词和浏览器样式 |
| [`run/core.py`](src/swe_core/run/core.py) | CLI 参数、配置合并与组件装配 |
| [`run/utilities/`](src/swe_core/run/utilities) | 辅助命令与轨迹浏览器 |
| [`paths.py`](src/swe_core/paths.py) | 用户配置和运行产物路径 |
| [`exceptions.py`](src/swe_core/exceptions.py) | 格式错误、提交和限制退出的控制流异常 |
| [`utils/`](src/swe_core/utils) | 序列化、配置合并与日志工具 |

建议从 `run/core.py` 了解装配过程，再阅读 `agents/default.py` 的执行循环，最后查看模型和环境实现。

## 开发与使用边界

安装开发依赖后运行测试或构建发行包：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

当前模型适配以文本和 bash tool calling 为主，尚未提供专门的多模态或 Responses API 适配。

本地环境以当前用户权限执行命令；Docker 的实际隔离范围取决于挂载与运行参数。轨迹会保存任务、配置和命令输出，分享前应检查其中的凭据或私有内容。

## 参考与致谢

项目的极简 Agent 设计参考了 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent/blob/main/README.md)，尤其是以 Shell 为主要执行接口、保持线性消息历史和分离执行环境的思路。本 README 中的功能与使用方式以 SWE-Core 当前源码为准。
