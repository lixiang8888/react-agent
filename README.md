# ReAct Agent 个人实践

> 复现 ReAct（Reasoning + Acting）范式：**DeepSeek 当大脑、Tavily 当手**。
> 单文件实现，支持联网搜索、多轮记忆与离线自测。

### 快速开始

依赖：**Python ≥ 3.10**（代码用了 `X | None`、`dict[str, Tool]` 这类注解）+ `requests`（唯一第三方依赖）。依赖已声明在 [pyproject.toml](pyproject.toml)，[uv.lock](uv.lock) 锁定了版本。

推荐用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync          # 建 .venv 并按 uv.lock 安装依赖
```

或者只用 pip：

```bash
python3 -m venv .venv
.venv/bin/pip install requests
```

**配置 key**（优先级：环境变量 > 本目录 `keys.py`）：

```bash
export DEEPSEEK_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
# 或在本目录建 keys.py（已 gitignore，模板见下方安全说明）
```

> 安全：`keys.py` 已在 [.gitignore](.gitignore) 中忽略、不会进 git。若曾把真实 key 填进文件并外传过，请到 DeepSeek / Tavily 控制台轮换重置。

**运行**（在 WSL 内；用 uv 时把 `python3` 换成 `uv run python`）：

```bash
python3 react_agent.py "问题"             # 单次提问
python3 react_agent.py --interactive      # 多轮对话（带记忆）
python3 react_agent.py --selftest         # 离线自测（不联网，不用 key）
```

| CLI | 说明 |
| --- | --- |
| `问题`（位置参数） | 单次提问 |
| `-i` / `--interactive` | 多轮对话，可追问上一句；`quit`/`exit` 退出 |
| `--steps N` | 每问最大思考步数（默认 6） |
| `--selftest` | 跑离线自测，桩 LLM/桩工具，不联网不用 key |

### 协议

每轮模型输出**二选一**：

```
Thought: 思考
Action: search("关键词")
```
或
```
Thought: 我已获得足够信息
Final Answer: 给用户的最终回答
```

执行 `Action` 后系统会回一条 `Observation`（搜索结果），模型看到后继续 `Thought → Action` 或直接给 `Final Answer`。约定**一次只做一个 Action**。

### agent架构

##### 1. agent提示词 —— [build_system_prompt()](react_agent.py)

* 定义 agent 的角色
* 工具清单（从注册表自动生成，加工具不用改提示词）
* 严格规定 thought 以及 action 格式
* 调用时喂入上文，让 agent 根据历史记录回答

##### 2. 定义类

###### run 类 —— `ReActAgent.run()`

* 1）拼装 system + 历史 + 提问
* 2）调用 LLM 思考（出错会被兜住，不崩会话）
* 3）解析 LLM 的输出（parse-output）
* 4）执行 action
* 5）将本轮的 action/observation 加入历史记录
* 6）重复 2~5，直到出现 `Final Answer` 或达到 `max_steps`

###### parse-output 类 —— `parse_output()`

分离出 thought 与 action / final answer；剥 ` ``` ` 代码围栏、兼容全角冒号。

###### parse-action 类 —— `parse_action()`

把 `search("关键词")` 进一步拆成 `(工具名, 参数)`。用"平衡括号扫描"，Action 后尾随的解释文字会被忽略，全角括号也能拆。

###### 工具类 —— `Tool` 基类 + `TOOL_REGISTRY` + `execute_tool()`

一个工具清单；加新工具只需 `TOOL_REGISTRY["xxx"] = XxxTool()` 注册一行。

###### 搜索工具 —— `SearchTool`

调 Tavily 的 API 搜全网，返回标题/摘要/来源并截断防撑爆上下文。

（扩展）会话记忆 `Session`：跨问题多轮记忆 + LLM 封装 `deepseek_chat()`。

##### 3. 文件结构（单文件 [react_agent.py](react_agent.py)）

| 段落 | 内容 |
| --- | --- |
| §0 | 密钥加载（环境变量 > keys.py） |
| §1 | 工具：`Tool` / `SearchTool` / `TOOL_REGISTRY` / `execute_tool` |
| §2 | 提示词：`build_system_prompt` |
| §3 | `parse_output` / `_strip_code_fence` |
| §4 | `parse_action` |
| §5 | 主循环：`ReActAgent.run` / `_log_step` |
| §6 | 会话记忆：`Session` |
| §7 | CLI：`main` |
| §8 | 离线自测：`_selftest` |

### 设计取舍 / 已知限制

* **记忆策略**：`Session` 只存"干净的最终问答"，单问的搜索轨迹随 `run` 结束即丢弃；达到最大步数或模型调用失败的结果**不会**写进记忆。上下文只保留最近 `max_rounds` 轮。
* **解析宽容性**：模型不老实是常态，parser 做了一系列兜底——剥代码围栏、兼容全角冒号/括号、`Action` 后跟解释文字会被忽略、单回合内"自己连写多轮"时只取首个可解析的 `Action`。极少数极端输出仍会触发一次"格式错误"提示让模型重写，`max_steps` 兜底防死循环。
* **Final Answer 截断**：正文之后若出现"真·新回合"标记（如再次 `Thought → Action`），只保留第一个 Final 的正文；正文里举例式的 `Action: ...` 行（非函数调用形态）不会被误截断。
* **失败兜底**：工具执行出错、模型调用失败（限流/欠费/网络）都会转成可读提示，交互会话不会崩；`parse_action` 拆不出的内容也作为 Observation 喂回请模型纠正。
* **纯内存**：单文件实践版，进程结束记忆即清空，无持久化。

### 相关文件

* [react_agent.py](react_agent.py) —— 全部逻辑（工具、提示词、parser、主循环、记忆、CLI、自测）
* [pyproject.toml](pyproject.toml) —— 项目元信息与依赖声明
* [uv.lock](uv.lock) —— uv 锁定的依赖版本（勿手改）
* [keys.py](keys.py) —— 本地密钥模板（已 gitignore，**勿提交**；推荐改用环境变量）
* [.gitignore](.gitignore) —— 忽略 `keys.py`、`.env`、`.venv/`、`__pycache__/` 等
