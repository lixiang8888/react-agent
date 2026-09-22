# ReAct Agent 个人实践

> 复现 ReAct（Reasoning + Acting）范式：**DeepSeek 当大脑、Tavily 当手**。

一个分层实现的 ReAct agent：模型先思考、再决定调哪个工具、看到结果后继续思考，
直到能给出最终回答。支持联网搜索、多轮记忆、离线自测。

**这个仓库的定位是母本**——后续需要新的 agent 时，复制出去改几个文件就是另一个人格。
所以代码的边界是按"改起来方不方便"来划的，不是按"跑起来对不对"来划的。

它的两个设计目标：

1. **换人格不用改内核**。角色提示词、工具集、输出协议、循环策略、模型参数，
   全部从 [persona.py](persona.py) 声明，`core/` 一行都不用动。
2. **能往上加东西**。长文本记忆、多 agent 编排这类需求，是靠写新的子类挂上去的，
   不是改进内核里。

想动手改东西，看 **[MANUAL.md](MANUAL.md)**（操作手册：想改 X 就动哪个文件、
钩子怎么用、并行/持久化/多 agent 怎么升级）。

### 快速开始

依赖：**Python ≥ 3.10** + `requests`（唯一第三方依赖）。

```bash
uv sync          # 建 .venv 并按 uv.lock 安装依赖
```

配 key（优先级：环境变量 > 本目录 `keys.py`）：

```bash
export DEEPSEEK_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
```

跑起来：

```bash
python3 main.py "问题"             # 单次提问
python3 main.py --interactive      # 多轮对话（带记忆）
python3 main.py --selftest         # 离线自测（不联网，不用 key）
```

> 安全：`keys.py` 已在 [.gitignore](.gitignore) 中忽略、不会进 git。若曾把真实 key 填进文件并外传过，请到 DeepSeek / Tavily 控制台轮换重置。

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

执行 `Action` 后系统会回一条 `Observation`，模型看到后继续 `Thought → Action` 或直接给
`Final Answer`。约定**一次只做一个 Action**。

输出格式和解析逻辑打包在同一个协议对象里（[core/protocol.py](core/protocol.py)），
所以换协议时格式说明和 parser 不会脱节。

### 结构一览

```
core/          内核：协议契约 + 主循环与钩子（换人格通常不用动这里）
persona.py     人格定义 —— 换人格只改这个文件
llm.py         LLM 基类 + DeepSeek 实现
tools.py       工具基类 + 搜索工具 + 注册表工厂
memory.py      记忆基类 + 滑动窗口实现
agent.py       门面：组装上述部件，自动读写记忆
main.py        CLI + 离线自测
legacy/        重构前的单文件版，原样保留作参照
```

内核和外围的分界线是**耦合度**：循环调 `protocol.parse()` 拿结果，两者接口必须一起改，
所以都在 `core/`；LLM / 工具 / 记忆 / 人格跟循环只有松耦合，各自待在根目录自己的文件里。

### 文档

| 文件 | 内容 |
| --- | --- |
| 本文件 | 定位、快速开始、协议、结构一览 |
| [MANUAL.md](MANUAL.md) | 操作手册：改哪里、钩子、升级路径、设计取舍 |
