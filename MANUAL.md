# 操作手册

> 本文是 [README.md](README.md) 的展开：**想改什么、动哪个文件、怎么加东西**。
> 只想知道怎么跑，看 README 就够了。

## 目录

- [目录结构与职责](#目录结构与职责)
- [想改 X，就动 Y](#想改-x就动-y)
- [两层设计：门面 vs 内核](#两层设计门面-vs-内核)
- [三个钩子](#三个钩子)
- [升级路径](#升级路径)
- [离线自测](#离线自测)
- [设计取舍 / 已知限制](#设计取舍--已知限制)
- [旧版参照](#旧版参照)

---

## 目录结构与职责

```
core/                       内核：与主循环强耦合的两样东西
  protocol.py                 协议契约 + StepResult + 默认文本协议（含 parser）
  loop.py                     ReAct 主循环骨架 + 三个钩子
persona.py                  人格定义 —— 换人格只改这个文件
llm.py                      LLM 基类 + DeepSeekLLM + 密钥加载
tools.py                    Tool 基类 + SearchTool + 注册表工厂
memory.py                   Memory 基类 + WindowMemory + NoMemory
agent.py                    门面：组装 persona/llm/工具/记忆，自动读写记忆
main.py                     CLI + 离线自测
legacy/                     旧单文件版，原样保留作参照
  react_agent_single_file.py
```

**边界是按耦合度切的，不是按文件数。** 循环调 `protocol.parse()` 拿 `StepResult`，
两者接口必须一起改，所以都在 `core/`；而 LLM / 工具 / 记忆 / 人格跟循环只有松耦合
（循环对 LLM 只要求"可调用、收 messages 返 str"），所以各自待在根目录自己的文件里。

## 想改 X，就动 Y

| 想做什么 | 动哪个文件 |
| --- | --- |
| 换人格（角色、语气、职责、用哪些工具、用哪个模型） | [persona.py](persona.py) —— `PERSONA` |
| 加 / 改工具 | [tools.py](tools.py) —— 写 `Tool` 子类，登记进 `ALL_TOOLS`，再填进 `persona.tool_names` |
| 换输出协议（JSON、原生 tool_calls…） | [core/protocol.py](core/protocol.py) —— 写 `Protocol` 子类，赋给 `persona.protocol` |
| 换记忆策略（摘要、落盘、检索） | [memory.py](memory.py) —— 写 `Memory` 子类，传 `ReActAgent(memory=...)` |
| 换循环行为（反思、动态步数、人工确认、工具白名单） | 继承 [core/loop.py](core/loop.py) 的 `ReActLoop` 重写钩子，赋给 `persona.loop_cls` |
| 换模型 / 供应商 | [llm.py](llm.py) —— 写 `LLM` 子类，赋给 `persona.llm` |
| 改主循环本身（很少需要） | [core/loop.py](core/loop.py) |

**加人格的最小改法**：复制整个仓库 → 改 `persona.py` 里的 `PERSONA` → 按需加工具。
内核（`core/`）通常一行都不用动。

### 人格里有什么

```python
@dataclass
class Persona:
    name: str
    system_prompt: str                  # 只写角色与要求，不写输出格式
    tool_names: list[str]               # 名字要能在 tools.ALL_TOOLS 里找到
    llm: LLM
    protocol: Protocol = TextReActProtocol()   # 换协议
    loop_cls: type = ReActLoop                 # 换循环策略
    max_steps: int = 6                         # 每问最大思考步数
    max_rounds: int = 10                       # 记忆保留最近多少轮
```

`system_prompt` 里**不要写格式要求**——那是 `protocol.format_instructions()` 的活。
分开写，换协议时格式说明会自动跟着变，不会漏改。最终拼给模型的 system prompt 是：

```
persona.system_prompt
+ 可用工具清单（从注册表渲染，加工具不用改提示词）
+ protocol.format_instructions(tools)
```

### 加一个工具

```python
# tools.py
class WeatherTool(Tool):
    name = "weather"
    usage_hint = 'weather("城市名")'
    description = "查某个城市的实时天气。"
    def run(self, arg: str) -> str:
        return f"{arg}：晴，23 度"

ALL_TOOLS["weather"] = WeatherTool          # ← 登记一行
```

再把名字填进 `persona.tool_names = ["search", "weather"]`。提示词里的工具清单是渲染出来的，不用手改。

工具参数是**单个字符串**（`run(arg: str) -> str`）。要多参数就往里传 JSON 字符串、
工具自己 `json.loads`——好处是 parser 和内核都不用动。

## 两层设计：门面 vs 内核

```python
# 单 agent 应用：用门面，最省事
from agent import ReActAgent
agent = ReActAgent()
answer, trace = agent.run("上海今天天气")      # 记忆自动读写

# 辩论 / 编排：绕过门面，直接用内核，记忆由你自己调度
from core.loop import ReActLoop
loop = ReActLoop(llm=..., tools=..., protocol=..., system_prompt=...)
answer, trace = loop.run("问题", history=[...])  # 内核不知道"记忆"是什么
```

`core/loop.py` 只收一个 `history` 参数，整份代码里搜不到"记忆"这个词。
所以想让 agent A 看见 agent B 的发言，往 A 的 memory 里塞一条就行，不用碰内核。

**「什么算成功的一轮」的策略只在 [agent.py](agent.py) 一处**：只有模型真正给出
`Final Answer` 的轮次才写进记忆；格式错误、工具失败、步数耗尽的轮次一律不记。

## 三个钩子

`core/loop.py` 的骨架是固定的，想改行为就继承 `ReActLoop` 重写钩子——

```
┌─ on_step_start ─┐  调模型之前。可插消息（Reflection）、改 ctx.max_steps
│   ① 调 LLM      │  （动态步数）、置 ctx.stop 提前结束
│   ② 解析输出     │  ← protocol.parse()
│   ③ final？      │  → 返回
│   ④ malformed？  │  → 喂回纠正提示，下一步重来
│   ⑤ 执行工具     │  ← before_tool 能否决 / 替换这次执行
│   ⑥ 回写历史     │
└─ on_step_end ───┘  Observation 回写之后
```

钩子拿到的是 `LoopContext`：

```python
@dataclass
class LoopContext:
    step: int
    max_steps: int              # 钩子可改，实现「跑到一半提高步数上限」
    messages: list              # 钩子可插可改
    question: str
    trace: list
    stop: bool = False          # 置 True → 本轮结束后收工
    scratch: dict               # 钩子的私有寄存处
```

例子：

```python
class FirstStepReflect(ReActLoop):
    def on_step_start(self, ctx):
        if ctx.step == 1:
            ctx.messages.append({"role": "user", "content": "（反思：先想清楚要什么）"})

class Whitelist(ReActLoop):
    def before_tool(self, ctx, name, arg):
        if name not in ("search",):
            return f"[被拦截] {name} 不在白名单"
        return None            # None = 照常执行
```

挂上去：`Persona(..., loop_cls=FirstStepReflect)`。

`before_tool` 的三种返回值：

| 返回 | 效果 |
| --- | --- |
| `None` | 照常执行工具 |
| 字符串 | **跳过执行**，拿这个字符串当 Observation |
| 抛异常 | 不执行，异常信息作为 Observation 喂回 |

典型用法：人工确认、工具白名单、结果缓存。

## 升级路径

模板只做到"够用"，下面是几种常见需求的加法。**这些都是往上加，不是改内核。**

### 1. 并行工具调用（一回合多个 Action）

现在一次只做一个 Action（parser 只取第一个）。想并行，需要三处一起改：

```python
class ParallelLoop(ReActLoop):
    def execute_actions(self, ctx, actions):        # ← 新增的可重写方法
        with ThreadPoolExecutor(8) as p:
            return list(p.map(lambda a: self._run_tool(ctx, *a), actions))
```

配套改动：
1. `StepResult` 加 `actions: list`；
2. `TextReActProtocol.format_instructions` 允许模型写多个 `Action:`，`parse()` 返回全部；
3. 主循环把 `_run_tool` 那一行换成 `execute_actions(ctx, actions)`；
4. 自测里"一次只做一个 Action"的断言要跟着改。

### 2. 长文本记忆（上下文超长时摘要）

写一个 `Memory` 子类，在 `_trim()` 里先调 LLM 把要丢掉的部分压成要点：

```python
class SummaryMemory(Memory):
    def _trim(self):
        if len(self._msgs) > self.max_rounds * 2:
            old, self._msgs = self._msgs[:-4], self._msgs[-4:]
            self._msgs.insert(0, {"role": "user",
                                  "content": "（更早的对话摘要）" + self._summarize(old)})
```

内核不知道这件事，`ReActAgent(memory=SummaryMemory())` 就生效了。

### 3. 跨会话持久化

同样在 `Memory` 子类里做：`record()` 落盘（JSON / SQLite / 向量库），
`history()` 读回并按需检索。接口不变。

### 4. 多 agent（orchestrator / 辩论）

内核**不管** agent 之间怎么通信，它只保证三件事：**无全局状态、可重入、可多实例共存**。
所以编排逻辑写在你的调用方里：

```python
a = ReActAgent(persona=PERSONA_A, memory=NoMemory())
b = ReActAgent(persona=PERSONA_B, memory=NoMemory())

r1, _ = a.run("任务")
r2, _ = b.run(f"请评估这个结论：{r1}")      # 谁看得到谁的输出，你来定
```

`NoMemory()` 是显式表示"不要记忆"（传 `memory=None` 也是同样效果）。
辩论场景想让 A 看见 B 的发言，把 B 的输出塞进 A 的 memory 里就行——这正是把
"记忆"和"循环"分开的收益。

### 5. 换输出协议

```python
class JsonProtocol(Protocol):
    def format_instructions(self, tools):
        return '输出 JSON：{"thought": "...", "action": {"name": "...", "arg": "..."}}'
    def parse(self, text):
        ...   # 返回 StepResult(kind=..., tool_name=..., tool_arg=...)
```

赋给 `persona.protocol` 即可。**注意**：Observation 的回喂格式
（`f"Observation: {obs}"`）目前写死在 `core/loop.py`，JSON 协议下这里要一起改。

## 离线自测

```bash
python3 main.py --selftest        # 36 项，不联网、不用 key
```

覆盖：parser 全部边界情况（剥围栏、全角冒号/括号、尾随解释、空答案、截断启发式）、
主循环、格式错误纠错、模型调用失败兜底、记忆读写与裁剪、**换人格不碰内核**、
**钩子（注入级 + 否决级）**、无记忆模式。

自测全部用桩 LLM 和桩工具，所以不需要任何 key。加新功能时**先在这里加一条断言**。

## 设计取舍 / 已知限制

* **记忆策略**：只有给出 `Final Answer` 的轮次才写进记忆（见 [agent.py](agent.py)）；
  上下文只保留最近 `max_rounds` 轮，更早的直接丢。
* **Observation 回喂格式写死在循环里**：`core/loop.py` 用 `f"Observation: {obs}"` 回喂。
  文本协议够用，但换成 JSON 协议或原生 `tool_calls` 时这里要一起改。
* **钩子只做到第 2 级**：能注入、能否决，**不能替换工具的执行方式**，所以并行工具
  做不了（见升级路径 1）。
* **工具参数是单个字符串**：文本协议天然只能给一个字符串；好处是 parser 和内核都不用动。
* **解析宽容性**：模型不老实是常态，parser 做了一系列兜底——剥代码围栏、兼容全角
  冒号/括号、`Action` 后跟解释文字会被忽略、单回合内"自己连写多轮"时只取首个可解析的
  `Action`。极少数极端输出仍会触发一次"格式错误"提示让模型重写，`max_steps` 兜底防死循环。
* **Final Answer 截断**：正文之后若出现"真·新回合"标记（如再次 `Thought → Action`），
  只保留第一个 Final 的正文；正文里举例式的 `Action: ...` 行（非函数调用形态）不会被误截断。
* **失败兜底**：工具执行出错、模型调用失败（限流/欠费/网络）都会转成可读提示，交互
  会话不会崩；拆不出的 Action 也作为 Observation 喂回请模型纠正。
* **纯内存**：`WindowMemory` 是进程内的，进程结束即清空（持久化见升级路径 3）。
* **`load_key()` 放在 `llm.py`**：没单独设 `config.py`，所以 [tools.py](tools.py) 取
  Tavily key 时要 `from llm import load_key`。轻微的反向依赖，8 行函数，密钥变多了再提升。

## 旧版参照

[legacy/react_agent_single_file.py](legacy/react_agent_single_file.py) 是重构前的单文件版，
**原样保留、仍可运行**，方便对比"单文件 vs 分层"两种写法：

```bash
PYTHONPATH=. python3 legacy/react_agent_single_file.py --selftest
```

（要 `PYTHONPATH` 是因为旧文件 `import keys` 依赖仓库根在 `sys.path` 上。）
