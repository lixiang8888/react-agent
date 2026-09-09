# -*- coding: utf-8 -*-
"""
ReAct Agent —— 个人实践（单文件版）
====================================

复现 ReAct（Reasoning + Acting）范式：DeepSeek 当大脑、Tavily 当手。
README 里的逻辑模块在此文件中的落点：

    README 模块      →  本文件
    -------------------------------
    agent 提示词     →  SYSTEM_PROMPT / build_system_prompt()
    run 类           →  ReActAgent.run()   （思考→行动→观察 主循环）
    parse-output 类  →  parse_output()
    parse-action 类  →  parse_action()
    工具类(清单管理) →  Tool 基类 + TOOL_REGISTRY + execute_tool()
    搜索工具         →  SearchTool(Tavily)
    （扩展）会话记忆 →  Session              （跨问题的多轮记忆）
    （支撑）LLM 封装 →  deepseek_chat()

结构化协议（每轮 LLM 输出二选一）：
    Thought: 思考
    Action: search("关键词")
 或
    Thought: 思考
    Final Answer: 最终回答

运行（在 WSL 内）：
    python3 react_agent.py "问题"          # 单次提问
    python3 react_agent.py --interactive   # 多轮对话（带记忆）
    python3 react_agent.py --selftest      # 离线自测（不联网，不用 key）
"""

import ast
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台保险
except Exception:
    pass

import requests

# ---------------------------------------------------------------------------
# 0. 密钥加载：环境变量优先，其次读 keys.py（已 gitignore）
# ---------------------------------------------------------------------------

def _load_key(env_name: str, key_module_attr: str) -> str:
    val = os.environ.get(env_name, "").strip()
    if not val:
        try:
            import keys as _keys
            val = str(getattr(_keys, key_module_attr, "") or "").strip()
        except ImportError:
            pass
    return val

DEEPSEEK_API_KEY = _load_key("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY")
TAVILY_API_KEY = _load_key("TAVILY_API_KEY", "TAVILY_API_KEY")

DEEPSEEK_BASE = "https://api.deepseek.com"
TAVILY_BASE = "https://api.tavily.com"

HTTP_TIMEOUT = (10, 60)   # (连接超时, 读取超时) 秒


def _post_json(url: str, headers: dict, body: dict) -> dict:
    """通用 POST JSON，非 2xx 抛带状态码异常。"""
    resp = requests.post(url, headers=headers, json=body, timeout=HTTP_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _http_headers() -> dict:
    return {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}


def deepseek_chat(messages: list, temperature: float = 0.3, max_tokens: int = 600) -> str:
    """调用 DeepSeek chat（OpenAI 兼容）。返回 assistant 文本。"""
    body = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = _post_json(f"{DEEPSEEK_BASE}/chat/completions", _http_headers(), body)
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 1. 工具类：一个工具清单（Tool 基类 + 注册表），加新工具只需注册一行
# ---------------------------------------------------------------------------

class Tool:
    """工具基类。run() 的返回值直接当作 Observation 喂回给模型。"""
    name: str = ""
    usage_hint: str = ""          # 例如 search("关键词")
    description: str = ""

    def run(self, arg: str) -> str:
        raise NotImplementedError


class SearchTool(Tool):
    """搜索工具：调 Tavily API 搜全网。"""
    name = "search"
    usage_hint = 'search("关键词")'
    description = "联网搜索，返回若干条网页标题、摘要与来源链接。用于需要最新、实时或我知识之外的信息。"

    def run(self, query: str) -> str:
        if not TAVILY_API_KEY:
            return "错误：未配置 Tavily API key（环境变量 TAVILY_API_KEY 或 keys.py）"
        body = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
        }
        data = _post_json(f"{TAVILY_BASE}/search",
                          {"Content-Type": "application/json"}, body)
        results = data.get("results", [])
        if not results:
            return "搜索无结果，请换一个更具体的关键词重试。"

        lines = []
        for i, r in enumerate(results[:5], 1):
            title = (r.get("title") or "").strip()
            content = (r.get("content") or "").strip()[:200]
            url = (r.get("url") or "").strip()
            lines.append(f"{i}. {title}\n   {content}\n   来源: {url}")
        obs = "\n".join(lines)
        return obs[:2000]   # 截断，防 Observation 撑爆上下文


TOOL_REGISTRY: dict[str, Tool] = {"search": SearchTool()}
# 以后加工具：TOOL_REGISTRY["weather"] = WeatherTool()


def execute_tool(name: str, arg: str, registry: dict | None = None) -> str:
    """执行工具（若名字非法返回错误 Observation，让模型自己纠正）。

    registry 缺省为全局 TOOL_REGISTRY；传入桩工具注册表即可离线测试。
    """
    registry = TOOL_REGISTRY if registry is None else registry
    tool = registry.get(name)
    if tool is None:
        available = ", ".join(t.usage_hint for t in registry.values())
        return f"错误：未知工具 {name!r}。可用工具格式：{available}"
    try:
        return tool.run(arg)
    except Exception as e:      # 网络/限流/欠费都兜住，转成 Observation
        return f"工具 {name} 执行出错：{e}。请换措辞重试，或直接基于已有信息作答。"


# ---------------------------------------------------------------------------
# 2. agent 提示词：角色 + 工具清单（自动生成）+ 严格格式
# ---------------------------------------------------------------------------

def build_system_prompt(registry: dict[str, Tool] | None = None) -> str:
    registry = registry or TOOL_REGISTRY
    first = next(iter(registry.values()))
    tools_desc = "\n".join(f"- {t.usage_hint}：{t.description}"
                           for t in registry.values())
    return f"""你是「ReAct Agent —— 个人实践（单文件版）」的智能体，用中文回答问题。

可用工具：
{tools_desc}

你必须严格按下面的格式思考与行动，不要输出额外内容：

Thought: 你针对当前问题的思考，说明下一步做什么
Action: {first.usage_hint}

每执行一次 Action，系统会返回一条 Observation（结果）。
看到 Observation 后继续输出 Thought -> Action，直到信息足够，
最后一行不再调用工具，输出：

Thought: 我已获得足够信息
Final Answer: 给用户的最终回答

要求：
1. Action 参数必须是双引号包裹的字符串字面量，例如 {first.usage_hint}。
2. 一次只做一个 Action，不要连续输出多个 Action。
3. 不确定、过时、无法凭记忆回答的信息就调用 search，不要编造。
4. 回答尽量带上关键来源。"""


# ---------------------------------------------------------------------------
# 3. parse-output：把模型一回合的输出拆成 thought + (action | final)
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    """剥掉模型偶尔包的 markdown 代码围栏 ```。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_output(text: str) -> dict:
    """
    返回 {"kind": "action"|"final"|"malformed", "thought": str|None, "payload": str}
    kind=action → payload 为 `search("...")` 这一行；final → payload 为回答正文。
    """
    text = _strip_code_fence(text or "")

    # 归一化全角冒号，兼容 Thought：/ Action：/ Final Answer：
    text = re.sub(r"(?i)(Thought|Final Answer|Action|Answer)\s*[：:]\s*",
                  lambda m: m.group(1) + ": ", text)

    # 找“最先出现”的结构标记：Thought 之后要么 Action、要么 (Final) Answer。
    marker_re = re.compile(r"(?i)\b(?:Action|Final Answer|(?<!Final\s)Answer)\s*:")
    matches = list(marker_re.finditer(text))
    if not matches:
        return {"kind": "malformed", "thought": None, "payload": text}
    m = matches[0]
    marker = m.group(0).lower()
    is_action = marker.startswith("action")
    payload = text[m.end():].strip()

    # 取出该标记之前、最后一个 Thought: 之后的内容作为思考
    thought = None
    thought_hits = list(re.finditer(r"(?i)\bThought\s*:", text))
    if thought_hits and thought_hits[-1].end() <= m.start():
        thought = text[thought_hits[-1].end():m.start()].strip() or None

    if is_action:
        if not payload:
            return {"kind": "malformed", "thought": thought, "payload": text}
        return {"kind": "action", "thought": thought, "payload": payload}

    # final：回答正文默认读到结尾，但若之后还有"真·新回合"的协议标记则截断。
    # 启发式（正文里举例式的 Action:/Observation: 不该被误当回合头）：
    #   尾部 ≥2 个标记  → 判为链式回合，在第一个处截断；
    #   尾部只有 1 个 Action 标记 → 仅当该行形如 search(...) 调用才截断；
    #   尾部只有 1 个 Thought/Final Answer/Observation 标记 → 截断；
    #   尾部无标记 → 整段作为答案保留。
    tail = text[m.end():]
    tail_markers = list(re.finditer(
        r"(?m)^\s*(?:Thought|Action|Final Answer|Observation)\s*[:：]", tail))
    cut = None
    if len(tail_markers) >= 2:
        cut = tail_markers[0].start()
    elif len(tail_markers) == 1:
        line = tail[tail_markers[0].start():].splitlines()[0].lstrip() \
            if tail[tail_markers[0].start():] else ""
        if re.match(r"(?i)^(Thought|Final Answer|Observation)\b", line):
            cut = tail_markers[0].start()
        else:                                   # 剩下的单标记是 Action
            rest = re.sub(r"(?i)^Action\s*[:：]\s*", "", line, count=1)
            if re.match(r"^[A-Za-z_]\w*\s*[(（]", rest):
                cut = tail_markers[0].start()
    payload = tail[:cut].strip() if cut is not None else tail.strip()
    if not payload:
        return {"kind": "malformed", "thought": thought, "payload": text}
    return {"kind": "final", "thought": thought, "payload": payload}


# ---------------------------------------------------------------------------
# 4. parse-action：把 `search("关键词")` 拆成 (工具名, 参数)
# ---------------------------------------------------------------------------

def parse_action(action_text: str) -> tuple[str | None, str]:
    """
    拆 `search("关键词")` → ("search", "关键词")。

    用"平衡括号扫描"解析：从工具名后第一个括号开始，跳过引号内的内容、
    计数括号直到归零；**括号之后的解释性文字一律忽略**（模型常在 Action 后
    追加一句说明）。全角括号（）先归一化成半角。参数是字符串字面量时用
    ast.literal_eval 安全求值（兼容单引号/多行/转义）。
    拆不出来返回 (None, 原文)，由调用方转成"无法解析"的 Observation。
    """
    text = (action_text or "").strip().replace("（", "(").replace("）", ")")
    m = re.match(r"^([A-Za-z_]\w*)\s*\(", text)
    if not m:
        return None, action_text
    name = m.group(1)

    # 逐字符扫描：跳过引号包裹的内容，括号配对归零即找到参数结尾
    i = m.end()
    depth = 1
    quote: str | None = None
    j = i
    while j < len(text) and depth > 0:
        ch = text[j]
        if quote:
            if ch == quote and text[j - 1] != "\\":   # 忽略 \" 这类转义引号
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        j += 1
    if depth != 0:                  # 括号不平衡，无法可靠解析
        return None, action_text

    raw = text[i:j - 1].strip()
    arg = raw
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, str):
            arg = parsed
    except Exception:
        pass
    return name, arg


# ---------------------------------------------------------------------------
# 5. run 类：主循环（思考 -> 行动 -> 观察 -> … -> 最终回答）
# ---------------------------------------------------------------------------

class ReActAgent:
    def __init__(self, llm=deepseek_chat, registry: dict | None = None,
                 max_steps: int = 6, verbose: bool = True):
        # 只有真正会用默认 deepseek_chat 时才需要 key；注入桩 llm（自测）可无 key
        if llm is deepseek_chat and not DEEPSEEK_API_KEY:
            raise RuntimeError(
                "未配置 DeepSeek API key：请设置环境变量 DEEPSEEK_API_KEY，或在本目录放 keys.py")
        self.llm = llm
        self.registry = registry or TOOL_REGISTRY
        self.max_steps = max_steps
        self.verbose = verbose
        self.system_prompt = build_system_prompt(self.registry)

    # ---- 日志（verbose 时边跑边打印，方便看循环） ----
    def _log_step(self, step: int, thought: str | None, action_disp: str,
                  observation: str | None, max_steps: int):
        print(f"\n[Step {step}/{max_steps}]")
        if thought:
            print(f"  Thought     : {thought}")
        if action_disp:
            print(f"  Action      : {action_disp}")
        if observation is not None:
            obs = observation.replace("\n", "\n               ")
            print(f"  Observation : {obs[:600]}")   # 只展示前 600 字符

    def run(self, question: str, session_messages: list | None = None,
            max_steps: int | None = None) -> tuple[str | None, list]:
        """
        README run 类的落地。返回 (最终回答, trace)。
        - session_messages：多轮记忆前缀（见 Session），run 本身不保存记忆。
        - trace：每一步的 (thought, action, observation) 列表，供展示/测试。
        """
        max_steps = self.max_steps if max_steps is None else max_steps
        messages = [{"role": "system", "content": self.system_prompt}]
        if session_messages:
            messages += list(session_messages)
        messages.append({"role": "user", "content": question})

        trace: list[dict] = []
        last_reply = ""

        for step in range(1, max_steps + 1):
            # ② 调 LLM 思考
            try:
                reply = (self.llm(messages) or "").strip()
            except Exception as e:      # 网络/限流/欠费别让整个会话崩掉
                err = f"[模型调用失败] {e}，可稍后重试。"
                trace.append({"type": "error", "error": str(e)})
                if self.verbose:
                    print(f"\n[!] {err}")
                return err, trace
            last_reply = reply
            if not reply:
                if self.verbose:
                    print("\n[!] 模型返回空内容，终止。")
                break

            # ③ 解析输出（parse-output）
            parsed = parse_output(reply)
            kind, thought, payload = parsed["kind"], parsed["thought"], parsed["payload"]

            if kind == "final":
                trace.append({"type": "final", "thought": thought, "answer": payload})
                if self.verbose:
                    self._log_step(step, thought, "Final Answer", None, max_steps)
                return payload, trace

            if kind == "malformed":
                # 兜底：把“格式错误”也当 Observation 喂回，请模型重写（有 max_steps 硬上限防死循环）
                obs = "解析不到 Action 或 Final Answer。请严格按格式输出一行 Action，或直接输出 Final Answer。"
                trace.append({"type": "malformed", "reply": reply})
                if self.verbose:
                    self._log_step(step, thought, "(格式错误，要求重写)", obs, max_steps)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
                continue

            # kind == "action"：④ 拆解并执行 action
            name, arg = parse_action(payload)
            if name is None:
                obs = f'无法解析 Action：{payload!r}。格式应为 search("关键词")，且一次只做一个 Action。'
            else:
                obs = execute_tool(name, arg, self.registry)   # 工具类执行（默认走 Tavily 搜索）

            action_disp = payload if name is None else f'{name}({arg!r})'
            trace.append({"type": "action", "step": step, "thought": thought,
                          "action": action_disp, "action_raw": payload,
                          "observation": obs})
            if self.verbose:
                self._log_step(step, thought, action_disp, obs, max_steps)

            # ⑤ 把本轮 assistant 回复 + Observation 追加进历史，进入下一轮
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        # 达到 max_steps 还没 Final Answer：兜底返回最后一段内容
        fallback = f"⚠ 已达最大步数({max_steps})仍无最终答案。最后给出的内容：\n{last_reply or '(空)'}"
        trace.append({"type": "fallback", "answer": fallback})
        if self.verbose:
            print(f"\n[!] {fallback}")
        return fallback, trace


# ---------------------------------------------------------------------------
# 6. Session：多轮对话记忆（扩展）
#    记忆只存“问题 -> 最终回答”，单问题的搜索轨迹随 run 结束即丢弃。
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, agent: ReActAgent | None = None, max_rounds: int = 10):
        self.agent = agent or ReActAgent()
        self.memory: list[dict] = []     # 交替的 user/assistant Q&A
        self.max_rounds = max_rounds

    def _trim(self):
        """上下文增长控制：只保留最近 max_rounds 轮 Q/A。"""
        limit = self.max_rounds * 2
        while len(self.memory) > limit:
            self.memory.pop(0)

    def ask(self, question: str) -> tuple[str | None, list]:
        answer, trace = self.agent.run(question, session_messages=self.memory)
        # 只回写"真正给出最终答案"的轮次：兜底文本 / 模型调用失败 / 空答案不入记忆
        if answer and trace and trace[-1].get("type") == "final":
            self.memory.append({"role": "user", "content": question})
            self.memory.append({"role": "assistant", "content": answer})
            self._trim()
        return answer, trace


# ---------------------------------------------------------------------------
# 7. CLI 入口
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="react_agent", description="ReAct 智能体：DeepSeek 思考 + Tavily 搜索")
    parser.add_argument("question", nargs="?", help="单次提问内容")
    parser.add_argument("-i", "--interactive", action="store_true", help="多轮对话（带记忆）")
    parser.add_argument("--steps", type=int, default=6, help="每问最大思考步数（默认 6）")
    parser.add_argument("--selftest", action="store_true", help="跑离线自测，不联网不用 key")
    args = parser.parse_args(argv)

    if args.selftest:
        sys.exit(0 if _selftest() else 1)

    try:
        agent = ReActAgent(max_steps=args.steps)
    except RuntimeError as e:
        print(f"启动失败：{e}")
        sys.exit(1)

    if args.interactive:
        print("多轮对话模式（有记忆，可追问上一句）。输入 quit / exit 退出。\n")
        session = Session(agent)
        while True:
            try:
                q = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in {"quit", "exit", "q"}:
                break
            answer, _ = session.ask(q)
            print("\n" + "=" * 40)
            print(f"Agent > {answer}")
            print("=" * 40 + "\n")
    elif args.question:
        answer, _ = agent.run(args.question)
        print("\n" + "=" * 40)
        print(f"Agent > {answer}")
        print("=" * 40)
    else:
        parser.print_help()


# ---------------------------------------------------------------------------
# 8. 离线自测（--selftest）：解析器 + 主循环 + 记忆，全部用桩，不联网
# ---------------------------------------------------------------------------

def _selftest() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("== parse_output / parse_action ==")
    r = parse_output('Thought: 需要查天气\nAction: search("上海今天天气")')
    check("action 正常解析", r["kind"] == "action" and "查天气" in (r["thought"] or "")
          and r["payload"].startswith('search('))
    r = parse_output("Thought: 好了\nFinal Answer: 上海多云 23 度")
    check("final 正常解析", r["kind"] == "final" and r["payload"] == "上海多云 23 度")
    r = parse_output('```\nThought：先搜\nAction：search(\'x\')\n```')
    check("剥代码围栏 + 全角冒号/单引号", r["kind"] == "action"
          and parse_action(r["payload"]) == ("search", "x"))
    r = parse_output("随便说点什么")
    check("无标记 → malformed", r["kind"] == "malformed")
    r = parse_output("Thought: 一步到位\nFinal Answer: 对\nAction: search(\"y\")")
    check("同回合多标记取最先（final 优先于后置 action）",
          r["kind"] == "final" and r["payload"] == "对")
    check("未知工具字符串", parse_action('foo("x")') == ("foo", "x"))
    check("非法 action 拆不出", parse_action("没有括号的内容")[0] is None)

    print("== ReActAgent.run（桩 LLM / 桩工具） ==")
    calls = []

    class StubSearch(Tool):
        name, usage_hint, description = "search", 'search("q")', "stub"
        def run(self, arg):
            calls.append(arg)
            return f"假结果：{arg}"

    registry = {"search": StubSearch()}

    def stub_llm(messages):
        # 第 1 次说去搜，之后看到 Observation 就直接给最终回答
        if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("Observation"):
            return "Thought: 有结果了\nFinal Answer: 答案来自搜索"
        return 'Thought: 需要搜索\nAction: search("测试词")'

    agent = ReActAgent(llm=stub_llm, registry=registry, verbose=False)
    answer, trace = agent.run("问题")
    check("循环执行了 search", calls == ["测试词"])
    check("最终答案正确", answer == "答案来自搜索")
    check("trace 记录 action 与 observation", len(trace) >= 2
          and any(t["type"] == "action" for t in trace))

    print("== 格式错误纠错路径 ==")
    bad_then_ok = [
        "没有格式的内容",
        "Thought: 好的\nFinal Answer: 纠错后答上来了",
    ]
    i = [0]
    def bad_llm(messages):
        reply = bad_then_ok[i[0] % len(bad_then_ok)]
        i[0] += 1
        return reply
    agent2 = ReActAgent(llm=bad_llm, registry=registry, verbose=False, max_steps=3)
    answer2, trace2 = agent2.run("问题")
    check("malformed 后能纠错并给答案",
          answer2 == "纠错后答上来了"
          and any(t["type"] == "malformed" for t in trace2))

    print("== Session 多轮记忆 ==")
    seen_questions = []

    def memory_llm(messages):
        # 记下第 2 问时喂进来的前缀里是否带第 1 问
        roles = [m["role"] for m in messages]
        if any(m["content"] == "第一问：上海天气" for m in messages):
            seen_questions.append("Q1_in_context")
        if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("Observation"):
            return "Thought: ok\nFinal Answer: 有结果"
        if messages[-1]["content"].startswith("第二问"):
            seen_questions.append("Q2_in_context")
            return "Thought: 靠记忆直接答\nFinal Answer: 记得上海，第二问同理"
        return 'Thought: 搜\nAction: search("上海天气")'

    session = Session(agent=ReActAgent(llm=memory_llm, registry=registry, verbose=False))
    session.ask("第一问：上海天气")
    answer3, _ = session.ask("第二问：那北京呢")
    check("记忆前缀把上一轮 Q&A 喂给模型",
          "Q1_in_context" in seen_questions and "Q2_in_context" in seen_questions)
    check("第二轮靠记忆给答案", answer3 == "记得上海，第二问同理")
    check("记忆只存 Q/A、上限内不裁剪", len(session.memory) == 4)

    session.max_rounds = 1
    session.ask("第三问")
    check("超过轮数上限会裁剪旧记忆", len(session.memory) == 2)

    print("== parser 鲁棒性（尾随文字 / 全角括号 / 空答案） ==")
    check("Action 同行尾随解释仍可拆",
          parse_action('search("上海天气")（需要最新信息）') == ("search", "上海天气"))
    check("Action 换行尾随解释仍可拆",
          parse_action('search("上海天气")\n因为要查最新') == ("search", "上海天气"))
    check("全角括号可拆", parse_action('search（"上海"）') == ("search", "上海"))
    check("引号内含括号不计深度",
          parse_action('search("吃(什么)比较好")') == ("search", "吃(什么)比较好"))
    check("括号不平衡 → 拆不出",
          parse_action('search("上海"') == (None, 'search("上海"'))
    check("空 Final 视为 malformed",
          parse_output("Thought: 好\nFinal Answer:")["kind"] == "malformed")

    print("== Final 正文截断启发式 ==")
    r = parse_output('Thought: ok\nFinal Answer: 多行正文\nAction: 这里只是举例格式')
    check("正文中的举例式 Action 行不被截断",
          r["kind"] == "final" and r["payload"].count("Action:") == 1)
    r = parse_output("Thought: 一步\nFinal Answer: 总结\nThought: 补充\nAction: search(\"y\")")
    check("正文后跟真·链式回合才截断",
          r["kind"] == "final" and r["payload"] == "总结")

    print("== 模型调用失败兜底 ==")
    def boom_llm(messages):
        raise RuntimeError("429 rate limit")
    boom_agent = ReActAgent(llm=boom_llm, registry=registry, verbose=False)
    boom_answer, boom_trace = boom_agent.run("问题")
    check("LLM 异常不抛出、返回友好提示",
          str(boom_answer).startswith("[模型调用失败]")
          and any(t["type"] == "error" for t in boom_trace))

    print("== Session 不存非最终结果 ==")
    boom_session = Session(agent=boom_agent)
    boom_answer2, _ = boom_session.ask("问题")
    check("失败轮次不入记忆", len(boom_session.memory) == 0)

    print()
    return ok


if __name__ == "__main__":
    main()
