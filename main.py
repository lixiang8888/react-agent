# -*- coding: utf-8 -*-
"""
main.py —— CLI 入口
====================

运行（在 WSL 内；用 uv 时把 python3 换成 `uv run python`）：

    python3 main.py "问题"          # 单次提问
    python3 main.py --interactive   # 多轮对话（带记忆）
    python3 main.py --selftest      # 离线自测（不联网，不用 key）
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台保险
except Exception:
    pass

import tools as tools_mod
from agent import ReActAgent
from core.loop import ReActLoop
from core.protocol import parse_action, parse_output
from persona import PERSONA, Persona
from tools import Tool


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="react_agent", description="ReAct 智能体：DeepSeek 思考 + Tavily 搜索")
    parser.add_argument("question", nargs="?", help="单次提问内容")
    parser.add_argument("-i", "--interactive", action="store_true", help="多轮对话（带记忆）")
    parser.add_argument("--steps", type=int, default=None, help="每问最大思考步数（默认取人格配置）")
    parser.add_argument("--selftest", action="store_true", help="跑离线自测，不联网不用 key")
    args = parser.parse_args(argv)

    if args.selftest:
        sys.exit(0 if _selftest() else 1)

    # 密钥缺失只在真的要跑的时候才拦（import 阶段不拦，否则 --selftest 也起不来）
    if not PERSONA.llm.ready():
        print("启动失败：未配置 DeepSeek API key。"
              "请设置环境变量 DEEPSEEK_API_KEY，或在本目录放 keys.py")
        sys.exit(1)

    agent = ReActAgent(max_steps=args.steps)

    if args.interactive:
        print(f"多轮对话模式（人格：{PERSONA.name}，有记忆，可追问上一句）。"
              f"输入 quit / exit 退出。\n")
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
            answer, _ = agent.run(q)
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
# 离线自测：解析器 + 主循环 + 记忆 + 钩子 + 换人格，全部用桩，不联网
# ---------------------------------------------------------------------------

def _stub_persona(tool_names=("search",)) -> Persona:
    """桩人格。llm 由测试显式传入，所以这里给 None。"""
    return Persona(name="桩人格", system_prompt="你是桩人格，用中文回答。",
                   tool_names=list(tool_names), llm=None)


def _selftest() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # ---- 1. 解析器（与旧单文件版一一对应）-------------------------------
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
    r = parse_output('Thought: 一步到位\nFinal Answer: 对\nAction: search("y")')
    check("同回合多标记取最先（final 优先于后置 action）",
          r["kind"] == "final" and r["payload"] == "对")
    check("未知工具字符串", parse_action('foo("x")') == ("foo", "x"))
    check("非法 action 拆不出", parse_action("没有括号的内容")[0] is None)

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

    # ---- 2. 主循环（桩 LLM / 桩工具）-----------------------------------
    print("== ReActLoop.run（桩 LLM / 桩工具） ==")
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

    agent = ReActAgent(persona=_stub_persona(), llm=stub_llm, tools=registry, verbose=False)
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
    agent2 = ReActAgent(persona=_stub_persona(), llm=bad_llm, tools=registry,
                        memory=None, verbose=False, max_steps=3)
    answer2, trace2 = agent2.run("问题")
    check("malformed 后能纠错并给答案",
          answer2 == "纠错后答上来了"
          and any(t["type"] == "malformed" for t in trace2))

    print("== 模型调用失败兜底 ==")
    def boom_llm(messages):
        raise RuntimeError("429 rate limit")
    boom_agent = ReActAgent(persona=_stub_persona(), llm=boom_llm, tools=registry, verbose=False)
    boom_answer, boom_trace = boom_agent.run("问题")
    check("LLM 异常不抛出、返回友好提示",
          str(boom_answer).startswith("[模型调用失败]")
          and any(t["type"] == "error" for t in boom_trace))
    check("失败轮次不入记忆", len(boom_agent.memory) == 0)

    # ---- 3. 记忆 -------------------------------------------------------
    print("== WindowMemory 多轮记忆 ==")
    seen_questions = []

    def memory_llm(messages):
        # 记下第 2 问时喂进来的前缀里是否带第 1 问
        if any(m["content"] == "第一问：上海天气" for m in messages):
            seen_questions.append("Q1_in_context")
        if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("Observation"):
            return "Thought: ok\nFinal Answer: 有结果"
        if messages[-1]["content"].startswith("第二问"):
            seen_questions.append("Q2_in_context")
            return "Thought: 靠记忆直接答\nFinal Answer: 记得上海，第二问同理"
        return 'Thought: 搜\nAction: search("上海天气")'

    agent3 = ReActAgent(persona=_stub_persona(), llm=memory_llm, tools=registry, verbose=False)
    agent3.run("第一问：上海天气")
    answer3, _ = agent3.run("第二问：那北京呢")
    check("记忆前缀把上一轮 Q&A 喂给模型",
          "Q1_in_context" in seen_questions and "Q2_in_context" in seen_questions)
    check("第二轮靠记忆给答案", answer3 == "记得上海，第二问同理")
    check("记忆只存 Q/A、上限内不裁剪", len(agent3.memory) == 4)

    agent3.memory.max_rounds = 1
    agent3.run("第三问")
    check("超过轮数上限会裁剪旧记忆", len(agent3.memory) == 2)

    # ---- 4. 换人格：适配度 + 无全局状态（新增）--------------------------
    print("== 换人格不碰内核（适配度 + 无全局状态） ==")
    calls_a, calls_b = [], []

    class StubSearchA(Tool):
        name, usage_hint, description = "search", 'search("q")', "桩搜索"
        def run(self, arg):
            calls_a.append(arg)
            return f"假搜索：{arg}"

    class StubCalcB(Tool):
        name, usage_hint, description = "calc", 'calc("算式")', "桩计算器"
        def run(self, arg):
            calls_b.append(arg)
            return f"假计算：{arg}"

    persona_a = Persona(name="搜索人格", system_prompt="你只会搜索。",
                        tool_names=["search"], llm=None)
    persona_b = Persona(name="计算人格", system_prompt="你只会计算。",
                        tool_names=["calc"], llm=None)

    def llm_a(messages):
        if messages[-1]["content"].startswith("Observation"):
            return "Thought: 够了\nFinal Answer: 搜索人格的答案"
        return 'Thought: 搜\nAction: search("甲")'

    def llm_b(messages):
        if messages[-1]["content"].startswith("Observation"):
            return "Thought: 够了\nFinal Answer: 计算人格的答案"
        return 'Thought: 算\nAction: calc("1+1")'

    # 两个 agent 在同一个进程里并存，各自一套工具、一份记忆
    ag_a = ReActAgent(persona=persona_a, llm=llm_a,
                      tools={"search": StubSearchA()}, verbose=False)
    ag_b = ReActAgent(persona=persona_b, llm=llm_b,
                      tools={"calc": StubCalcB()}, verbose=False)

    a1, _ = ag_a.run("甲问题")
    b1, _ = ag_b.run("乙问题")
    a2, _ = ag_a.run("甲问题2")

    check("两个不同人格都能跑通",
          a1 == "搜索人格的答案" and b1 == "计算人格的答案" and a2 == "搜索人格的答案")
    check("交叉执行不串：各自的工具只被自己调用",
          calls_a == ["甲", "甲"] and calls_b == ["1+1"])
    check("记忆互不串",
          len(ag_a.memory) == 4 and len(ag_b.memory) == 2
          and all("计算人格" not in m["content"] for m in ag_a.memory.history())
          and all("搜索人格" not in m["content"] for m in ag_b.memory.history()))
    check("同一份 ReActLoop 服务两个人格（内核未被改动）",
          persona_a.loop_cls is persona_b.loop_cls is ReActLoop)
    check("人格提示词独立拼装，协议格式说明自动接上",
          "你只会搜索" in ag_a.system_prompt
          and "你只会计算" in ag_b.system_prompt
          and "你只会计算" not in ag_a.system_prompt
          and "Thought:" in ag_a.system_prompt
          and "calc(" in ag_b.system_prompt)      # 工具清单也是渲染出来的
    check("没有模块级全局工具注册表（多实例共存的前提）",
          not hasattr(tools_mod, "TOOL_REGISTRY"))

    # ---- 5. 钩子：注入级 + 否决级（新增）--------------------------------
    print("== 钩子（on_step_start 注入 / before_tool 否决） ==")

    class FirstStepReflect(ReActLoop):
        """每个 run 的第一步插一句反思 —— 决策 7 的「注入级」。"""
        def on_step_start(self, ctx):
            if ctx.step == 1:
                ctx.messages.append({"role": "user", "content": "（反思：先想清楚要什么）"})

    seen_msgs = []

    def reflect_llm(messages):
        seen_msgs.append([m["content"] for m in messages])
        if messages[-1]["content"].startswith("Observation"):
            return "Thought: 够了\nFinal Answer: 反思后答上来了"
        return 'Thought: 搜\nAction: search("去")'

    p_reflect = Persona(name="爱反思", system_prompt="你先反思再行动。",
                        tool_names=["search"], llm=None, loop_cls=FirstStepReflect)
    ag_r = ReActAgent(persona=p_reflect, llm=reflect_llm,
                      tools={"search": StubSearchA()}, verbose=False)
    r_ans, _ = ag_r.run("问题")
    check("on_step_start 能往消息流插内容（Reflection）",
          any("（反思：先想清楚要什么）" in msgs for msgs in seen_msgs))
    check("换了 loop_cls 仍跑通（内核一行没改）", r_ans == "反思后答上来了")

    class DenyAll(ReActLoop):
        """否决所有工具调用 —— 「否决级」。"""
        def before_tool(self, ctx, name, arg):
            return f"[被策略拦截] {name} 不在白名单"

    denied = []

    def deny_llm(messages):
        last = messages[-1]["content"]
        if last.startswith("Observation"):
            if "被策略拦截" in last:
                denied.append(True)
                return "Thought: 工具被拦了\nFinal Answer: 我直接答"
            return "Thought: 竟然真搜了\nFinal Answer: 不该走到这"
        return 'Thought: 搜\nAction: search("去")'

    p_deny = Persona(name="严格人格", system_prompt="你只能直接回答。",
                     tool_names=["search"], llm=None, loop_cls=DenyAll)
    calls_before = len(calls_a)
    ag_d = ReActAgent(persona=p_deny, llm=deny_llm,
                      tools={"search": StubSearchA()}, verbose=False)
    d_ans, d_trace = ag_d.run("问题")
    check("before_tool 能否决工具执行（真工具没被调用）",
          bool(denied) and len(calls_a) == calls_before
          and any("被策略拦截" in t.get("observation", "") for t in d_trace))
    check("被否决后模型仍能收敛出答案", d_ans == "我直接答")

    print("== 无记忆模式 ==")
    ag_nomem = ReActAgent(persona=_stub_persona(), llm=stub_llm, tools=registry,
                          memory=None, verbose=False)
    ag_nomem.run("第一问")
    ag_nomem.run("第二问")
    check("memory=None 时不带记忆", ag_nomem.memory is None)

    print()
    return ok


if __name__ == "__main__":
    main()
