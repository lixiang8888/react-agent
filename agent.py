# -*- coding: utf-8 -*-
"""
agent.py —— 门面层：把 persona / llm / 工具 / 记忆 组装成一个能直接 run 的 agent
==================================================================================

两层设计（想用哪层用哪层）：

    core/loop.py  ReActLoop    纯骨架。收 history 参数，**不知道「记忆」是什么**
    agent.py      ReActAgent   门面。持有 memory，自动读写，一步到位

    # 单 agent 应用：用门面，最省事
    agent = ReActAgent()
    answer, trace = agent.run("上海今天天气")

    # 辩论 / 编排：绕过门面，直接用内核，记忆由你自己调度
    loop = ReActLoop(llm=..., tools=..., protocol=..., system_prompt=...)
    answer, trace = loop.run("问题", history=[...])

**「什么算成功的一轮」的策略在这个文件里**，只有这一处：
只有模型真正给出 Final Answer 的轮次才写进记忆；格式错误、工具失败、
步数耗尽的轮次一律不记——避免把半成品和报错当成「聊过的事」。
"""

from __future__ import annotations

from core.protocol import Protocol
from memory import Memory, WindowMemory
from persona import PERSONA, Persona
from tools import build_registry

_DEFAULT = object()      # 哨兵：区分「没传」和「显式传 None（=不要记忆）」


def build_system_prompt(persona: Persona, tools: dict) -> str:
    """人格提示词 + 工具清单（从注册表自动生成）+ 协议格式说明。

    加工具不用改提示词——工具清单是渲染出来的，不是手写的。
    """
    if not tools:
        raise ValueError(f"人格 {persona.name!r} 没配任何工具，无法生成提示词")
    tools_desc = "\n".join(f"- {t.usage_hint}：{t.description}" for t in tools.values())
    return f"""{persona.system_prompt}

可用工具：
{tools_desc}

{persona.protocol.format_instructions(tools)}"""


class ReActAgent:
    """门面。一个实例 = 一个人格 + 一份记忆，**没有全局状态**。

    所以同一个进程里可以并存多个实例：两个人格、两套工具、两份记忆，互不干扰。
    这是多 agent（orchestrator / 辩论）能落地的前提。
    """

    def __init__(self, persona: Persona = PERSONA, llm=None, tools: dict | None = None,
                 memory: Memory | None = _DEFAULT, max_steps: int | None = None,
                 verbose: bool = True):
        self.persona = persona
        self.llm = persona.llm if llm is None else llm
        # 每次 build_registry 都造新字典新实例 —— 没有模块级全局注册表
        self.tools = build_registry(persona.tool_names) if tools is None else tools
        self.memory = WindowMemory(persona.max_rounds) if memory is _DEFAULT else memory
        self.max_steps = persona.max_steps if max_steps is None else max_steps
        self.verbose = verbose

        self.system_prompt = build_system_prompt(persona, self.tools)
        self.loop = persona.loop_cls(
            llm=self.llm,
            tools=self.tools,
            protocol=persona.protocol,
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            verbose=verbose,
        )

    def run(self, question: str) -> tuple[str | None, list]:
        """提问一轮。带记忆时自动读历史、写回结果。"""
        history = self.memory.history() if self.memory is not None else None
        answer, trace = self.loop.run(question, history=history)
        # 「什么算成功的一轮」—— 唯一真源
        if self.memory is not None and trace and trace[-1].get("type") == "final":
            self.memory.record(question, answer)
        return answer, trace

    # 旧名字（原来的 Session.ask），保留以免已有的调用方断掉
    ask = run
