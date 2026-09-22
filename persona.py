# -*- coding: utf-8 -*-
"""
persona.py —— 人格定义。**换人格 = 只改这个文件。**
====================================================

一个 Agent 会变的东西，全在这里声明：角色提示词、用哪些工具、走什么输出协议、
用哪种循环、哪个模型、多少步。内核（core/）不认识「人格」这个词，它只收
拼装好的 system_prompt / 工具表 / 协议对象。

注意 system_prompt 里**不要写格式要求**——那是 protocol.format_instructions()
的活。分开写，换协议时格式说明会自动跟着变，不会漏改。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.loop import ReActLoop
from core.protocol import Protocol, TextReActProtocol
from llm import DeepSeekLLM, LLM


@dataclass
class Persona:
    # ---- 必填：这三样决定了「你是谁、你会什么、你多聪明」----
    name: str
    system_prompt: str                  # 只写角色与要求，不写输出格式
    tool_names: list[str]               # 名字要能在 tools.ALL_TOOLS 里找到
    llm: LLM

    # ---- 选填：想换就换，不换就用默认 ----
    protocol: Protocol = field(default_factory=TextReActProtocol)
    loop_cls: type = ReActLoop          # ← 换「循环策略」的入口：继承 ReActLoop 重写钩子
    max_steps: int = 6                  # 每问最大思考步数
    max_rounds: int = 10                # 记忆保留最近多少轮 Q/A


# ---------------------------------------------------------------------------
# 当前人格
# ---------------------------------------------------------------------------

PERSONA = Persona(
    name="搜索助手",
    system_prompt="你是「ReAct Agent —— 个人实践」的智能体，用中文回答问题。",
    tool_names=["search"],
    llm=DeepSeekLLM(),
)
