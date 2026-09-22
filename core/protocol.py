# -*- coding: utf-8 -*-
"""
core/protocol.py —— 输出协议契约 + 默认的文本协议
==================================================

协议负责**两件必须保持一致的事**，所以打包在同一个对象里：

    format_instructions(tools)  →  告诉模型该怎么输出（拼进 system prompt）
    parse(text)                 →  把模型输出拆成 StepResult（循环据此行动）

这两件事一旦分家（比如格式说明写在人格提示词里、解析器写在别处），
改了一个忘了另一个就会**静默出错**——模型照着旧格式输出，解析器按新格式拆，
最后表现为「模型不听话」。成对打包就是为了让这种错误不可能发生。

换协议 = 换一个 Protocol 对象（见 persona.py）。想看怎么加 JSON 协议，
读 TextReActProtocol 就够了——它把该做的事演示全了。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. 契约
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """parse() 的返回值：模型这一回合被**理解**成了什么。"""
    kind: str                        # "action" | "final" | "malformed"
    thought: str | None = None
    payload: str = ""                # 原始文本（action 行 / final 正文）
    tool_name: str | None = None     # kind == "action" 时：工具名
    tool_arg: str | None = None      # kind == "action" 时：参数
    error: str | None = None         # kind == "malformed" 时：喂回给模型的纠正提示


class Protocol:
    """输出协议基类。

    子类要做的就两件事，而且**必须同时做**：
      1. format_instructions：把格式要求写清楚
      2. parse：按同一套格式拆解
    """

    def format_instructions(self, tools: dict) -> str:
        """返回格式说明，会被拼进 system prompt。tools 形如 {name: Tool}。"""
        raise NotImplementedError

    def parse(self, text: str) -> StepResult:
        """把模型一回合的输出拆成 StepResult。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. 默认实现：Thought / Action / Final Answer 三行式文本协议
# ---------------------------------------------------------------------------

class TextReActProtocol(Protocol):
    """默认协议。parse() 内部复用 module 级的 parse_output / parse_action。"""

    def format_instructions(self, tools: dict) -> str:
        first = next(iter(tools.values())).usage_hint
        return f"""你必须严格按下面的格式思考与行动，不要输出额外内容：

Thought: 你针对当前问题的思考，说明下一步做什么
Action: {first}

每执行一次 Action，系统会返回一条 Observation（结果）。
看到 Observation 后继续输出 Thought -> Action，直到信息足够，
最后一行不再调用工具，输出：

Thought: 我已获得足够信息
Final Answer: 给用户的最终回答

要求：
1. Action 参数必须是双引号包裹的字符串字面量，例如 {first}。
2. 一次只做一个 Action，不要连续输出多个 Action。
3. 拿不准、过时、或超出你知识范围的信息就调用工具，不要编造。
4. 回答尽量带上关键来源。"""

    def parse(self, text: str) -> StepResult:
        raw = parse_output(text)
        kind, thought, payload = raw["kind"], raw["thought"], raw["payload"]

        if kind == "action":
            name, arg = parse_action(payload)
            if name is None:      # 括号不平衡之类，拆不出工具名
                return StepResult(
                    kind="malformed", thought=thought, payload=payload,
                    error=f'无法解析 Action：{payload!r}。格式应为 search("关键词")，'
                          f"且一次只做一个 Action。")
            return StepResult(kind="action", thought=thought, payload=payload,
                              tool_name=name, tool_arg=arg)

        if kind == "final":
            return StepResult(kind="final", thought=thought, payload=payload)

        return StepResult(
            kind="malformed", thought=thought, payload=payload,
            error="解析不到 Action 或 Final Answer。"
                  "请严格按格式输出一行 Action，或直接输出 Final Answer。")


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
