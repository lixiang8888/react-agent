# -*- coding: utf-8 -*-
"""
core/loop.py —— ReAct 主循环骨架 + 三个钩子
============================================

这是内核。骨架**固定**（思考 → 行动 → 观察 → … → 最终回答），
想改行为就重写钩子，而不是改这个文件。

    ┌─ on_step_start ─┐  调模型之前。可以往 ctx.messages 插内容
    │                 │  （Reflection）、改 ctx.max_steps（动态步数）、
    │                 │  置 ctx.stop 提前结束。
    │   ① 调 LLM      │
    │   ② 解析输出     │  ← 交给 protocol.parse()，内核不碰文本格式
    │   ③ final？      │  → 返回
    │   ④ malformed？  │  → 喂回纠正提示，下一步重来
    │   ⑤ 执行工具     │  ← before_tool 可以否决 / 替换这次执行
    │   ⑥ 回写历史     │
    └─ on_step_end ───┘  Observation 回写之后

**第 3 级扩展（并行工具）怎么做**：加一个 `execute_actions(ctx, actions)` 可重写
方法，默认串行执行；同时 StepResult 要扩出 `actions: list`，协议说明要允许模型
写多个 Action，自测里「一次只做一个 Action」的断言要跟着改。README 有详述。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.protocol import Protocol, StepResult
from tools import execute_tool

_MALFORMED_FALLBACK = ("解析不到 Action 或 Final Answer。"
                       "请严格按格式输出一行 Action，或直接输出 Final Answer。")


@dataclass
class LoopContext:
    """钩子能碰到的一切。想给钩子更多信息，就往这个 dataclass 上加字段。"""
    step: int
    max_steps: int              # 钩子可改，实现「跑到一半提高步数上限」
    messages: list              # 钩子可插可改
    question: str
    trace: list
    stop: bool = False          # 钩子置 True → 本轮结束后收工
    scratch: dict = field(default_factory=dict)   # 钩子的私有寄存处


class ReActLoop:
    """默认循环。人格想换循环策略就继承它、重写钩子（见 persona.loop_cls）。"""

    def __init__(self, llm, tools: dict, protocol: Protocol,
                 system_prompt: str, max_steps: int = 6, verbose: bool = True):
        self.llm = llm
        self.tools = tools
        self.protocol = protocol
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.verbose = verbose

    # ------------------------------------------------------------------
    # 钩子：默认什么都不做。子类重写。
    # ------------------------------------------------------------------

    def on_step_start(self, ctx: LoopContext) -> None:
        """调 LLM 之前。典型用法：每 N 步插入一句反思、动态改步数、提前收工。"""

    def on_step_end(self, ctx: LoopContext) -> None:
        """Observation 回写之后。典型用法：埋点、日志、统计工具调用次数。"""

    def before_tool(self, ctx: LoopContext, name: str, arg: str) -> str | None:
        """拦在工具执行之前。

            return None      → 照常执行
            return "文本"    → **跳过**执行，拿这个字符串当 Observation
            raise Exception  → 不执行，异常信息作为 Observation 喂回

        典型用法：人工确认、工具白名单、结果缓存。
        """
        return None

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, question: str, history: list | None = None,
            max_steps: int | None = None) -> tuple[str | None, list]:
        """
        返回 (最终回答, trace)。

        - history：拼在 system 之后、本次提问之前的消息前缀。**循环不知道
          「记忆」这个概念**，它只收一段现成的历史——记忆怎么存、什么时候
          写回，是 memory.py / agent.py 的事。
        - trace：每一步的 thought / action / observation，供展示与测试。
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages += list(history)
        messages.append({"role": "user", "content": question})

        ctx = LoopContext(
            step=0,
            max_steps=self.max_steps if max_steps is None else max_steps,
            messages=messages,
            question=question,
            trace=[],
        )
        last_reply = ""
        stop_reason: str | None = None

        # 用 while 而不是 for：钩子可能中途调大 ctx.max_steps
        while ctx.step < ctx.max_steps:
            ctx.step += 1
            self.on_step_start(ctx)          # 钩子①
            if ctx.stop:
                stop_reason = "循环被钩子提前终止"
                break

            # ① 调模型思考
            try:
                reply = (self.llm(ctx.messages) or "").strip()
            except Exception as e:      # 网络/限流/欠费别让整个会话崩掉
                err = f"[模型调用失败] {e}，可稍后重试。"
                ctx.trace.append({"type": "error", "error": str(e)})
                if self.verbose:
                    print(f"\n[!] {err}")
                return err, ctx.trace
            last_reply = reply
            if not reply:
                stop_reason = "模型返回空内容"
                if self.verbose:
                    print("\n[!] 模型返回空内容，终止。")
                break

            # ② 解析输出（协议的事，内核不碰文本格式）
            result = self.protocol.parse(reply)

            if result.kind == "action" and result.tool_name is None:
                # 自定义协议漏填了工具名，当格式错误处理，别让内核崩
                result = StepResult(kind="malformed", thought=result.thought,
                                    payload=result.payload,
                                    error=result.error or f"无法解析 Action：{result.payload!r}。")

            # ③ final：收工
            if result.kind == "final":
                ctx.trace.append({"type": "final", "thought": result.thought,
                                  "answer": result.payload})
                if self.verbose:
                    self._log_step(ctx.step, result.thought, "Final Answer",
                                   None, ctx.max_steps)
                return result.payload, ctx.trace

            # ④ malformed：把「格式错误」也当 Observation 喂回请模型重写
            #    （有 ctx.max_steps 硬上限防死循环）
            if result.kind == "malformed":
                obs = result.error or _MALFORMED_FALLBACK
                ctx.trace.append({"type": "malformed", "reply": reply})
                if self.verbose:
                    self._log_step(ctx.step, result.thought, "(格式错误，要求重写)",
                                   obs, ctx.max_steps)
                self._append_observation(ctx, reply, obs)
                self.on_step_end(ctx)        # 钩子③
                continue

            # ⑤ 执行 action（before_tool 有机会否决或替换）
            name, arg = result.tool_name, result.tool_arg
            obs = self._run_tool(ctx, name, arg)
            action_disp = f"{name}({arg!r})"
            ctx.trace.append({"type": "action", "step": ctx.step,
                              "thought": result.thought, "action": action_disp,
                              "action_raw": result.payload, "observation": obs})
            if self.verbose:
                self._log_step(ctx.step, result.thought, action_disp, obs, ctx.max_steps)

            # ⑥ 回写历史，进入下一轮
            self._append_observation(ctx, reply, obs)
            self.on_step_end(ctx)            # 钩子③

        # 没拿到 Final Answer 就退出了：兜底返回最后一段内容
        if stop_reason:
            fallback = (f"⚠ {stop_reason}，未给出最终答案。"
                        f"最后给出的内容：\n{last_reply or '(空)'}")
        else:
            fallback = (f"⚠ 已达最大步数({ctx.max_steps})仍无最终答案。"
                        f"最后给出的内容：\n{last_reply or '(空)'}")
        ctx.trace.append({"type": "fallback", "answer": fallback})
        if self.verbose:
            print(f"\n[!] {fallback}")
        return fallback, ctx.trace

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _run_tool(self, ctx: LoopContext, name: str, arg: str) -> str:
        """过一遍 before_tool 钩子，再执行工具。"""
        try:
            override = self.before_tool(ctx, name, arg)
        except Exception as e:          # 钩子抛异常 = 否决这次执行
            return f"工具 {name} 被拦截：{e}"
        if override is not None:
            return override
        return execute_tool(name, arg, self.tools)

    @staticmethod
    def _append_observation(ctx: LoopContext, reply: str, obs: str) -> None:
        ctx.messages.append({"role": "assistant", "content": reply})
        ctx.messages.append({"role": "user", "content": f"Observation: {obs}"})

    def _log_step(self, step: int, thought: str | None, action_disp: str,
                  observation: str | None, max_steps: int) -> None:
        print(f"\n[Step {step}/{max_steps}]")
        if thought:
            print(f"  Thought     : {thought}")
        if action_disp:
            print(f"  Action      : {action_disp}")
        if observation is not None:
            obs = observation.replace("\n", "\n               ")
            print(f"  Observation : {obs[:600]}")   # 只展示前 600 字符
