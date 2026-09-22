# -*- coding: utf-8 -*-
"""
memory.py —— 记忆接缝
======================

这一层只解决**上下文窗口管理**：会话变长时，喂回模型的 history 该保留哪些。
跨会话持久化、语义检索、轨迹记忆都还没做——那是另一个维度的东西，
要加就照着 Memory 基类再写一个子类，内核不用动。

注意分工：
    core/loop.py   只收一个 history 参数，整份代码里没有「记忆」这个词
    memory.py      负责存、取、裁剪
    agent.py       负责把两者粘起来，并执行「什么算成功的一轮」的策略

所以想在辩论/多 agent 场景里让 A 看见 B 的发言，往 A 的 memory 里塞一条就行，
不需要碰内核。
"""

from __future__ import annotations


class Memory:
    """记忆基类。两个方法：取出该喂进上下文的历史、记录一轮问答。"""

    def history(self) -> list[dict]:
        """返回要拼在 system 之后、本次提问之前的消息（[{role, content}, ...]）。"""
        raise NotImplementedError

    def record(self, question: str, answer: str) -> None:
        """记下一轮问答。什么时候调由 agent.py 决定。"""
        raise NotImplementedError


class WindowMemory(Memory):
    """滑动窗口：只保留最近 max_rounds 轮 Q/A，更早的直接丢掉。

    想换成「超长时调 LLM 摘要成要点」，把 _trim() 改掉即可；
    想换成「存磁盘、下次接着聊」，在 record/history 里读写文件即可。
    """

    def __init__(self, max_rounds: int = 10):
        self.max_rounds = max_rounds
        self._msgs: list[dict] = []      # 交替的 user/assistant Q&A

    def _trim(self) -> None:
        limit = self.max_rounds * 2
        while len(self._msgs) > limit:
            self._msgs.pop(0)

    def history(self) -> list[dict]:
        return list(self._msgs)          # 拷一份，防调用方改到内部状态

    def record(self, question: str, answer: str) -> None:
        self._msgs.append({"role": "user", "content": question})
        self._msgs.append({"role": "assistant", "content": answer})
        self._trim()

    def __len__(self) -> int:
        return len(self._msgs)


class NoMemory(Memory):
    """不带记忆：每次提问都是全新的。辩论/无状态批处理时用。"""

    def history(self) -> list[dict]:
        return []

    def record(self, question: str, answer: str) -> None:
        pass
