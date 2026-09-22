# -*- coding: utf-8 -*-
"""
llm.py —— LLM 基类 + DeepSeek 默认实现 + 密钥加载
==================================================

内核只依赖一条约定：**llm 是一个可调用对象，收 messages 返回 str**。
怎么发请求、用哪家、带什么参数，全在这个文件里，内核不知道。

换供应商 = 写一个 LLM 子类，把 `chat()` 实现掉。参考 DeepSeekLLM，它演示了：
构造时读密钥 → 组装 body → 发 POST → 取出 assistant 文本。
"""

from __future__ import annotations

import os

import requests

HTTP_TIMEOUT = (10, 60)   # (连接超时, 读取超时) 秒


# ---------------------------------------------------------------------------
# 1. 密钥加载：环境变量优先，其次读仓库根的 keys.py（已 gitignore）
# ---------------------------------------------------------------------------

def load_key(env_name: str, key_module_attr: str | None = None) -> str:
    """读密钥。找不到返回空串（由调用方决定怎么报错）。

    放在这里而不是单独的 config.py：目前只有两个密钥（DeepSeek / Tavily），
    多一个文件多一层跳转。tools.py 需要 Tavily key 时 `from llm import load_key`。
    密钥变多了再提升成 config.py。
    """
    val = os.environ.get(env_name, "").strip()
    if val:
        return val
    try:
        import keys as _keys
    except ImportError:
        return ""
    return str(getattr(_keys, key_module_attr or env_name, "") or "").strip()


def post_json(url: str, headers: dict, body: dict) -> dict:
    """通用 POST JSON，非 2xx 抛带状态码异常。"""
    resp = requests.post(url, headers=headers, json=body, timeout=HTTP_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


# ---------------------------------------------------------------------------
# 2. LLM 基类
# ---------------------------------------------------------------------------

class LLM:
    """LLM 基类。子类实现 chat()；__call__ 让实例本身可调用，方便当函数传。"""

    name: str = "llm"

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def __call__(self, messages: list[dict]) -> str:
        return self.chat(messages)

    def ready(self) -> bool:
        """能不能开工（密钥配没配）。CLI 启动前用它给出友好提示。"""
        return True


# ---------------------------------------------------------------------------
# 3. DeepSeek 实现（OpenAI 兼容）
# ---------------------------------------------------------------------------

class DeepSeekLLM(LLM):
    name = "deepseek"

    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None,
                 base_url: str = "https://api.deepseek.com",
                 temperature: float = 0.3, max_tokens: int = 600):
        # 密钥在构造时读一次存进实例 —— 不是模块级常量，所以同进程能共存多个实例
        self.api_key = load_key("DEEPSEEK_API_KEY") if api_key is None else api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def ready(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict]) -> str:
        # 构造不报错、调用才报错：这样没密钥也能 import persona / 跑 --selftest
        if not self.api_key:
            raise RuntimeError(
                "未配置 DeepSeek API key：请设置环境变量 DEEPSEEK_API_KEY，"
                "或在本目录放 keys.py")
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        data = post_json(f"{self.base_url}/chat/completions",
                         {"Authorization": f"Bearer {self.api_key}",
                          "Content-Type": "application/json"},
                         body)
        return data["choices"][0]["message"]["content"]
