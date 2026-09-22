# -*- coding: utf-8 -*-
"""
tools.py —— 工具基类 + 搜索工具 + 注册表工厂
==============================================

工具的参数签名是**单个字符串** `run(arg: str) -> str`——这是刻意的：
文本协议天然只能给一个字符串，保持这个签名，parser 和内核都不用动。
需要多参数就往里传 JSON 字符串，工具自己 `json.loads`。

加工具三步：
    1. 写一个 Tool 子类，填 name / usage_hint / description，实现 run()
    2. 在 ALL_TOOLS 里登记一行
    3. 在 persona.py 的 tool_names 里写上它的名字
"""

from __future__ import annotations

from llm import load_key, post_json

TAVILY_BASE = "https://api.tavily.com"


# ---------------------------------------------------------------------------
# 1. 工具基类
# ---------------------------------------------------------------------------

class Tool:
    """工具基类。run() 的返回值直接当作 Observation 喂回给模型。"""
    name: str = ""
    usage_hint: str = ""          # 例如 search("关键词") —— 会被写进提示词
    description: str = ""         # 什么时候该用它

    def run(self, arg: str) -> str:
        raise NotImplementedError


class SearchTool(Tool):
    """搜索工具：调 Tavily API 搜全网。"""
    name = "search"
    usage_hint = 'search("关键词")'
    description = "联网搜索，返回若干条网页标题、摘要与来源链接。用于需要最新、实时或我知识之外的信息。"

    def __init__(self, api_key: str | None = None):
        self.api_key = load_key("TAVILY_API_KEY") if api_key is None else api_key

    def run(self, query: str) -> str:
        if not self.api_key:
            return "错误：未配置 Tavily API key（环境变量 TAVILY_API_KEY 或 keys.py）"
        body = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
        }
        data = post_json(f"{TAVILY_BASE}/search",
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


# ---------------------------------------------------------------------------
# 2. 注册表工厂
# ---------------------------------------------------------------------------

# 全部可用工具：名字 → 类。注意登记的是**类**不是实例。
ALL_TOOLS: dict[str, type[Tool]] = {
    "search": SearchTool,
    # 以后加工具：ALL_TOOLS["weather"] = WeatherTool
}


def build_registry(names: list[str]) -> dict[str, Tool]:
    """按名字造一份**新的**工具注册表。

    每次调用返回新字典、新实例——没有模块级全局注册表，所以同一个进程里
    可以同时存在多个工具集不同的 agent，互不干扰（多 agent 的前提）。
    """
    registry: dict[str, Tool] = {}
    for n in names:
        cls = ALL_TOOLS.get(n)
        if cls is None:
            raise KeyError(f"未知工具 {n!r}；ALL_TOOLS 里可用的有：{', '.join(ALL_TOOLS) or '(空)'}")
        registry[n] = cls()
    return registry


def execute_tool(name: str, arg: str, registry: dict[str, Tool]) -> str:
    """执行工具。名字非法 / 执行出错都转成可读的 Observation，让模型自己纠正。"""
    tool = registry.get(name)
    if tool is None:
        available = ", ".join(t.usage_hint for t in registry.values())
        return f"错误：未知工具 {name!r}。可用工具格式：{available}"
    try:
        return tool.run(arg)
    except Exception as e:      # 网络/限流/欠费都兜住，转成 Observation
        return f"工具 {name} 执行出错：{e}。请换措辞重试，或直接基于已有信息作答。"
