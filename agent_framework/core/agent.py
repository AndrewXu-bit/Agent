"""
通用编排引擎 - 与具体模型底座和工具来源完全解耦的 Agent 循环。

根据 framework.md 第 3 节设计：
引擎仅负责控制 "推理 -> 工具调用与等待 -> 再次推理" 的闭环，
不关心模型是 OpenAI 还是 Claude，也不关心工具是本地函数还是 MCP 进程。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .llm_client import BaseLLMClient
from .mcp_adapter import MCPClientAdapter
from .standard_tool import StandardTool

logger = logging.getLogger(__name__)


class UniversalAgent:
    """通用编排引擎。

    核心职责：
    1. 管理对话历史上下文
    2. 统一调度 LLM 推理
    3. 分发工具执行并收集结果
    4. 循环直到模型给出最终答案或达到最大步数

    Args:
        llm: LLM 客户端实例（OpenAI / Anthropic / Mock）。
        name: Agent 名称，用于日志标识。
        system_prompt: 系统提示词，可自定义 Agent 角色。
        max_steps: 最大推理-工具调用循环步数。
        verbose: 是否打印详细执行日志。
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        name: str = "UniversalAgent",
        system_prompt: Optional[str] = None,
        max_steps: int = 10,
        verbose: bool = True,
    ):
        self.llm = llm
        self.name = name
        self.max_steps = max_steps
        self.verbose = verbose

        self.tools: Dict[str, StandardTool] = {}
        self.history: List[Dict[str, Any]] = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    # ----------------------------------------------------------------
    # 工具挂载
    # ----------------------------------------------------------------

    def mount_tool(self, tool: StandardTool) -> None:
        """挂载单个 StandardTool。

        Args:
            tool: StandardTool 实例。
        """
        self.tools[tool.name] = tool
        self._log(f"📦 挂载工具: {tool.name}")

    def mount_tools(self, tools: List[StandardTool]) -> None:
        """批量挂载工具。"""
        for t in tools:
            self.mount_tool(t)

    def mount_tool_registry(self, registry: Any) -> None:
        """挂载 LocalToolRegistry 中的所有工具。"""
        for t in registry.list_tools():
            self.mount_tool(t)

    async def mount_mcp_server(self, mcp_adapter: MCPClientAdapter) -> None:
        """挂载 MCP Server 提供的整套工具集。

        Args:
            mcp_adapter: MCPClientAdapter 实例。
        """
        self._log(f"🔗 连接 MCP 服务: {mcp_adapter.server_name} ({mcp_adapter.transport})")
        mcp_tools = await mcp_adapter.connect_and_fetch_tools()
        for t in mcp_tools:
            self.mount_tool(t)
        self._log(f"✅ MCP 服务 {mcp_adapter.server_name} 加载完成 ({len(mcp_tools)} 个工具)")

    # ----------------------------------------------------------------
    # 核心运行循环
    # ----------------------------------------------------------------

    async def run(
        self,
        user_query: str,
        max_steps: Optional[int] = None,
    ) -> str:
        """运行 Agent 主循环。

        流程：
        1. 将用户消息加入历史
        2. LLM 推理 -> 返回文本或工具调用
        3. 如果返回工具调用 -> 执行工具 -> 将结果写回历史 -> 回到步骤 2
        4. 如果返回文本 -> 结束循环

        Args:
            user_query: 用户输入。
            max_steps: 本轮最大步数（覆盖初始化时的设置）。

        Returns:
            Agent 最终回复文本。
        """
        self.history.append({"role": "user", "content": user_query})
        steps = max_steps or self.max_steps

        self._log(f"\n{'='*60}")
        self._log(f"🤖 Agent [{self.name}] 开始运行 | 最大步数: {steps}")
        self._log(f"📝 用户: {user_query}")
        self._log(f"{'='*60}")

        for step in range(1, steps + 1):
            self._log(f"\n{'─'*40}  第 {step}/{steps} 步  {'─'*40}")

            # 1. 调用 LLM 推理
            self._log(f"🧠 调用模型: {self.llm.model}")
            available_tools = list(self.tools.values())

            try:
                response = await self.llm.chat(
                    messages=self.history,
                    tools=available_tools if available_tools else None,
                )
            except Exception as exc:
                error_msg = f"LLM 调用失败: {exc}"
                self._log(f"❌ {error_msg}")
                self.history.append({"role": "assistant", "content": error_msg})
                return error_msg

            # 2. 将模型响应加入历史
            self.history.append(response)
            content = response.get("content", "")
            if content:
                self._log(f"💬 模型: {content[:200]}{'...' if len(content) > 200 else ''}")

            # 3. 判断是否触发工具调用
            tool_calls = response.get("tool_calls")
            if not tool_calls:
                self._log(f"\n✅ Agent 完成推理，返回最终结果")
                self._log(f"{'='*60}\n")
                return content or "（模型未返回文本内容）"

            # 4. 执行工具调用
            self._log(f"🔧 模型请求调用 {len(tool_calls)} 个工具:")
            for call in tool_calls:
                func_name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                self._log(f"  → {func_name}({json.dumps(arguments, ensure_ascii=False)})")

                if func_name in self.tools:
                    result = await self.tools[func_name].run(arguments)
                    self._log(f"  ← 结果: {result[:150]}{'...' if len(result) > 150 else ''}")
                else:
                    result = f"Error: 工具 '{func_name}' 未注册。可用工具: {list(self.tools.keys())}"
                    self._log(f"  ⚠️ {result}")

                # 5. 将工具执行结果写回历史
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", f"call_{step}_{func_name}"),
                        "name": func_name,
                        "content": result,
                        "type": "function",  # DeepSeek API 需要此字段
                    }
                )

        # 达到最大步数
        msg = f"⚠️ 已达到最大迭代步数 ({steps})，无法生成终局结论。"
        self._log(f"\n{msg}")
        self.history.append({"role": "assistant", "content": msg})
        return msg

    # ----------------------------------------------------------------
    # 历史管理
    # ----------------------------------------------------------------

    def get_history(self) -> List[Dict[str, Any]]:
        """获取完整对话历史。"""
        return self.history.copy()

    def clear_history(self) -> None:
        """清空对话历史（保留 system prompt）。"""
        system_prompts = [m for m in self.history if m.get("role") == "system"]
        self.history = system_prompts

    def reset(self) -> None:
        """完全重置 Agent 状态。"""
        self.history = []
        self.tools = {}

    # ----------------------------------------------------------------
    # 工具 & 状态查询
    # ----------------------------------------------------------------

    def list_tools(self) -> List[StandardTool]:
        """列出所有已挂载工具。"""
        return list(self.tools.values())

    def get_tool(self, name: str) -> Optional[StandardTool]:
        """按名称查找工具。"""
        return self.tools.get(name)

    def _log(self, message: str) -> None:
        """内部日志输出。"""
        if self.verbose:
            print(message)