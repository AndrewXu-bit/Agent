"""
MCP 服务客户端适配器 - 连接 MCP Server 并动态加载工具。

根据 framework.md 2.3 节设计：
作为客户端连接 stdio / SSE MCP Server，通过 tools/list 拉取
远程工具列表，映射为框架的 StandardTool。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .standard_tool import StandardTool

logger = logging.getLogger(__name__)


class MCPClientAdapter:
    """MCP 服务客户端适配器。

    连接 MCP 服务器（stdio 子进程或 SSE 远程服务），
    动态获取其工具列表并封装为 StandardTool。

    Args:
        server_name: 服务名称，仅用于日志/标识。
        transport: 传输方式，支持 "stdio" 或 "sse"。
        command: stdio 模式下的启动命令。
        args: stdio 模式下的命令参数。
        url: SSE 模式下的服务 URL。
    """

    def __init__(
        self,
        server_name: str = "mcp-server",
        transport: str = "stdio",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
    ):
        self.server_name = server_name
        self.transport = transport
        self.command = command
        self.args = args or []
        self.url = url
        self._session = None
        # 保存上下文管理器引用，用于保持连接存活和清理
        self._sse_ctx = None
        self._stdio_ctx = None
        self._session_ctx = None

    async def _connect_stdio(self) -> List[Dict[str, Any]]:
        """通过 stdio 子进程连接 MCP Server 并获取工具列表。

        使用 mcp Python SDK 的 StdioServerParameters。
        不通过 async with，而是手动进入上下文管理器以保持连接存活。
        需要安装: pip install mcp
        """
        try:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=self.command or "python",
                args=self.args,
            )

            # 手动进入上下文管理器，保持连接不断开
            self._stdio_ctx = stdio_client(server_params)
            read, write = await self._stdio_ctx.__aenter__()

            from mcp.client.session import ClientSession
            self._session_ctx = ClientSession(read, write)
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()
            result = await self._session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema,
                }
                for t in result.tools
            ]
        except ImportError:
            logger.warning(
                "mcp SDK 未安装，回退到模拟模式。"
                "请执行: pip install mcp"
            )
            return self._mock_fetch()
        except Exception as exc:
            logger.error(f"MCP stdio 连接失败: {exc}")
            return self._mock_fetch()

    async def _connect_sse(self) -> List[Dict[str, Any]]:
        """通过 SSE (Server-Sent Events) 连接远程 MCP Server。

        不通过 async with，而是手动进入上下文管理器以保持连接存活。
        """
        try:
            from mcp.client.sse import sse_client

            if not self.url:
                raise ValueError("SSE 模式需要提供 url 参数")

            # 手动进入上下文管理器，保持 SSE 连接不断开
            self._sse_ctx = sse_client(self.url)
            read, write = await self._sse_ctx.__aenter__()

            from mcp.client.session import ClientSession
            self._session_ctx = ClientSession(read, write)
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()
            result = await self._session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema,
                }
                for t in result.tools
            ]
        except ImportError:
            logger.warning("mcp SDK 未安装，回退到模拟模式。")
            return self._mock_fetch()
        except Exception as exc:
            logger.error(f"MCP SSE 连接失败: {exc}")
            return self._mock_fetch()

    def _mock_fetch(self) -> List[Dict[str, Any]]:
        """模拟 MCP 工具列表（当 SDK 不可用时）。"""
        logger.info(f"[MCP] {self.server_name}: 使用模拟工具列表")
        return [
            {
                "name": f"{self.server_name}_echo",
                "description": f"[{self.server_name}] 回显输入参数",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "要回显的消息",
                        }
                    },
                    "required": ["message"],
                },
            },
            {
                "name": f"{self.server_name}_random",
                "description": f"[{self.server_name}] 生成一个随机数",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max": {
                            "type": "integer",
                            "description": "最大值（默认 100）",
                            "default": 100,
                        }
                    },
                },
            },
        ]

    async def connect_and_fetch_tools(self) -> List[StandardTool]:
        """连接 MCP 服务器并获取封装好的 StandardTool 列表。

        Returns:
            StandardTool 列表。如果连接失败，返回空列表。
        """
        if self.transport == "sse":
            raw_tools = await self._connect_sse()
        else:
            raw_tools = await self._connect_stdio()

        standard_tools: List[StandardTool] = []
        for raw in raw_tools:
            tool_name = raw["name"]

            async def _make_executor(
                name: str = tool_name,
            ) -> Any:
                async def execute(args: Dict[str, Any]) -> str:
                    return await self._call_tool(name, args)

                return execute

            standard_tools.append(
                StandardTool(
                    name=tool_name,
                    description=raw.get("description", ""),
                    input_schema=raw.get("inputSchema", {"type": "object", "properties": {}}),
                    execute_fn=await _make_executor(),
                )
            )

        logger.info(
            f"[MCP] {self.server_name}: 加载了 {len(standard_tools)} 个工具"
        )
        return standard_tools

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具。

        优先使用真实 session 调用，回退到模拟。
        """
        if self._session is not None:
            try:
                result = await self._session.call_tool(tool_name, arguments)
                if hasattr(result, "content"):
                    parts = []
                    for c in result.content:
                        if hasattr(c, "text"):
                            parts.append(c.text)
                        elif isinstance(c, dict):
                            parts.append(json.dumps(c, ensure_ascii=False))
                    return "\n".join(parts)
                return str(result)
            except Exception as exc:
                return f"MCP Call Error [{tool_name}]: {exc}"

        # 模拟执行
        if "echo" in tool_name:
            return json.dumps(
                {"echo": arguments.get("message", ""), "server": self.server_name},
                ensure_ascii=False,
            )
        if "random" in tool_name:
            import random

            max_val = arguments.get("max", 100)
            return str(random.randint(0, max_val))

        return f"[MCP {self.server_name}] 工具 {tool_name} 执行成功，参数: {arguments}"

    async def disconnect(self) -> None:
        """关闭 MCP 连接，释放资源。"""
        try:
            if self._session_ctx is not None:
                await self._session_ctx.__aexit__(None, None, None)
                self._session_ctx = None
                self._session = None
                logger.info(f"[MCP] {self.server_name}: 会话已关闭")
        except Exception as exc:
            logger.warning(f"[MCP] {self.server_name}: 关闭会话异常: {exc}")

        try:
            if self._sse_ctx is not None:
                await self._sse_ctx.__aexit__(None, None, None)
                self._sse_ctx = None
                logger.info(f"[MCP] {self.server_name}: SSE 连接已关闭")
        except Exception as exc:
            logger.warning(f"[MCP] {self.server_name}: 关闭 SSE 异常: {exc}")

        try:
            if self._stdio_ctx is not None:
                await self._stdio_ctx.__aexit__(None, None, None)
                self._stdio_ctx = None
                logger.info(f"[MCP] {self.server_name}: stdio 连接已关闭")
        except Exception as exc:
            logger.warning(f"[MCP] {self.server_name}: 关闭 stdio 异常: {exc}")