"""
标准工具接口 - 统一所有工具的数据结构与执行契约。

根据 framework.md 2.1 节设计：
无论是本地 Python 函数还是 MCP 远程服务，在框架内部
统一表现为 StandardTool 这一标准化数据结构。
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable, Dict

from pydantic import BaseModel, Field


class StandardTool(BaseModel):
    """统一工具接口 —— 框架内所有工具的唯一表示形式。

    Attributes:
        name: 工具名称，模型通过此名称引用工具。
        description: 工具描述，帮助模型理解何时调用此工具。
        input_schema: JSON Schema 格式的入参描述。
        execute_fn: 实际的执行函数，接收 Dict 参数，返回任意类型。
    """

    name: str = Field(..., description="工具名称")
    description: str = Field("", description="工具描述")
    input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema 格式参数定义",
    )
    execute_fn: Callable[[Dict[str, Any]], Any] = Field(
        ..., exclude=True, description="实际执行函数"
    )

    class Config:
        arbitrary_types_allowed = True

    async def run(self, arguments: Dict[str, Any]) -> str:
        """执行工具调用，将返回值转为字符串供 LLM 消费。

        Args:
            arguments: 工具参数，Dict[str, Any]。

        Returns:
            字符串形式的执行结果或错误信息。
        """
        try:
            result = await self.execute_fn(arguments)
            if result is None:
                return "执行成功（无返回值）"
            if isinstance(result, str):
                return result
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, default=str)
            return str(result)
        except Exception as exc:
            tb = traceback.format_exc()
            return f"Tool Execution Error [{self.name}]: {exc}\n{tb}"

    def to_openai_tool(self) -> Dict[str, Any]:
        """转换为 OpenAI 格式的工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """转换为 Anthropic Claude 格式的工具定义。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        return (
            f"StandardTool(name={self.name!r}, "
            f"schema_keys=list({self.input_schema.get('properties', {})}))"
        )