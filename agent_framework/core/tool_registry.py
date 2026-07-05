"""
本地工具注册器 - 通过装饰器一键注册 Python 本地函数。

根据 framework.md 2.2 节设计：
支持 @tool 装饰器，将普通 Python 函数快速注册为框架
的 StandardTool，自动从函数签名和 docstring 推断 schemas。
"""

from __future__ import annotations

import inspect
import json
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, get_type_hints

from pydantic import create_model

from .standard_tool import StandardTool


def _parse_docstring(doc: Optional[str]) -> str:
    """从 docstring 提取描述（第一行）。"""
    if not doc:
        return ""
    lines = doc.strip().split("\n")
    # 去掉可能的 """ 和前后空格
    cleaned = []
    for line in lines:
        line = line.strip()
        if line in ('"""', "'''"):
            continue
        cleaned.append(line)
    return " ".join(cleaned) if cleaned else ""


def _type_to_json_schema(tp: type) -> Dict[str, Any]:
    """Python 类型 → JSON Schema 类型映射。"""
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        Any: {"type": "string"},
    }
    # 处理 Optional[X] → 加上 nullable
    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", ())
    if origin is type(None) or origin is None:
        return {"type": "null"}
    if origin is list:
        item_type = args[0] if args else str
        return {"type": "array", "items": _type_to_json_schema(item_type)}
    if origin is dict:
        return {"type": "object"}
    return mapping.get(tp, {"type": "string"})


def _func_to_schema(func: Callable) -> Dict[str, Any]:
    """从函数签名自动生成 JSON Schema。"""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name == "self" or name == "cls":
            continue
        # 推断类型
        tp = hints.get(name, str)
        prop = _type_to_json_schema(tp)
        # 从 param.default 判断是否必填
        if param.default is inspect.Parameter.empty:
            required.append(name)
            prop["description"] = f"参数 {name}（必填）"
        else:
            prop["default"] = (
                param.default if not isinstance(param.default, type) else None
            )
            prop["description"] = f"参数 {name}（可选，默认: {param.default}）"
        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


class LocalToolRegistry:
    """本地工具注册器。

    管理一组 StandardTool，提供装饰器注册和批量注册 API。

    Usage:
        registry = LocalToolRegistry()

        @registry.register("calculator", "执行数学运算", {
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expr"]
        })
        async def calculator(args):
            return eval(args["expr"])
    """

    def __init__(self):
        self.tools: Dict[str, StandardTool] = {}

    def register(
        self,
        name: str,
        description: str = "",
        schema: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """工具注册装饰器。

        Args:
            name: 工具名称。
            description: 工具描述。留空则从函数 docstring 提取。
            schema: JSON Schema。留空则从函数签名自动推断。

        Returns:
            装饰器函数。
        """

        def decorator(func: Callable) -> Callable:
            nonlocal description, schema
            if not description:
                description = _parse_docstring(func.__doc__)
            if not schema:
                schema = _func_to_schema(func)

            @wraps(func)
            async def wrapper(args: Dict[str, Any]) -> Any:
                if inspect.iscoroutinefunction(func):
                    return await func(**args)
                return func(**args)

            tool = StandardTool(
                name=name,
                description=description,
                input_schema=schema or {"type": "object", "properties": {}},
                execute_fn=wrapper,
            )
            self.tools[name] = tool
            return func

        return decorator

    def register_func(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable:
        """直接注册一个函数（无需装饰器语法糖）。"""
        tool_name = name or func.__name__
        tool_desc = description or _parse_docstring(func.__doc__)
        schema = _func_to_schema(func)

        return self.register(tool_name, tool_desc, schema)(func)

    def get_tool(self, name: str) -> Optional[StandardTool]:
        """按名称获取工具。"""
        return self.tools.get(name)

    def list_tools(self) -> List[StandardTool]:
        """返回所有已注册工具列表。"""
        return list(self.tools.values())

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        return name in self.tools


# ======== 全局快捷装饰器 ========

_default_registry = LocalToolRegistry()


def tool(
    name: Optional[str] = None,
    description: str = "",
    schema: Optional[Dict[str, Any]] = None,
):
    """全局快捷装饰器，自动注册到默认全局 Registry。

    Usage:
        @tool("my_func", "我的工具函数")
        async def my_func(a: int, b: str) -> str:
            return f"{a}: {b}"
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name if name else func.__name__
        _default_registry.register(tool_name, description or "", schema)(func)
        return func

    return decorator


def get_default_registry() -> LocalToolRegistry:
    """获取默认全局 Registry。"""
    return _default_registry


# 全局快捷引用（用于 import registry）
registry = _default_registry