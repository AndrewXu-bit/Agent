"""
TaskAssistant - 智能任务助手框架
==================================

基于 StandardTool 标准化抽象与 Unified Tool Interface 设计哲学，
支持多种大语言模型后端、本地工具注册和 MCP 服务挂载。
通过 YAML 配置文件集中管理所有应用配置。
"""

__version__ = "1.0.0"

from .core.standard_tool import StandardTool
from .core.tool_registry import LocalToolRegistry, get_default_registry, registry, tool
from .core.llm_client import BaseLLMClient, OpenAICompatibleClient, AnthropicClient, MockLLMClient
from .core.agent import UniversalAgent
from .core.mcp_adapter import MCPClientAdapter
from .core.config import Config, get_config

__all__ = [
    "StandardTool",
    "LocalToolRegistry",
    "get_default_registry",
    "registry",
    "tool",
    "BaseLLMClient",
    "OpenAICompatibleClient",
    "AnthropicClient",
    "MockLLMClient",
    "UniversalAgent",
    "MCPClientAdapter",
    "Config",
    "get_config",
]