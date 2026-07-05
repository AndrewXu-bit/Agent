#!/usr/bin/env python
"""
TaskAssistant - 智能任务助手主入口。

用法:
    python run_demo.py              # 启动交互式对话模式
    python run_demo.py --query "查询北京天气"   # 单次查询
    python run_demo.py --config config.yaml     # 指定配置文件
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from agent_framework import (
    AnthropicClient,
    MockLLMClient,
    OpenAICompatibleClient,
    UniversalAgent,
    get_config,
)
from agent_framework.core.tool_registry import get_default_registry


def setup_logging():
    """配置日志系统。"""
    import logging
    
    config = get_config()
    log_config = config.get_logging_config()
    
    level = getattr(logging, log_config.get("level", "INFO"))
    format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logging.basicConfig(level=level, format=format_str)
    
    if log_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(format_str))


def create_llm_client():
    """根据配置创建 LLM 客户端。"""
    config = get_config()
    llm_config = config.get_llm_config()
    provider = llm_config.get("provider", "mock")
    
    if provider == "openai":
        openai_cfg = llm_config.get("openai", {})
        return OpenAICompatibleClient(
            api_key=openai_cfg.get("api_key", ""),
            base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
            model=openai_cfg.get("model", "gpt-4o"),
            max_tokens=openai_cfg.get("max_tokens", 4096),
            temperature=openai_cfg.get("temperature", 0.7),
            http_timeout=openai_cfg.get("timeout", 120),
        )
    elif provider == "anthropic":
        anthropic_cfg = llm_config.get("anthropic", {})
        return AnthropicClient(
            api_key=anthropic_cfg.get("api_key", ""),
            model=anthropic_cfg.get("model", "claude-sonnet-5-20251001"),
            max_tokens=anthropic_cfg.get("max_tokens", 4096),
            anthropic_version=anthropic_cfg.get("version", "2023-06-01"),
        )
    else:  # mock
        mock_cfg = llm_config.get("mock", {})
        return MockLLMClient(
            model=mock_cfg.get("model", "mock-model"),
            mock_response=mock_cfg.get("response"),
        )


async def load_tools(agent: UniversalAgent):
    """加载配置中启用的工具。"""
    config = get_config()
    enabled_tools = config.get("tools.enabled", [])
    
    # 导入所有工具模块使其注册
    from agent_framework.examples import web_search_tool  # noqa: F401
    from agent_framework.tools import productivity  # noqa: F401
    from agent_framework.tools import rag_tool  # noqa: F401
    
    registry = get_default_registry()
    
    # 只挂载配置中启用的工具
    for tool_name in enabled_tools:
        tool = registry.get_tool(tool_name)
        if tool:
            agent.mount_tool(tool)
        else:
            print(f"⚠️ 警告: 工具 '{tool_name}' 未找到，跳过")
    
    # 挂载 MCP 服务
    mcp_config = config.get("mcp", {})
    if mcp_config.get("enabled", False):
        servers = mcp_config.get("servers", [])
        for server_cfg in servers:
            name = server_cfg.get("name", "mcp-server")
            transport = server_cfg.get("transport", "stdio")
            try:
                from agent_framework.core.mcp_adapter import MCPClientAdapter
                adapter = MCPClientAdapter(
                    server_name=name,
                    transport=transport,
                    command=server_cfg.get("command"),
                    args=server_cfg.get("args"),
                    url=server_cfg.get("url"),
                )
                await agent.mount_mcp_server(adapter)
                print(f"✅ MCP 服务挂载成功: {name}")
            except Exception as e:
                print(f"⚠️ MCP 服务挂载失败 {name}: {e}")


async def interactive_mode():
    """交互式对话模式。"""
    config = get_config()
    agent_config = config.get_agent_config()
    
    print("=" * 60)
    print(f"🤖 {agent_config.get('name', 'TaskAssistant')} 已启动")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    llm = create_llm_client()
    agent = UniversalAgent(
        llm=llm,
        name=agent_config.get("name", "TaskAssistant"),
        system_prompt=agent_config.get("system_prompt"),
        max_steps=agent_config.get("max_steps", 10),
        verbose=agent_config.get("verbose", True),
    )
    
    await load_tools(agent)
    print(f"\n✅ 已加载 {len(agent.list_tools())} 个工具")

    while True:
        try:
            user_input = input("\n👤 您: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 再见！")
                break
            
            if not user_input.strip():
                continue
            
            result = await agent.run(user_input)
            print(f"\n🤖 助手: {result}")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


async def single_query(query: str):
    """执行单次查询。"""
    config = get_config()
    agent_config = config.get_agent_config()
    
    llm = create_llm_client()
    agent = UniversalAgent(
        llm=llm,
        name=agent_config.get("name", "TaskAssistant"),
        system_prompt=agent_config.get("system_prompt"),
        max_steps=agent_config.get("max_steps", 10),
        verbose=agent_config.get("verbose", True),
    )
    
    await load_tools(agent)
    
    print(f"\n{'='*60}")
    print(f"👤 查询: {query}")
    print(f"{'='*60}\n")
    
    result = await agent.run(query)
    print(f"\n🤖 结果: {result}")


def main():
    parser = argparse.ArgumentParser(description="TaskAssistant - 智能任务助手")
    parser.add_argument("--query", "-q", type=str, help="执行单次查询")
    parser.add_argument("--config", "-c", type=str, help="指定配置文件路径")
    
    args = parser.parse_args()
    
    # 初始化配置
    if args.config:
        from agent_framework.core.config import Config
        Config.reset()
        get_config(args.config)
    
    # 设置日志
    setup_logging()
    
    # 运行模式
    if args.query:
        asyncio.run(single_query(args.query))
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()