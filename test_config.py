#!/usr/bin/env python
"""
测试配置管理模块。
"""

from agent_framework.core.config import get_config


def test_config_loading():
    """测试配置加载。"""
    config = get_config()
    
    print("=" * 60)
    print("测试配置加载")
    print("=" * 60)
    
    # 测试 LLM 配置
    print("\n1. LLM 配置:")
    llm_config = config.get_llm_config()
    print(f"   Provider: {llm_config.get('provider')}")
    print(f"   Model: {llm_config.get('openai', {}).get('model')}")
    
    # 测试 Agent 配置
    print("\n2. Agent 配置:")
    agent_config = config.get_agent_config()
    print(f"   Name: {agent_config.get('name')}")
    print(f"   Max Steps: {agent_config.get('max_steps')}")
    
    # 测试工具配置
    print("\n3. 工具配置:")
    tools_config = config.get_tools_config()
    print(f"   Enabled Tools: {tools_config.get('enabled')}")
    print(f"   Weather Default City: {tools_config.get('weather', {}).get('default_city')}")
    
    # 测试日志配置
    print("\n4. 日志配置:")
    log_config = config.get_logging_config()
    print(f"   Level: {log_config.get('level')}")
    print(f"   Console: {log_config.get('console')}")
    
    # 测试点号路径访问
    print("\n5. 点号路径访问测试:")
    print(f"   llm.provider = {config.get('llm.provider')}")
    print(f"   agent.name = {config.get('agent.name')}")
    print(f"   tools.weather.default_city = {config.get('tools.weather.default_city')}")
    
    print("\n✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_config_loading()
