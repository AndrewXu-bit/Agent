# TaskAssistant - 智能任务助手

一个基于多模型与 MCP 协议的通用 Agent 框架，支持多种大语言模型后端、本地工具注册和 MCP 服务挂载。

## ✨ 特性

- 🤖 **多模型支持**: OpenAI GPT、Anthropic Claude、DeepSeek、通义千问等
- 🔧 **统一工具接口**: 标准化的工具注册和调用机制
- ⚙️ **YAML 配置管理**: 所有配置集中管理，支持环境变量替换
- 🔌 **MCP 协议集成**: 可连接 Model Context Protocol 服务
- 🎯 **灵活扩展**: 通过装饰器快速添加工具

## 📋 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置应用

编辑 `config.yaml` 文件：

```yaml
llm:
  provider: "mock"  # mock, openai, anthropic
  
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o"
```

### 3. 运行应用

**交互式模式：**
```bash
python main.py
```

**单次查询：**
```bash
python main.py --query "北京今天的天气怎么样？"
```

## 📁 项目结构

```
Agent/
├── config.yaml                 # 主配置文件
├── main.py                     # 应用入口
├── requirements.txt            # 依赖列表
├── framework.md                # 详细文档
├── agent_framework/
│   ├── core/                   # 核心模块
│   │   ├── agent.py            # Agent 编排引擎
│   │   ├── llm_client.py       # LLM 客户端适配层
│   │   ├── standard_tool.py    # 标准工具接口
│   │   ├── tool_registry.py    # 工具注册器
│   │   ├── mcp_adapter.py      # MCP 适配器
│   │   └── config.py           # 配置管理
│   ├── tools/                  # 业务工具集
│   └── examples/               # 示例工具
└── logs/                       # 日志目录
```

## 🔧 添加工具

使用 `@tool` 装饰器快速注册：

```python
from agent_framework.core.tool_registry import tool

@tool("my_tool", "我的工具描述")
async def my_tool(param1: str, param2: int = 10) -> str:
    """工具函数实现。"""
    return f"结果: {param1}, {param2}"
```

在 `config.yaml` 中启用：

```yaml
tools:
  enabled:
    - my_tool
    - get_time
    - get_weather
```

## 📖 文档

详细文档请查看 [framework.md](framework.md)

## 🛠️ 技术栈

- **Python**: 开发语言
- **Pydantic**: 数据验证和序列化
- **httpx**: HTTP 客户端
- **PyYAML**: YAML 配置解析

## 📝 许可证

MIT License
