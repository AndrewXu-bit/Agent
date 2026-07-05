# TaskAssistant - 智能任务助手

TaskAssistant 是一个基于多模型与 MCP 协议的通用 Agent 框架，原生支持**多种第三方大语言模型 (LLM)**、**自定义本地工具注册**，并能直接挂载 **MCP (Model Context Protocol)** 服务。

框架的最核心设计哲学是：**标准化抽象（Normalization）与统一调用协议（Unified Tool Interface）**。

---

## 1. 核心架构与设计理念

为保证框架的高内聚与低耦合，整体系统分为四大核心模块：

+-------------------------------------------------------------------------+
|                        Agent Orchestrator (编排引擎)                     |
|    - 上下文历史管理 (Context)   - ReAct / Function Calling 循环决策     |
+------------------------------------+------------------------------------+
| 统一调用抽象 (Standard Tool Schema)
v
+-------------------------------------------------------------------------+
|                        Unified Tool Manager (统一工具中心)               |
+------------------------------------+------------------------------------+
|                                   |
| 继承统一 Tool 接口                  | JSON-RPC / SSE 适配
v                                   v
+------------------------------------+ +----------------------------------+
|      Local Custom Tool Registry    | |         MCP Client Adapter       |
|  - @tool 装饰器函数 / Pydantic 校验 | |  - 连接 stdio / sse MCP Servers   |
|  - 本地 API 请求 / 数据库操作       | |  - 动态拉取 tools/list 并映射为工具|
+------------------------------------+ +----------------------------------+
^
| 抹平工具定义差异 (OpenAI/Anthropic格式)
v
+-------------------------------------------------------------------------+
|                    Model I/O Layer (模型适配层)                          |
+-------------------------------------------------------------------------+

### 配置管理

所有配置项集中管理在 `config.yaml` 文件中，包括：
- LLM 模型提供商及参数配置
- Agent 行为和系统提示词
- 工具启用列表和特定参数
- MCP 服务连接信息
- 日志级别和输出格式

支持环境变量替换（`${VAR_NAME}` 格式），方便在不同环境中部署。

---

## 2. 核心模块代码实现

### 2.1 统一工具接口 (`StandardTool`)
无论是用户在 Python 中自定义的本地函数，还是通过 MCP 协议挂载的远程服务，在框架内部必须抹平差异，表现为同一种标准化的数据结构。

```python
from pydantic import BaseModel
from typing import Callable, Dict, Any, List
import inspect
import json

class StandardTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]  # 严格符合 JSON Schema 标准
    execute_fn: Callable[[Dict[str, Any]], Any]

    async def run(self, arguments: Dict[str, Any]) -> str:
        """执行工具，确保返回值转化为标准的字符串格式以供模型理解"""
        try:
            result = await self.execute_fn(arguments)
            return str(result) if not isinstance(result, str) else result
        except Exception as e:
            return f"Tool Execution Error: {str(e)}"
2.2 本地工具注册器 (LocalToolRegistry)
支持通过装饰器一键注册自定义的 Python 本地函数（如读写本地数据库、直接调用三方 API 等）：

Python
class LocalToolRegistry:
    def __init__(self):
        self.tools: Dict[str, StandardTool] = {}

    def register(self, name: str, description: str, schema: Dict[str, Any]):
        """工具注册装饰器"""
        def decorator(func):
            async def wrapper(args):
                if inspect.iscoroutinefunction(func):
                    return await func(**args)
                return func(**args)
            
            tool = StandardTool(
                name=name,
                description=description,
                input_schema=schema,
                execute_fn=wrapper
            )
            self.tools[name] = tool
            return func
        return decorator
2.3 MCP 服务客户端适配器 (MCPClientAdapter)
作为客户端连接官方或第三方的 MCP Server（通过 stdio 或 SSE），发送 tools/list 请求将服务端的工具映射为框架标准工具。

Python
class MCPClientAdapter:
    def __init__(self, server_command: str, server_args: List[str]):
        self.server_command = server_command
        self.server_args = server_args

    async def connect_and_fetch_tools(self) -> List[StandardTool]:
        """
        连接 MCP 进程/远程服务，动态获取其能够提供的工具列表，
        并将其封装为框架统一样式的 StandardTool
        """
        standard_tools = []
        # 此处使用伪代码展示核心流程，实际可集成官方 Python MCP SDK:
        # session = await StdioServerParameters(...).connect()
        # mcp_tools = await session.list_tools()
        
        mock_mcp_tools = [] # 模拟获取到的列表
        for mcp_tool in mock_mcp_tools:
            async def create_executor(tool_name):
                async def execute_mcp(args: Dict[str, Any]):
                    # 委托给 MCP Client 发起 JSON-RPC 协议调用: tools/call
                    # return await session.call_tool(tool_name, arguments=args)
                    return f"MCP [{tool_name}] 远程进程执行结果"
                return execute_mcp

            standard_tools.append(
                StandardTool(
                    name=mcp_tool["name"],
                    description=mcp_tool["description"],
                    input_schema=mcp_tool["inputSchema"],
                    execute_fn=await create_executor(mcp_tool["name"])
                )
            )
        return standard_tools
2.4 多模型统一适配层 (Model I/O Layer)
抹平 OpenAI、Claude 等不同模型在 Tool Calling 上的 JSON 定义协议差异。

Python
class BaseLLMClient:
    async def chat(self, messages: List[Dict[str, Any]], tools: List[StandardTool]) -> Dict[str, Any]:
        """统一返回结构：{'role': 'assistant', 'content': '...', 'tool_calls': [...]}"""
        raise NotImplementedError

class OpenAICompatibleClient(BaseLLMClient):
    """适配兼容 OpenAI 格式的大模型（例如 OpenAI, DeepSeek, Qwen 等）"""
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    async def chat(self, messages: List[Dict[str, Any]], tools: List[StandardTool]) -> Dict[str, Any]:
        # 将标准工具定义自动转换成 OpenAI 要求的 specific Schema
        openai_tools = [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema
            }
        } for t in tools] if tools else None
        
        # 发送真实的 API 请求并获得回应 (这里作 Mock 展示)
        return {
            "role": "assistant",
            "content": "我决定调用工具处理用户的请求。",
            "tool_calls": [
                {
                    "id": "call_123", 
                    "function": {"name": "sample_tool", "arguments": '{"query": "test"}'}
                }
            ]
        }
3. 通用编排引擎 (UniversalAgent)
引擎的核心在于完全与“具体的模型底座”、“工具是本地还是远程 MCP”完全解耦，仅负责控制 “推理 -> 工具调用与等待 -> 再次推理” 的闭环控制。

Python
class UniversalAgent:
    def __init__(self, llm: BaseLLMClient):
        self.llm = llm
        self.tools: Dict[str, StandardTool] = {}
        self.history: List[Dict[str, Any]] = []

    def mount_tool(self, tool: StandardTool):
        """挂载单个工具（多用于本地注册工具）"""
        self.tools[tool.name] = tool

    async def mount_mcp_server(self, mcp_adapter: MCPClientAdapter):
        """挂载由远程或本地子进程 MCP Server 提供的整套工具集"""
        mcp_tools = await mcp_adapter.connect_and_fetch_tools()
        for t in mcp_tools:
            self.mount_tool(t)

    async def run(self, user_query: str, max_steps: int = 10) -> str:
        self.history.append({"role": "user", "content": user_query})
        available_tools = list(self.tools.values())

        for step in range(max_steps):
            # 1. 统一调用模型进行推理决策
            response = await self.llm.chat(self.history, tools=available_tools)
            self.history.append(response)

            # 2. 判断当前轮是否触发了工具调用
            tool_calls = response.get("tool_calls")
            if not tool_calls:
                # 若模型不调用工具，直接返回文本对话
                return response.get("content", "")

            # 3. 执行所有工具请求（完全无感于是 Python 内部逻辑还是 MCP 进程协议）
            for call in tool_calls:
                func_name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"])

                if func_name in self.tools:
                    print(f"[分发执行工具]: {func_name} | 参数: {arguments}")
                    result = await self.tools[func_name].run(arguments)
                else:
                    result = f"Error: Tool {func_name} not found."

                # 4. 把执行返回值写回上下文，让模型继续下一次推理
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": func_name,
                    "content": result
                })

        return "系统已达到最高迭代步骤限制，无法生成终局结论。"

## 4. 架构总结

构建面向生产级、具备 MCP 协议特性的 Agent 系统的核心在于“中间抽象层的设计”：

统一模型输入输出格式 (Schema API)：无论底座是大厂 API 还是私有化开源模型，都强制经过 Adapter 转换为同一格式。

构建标准化的 StandardTool 中间件：打通了直接写死在项目里的 Local Python Tools 与处于完全隔离环境的 MCP Server Tools 之间的壁垒，真正让 Agent 框架变成了一个“可任意拔插拓展接口”的超级枢纽。

---

## 5. 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置应用

编辑 `config.yaml` 文件，设置你的 LLM 提供商和 API 密钥：

```yaml
llm:
  provider: "openai"  # 或 "anthropic", "mock"
  
  openai:
    api_key: "${OPENAI_API_KEY}"  # 从环境变量读取
    model: "gpt-4o"
```

或使用环境变量：

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 运行应用

**交互式对话模式：**

```bash
python main.py
```

**单次查询：**

```bash
python main.py --query "北京今天的天气怎么样？"
```

**指定配置文件：**

```bash
python main.py --config path/to/config.yaml
```

---

## 6. 添加工具

### 方法一：使用装饰器

```python
from agent_framework.core.tool_registry import tool

@tool("my_tool", "我的工具描述")
async def my_tool(param1: str, param2: int = 10) -> str:
    """工具函数实现。"""
    return f"结果: {param1}, {param2}"
```

### 方法二：从配置文件启用

在 `config.yaml` 中添加：

```yaml
tools:
  enabled:
    - my_tool
    - get_time
    - get_weather
```

---

## 7. 项目结构

```
Agent/
├── config.yaml                 # 主配置文件
├── main.py                     # 应用入口
├── requirements.txt            # 依赖列表
├── framework.md                # 项目文档
├── agent_framework/
│   ├── __init__.py
│   ├── core/                   # 核心模块
│   │   ├── agent.py            # Agent 编排引擎
│   │   ├── llm_client.py       # LLM 客户端适配层
│   │   ├── standard_tool.py    # 标准工具接口
│   │   ├── tool_registry.py    # 工具注册器
│   │   ├── mcp_adapter.py      # MCP 适配器
│   │   └── config.py           # 配置管理
│   ├── tools/                  # 业务工具集
│   │   ├── __init__.py
│   │   └── productivity.py     # 生产力工具
│   └── examples/               # 示例工具
│       ├── web_search_tool.py  # 网络工具
│       └── ...
└── logs/                       # 日志目录
```

---

## 8. 开发指南

### 创建新工具

1. 在 `agent_framework/tools/` 下创建新模块
2. 使用 `@tool` 装饰器注册函数
3. 在 `config.yaml` 中启用工具
4. 在 `run_demo.py` 的 `load_tools()` 中导入模块

### 切换 LLM 提供商

修改 `config.yaml` 中的 `llm.provider` 字段：
- `openai`: OpenAI 兼容 API（GPT-4, DeepSeek, Qwen 等）
- `anthropic`: Anthropic Claude
- `mock`: 测试用模拟客户端

### 配置 MCP 服务

在 `config.yaml` 中启用 MCP：

```yaml
mcp:
  enabled: true
  servers:
    - name: "filesystem"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
      transport: "stdio"
```