# TaskAssistant — 智能任务助手

基于多模型与 MCP 协议的通用 Agent 框架，集成飞书 Bot、本地 RAG 知识库和高德地图 MCP 服务，支持自然语言交互式任务处理。

---

## ✨ 核心特性

| 能力 | 说明 |
|------|------|
| 🤖 **多模型支持** | OpenAI GPT、Anthropic Claude、DeepSeek (含 v4 推理模型)、智谱 GLM 等 |
| 🔧 **统一工具接口** | 基于 Pydantic 的 `StandardTool`，抹平本地函数和 MCP 远程工具的差异 |
| 🧠 **RAG 本地知识库** | BGE-M3 多语言嵌入 + BGE-Reranker-Large 重排序 + Chroma 向量存储 |
| 🔌 **MCP 协议集成** | 支持 stdio / SSE 两种传输方式，动态拉取远程工具 |
| 🗺️ **高德地图服务** | 通过 MCP SSE 接入 15 个地图工具（地理编码、路线规划、POI 搜索等） |
| 🕊️ **飞书 Bot 接入** | 完整的消息接收 / Webhook 验证 / API 回复能力 |
| ⚙️ **YAML 配置管理** | 所有配置集中管理，支持环境变量替换和点号路径访问 |
| 🔄 **多轮推理** | 编排引擎自动管理 `推理 → 工具调用 → 结果反馈 → 再推理` 闭环 |

---

## 🏗️ 系统架构

```
用户输入 ──→ 飞书 Bot (HTTP Webhook) / CLI (终端)
                  │
                  ▼
         ┌────────────────────┐
         │   UniversalAgent   │  ← 编排引擎 (推理-工具调用 闭环)
         │   (agent.py)       │
         └───────┬────────────┘
                 │
    ┌────────────┼──────────────┐
    ▼            ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────────┐
│ LLM    │ │ 工具系统  │ │ MCP 适配器    │
│ 客户端  │ │          │ │              │
├────────┤ ├──────────┤ ├──────────────┤
│OpenAI  │ │本地工具:  │ │ 高德地图 SSE  │
│DeepSeek│ │ get_time  │ │ (15个工具)   │
│Claude  │ │get_weather│ │              │
│Qwen等  │ │fetch_url  │ │ 可扩展其他   │
└────────┘ │calculate │ │ MCP Server   │
           │          │ └──────────────┘
           │RAG桥接:  │
           │query_    │
           │knowledge_│────→ RAG 知识库
           │base      │     ├─ BGE-M3 嵌入
           └──────────┘     ├─ BGE-Reranker 重排序
                            ├─ Chroma 向量存储
                            └─ 混合检索 (向量+BM25)
```

---

## 📁 项目结构

```
Agent/
├── main.py                          # CLI 启动入口（交互式 / 单次查询）
├── feishu_bot.py                    # 飞书 Bot 启动入口
├── config.yaml                      # 主配置文件
├── requirements.txt                 # Python 依赖
├── framework.md                     # 框架设计文档
│
├── agent_framework/                 # 核心框架层
│   ├── core/
│   │   ├── agent.py                 # UniversalAgent — 编排引擎
│   │   ├── llm_client.py            # 多模型适配（OpenAI/DeepSeek/Claude）
│   │   ├── standard_tool.py         # 统一工具接口定义
│   │   ├── tool_registry.py         # 工具注册器 + @tool 装饰器
│   │   ├── mcp_adapter.py           # MCP 客户端适配器（stdio/SSE）
│   │   └── config.py                # YAML 配置管理 + 环境变量替换
│   ├── tools/
│   │   ├── productivity.py          # 生产力工具（时间/日期计算）
│   │   └── rag_tool.py              # RAG 知识库桥接工具
│   └── examples/
│       └── web_search_tool.py       # 信息获取工具（天气/URL抓取/计算器）
│
├── RAG/                             # RAG 知识库子系统
│   ├── engine.py                    # RAGEngine 总控
│   ├── embedding.py                 # BGE-M3 嵌入模型封装
│   ├── reranker.py                  # BGE-Reranker-Large 重排序器
│   ├── vector_store.py              # Chroma 向量数据库
│   ├── retriever.py                 # 混合检索器（向量 + BM25）
│   ├── document_loader.py           # 多格式文档加载器
│   ├── text_splitter.py             # 智能文本分块
│   ├── build_index.py               # 文档入库 CLI
│   ├── query.py                     # 知识库查询 CLI
│   └── data/documents/              # 待索引文档目录
│
├── feishu/                          # 飞书集成子系统
│   ├── bot.py                       # FeishuBot — 消息处理核心
│   ├── client.py                    # FeishuClient — API 客户端
│   └── server.py                    # FeishuWebhookServer — FastAPI 服务
│
└── logs/                            # 日志目录
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 `config.yaml`

最小配置仅需设置 LLM：

```yaml
llm:
  provider: "openai"
  openai:
    api_key: "your-deepseek-api-key"
    base_url: "https://api.deepseek.com"
    model: "deepseek-v4-flash"
    max_tokens: 4096
    temperature: 0.7

agent:
  name: "TaskAssistant"
  max_steps: 10
  verbose: true
```

> 💡 支持环境变量：`api_key: "${DEEPSEEK_API_KEY}"` 会从环境变量读取

### 3. 运行

**CLI 交互式模式：**
```bash
python main.py
```

**单次查询：**
```bash
python main.py --query "北京今天天气怎么样？"
```

**启动飞书 Bot：**
```bash
python feishu_bot.py
```

---

## ⚙️ 配置详解

### LLM 模型

```yaml
llm:
  provider: "openai"          # openai / anthropic / mock
  openai:
    api_key: "${API_KEY}"     # 支持环境变量 ${VAR}
    base_url: "https://api.deepseek.com"
    model: "deepseek-v4-flash"
    max_tokens: 4096
    temperature: 0.7
```

支持的模型后端：DeepSeek (v3/v4)、OpenAI GPT-4o、智谱 GLM、通义千问、Claude 等。

### Agent 行为

```yaml
agent:
  name: "TaskAssistant"
  system_prompt: "你是一个智能任务助手..."
  max_steps: 10               # 最大工具调用轮次
  verbose: true               # 是否打印详细执行日志
```

### 工具启用

```yaml
tools:
  enabled:
    - query_knowledge_base    # RAG 本地知识库查询
    - get_time                # 当前时间
    - get_weather             # 天气查询（支持多日预报）
    - fetch_url               # URL 内容抓取
    - calculate               # 安全计算器

  weather:
    default_city: "Beijing"
    timeout: 15
```

### RAG 知识库

```yaml
rag:
  embedding:
    model_path: "RAG/model/BAAI/bge-m3"    # 本地 BGE-M3 路径
    device: null                            # null=自动; cuda/cpu
    max_length: 8192

  reranker:
    model_path: "RAG/model/BAAI/bge-reranker-large"

  docs_dir: "RAG/data/documents"            # 文档目录
  chunk_size: 512                           # 分块大小

  chroma:
    collection: "rag_knowledge"
    persist_dir: "./chroma_db"

  similarity_top_k: 10
  rerank_top_k: 5
  similarity_threshold: 0.5                 # 相关性阈值
  use_reranker: true
  use_hybrid_retrieval: true                # 向量 + BM25 混合检索
```

### MCP 服务

```yaml
mcp:
  enabled: true
  servers:
    - name: "amap-map"
      transport: "sse"
      url: "https://mcp.amap.com/sse?key=YOUR_AMAP_KEY"
```

### 飞书 Bot

```yaml
feishu:
  app_id: "cli_xxxxxxxx"
  app_secret: "your-app-secret"
  verification_token: "your-verification-token"
  host: "0.0.0.0"
  port: 8000
```

---

## 🧠 RAG 知识库

### 构建索引

将文档放入 `RAG/data/documents/`，然后执行：

```bash
python RAG/build_index.py
```

支持的文档格式：`.txt`、`.md`、`.html`、`.pdf`、`.docx`、`.pptx` 等。

### 工作原理

```
用户提问
  → BGE-M3 嵌入模型编码查询向量
  → Chroma 向量数据库检索 top-10
  → BM25 关键词检索 top-10（混合检索）
  → QueryFusion 融合双路结果
  → BGE-Reranker-Large 重排序 top-5
  → 相似度阈值过滤 (默认 0.5)
  → 返回相关结果 / "未找到"
```

### 相关性过滤

知识库只返回相似度 ≥ 0.5 的结果。当用户问题与本地文档无关时，Agent 会自动转向联网工具（`fetch_url` 等）获取信息，避免"强行匹配"不相关内容。

---

## 🕊️ 飞书 Bot 集成

### 消息处理流程

```
飞书服务器 → POST /webhook/event → FeishuWebhookServer
  → URL 验证（启动时） / 事件验证（运行时）
  → 文本消息 → FeishuBot.handle_message_event()
    → UniversalAgent.run(user_text)
    → 工具调用 (MCP/RAG/本地)
    → LLM 生成回复
  → FeishuClient.send_text_message() → 飞书 API → 用户
```

### 部署

1. 在飞书开放平台创建应用，启用机器人能力
2. 配置事件订阅 URL: `https://your-domain/webhook/event`
3. 填写 `config.yaml` 中的飞书参数
4. 运行 `python feishu_bot.py`

---

## 🗺️ 高德地图 MCP 服务

通过 MCP SSE 协议接入高德地图，提供 15 个工具：

| 工具 | 说明 |
|------|------|
| `maps_geo` | 地理编码（地址 → 坐标） |
| `maps_regeocode` | 逆地理编码（坐标 → 地址） |
| `maps_text_search` | POI 文本搜索 |
| `maps_around_search` | 周边搜索 |
| `maps_search_detail` | POI 详情查询 |
| `maps_direction_driving` | 驾车路线规划 |
| `maps_direction_transit_integrated` | 公交路线规划 |
| `maps_direction_walking` | 步行路线规划 |
| `maps_direction_bicycling` | 骑行路线规划 |
| `maps_distance` | 距离测量 |
| `maps_weather` | 天气查询 |
| `maps_ip_location` | IP 定位 |
| `maps_schema_navi` | 调起导航 |
| `maps_schema_take_taxi` | 调起打车 |
| `maps_schema_personal_map` | 个人地图 |

配置只需在 `config.yaml` 中填写高德 MCP 的 SSE URL 和 Key，Agent 会自动挂载所有工具。

---

## 🔧 添加自定义工具

使用 `@tool` 装饰器快速注册：

```python
from agent_framework.core.tool_registry import tool

@tool("my_tool", "我的自定义工具描述")
async def my_tool(param1: str, param2: int = 10) -> str:
    """工具函数实现。"""
    return f"执行结果: {param1} × {param2} = {param1 * (param2 if isinstance(param2, int) else 10)}"
```

然后在 `config.yaml` 中启用：

```yaml
tools:
  enabled:
    - my_tool
    - get_time
```

---

## 🔌 接入外部 MCP Server

除了高德地图，可以接入任何符合 MCP 协议的服务：

```yaml
mcp:
  enabled: true
  servers:
    - name: "my-service"
      transport: "stdio"            # 或 "sse"
      command: "python"
      args: ["-m", "my_mcp_server"]
```

---

## 📖 文档

- 框架设计：[framework.md](framework.md)
- RAG 使用指南：[RAG/USAGE_GUIDE.md](RAG/USAGE_GUIDE.md)
- RAG 快速入门：[RAG/QUICKSTART.md](RAG/QUICKSTART.md)
- 飞书接入：[feishu/README.md](feishu/README.md)

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| LLM 客户端 | httpx (异步 HTTP) |
| 工具接口 | Pydantic v2 |
| 向量嵌入 | FlagEmbedding (BGE-M3) |
| 重排序 | Transformers (BGE-Reranker-Large) |
| 向量数据库 | ChromaDB |
| 文本处理 | LlamaIndex |
| MCP 协议 | mcp Python SDK |
| 飞书 API | httpx + FastAPI + uvicorn |
| 配置管理 | PyYAML |

---

## 📝 License

MIT License
