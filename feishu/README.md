# 飞书机器人模块

这是一个集成 Agent Framework 的飞书机器人模块，可以接收飞书消息，通过 AI Agent 智能处理后自动回复。

## 功能特性

- ✅ 接收飞书单聊和群聊消息
- ✅ 自动调用 Agent 进行智能处理
- ✅ 支持工具调用（时间查询、天气、URL抓取等）
- ✅ 消息去重，避免重复处理
- ✅ Webhook 事件验证
- ✅ 异步高性能处理

## 前置要求

### 1. 安装依赖

```bash
pip install fastapi uvicorn httpx
```

### 2. 创建飞书应用

1. 访问 [飞书开放平台](https://open.feishu.cn/app)
2. 创建企业自建应用
3. 开启**机器人能力**
4. 配置**事件订阅**：
   - 订阅 `im.message.receive_v1` 事件（接收消息v2.0）
   - 设置请求网址为你的服务器地址（如 `http://your-domain.com/webhook/event`）
5. 获取以下凭证：
   - App ID
   - App Secret
   - Verification Token

### 3. 配置权限

在飞书开发者后台申请以下权限：
- `im:message.p2p_msg:readonly` - 接收单聊消息
- `im:message.group_at_msg:readonly` - 接收群聊@消息
- 或 `im:message.group_msg` - 接收群聊所有消息

## 配置方法

### 方式一：环境变量（推荐）

在项目根目录创建 `.env` 文件：

```bash
FEISHU_APP_ID=cli_xxxxxxxxxxxxx
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxx
FEISHU_VERIFICATION_TOKEN=xxxxxxxxxxxxxxxx
```

### 方式二：直接修改 config.yaml

编辑 `config.yaml` 文件：

```yaml
feishu:
  app_id: "cli_xxxxxxxxxxxxx"
  app_secret: "xxxxxxxxxxxxxxxx"
  verification_token: "xxxxxxxxxxxxxxxx"
  encrypt_key: ""  # 可选
  host: "0.0.0.0"
  port: 8000
```

## 启动机器人

```bash
python feishu_bot.py
```

启动后会看到类似输出：

```
============================================================
启动飞书机器人
============================================================

配置信息:
  Webhook URL: http://0.0.0.0:8000/webhook/event
  健康检查: http://0.0.0.0:8000/health

请在飞书开发者后台配置事件订阅 URL
============================================================
```

## 部署到公网

要让飞书能够访问你的 Webhook，需要将服务暴露到公网。有以下几种方式：

### 方式一：使用内网穿透（开发测试）

使用 ngrok：

```bash
# 安装 ngrok
# 访问 https://ngrok.com 下载

# 启动内网穿透
ngrok http 8000
```

会获得一个公网 URL，如 `https://xxxx.ngrok.io`，将其配置到飞书后台。

### 方式二：部署到云服务器

1. 购买云服务器（阿里云、腾讯云等）
2. 部署应用并绑定域名
3. 配置 HTTPS（飞书要求 HTTPS）

### 方式三：使用 Serverless 平台

可以部署到 Vercel、Railway 等平台。

## 工作流程

```
用户发送消息 → 飞书服务器 → Webhook 服务器 → FeishuBot 
    ↓
Agent 处理 → 调用工具（可选）→ 生成回复
    ↓
飞书 API → 飞书服务器 → 用户收到回复
```

## 消息处理逻辑

1. **接收消息**：Webhook 服务器接收飞书推送的事件
2. **验证签名**：验证请求来源的合法性
3. **消息去重**：使用 message_id 避免重复处理
4. **过滤消息**：只处理文本消息，忽略机器人消息
5. **调用 Agent**：将消息交给 AI Agent 处理
6. **发送回复**：根据聊天类型（单聊/群聊）发送回复

## 自定义扩展

### 添加工具

在 `config.yaml` 中启用更多工具：

```yaml
tools:
  enabled:
    - get_time
    - get_weather
    - fetch_url
    - calculate
    - your_custom_tool  # 添加自定义工具
```

### 修改系统提示词

编辑 `config.yaml`：

```yaml
agent:
  system_prompt: |
    你是飞书智能助手，可以帮助用户解答问题、完成任务。
    你可以使用多种工具来协助工作。
```

### 处理其他事件类型

在 `server.py` 中添加新的事件处理：

```python
if event_type == "im.message.receive_v1":
    # 处理消息
    ...
elif event_type == "your_event_type":
    # 处理其他事件
    ...
```

## 日志查看

日志文件位于 `logs/feishu_bot.log`：

```bash
tail -f logs/feishu_bot.log
```

## 常见问题

### Q: 收不到消息推送？

A: 检查以下几点：
1. 确认飞书后台配置了正确的 Webhook URL
2. 确认 URL 可以通过公网访问
3. 确认已订阅 `im.message.receive_v1` 事件
4. 查看日志是否有错误信息

### Q: 消息回复失败？

A: 检查：
1. 飞书应用是否有发送消息权限
2. 接收者是否在应用可用范围内
3. 查看日志中的错误信息

### Q: 如何实现多轮对话？

A: 目前每次对话都会清空历史。如需多轮对话，注释掉 `bot.py` 中的：

```python
# self.agent.clear_history()
```

### Q: 如何限制某些用户或群组？

A: 在 `bot.py` 的 `handle_message_event` 方法中添加过滤逻辑：

```python
# 只响应特定用户
if open_id not in allowed_users:
    return None

# 只响应特定群组
if chat_type == "group" and message.get("chat_id") not in allowed_groups:
    return None
```

## 技术架构

- **FastAPI**: Web 框架，提供高性能异步 HTTP 服务
- **FeishuClient**: 封装飞书 API 调用（获取 token、发送消息）
- **FeishuBot**: 核心业务逻辑，协调消息处理和 Agent 调用
- **FeishuWebhookServer**: HTTP 服务器，接收飞书事件推送
- **Agent Framework**: AI Agent 引擎，负责智能处理和工具调用

## 安全建议

1. **不要泄露凭证**：App Secret 和 Verification Token 要妥善保管
2. **使用 HTTPS**：生产环境必须使用 HTTPS
3. **严格验证签名**：生产环境应实现完整的签名验证逻辑
4. **限流保护**：添加请求频率限制，防止滥用
5. **日志脱敏**：不要在日志中记录敏感信息

## 许可证

与主项目保持一致
