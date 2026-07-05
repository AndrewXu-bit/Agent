"""
飞书机器人核心逻辑 - 接收消息、调用 Agent、回复消息
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from agent_framework.core.agent import UniversalAgent
from agent_framework.core.config import Config

from .client import FeishuClient

logger = logging.getLogger(__name__)


class FeishuBot:
    """飞书机器人
    
    负责处理飞书消息事件，调用 Agent 进行智能回复
    """
    
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        config_path: Optional[str] = None,
    ):
        """初始化飞书机器人
        
        Args:
            app_id: 飞书应用 App ID
            app_secret: 飞书应用 App Secret
            config_path: Agent 配置文件路径
        """
        self.client = FeishuClient(app_id, app_secret)
        
        # 加载 Agent 配置
        if config_path:
            self.config = Config(config_path)
        else:
            self.config = Config.instance()
        
        # 创建 Agent 实例
        self.agent = self._create_agent()
        
        # 消息去重缓存（message_id -> True）
        self._processed_messages: set = set()
        # 最多保留 1000 条记录
        self._max_cache_size = 1000
    
    def _create_agent(self) -> UniversalAgent:
        """根据配置创建 Agent 实例
        
        Returns:
            UniversalAgent 实例
        """
        llm_config = self.config.get_llm_config()
        provider = llm_config.get("provider", "mock")
        
        # 根据 provider 创建对应的 LLM 客户端
        if provider == "openai":
            from agent_framework.core.llm_client import OpenAICompatibleClient
            
            openai_cfg = llm_config.get("openai", {})
            llm_client = OpenAICompatibleClient(
                api_key=openai_cfg.get("api_key", ""),
                base_url=openai_cfg.get("base_url", ""),
                model=openai_cfg.get("model", "gpt-4"),
                max_tokens=openai_cfg.get("max_tokens", 4096),
                temperature=openai_cfg.get("temperature", 0.7),
            )
        elif provider == "anthropic":
            from agent_framework.core.llm_client import AnthropicClient
            
            anthropic_cfg = llm_config.get("anthropic", {})
            llm_client = AnthropicClient(
                api_key=anthropic_cfg.get("api_key", ""),
                model=anthropic_cfg.get("model", "claude-3-opus-20240229"),
                max_tokens=anthropic_cfg.get("max_tokens", 4096),
            )
        else:
            from agent_framework.core.llm_client import MockLLMClient
            
            llm_client = MockLLMClient()
        
        # 创建 Agent
        agent_config = self.config.get_agent_config()
        agent = UniversalAgent(
            llm=llm_client,
            name=agent_config.get("name", "FeishuAssistant"),
            system_prompt=agent_config.get("system_prompt", ""),
            max_steps=agent_config.get("max_steps", 10),
            verbose=agent_config.get("verbose", False),
        )
        
        # 挂载工具
        enabled_tools = self.config.get("tools.enabled", [])
        
        # 导入所有工具模块使其注册
        from agent_framework.examples import web_search_tool
        from agent_framework.tools import productivity
        from agent_framework.tools import rag_tool  # RAG 知识库查询工具
        from agent_framework.core.tool_registry import get_default_registry
        tool_registry = get_default_registry()
        
        for tool_name in enabled_tools:
            tool = tool_registry.get_tool(tool_name)
            if tool:
                agent.mount_tool(tool)
                logger.info(f"挂载工具: {tool_name}")
            else:
                logger.warning(f"工具未找到: {tool_name}")
        
        return agent
    
    async def _ensure_mcp_mounted(self):
        """确保 MCP 服务已挂载（懒加载，只执行一次）。"""
        if hasattr(self, '_mcp_mounted') and self._mcp_mounted:
            return
        
        self._mcp_mounted = False
        mcp_config = self.config.get("mcp", {})
        if not mcp_config.get("enabled", False):
            logger.info("MCP 服务未启用，跳过")
            self._mcp_mounted = True
            return
        
        servers = mcp_config.get("servers", [])
        if not servers:
            logger.info("MCP 服务列表为空，跳过")
            self._mcp_mounted = True
            return
        
        from agent_framework.core.mcp_adapter import MCPClientAdapter
        
        for server_cfg in servers:
            name = server_cfg.get("name", "mcp-server")
            transport = server_cfg.get("transport", "stdio")
            
            try:
                adapter = MCPClientAdapter(
                    server_name=name,
                    transport=transport,
                    command=server_cfg.get("command"),
                    args=server_cfg.get("args"),
                    url=server_cfg.get("url"),
                )
                await self.agent.mount_mcp_server(adapter)
                logger.info(f"MCP 服务挂载成功: {name} ({transport})")
            except Exception as e:
                logger.error(f"MCP 服务挂载失败 {name}: {e}", exc_info=True)
        
        self._mcp_mounted = True
    
    async def handle_message_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """处理飞书消息事件
        
        Args:
            event_data: 飞书事件数据
            
        Returns:
            回复的消息 ID，如果不需要回复则返回 None
        """
        try:
            # 解析事件
            event = event_data.get("event", {})
            message = event.get("message", {})
            sender = event.get("sender", {})
            
            message_id = message.get("message_id")
            chat_type = message.get("chat_type")
            message_type = message.get("message_type")
            content_str = message.get("content", "{}")
            
            # 消息去重
            if message_id in self._processed_messages:
                logger.info(f"消息已处理，跳过: {message_id}")
                return None
            
            # 只处理文本消息
            if message_type != "text":
                logger.info(f"跳过非文本消息: {message_type}")
                return None
            
            # 获取发送者信息
            sender_id = sender.get("sender_id", {})
            open_id = sender_id.get("open_id")
            sender_type = sender.get("sender_type")
            
            # 忽略机器人发送的消息
            if sender_type == "bot":
                logger.info("忽略机器人消息")
                return None
            
            # 解析消息内容
            content = json.loads(content_str)
            text = content.get("text", "")
            
            if not text:
                logger.warning("消息内容为空")
                return None
            
            logger.info(f"收到消息: chat_type={chat_type}, open_id={open_id}, text={text[:50]}")
            
            # 调用 Agent 处理
            response_text = await self._process_with_agent(text)
            
            if not response_text:
                logger.warning("Agent 未生成回复")
                return None
            
            # 确定接收者 ID 和类型
            if chat_type == "p2p":
                # 单聊：回复给发送者
                receive_id = open_id
                receive_id_type = "open_id"
            else:
                # 群聊：回复到群组
                receive_id = message.get("chat_id")
                receive_id_type = "chat_id"
            
            # 发送回复
            result = await self.client.send_text_message(
                receive_id=receive_id,
                text=response_text,
                receive_id_type=receive_id_type,
            )
            
            # 记录已处理的消息
            self._add_to_processed(message_id)
            
            logger.info(f"回复成功: message_id={result.get('message_id')}")
            return result.get("message_id")
            
        except Exception as e:
            logger.error(f"处理消息事件失败: {e}", exc_info=True)
            return None
    
    async def _process_with_agent(self, user_message: str) -> Optional[str]:
        """使用 Agent 处理用户消息
        
        Args:
            user_message: 用户消息文本
            
        Returns:
            Agent 生成的回复文本
        """
        try:
            logger.info(f"调用 Agent 处理消息: {user_message[:50]}")
            
            # 确保 MCP 服务已挂载（懒加载）
            await self._ensure_mcp_mounted()
            
            # 运行 Agent
            result = await self.agent.run(user_message)
            
            # 清空历史，避免上下文过长
            # 如果需要多轮对话，可以注释掉这行
            self.agent.clear_history()
            
            return result
        except Exception as e:
            logger.error(f"Agent 处理失败: {e}", exc_info=True)
            return f"抱歉，我遇到了一些问题：{str(e)}"
    
    def _add_to_processed(self, message_id: str):
        """添加消息到已处理集合
        
        Args:
            message_id: 消息 ID
        """
        self._processed_messages.add(message_id)
        
        # 控制缓存大小
        if len(self._processed_messages) > self._max_cache_size:
            # 简单策略：清空一半
            items = list(self._processed_messages)
            self._processed_messages = set(items[len(items)//2:])
    
    def clear_processed_cache(self):
        """清空已处理消息缓存"""
        self._processed_messages.clear()
        logger.info("已清空消息缓存")
