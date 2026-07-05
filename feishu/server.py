"""
飞书 Webhook 服务器 - 接收飞书事件推送
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .bot import FeishuBot

logger = logging.getLogger(__name__)


class FeishuWebhookServer:
    """飞书 Webhook 服务器
    
    提供 HTTP 接口接收飞书事件推送，并调用 Bot 进行处理
    """
    
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        verification_token: str,
        encrypt_key: Optional[str] = None,
        config_path: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """初始化 Webhook 服务器
        
        Args:
            app_id: 飞书应用 App ID
            app_secret: 飞书应用 App Secret
            verification_token: 飞书事件订阅验证 Token
            encrypt_key: 可选，加密密钥（如果启用了加密）
            config_path: Agent 配置文件路径
            host: 服务器监听地址
            port: 服务器监听端口
        """
        self.app_id = app_id
        self.app_secret = app_secret
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        self.host = host
        self.port = port
        
        # 创建 Bot 实例
        self.bot = FeishuBot(app_id, app_secret, config_path)
        
        # 创建 FastAPI 应用
        self.app = FastAPI(title="Feishu Bot Webhook")
        
        # 注册路由
        self._register_routes()
    
    def _register_routes(self):
        """注册 HTTP 路由"""
        
        @self.app.post("/webhook/event")
        async def webhook_endpoint(request: Request):
            """接收飞书事件推送
            
            Args:
                request: FastAPI 请求对象
                
            Returns:
                JSON 响应
            """
            try:
                # 获取请求体
                body = await request.body()
                event_data = json.loads(body)
                
                # 验证请求
                if not self._verify_request(event_data):
                    logger.warning("请求验证失败")
                    raise HTTPException(status_code=403, detail="Invalid signature")
                
                # 处理 URL 验证挑战
                if event_data.get("type") == "url_verification":
                    challenge = event_data.get("challenge", "")
                    logger.info("URL 验证成功")
                    return {"challenge": challenge}
                
                # 处理事件
                event_type = event_data.get("header", {}).get("event_type")
                
                if event_type == "im.message.receive_v1":
                    # 异步处理消息事件
                    message_id = await self.bot.handle_message_event(event_data)
                    
                    return {
                        "code": 0,
                        "msg": "success",
                        "data": {"message_id": message_id},
                    }
                else:
                    logger.info(f"忽略事件类型: {event_type}")
                    return {"code": 0, "msg": "ignored"}
                    
            except json.JSONDecodeError:
                logger.error("无效的 JSON 数据")
                raise HTTPException(status_code=400, detail="Invalid JSON")
            except Exception as e:
                logger.error(f"处理 webhook 失败: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """健康检查接口"""
            return {"status": "healthy"}
    
    def _verify_request(self, event_data: Dict[str, Any]) -> bool:
        """验证飞书请求签名
        
        Args:
            event_data: 事件数据
            
        Returns:
            验证是否通过
        """
        # 如果是 URL 验证请求，只需要检查 token
        if event_data.get("type") == "url_verification":
            token = event_data.get("token", "")
            return token == self.verification_token
        
        # 验证事件请求
        header = event_data.get("header", {})
        token = header.get("token", "")
        
        # 简单验证：检查 token 是否匹配
        # 注意：生产环境应该使用更严格的签名验证
        return token == self.verification_token
    
    def run(self):
        """启动 Webhook 服务器"""
        import uvicorn
        
        logger.info(f"启动飞书 Webhook 服务器: http://{self.host}:{self.port}")
        logger.info(f"Webhook URL: http://{self.host}:{self.port}/webhook/event")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
        )
    
    @staticmethod
    def create_from_config(
        config_path: Optional[str] = None,
        feishu_config: Optional[Dict[str, Any]] = None,
    ) -> "FeishuWebhookServer":
        """从配置创建 Webhook 服务器
        
        Args:
            config_path: 配置文件路径
            feishu_config: 飞书配置字典（可选）
            
        Returns:
            FeishuWebhookServer 实例
        """
        if feishu_config is None:
            from agent_framework.core.config import Config
            
            config = Config(config_path) if config_path else Config.instance()
            feishu_config = config.data.get("feishu", {})
        
        app_id = feishu_config.get("app_id", "")
        app_secret = feishu_config.get("app_secret", "")
        verification_token = feishu_config.get("verification_token", "")
        encrypt_key = feishu_config.get("encrypt_key")
        host = feishu_config.get("host", "0.0.0.0")
        port = feishu_config.get("port", 8000)
        
        if not app_id or not app_secret or not verification_token:
            raise ValueError("飞书配置不完整：需要 app_id, app_secret, verification_token")
        
        return FeishuWebhookServer(
            app_id=app_id,
            app_secret=app_secret,
            verification_token=verification_token,
            encrypt_key=encrypt_key,
            config_path=config_path,
            host=host,
            port=port,
        )
