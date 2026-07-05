"""
飞书 API 客户端 - 封装飞书开放平台 API 调用
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class FeishuClient:
    """飞书 API 客户端
    
    负责获取 access_token 和发送消息
    """
    
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        base_url: str = "https://open.feishu.cn",
    ):
        """初始化飞书客户端
        
        Args:
            app_id: 飞书应用 App ID
            app_secret: 飞书应用 App Secret
            base_url: 飞书 API 基础 URL
        """
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = base_url
        self._tenant_access_token: Optional[str] = None
    
    async def get_tenant_access_token(self) -> str:
        """获取 tenant_access_token
        
        Returns:
            tenant_access_token 字符串
        """
        if self._tenant_access_token:
            return self._tenant_access_token
        
        url = f"{self.base_url}/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        }
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if data.get("code") != 0:
                    raise Exception(f"获取 token 失败: {data}")
                
                self._tenant_access_token = data["tenant_access_token"]
                logger.info("成功获取 tenant_access_token")
                return self._tenant_access_token
        except Exception as e:
            logger.error(f"获取 tenant_access_token 失败: {e}")
            raise
    
    async def send_message(
        self,
        receive_id: str,
        receive_id_type: str = "open_id",
        msg_type: str = "text",
        content: str = "",
        uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送消息
        
        Args:
            receive_id: 消息接收者 ID
            receive_id_type: ID 类型 (open_id/user_id/union_id/email/chat_id)
            msg_type: 消息类型 (text/post/image等)
            content: 消息内容（JSON 字符串）
            uuid: 可选，去重标识
            
        Returns:
            发送结果，包含 message_id
        """
        token = await self.get_tenant_access_token()
        
        url = f"{self.base_url}/open-apis/im/v1/messages"
        params = {"receive_id_type": receive_id_type}
        
        payload = {
            "receive_id": receive_id,
            "msg_type": msg_type,
            "content": content,
        }
        
        if uuid:
            payload["uuid"] = uuid
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url,
                    params=params,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("code") != 0:
                    logger.error(f"发送消息失败: {data}")
                    raise Exception(f"发送消息失败: {data}")
                
                logger.info(f"消息发送成功: message_id={data['data']['message_id']}")
                return data["data"]
        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            raise
    
    async def send_text_message(
        self,
        receive_id: str,
        text: str,
        receive_id_type: str = "open_id",
    ) -> Dict[str, Any]:
        """发送文本消息（便捷方法）
        
        Args:
            receive_id: 消息接收者 ID
            text: 文本内容
            receive_id_type: ID 类型
            
        Returns:
            发送结果
        """
        content = json.dumps({"text": text}, ensure_ascii=False)
        return await self.send_message(
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            msg_type="text",
            content=content,
        )
