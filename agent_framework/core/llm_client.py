"""
多模型统一适配层 - 抹平不同 LLM 的 Tool Calling 协议差异。

根据 framework.md 2.4 节设计：
- BaseLLMClient: 抽象基类，规定统一 chat 接口
- OpenAICompatibleClient: 兼容 OpenAI 格式的模型（OpenAI, DeepSeek, Qwen, GLM 等）
- AnthropicClient: Claude 模型适配（可选）
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from .standard_tool import StandardTool

logger = logging.getLogger(__name__)


# DeepSeek 等模型有时不通过标准 tool_calls 字段返回工具调用，
# 而是把 <｜tool_calls｜> ... </｜tool_calls> 标记塞进 content 文本。
# 注意：DSML 标签使用全角竖线 ｜ (U+FF5C)，不是半角 | (U+007C)。
# [｜|] 字符组同时匹配两种，兼容不同模型版本。
# 全部使用纯 raw string，不用 f-string，避免 {2} 被 Python 误解释。
_DSML_TOOL_CALL_PATTERN = re.compile(
    r"<[｜|]{2}DSML[｜|]{2}tool_calls>(.*?)</[｜|]{2}DSML[｜|]{2}tool_calls>",
    re.DOTALL,
)
_DSML_INVOKE_PATTERN = re.compile(
    r"<[｜|]{2}DSML[｜|]{2}invoke\s+name=\"([^\"]+)\">(.*?)</[｜|]{2}DSML[｜|]{2}invoke>",
    re.DOTALL,
)
_DSML_PARAM_PATTERN = re.compile(
    r"<[｜|]{2}DSML[｜|]{2}parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</[｜|]{2}DSML[｜|]{2}parameter>",
    re.DOTALL,
)
# 兜底：清理所有残留的 DSML 开标签和闭标签
_DSML_OPEN_TAG_PATTERN = re.compile(r"<[｜|]{2}DSML[｜|]{2}[^>]*>")
_DSML_CLOSE_TAG_PATTERN = re.compile(r"</[｜|]{2}DSML[｜|]{2}[^>]*>")


def _extract_dsml_tool_calls(content: str) -> tuple:
    """从 content 文本中提取 DeepSeek DSML 格式的工具调用。

    Args:
        content: 模型返回的原始文本内容。

    Returns:
        (cleaned_content, tool_calls_list)
        cleaned_content: 移除 DSML 标记后的纯文本。
        tool_calls_list: 提取出的工具调用列表，格式与 OpenAI tool_calls 一致。
    """
    tool_calls = []
    tc_blocks = _DSML_TOOL_CALL_PATTERN.findall(content)

    call_idx = 0
    for block in tc_blocks:
        for invoke_name, invoke_body in _DSML_INVOKE_PATTERN.findall(block):
            arguments: Dict[str, Any] = {}
            for param_name, param_value in _DSML_PARAM_PATTERN.findall(invoke_body):
                arguments[param_name] = param_value.strip()

            tool_calls.append(
                {
                    "id": f"dsml_call_{call_idx}",
                    "function": {
                        "name": invoke_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
            call_idx += 1

    # 清理所有 DSML 标记，得到纯文本
    cleaned = _DSML_OPEN_TAG_PATTERN.sub("", content)
    cleaned = _DSML_CLOSE_TAG_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned, tool_calls


# ====================================================================
# 抽象基类
# ====================================================================


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类。

    所有模型适配器必须实现 chat() 方法，返回统一格式：
    {
        "role": "assistant",
        "content": "...",
        "tool_calls": [
            {"id": "call_xxx", "function": {"name": "...", "arguments": "{}"}}
        ]
    }
    """

    model: str

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[StandardTool]] = None,
    ) -> Dict[str, Any]:
        """发送对话并返回统一格式的响应。"""
        ...


# ====================================================================
# OpenAI 兼容格式客户端
# ====================================================================


class OpenAICompatibleClient(BaseLLMClient):
    """兼容 OpenAI Chat Completion API 格式的模型客户端。

    支持：OpenAI、DeepSeek、通义千问 (Qwen)、智谱 GLM、零一万物 Yi 等。

    Args:
        api_key: API 密钥。
        base_url: API 基础 URL（如 https://api.openai.com/v1）。
        model: 模型名称（如 gpt-4o, deepseek-chat, qwen-plus 等）。
        max_tokens: 最大生成 token 数。
        temperature: 采样温度。
        http_timeout: HTTP 请求超时秒数。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        http_timeout: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.http_timeout = http_timeout

    def _convert_tools(
        self, tools: Optional[List[StandardTool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """将 StandardTool 列表转为 OpenAI 格式。"""
        if not tools:
            return None
        return [t.to_openai_tool() for t in tools]

    def _parse_response(
        self, raw: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将原始 API 响应解析为统一格式。"""
        choice = raw["choices"][0]
        msg = choice["message"]

        result: Dict[str, Any] = {
            "role": "assistant",
            "content": msg.get("content", ""),
        }
        # DeepSeek 推理模型 (v4) 需要 reasoning_content 原样传回，否则 400 错误
        if msg.get("reasoning_content"):
            result["reasoning_content"] = msg["reasoning_content"]

        # 解析 tool_calls
        raw_tool_calls = msg.get("tool_calls")
        if raw_tool_calls:
            parsed = []
            for tc in raw_tool_calls:
                parsed.append(
                    {
                        "id": tc["id"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                )
            result["tool_calls"] = parsed

        # DeepSeek 等模型可能把工具调用以 DSML 标记塞进 content，
        # 这里提取为标准 tool_calls 并清理 content，避免原始标记展示给用户。
        # 无论是否有标准 tool_calls，content 中的 DSML 标记都必须清理。
        if result.get("content"):
            cleaned, dsml_tool_calls = _extract_dsml_tool_calls(result["content"])
            # 如果没有标准 tool_calls，用 DSML 提取的替代
            if dsml_tool_calls and not result.get("tool_calls"):
                result["tool_calls"] = dsml_tool_calls
            result["content"] = cleaned

        return result

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[StandardTool]] = None,
    ) -> Dict[str, Any]:
        """发送对话请求，返回统一格式响应。"""
        # 清理消息格式，确保符合 DeepSeek API 要求
        cleaned_messages = []
        for msg in messages:
            cleaned_msg = {"role": msg["role"]}
            
            # 处理 tool 消息 - DeepSeek 需要 type 字段
            if msg["role"] == "tool":
                cleaned_msg["type"] = "function"  # 必须添加 type 字段
                cleaned_msg["tool_call_id"] = msg.get("tool_call_id", "")
                cleaned_msg["content"] = msg.get("content", "")
            # 处理 assistant 消息
            elif msg["role"] == "assistant":
                if msg.get("content"):
                    cleaned_msg["content"] = msg["content"]
                # DeepSeek 推理模型需要 reasoning_content 原样传回
                if msg.get("reasoning_content"):
                    cleaned_msg["reasoning_content"] = msg["reasoning_content"]
                if msg.get("tool_calls"):
                    # 给每个 tool_call 添加 type 字段
                    tool_calls_with_type = []
                    for tc in msg["tool_calls"]:
                        tc_copy = tc.copy()
                        tc_copy["type"] = "function"
                        tool_calls_with_type.append(tc_copy)
                    cleaned_msg["tool_calls"] = tool_calls_with_type
                # 只要有 content 或 tool_calls，就添加 type
                if msg.get("content") or msg.get("tool_calls"):
                    cleaned_msg["type"] = "text"  # DeepSeek 需要
            # 处理 user 和 system 消息
            else:
                cleaned_msg["content"] = msg.get("content", "")
                cleaned_msg["type"] = "text"  # DeepSeek 需要
            
            cleaned_messages.append(cleaned_msg)
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": cleaned_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        openai_tools = self._convert_tools(tools)
        # 检查是否有 tool 消息，如果有则不传 tools 参数
        has_tool_message = any(msg.get("role") == "tool" for msg in cleaned_messages)
        if openai_tools and not has_tool_message:
            payload["tools"] = openai_tools
            # OpenAI 需要 tool_choice 来强制/允许调用
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                
                # 如果是 400 错误，尝试不带 tools 参数重试
                if response.status_code == 400 and openai_tools:
                    logger.warning("带 tools 参数请求失败，尝试不带 tools 参数重试...")
                    # 打印调试信息
                    import json as json_module
                    print(f"\n=== DEBUG: 失败的消息 ===")
                    print(json_module.dumps(cleaned_messages, ensure_ascii=False, indent=2)[:2000])
                    print(f"=== END DEBUG ===\n")
                    payload_without_tools = {k: v for k, v in payload.items() if k not in ["tools", "tool_choice"]}
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload_without_tools,
                    )
                
                response.raise_for_status()
                raw = response.json()

            return self._parse_response(raw)
        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            raise Exception(f"LLM API 请求失败: {error_detail}") from e


# ====================================================================
# Anthropic Claude 客户端
# ====================================================================


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude 模型客户端。

    使用 Anthropic Messages API，自动适配 Claude 的 tool 格式。

    Args:
        api_key: Anthropic API 密钥。
        model: 模型名称（如 claude-sonnet-5-20251001）。
        max_tokens: 最大生成 token 数。
        anthropic_version: API 版本。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5-20251001",
        max_tokens: int = 4096,
        anthropic_version: str = "2023-06-01",
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.anthropic_version = anthropic_version

    def _convert_tools(
        self, tools: Optional[List[StandardTool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """将 StandardTool 转为 Anthropic 格式。"""
        if not tools:
            return None
        return [t.to_anthropic_tool() for t in tools]

    def _parse_response(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """解析 Anthropic Messages API 响应为统一格式。

        Anthropic 的 tool_use 是 content 块中的一种类型，需提取为
        OpenAI 兼容的 tool_calls 格式以便编排引擎统一处理。
        """
        content_blocks = raw.get("content", [])

        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if block["type"] == "text":
                text_content += block.get("text", "")
            elif block["type"] == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    }
                )

        result: Dict[str, Any] = {
            "role": "assistant",
            "content": text_content,
        }
        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[StandardTool]] = None,
    ) -> Dict[str, Any]:
        """发送对话请求，返回统一格式响应。"""
        # 构建 Anthropic 格式的消息
        anthropic_messages: List[Dict[str, Any]] = []
        system_prompt = None

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            # Anthropic 将 system 消息放在顶层参数
            if role == "system":
                system_prompt = content
                continue

            # 处理 tool 结果消息
            if role == "tool":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": content,
                            }
                        ],
                    }
                )
                continue

            # 处理 assistant 消息（可能带 tool_use）
            if role == "assistant":
                blocks: List[Dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]),
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": blocks})
                continue

            # 普通 user 消息
            anthropic_messages.append({"role": role, "content": content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt

        claude_tools = self._convert_tools(tools)
        if claude_tools:
            payload["tools"] = claude_tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()

        return self._parse_response(raw)


# ====================================================================
# Mock 客户端（用于测试 / 无 API Key 时快速验证）
# ====================================================================


class MockLLMClient(BaseLLMClient):
    """模拟 LLM 客户端，用于测试场景。

    不发送真实 API 请求，而是根据预设规则或用户输入返回响应。
    """

    def __init__(
        self,
        model: str = "mock-model",
        mock_response: Optional[str] = None,
    ):
        self.model = model
        self._mock_response = mock_response

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[StandardTool]] = None,
    ) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": self._mock_response
            or f"[Mock] 收到 {len(messages)} 条消息，{len(tools) if tools else 0} 个工具可用",
        }