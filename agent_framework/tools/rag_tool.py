"""
RAG 知识库查询工具 - 将本地 RAGEngine 包装为 Agent 可调用的工具。

提供 query_knowledge_base 工具：Agent 在回答问题时应优先调用此工具检索本地知识库，
当本地知识库有相关内容时基于检索结果作答，无相关内容时再考虑使用其他工具（如联网搜索）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from agent_framework.core.tool_registry import tool

logger = logging.getLogger(__name__)

# 项目根目录（agent_framework/tools/rag_tool.py -> 向上三级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 模块级单例：RAGEngine 初始化较重（需加载 BGE 模型），全局只创建一次
_rag_engine: Any = None
_rag_init_error: Optional[str] = None


def _resolve_path(p: str) -> str:
    """将相对路径解析为基于项目根目录的绝对路径。"""
    if not p:
        return p
    path = Path(p)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def _load_rag_config() -> dict:
    """从 config.yaml 读取 rag 配置，失败时返回空字典。"""
    try:
        config_path = _PROJECT_ROOT / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("rag", {}) or {}
    except Exception as e:
        logger.warning(f"读取 config.yaml 失败，RAG 工具将使用默认配置: {e}")
        return {}


def _get_rag_engine():
    """延迟初始化并返回 RAGEngine 单例。

    Returns:
        RAGEngine 实例；初始化失败时返回 None。
    """
    global _rag_engine, _rag_init_error

    if _rag_engine is not None:
        return _rag_engine

    if _rag_init_error is not None:
        # 之前已失败，避免每次调用都重复尝试加载模型
        return None

    try:
        from RAG.engine import RAGEngine

        rag_cfg = _load_rag_config()
        emb_cfg = rag_cfg.get("embedding", {}) or {}
        rer_cfg = rag_cfg.get("reranker", {}) or {}
        chroma_cfg = rag_cfg.get("chroma", {}) or {}

        logger.info("初始化 RAG 引擎（知识库查询工具）...")
        _rag_engine = RAGEngine(
            chunk_size=rag_cfg.get("chunk_size", 512),
            chunk_overlap=rag_cfg.get("chunk_overlap", 50),
            embedding_model_name=_resolve_path(
                emb_cfg.get("model_path", "BAAI/bge-m3")
            ),
            embedding_device=emb_cfg.get("device"),
            chroma_collection=chroma_cfg.get("collection", "rag_knowledge"),
            chroma_persist_dir=chroma_cfg.get("persist_dir", "./chroma_db"),
            reranker_model_name=_resolve_path(
                rer_cfg.get("model_path", "BAAI/bge-reranker-large")
            ),
            reranker_device=rer_cfg.get("device"),
            similarity_top_k=rag_cfg.get("similarity_top_k", 10),
            bm25_top_k=rag_cfg.get("bm25_top_k", 10),
            rerank_top_k=rag_cfg.get("rerank_top_k", 5),
            use_reranker=rag_cfg.get("use_reranker", True),
            use_hybrid_retrieval=rag_cfg.get("use_hybrid_retrieval", True),
        )
        logger.info("RAG 引擎初始化成功")
        return _rag_engine

    except Exception as e:
        _rag_init_error = str(e)
        logger.error(f"RAG 引擎初始化失败，知识库查询工具将不可用: {e}", exc_info=True)
        return None


@tool(
    "query_knowledge_base",
    "查询本地知识库，检索与问题最相关的文档片段。"
    "当用户提出的事实性问题可能涉及已索引的本地文档时，应优先调用此工具；"
    "只有当知识库无相关内容时，再考虑联网搜索等其他方式。",
)
async def query_knowledge_base(question: str, top_k: int = 5) -> str:
    """在本地知识库中检索与问题相关的内容。

    Args:
        question: 用户的问题或查询文本。
        top_k: 返回的最相关文档片段数量（默认 5）。

    Returns:
        检索到的文档内容摘要；若知识库不可用或无相关内容，返回提示信息。
    """
    engine = _get_rag_engine()

    if engine is None:
        return (
            f"知识库暂不可用：{_rag_init_error or '初始化失败'}。"
            "请检查 RAG 配置与模型路径，或改用其他工具。"
        )

    try:
        result = engine.query(question, top_k=top_k)

        if not result.get("has_answer"):
            return "未在本地知识库中找到与该问题相关的内容。"

        # 拼接检索到的上下文，供 LLM 据此回答
        contexts = result.get("contexts", [])
        if not contexts:
            return "本地知识库检索结果为空。"

        # 相关性阈值过滤：低于阈值的结果视为不相关，避免误导 LLM
        # 阈值从 config.yaml 读取，默认 0.5（余弦相似度）
        rag_cfg = _load_rag_config()
        similarity_threshold = rag_cfg.get("similarity_threshold", 0.5)

        filtered = [c for c in contexts if c.get("score", 0) >= similarity_threshold]

        if not filtered:
            # 最高分都低于阈值，说明本地知识库确实没有相关内容
            max_score = max((c.get("score", 0) for c in contexts), default=0)
            return (
                f"未在本地知识库中找到与该问题相关的内容"
                f"（最高相关性 {max_score:.3f}，低于阈值 {similarity_threshold}）。"
                "建议使用 fetch_url 等工具联网查询。"
            )

        lines = [f"从本地知识库检索到 {len(filtered)} 条相关内容：\n"]
        for ctx in filtered:
            score = ctx.get("score", 0)
            text = ctx.get("text", "")
            source = ctx.get("metadata", {}).get("filename", "未知来源")
            lines.append(f"【相关性: {score:.3f} | 来源: {source}】")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"知识库查询失败: {e}", exc_info=True)
        return f"知识库查询出错: {e}"
