"""
混合检索器 - 使用 LlamaIndex EnsembleRetriever 结合多种检索策略

结合向量检索和关键词检索，提升检索效果。
"""

from typing import List, Dict, Any, Optional
import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever

# 尝试导入 BM25Retriever（不同版本路径可能不同）
try:
    from llama_index.retrievers.bm25 import BM25Retriever
    BM25_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from llama_index.core.retrievers import BM25Retriever
        BM25_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        BM25Retriever = None
        BM25_AVAILABLE = False
        logging.warning("BM25Retriever 未安装，混合检索功能将降级为向量检索")

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器
    
    使用 EnsembleRetriever 结合向量检索（语义相似度）和 
    BM25检索（关键词匹配），提供更准确的检索结果。
    """
    
    def __init__(
        self,
        vector_store = None,
        embedding_model = None,
        documents: Optional[List[Any]] = None,
        similarity_top_k: int = 5,
        bm25_top_k: int = 5,
        mode: str = "reciprocal_rerank",
        num_queries_fusion: int = 4,
    ):
        """初始化混合检索器
        
        Args:
            vector_store: Chroma向量存储实例
            embedding_model: 嵌入模型实例
            documents: 文档列表（用于BM25索引）
            similarity_top_k: 向量检索返回的top_k数量
            bm25_top_k: BM25检索返回的top_k数量
            mode: 融合模式 (reciprocal_rerank/relative_score/dist_based_score)
            num_queries_fusion: 查询融合生成的查询数量
        """
        self.similarity_top_k = similarity_top_k
        self.bm25_top_k = bm25_top_k
        self.embedding_model = embedding_model
        
        try:
            # 创建向量检索器
            if vector_store and embedding_model:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embedding_model,
                )
                self.vector_retriever = index.as_retriever(
                    similarity_top_k=similarity_top_k
                )
                logger.info("向量检索器初始化成功")
            else:
                self.vector_retriever = None
                logger.warning("未提供向量存储或嵌入模型，向量检索器不可用")
            
            # 创建 BM25 检索器（关键词检索）
            if documents and BM25_AVAILABLE:
                self.bm25_retriever = BM25Retriever.from_defaults(
                    nodes=documents,
                    similarity_top_k=bm25_top_k,
                )
                logger.info("BM25 检索器初始化成功")
            else:
                self.bm25_retriever = None
                if not BM25_AVAILABLE:
                    logger.warning("BM25Retriever 未安装，跳过BM25检索")
                elif not documents:
                    logger.warning("未提供文档列表，BM25检索器不可用")
            
            # 创建融合检索器
            retrievers = []
            if self.vector_retriever:
                retrievers.append(self.vector_retriever)
            if self.bm25_retriever:
                retrievers.append(self.bm25_retriever)
            
            if retrievers:
                self.fusion_retriever = QueryFusionRetriever(
                    retrievers=retrievers,
                    similarity_top_k=similarity_top_k,
                    num_queries=num_queries_fusion,
                    mode=mode,
                    use_async=False,
                )
                logger.info(f"融合检索器初始化成功 (mode={mode})")
            else:
                self.fusion_retriever = None
                logger.error("没有可用的检索器")
                
        except Exception as e:
            logger.error(f"初始化混合检索器失败: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        use_fusion: bool = True,
    ) -> List[Dict[str, Any]]:
        """执行检索
        
        Args:
            query: 查询文本
            use_fusion: 是否使用融合检索
            
        Returns:
            检索结果列表
        """
        try:
            if use_fusion and self.fusion_retriever:
                logger.info(f"使用融合检索: query='{query[:50]}...'")
                nodes = self.fusion_retriever.retrieve(query)
            elif self.vector_retriever:
                logger.info(f"使用向量检索: query='{query[:50]}...'")
                nodes = self.vector_retriever.retrieve(query)
            else:
                raise ValueError("没有可用的检索器")
            
            # 格式化结果
            results = []
            for node in nodes:
                result = {
                    "text": node.node.text,
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
                results.append(result)
            
            logger.info(f"检索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise
    
    def get_config(self) -> dict:
        """获取检索器配置
        
        Returns:
            配置字典
        """
        return {
            "similarity_top_k": self.similarity_top_k,
            "bm25_top_k": self.bm25_top_k,
            "has_vector_retriever": self.vector_retriever is not None,
            "has_bm25_retriever": self.bm25_retriever is not None,
            "type": "HybridRetriever",
        }
