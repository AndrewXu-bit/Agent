"""
向量数据库 - 使用 Chroma 存储和检索向量

Chroma 是一个轻量级的向量数据库，支持持久化存储和高效相似度搜索。
"""

from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

logger = logging.getLogger(__name__)


class ChromaVectorDB:
    """基于 Chroma 的向量数据库
    
    提供向量的存储、检索和管理功能。
    """
    
    def __init__(
        self,
        collection_name: str = "rag_knowledge",
        persist_directory: str = "./chroma_db",
        embedding_model = None,
    ):
        """初始化 Chroma 向量数据库
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化存储目录
            embedding_model: 嵌入模型实例
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        try:
            # 创建持久化目录
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            # 初始化 Chroma 客户端
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory
            )
            
            # 获取或创建集合
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )
            
            # 创建 LlamaIndex VectorStore
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            logger.info(
                f"Chroma 向量数据库初始化成功: collection={collection_name}, "
                f"path={persist_directory}"
            )
            
        except Exception as e:
            logger.error(f"初始化 Chroma 数据库失败: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[Any],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """添加文档到向量数据库
        
        Args:
            documents: 文档列表（LlamaIndex Document）
            embeddings: 预计算的嵌入向量（保留参数兼容性，当前未使用；
                        LlamaIndex 会通过 embed_model 自动计算）
            
        Returns:
            添加的文档 ID 列表
        """
        try:
            if not documents:
                logger.warning("没有文档需要添加")
                return []
            
            logger.info(f"添加 {len(documents)} 个文档到向量数据库")
            
            # 创建索引：传入我们的 BGE 嵌入模型，由 LlamaIndex 自动计算向量
            # 注意：VectorStoreIndex.from_documents 不接受预计算的 embeddings，
            # 若 embed_model=None 会回退到 Settings.embed_model（默认 OpenAI），导致报错
            if self.embedding_model is None:
                raise ValueError(
                    "缺少嵌入模型 (embedding_model)，无法构建索引。"
                    "请在初始化 ChromaVectorDB 时传入 embedding_model。"
                )
            
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                embed_model=self.embedding_model,
            )
            
            # 获取文档 ID
            doc_ids = [doc.doc_id for doc in documents if hasattr(doc, 'doc_id')]
            
            logger.info(f"成功添加 {len(documents)} 个文档")
            return doc_ids
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回最相似的 K 个结果
            similarity_threshold: 相似度阈值
            
        Returns:
            搜索结果列表，每个结果包含 text、score、metadata
        """
        try:
            logger.info(f"执行相似度搜索: query='{query[:50]}...', top_k={top_k}")
            
            # 创建查询索引
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embedding_model,
            )
            
            # 执行查询
            retriever = index.as_retriever(
                similarity_top_k=top_k,
            )
            
            nodes = retriever.retrieve(query)
            
            # 格式化结果
            results = []
            for node in nodes:
                result = {
                    "text": node.node.text,
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
                
                # 应用相似度阈值过滤
                if node.score >= similarity_threshold:
                    results.append(result)
            
            logger.info(f"搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise
    
    def delete_collection(self):
        """删除当前集合"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"已删除集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            raise
    
    def get_document_count(self) -> int:
        """获取文档数量
        
        Returns:
            文档数量
        """
        return self.collection.count()
    
    def clear(self):
        """清空集合中的所有文档"""
        try:
            ids = self.collection.get()["ids"]
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"已清空集合中的所有文档")
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            raise
    
    def get_config(self) -> dict:
        """获取数据库配置
        
        Returns:
            配置字典
        """
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": self.get_document_count(),
            "type": "Chroma",
        }
