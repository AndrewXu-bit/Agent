"""
RAG 引擎 - 整合所有组件的完整 RAG 系统

提供文档加载、分割、索引、检索和问答的完整流程。
"""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .document_loader import DocumentLoader
from .text_splitter import TextSplitter
from .embedding import BGEEmbedding
from .vector_store import ChromaVectorDB
from .reranker import BGEReranker
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 知识库引擎
    
    整合文档加载、文本分割、向量嵌入、存储检索、重排序等完整流程。
    """
    
    def __init__(
        self,
        # 文档处理配置
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        
        # 嵌入模型配置
        embedding_model_name: str = "BAAI/bge-m3",
        embedding_device: Optional[str] = None,
        
        # 向量数据库配置
        chroma_collection: str = "rag_knowledge",
        chroma_persist_dir: str = "./chroma_db",
        
        # 重排序配置
        reranker_model_name: str = "BAAI/bge-reranker-large",
        reranker_device: Optional[str] = None,
        
        # 检索配置
        similarity_top_k: int = 10,
        bm25_top_k: int = 10,
        rerank_top_k: int = 5,
        use_reranker: bool = True,
        use_hybrid_retrieval: bool = True,
    ):
        """初始化 RAG 引擎
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠
            embedding_model_name: 嵌入模型名称
            embedding_device: 嵌入模型设备
            chroma_collection: Chroma 集合名称
            chroma_persist_dir: Chroma 持久化目录
            reranker_model_name: 重排序模型名称
            reranker_device: 重排序模型设备
            similarity_top_k: 向量检索 top_k
            bm25_top_k: BM25 检索 top_k
            rerank_top_k: 重排序后返回 top_k
            use_reranker: 是否使用重排序
            use_hybrid_retrieval: 是否使用混合检索
        """
        logger.info("初始化 RAG 引擎...")
        
        # 初始化组件
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        self.embedding_model = BGEEmbedding(
            model_name=embedding_model_name,
            device=embedding_device,
        )
        
        self.vector_db = ChromaVectorDB(
            collection_name=chroma_collection,
            persist_directory=chroma_persist_dir,
            embedding_model=self.embedding_model,
        )
        
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k
        
        if use_reranker:
            self.reranker = BGEReranker(
                model_name=reranker_model_name,
                device=reranker_device,
            )
        else:
            self.reranker = None
        
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.hybrid_retriever = None
        
        if use_hybrid_retrieval:
            try:
                self.hybrid_retriever = HybridRetriever(
                    vector_store=self.vector_db.vector_store,
                    embedding_model=self.embedding_model,
                    similarity_top_k=similarity_top_k,
                    bm25_top_k=bm25_top_k,
                )
            except Exception as e:
                logger.warning(f"混合检索器初始化失败，回退到向量检索: {e}")
                self.use_hybrid_retrieval = False
        
        logger.info("RAG 引擎初始化完成")
    
    def build_index(self, source: str, recursive: bool = True) -> int:
        """构建知识索引
        
        Args:
            source: 文件或目录路径
            recursive: 是否递归扫描子目录
            
        Returns:
            处理的文档数量
        """
        logger.info(f"开始构建索引: {source}")
        
        # 1. 加载文档
        documents = self.document_loader.load(source, recursive)
        logger.info(f"加载了 {len(documents)} 个文档片段")
        
        if not documents:
            logger.warning("没有加载到任何文档")
            return 0
        
        # 2. 分割文档
        chunked_docs = self.text_splitter.split_documents(documents)
        logger.info(f"分割为 {len(chunked_docs)} 个文本块")
        
        # 3. 添加到向量数据库
        doc_ids = self.vector_db.add_documents(chunked_docs)
        logger.info(f"成功添加 {len(doc_ids)} 个文档到向量数据库")
        
        return len(doc_ids)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranker: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """查询知识库
        
        Args:
            question: 问题文本
            top_k: 返回的结果数量
            use_reranker: 是否使用重排序（None 则使用初始化时的配置）
            
        Returns:
            包含答案和上下文的字典
        """
        logger.info(f"查询: {question[:50]}...")
        
        # 确定是否使用重排序
        if use_reranker is None:
            use_reranker = self.use_reranker
        
        # 1. 检索相关文档
        if self.use_hybrid_retrieval and self.hybrid_retriever:
            results = self.hybrid_retriever.retrieve(question, use_fusion=True)
        else:
            results = self.vector_db.search(question, top_k=top_k * 2)
        
        logger.info(f"检索到 {len(results)} 个相关文档")
        
        if not results:
            return {
                "question": question,
                "answer": "未找到相关知识库内容。",
                "contexts": [],
                "has_answer": False,
            }
        
        # 2. 重排序（如果启用）
        if use_reranker and self.reranker:
            results = self.reranker.rerank(question, results, top_k=top_k)
        else:
            results = results[:top_k]
        
        # 3. 准备上下文
        contexts = []
        for i, result in enumerate(results, 1):
            context = {
                "index": i,
                "text": result["text"],
                "score": result.get("rerank_score", result.get("score", 0)),
                "metadata": result.get("metadata", {}),
            }
            contexts.append(context)
        
        # 4. 生成答案（这里只返回上下文，实际应用中可以调用LLM生成答案）
        answer_text = self._generate_answer(question, contexts)
        
        return {
            "question": question,
            "answer": answer_text,
            "contexts": contexts,
            "has_answer": True,
        }
    
    def _generate_answer(self, question: str, contexts: List[Dict]) -> str:
        """基于检索结果生成答案
        
        Args:
            question: 问题
            contexts: 上下文列表
            
        Returns:
            生成的答案文本
        """
        # 这里可以集成 LLM 来生成更自然的答案
        # 目前返回格式化的上下文信息
        
        if not contexts:
            return "未找到相关信息。"
        
        answer_lines = ["基于知识库检索到的相关信息：\n"]
        
        for ctx in contexts:
            score = ctx.get('score', 0)
            text = ctx['text']
            source = ctx.get('metadata', {}).get('source', '未知来源')
            
            answer_lines.append(f"[相关性: {score:.3f}]")
            answer_lines.append(f"来源: {source}")
            answer_lines.append(f"内容: {text}\n")
        
        answer_lines.append("\n请根据以上信息回答问题。")
        
        return "\n".join(answer_lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "document_count": self.vector_db.get_document_count(),
            "use_reranker": self.use_reranker,
            "use_hybrid_retrieval": self.use_hybrid_retrieval,
            "embedding_model": self.embedding_model.get_config(),
            "vector_db": self.vector_db.get_config(),
        }
    
    def clear_index(self):
        """清空知识库索引"""
        self.vector_db.clear()
        logger.info("已清空知识库索引")
    
    def get_config(self) -> Dict[str, Any]:
        """获取引擎配置
        
        Returns:
            配置字典
        """
        return {
            "text_splitter": self.text_splitter.get_config(),
            "embedding": self.embedding_model.get_config(),
            "vector_db": self.vector_db.get_config(),
            "reranker": self.reranker.get_config() if self.reranker else None,
            "retriever": self.hybrid_retriever.get_config() if self.hybrid_retriever else None,
        }
