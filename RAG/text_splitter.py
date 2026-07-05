"""
文本分割器 - 使用 RecursiveCharacterTextSplitter 进行智能分块

支持按字符数、重叠窗口等方式分割文本。
"""

from typing import List, Optional
import logging

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

logger = logging.getLogger(__name__)


class TextSplitter:
    """基于 LlamaIndex 的递归字符文本分割器
    
    使用 SentenceSplitter 实现智能的文本分块，保持语义完整性。
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = " ",
        paragraph_separator: str = "\n\n\n",
        secondary_chunk_split_regex: str = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",
    ):
        """初始化文本分割器
        
        Args:
            chunk_size: 每个文本块的大小（字符数）
            chunk_overlap: 文本块之间的重叠大小（字符数）
            separator: 主要分隔符
            paragraph_separator: 段落分隔符
            secondary_chunk_split_regex: 二级分割正则表达式
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 创建 LlamaIndex 的 SentenceSplitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            paragraph_separator=paragraph_separator,
        )
        
        logger.info(
            f"文本分割器初始化: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )
    
    def split_text(self, text: str) -> List[str]:
        """分割单个文本字符串
        
        Args:
            text: 待分割的文本
            
        Returns:
            分割后的文本块列表
        """
        try:
            nodes = self.splitter.get_nodes_from_documents(
                [Document(text=text)]
            )
            chunks = [node.text for node in nodes]
            
            logger.info(f"文本分割完成: {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            logger.error(f"文本分割失败: {e}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表
        
        Args:
            documents: Document 对象列表
            
        Returns:
            分割后的 Document 对象列表（每个包含一个文本块）
        """
        try:
            nodes = self.splitter.get_nodes_from_documents(documents)
            
            # 转换回 Document 格式
            chunked_docs = []
            for node in nodes:
                doc = Document(
                    text=node.text,
                    metadata=node.metadata.copy(),
                )
                chunked_docs.append(doc)
            
            logger.info(f"文档分割完成: {len(documents)} -> {len(chunked_docs)} 个块")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"文档分割失败: {e}")
            raise
    
    def get_config(self) -> dict:
        """获取分割器配置
        
        Returns:
            配置字典
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "type": "RecursiveCharacterTextSplitter",
        }
