"""
文档加载器 - 使用 Unstructured 加载多种格式的文档

支持 PDF、Word、PowerPoint、Excel、TXT、Markdown 等格式。
"""

from pathlib import Path
from typing import List, Optional
import logging

from llama_index.core import Document
from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)

# 纯文本格式直接用 Python 读取，避免 unstructured 在 libmagic 缺失/中文文件名时
# 触发不必要的网络调用（曾出现 [WinError 10054] 远程主机强迫关闭连接）
PLAIN_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".rst", ".log", ".csv"}


class DocumentLoader:
    """基于 Unstructured 的文档加载器
    
    支持多种文档格式的自动解析和加载。
    """
    
    def __init__(self, supported_extensions: Optional[List[str]] = None):
        """初始化文档加载器
        
        Args:
            supported_extensions: 支持的文档扩展名列表
        """
        if supported_extensions is None:
            self.supported_extensions = [
                ".pdf", ".docx", ".doc", ".pptx", ".ppt",
                ".xlsx", ".xls", ".txt", ".md", ".html",
                ".htm", ".epub", ".odt"
            ]
        else:
            self.supported_extensions = supported_extensions
    
    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document 对象列表
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
        
        try:
            logger.info(f"加载文件: {file_path}")
            
            # 纯文本格式直接读取，避免 unstructured 触发网络/依赖问题
            if path.suffix.lower() in PLAIN_TEXT_EXTENSIONS:
                return self._load_plain_text(path)
            
            # 复杂格式使用 unstructured 解析文档
            elements = partition(
                filename=str(path),
                strategy="auto",  # 自动选择最佳策略
            )
            
            # 转换为 LlamaIndex Document
            documents = []
            for element in elements:
                if hasattr(element, 'text') and element.text:
                    doc = Document(
                        text=element.text,
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "page_number": getattr(element, 'metadata', {}).get('page_number', None),
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"成功加载 {len(documents)} 个文档片段")
            return documents
            
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            raise
    
    def _load_plain_text(self, path: Path) -> List[Document]:
        """直接读取纯文本文件（.txt/.md/.html 等）

        Args:
            path: 文件路径
            
        Returns:
            Document 对象列表
        """
        # 尝试多种编码，避免中文内容乱码
        text = None
        for encoding in ("utf-8", "gbk", "gb18030", "utf-16", "latin-1"):
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            raise ValueError(f"无法解码文件，已尝试多种编码: {path}")
        
        doc = Document(
            text=text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "page_number": None,
            }
        )
        logger.info(f"成功加载 1 个文档片段 (纯文本)")
        return [doc]
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """加载目录下的所有支持的文件
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归扫描子目录
            
        Returns:
            Document 对象列表
        """
        path = Path(directory_path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"路径不是目录: {directory_path}")
        
        all_documents = []
        
        # 扫描文件
        pattern = "**/*" if recursive else "*"
        files = list(path.glob(pattern))
        
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    docs = self.load_file(str(file_path))
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"跳过文件 {file_path}: {e}")
                    continue
        
        logger.info(f"目录加载完成，共 {len(all_documents)} 个文档片段")
        return all_documents
    
    def load(self, source: str, recursive: bool = True) -> List[Document]:
        """智能加载文件或目录
        
        Args:
            source: 文件或目录路径
            recursive: 是否递归（仅对目录有效）
            
        Returns:
            Document 对象列表
        """
        path = Path(source)
        
        if path.is_file():
            return self.load_file(str(path))
        elif path.is_dir():
            return self.load_directory(str(path), recursive)
        else:
            raise FileNotFoundError(f"路径不存在: {source}")
