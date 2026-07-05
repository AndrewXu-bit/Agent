"""
嵌入模型 - 使用 BGE-M3 多语言嵌入模型

BGE-M3 支持多语言、多粒度，具有强大的语义表示能力。
"""

from typing import List, Optional
import math
import os
import logging

from llama_index.core.embeddings import BaseEmbedding
from pydantic import ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)

# 默认 HuggingFace 镜像（国内访问加速），用户可通过环境变量 HF_ENDPOINT 覆盖
DEFAULT_HF_MIRROR = "https://hf-mirror.com"

def _ensure_hf_mirror():
    """如果未设置 HF_ENDPOINT，则使用国内镜像加速。"""
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = DEFAULT_HF_MIRROR
        logger.info(f"已设置 HF_ENDPOINT 镜像: {DEFAULT_HF_MIRROR}")


def _l2_normalize(vec: List[float]) -> List[float]:
    """对向量做 L2 归一化（返回单位向量）。

    新版 FlagEmbedding 的 BGEM3FlagModel.encode() 不再支持 normalize_embeddings 参数，
    因此在获取到 dense_vecs 后手动归一化。
    """
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class BGEEmbedding(BaseEmbedding):
    """基于 BGE-M3 的嵌入模型
    
    使用 FlagEmbedding 库加载 BGE-M3 模型，支持多语言文本嵌入。
    """
    
    # 允许任意属性
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
    
    # 声明为 Pydantic 字段，否则 Pydantic v2 不允许赋值未声明的属性
    max_length: int = Field(default=8192, description="最大序列长度")
    normalize_embeddings: bool = Field(default=True, description="是否归一化嵌入向量")
    # BGEM3FlagModel 是复杂对象，使用私有属性避免序列化问题
    _model: object = PrivateAttr(default=None)
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        max_length: int = 8192,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
    ):
        """初始化 BGE-M3 嵌入模型
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备 (cuda/cpu/mps)，None 为自动检测
            max_length: 最大序列长度
            normalize_embeddings: 是否归一化嵌入向量
            use_fp16: 是否使用半精度浮点数
        """
        # model_name 是 BaseEmbedding 自带字段，其余自定义字段也传入 super
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            normalize_embeddings=normalize_embeddings,
        )
        
        try:
            from FlagEmbedding import BGEM3FlagModel
            
            # 如果是本地路径且存在，则不需要 HF 镜像；否则使用镜像加速
            from pathlib import Path as _Path
            if not _Path(model_name).exists():
                _ensure_hf_mirror()
            
            # 自动检测设备
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"加载 BGE-M3 模型: {model_name} (device={device})")
            
            self._model = BGEM3FlagModel(
                model_name,
                use_fp16=use_fp16,
            )
            
            logger.info("BGE-M3 模型加载成功")
            
        except ImportError:
            logger.error("请安装 FlagEmbedding: pip install FlagEmbedding")
            raise
        except Exception as e:
            logger.error(f"加载 BGE-M3 模型失败: {e}")
            raise
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量列表
        """
        try:
            result = self._model.encode(
                [text],
                max_length=self.max_length,
                return_dense=True,
            )
            
            embedding = result["dense_vecs"][0].tolist()
            # 按需手动 L2 归一化（新版 FlagEmbedding 的 encode 不再支持 normalize_embeddings 参数）
            if self.normalize_embeddings:
                embedding = _l2_normalize(embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            raise
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量列表
        """
        return self._get_text_embedding(text)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询文本的嵌入向量
        
        Args:
            query: 查询文本
            
        Returns:
            嵌入向量列表
        """
        return self._get_text_embedding(query)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询文本的嵌入向量
        
        Args:
            query: 查询文本
            
        Returns:
            嵌入向量列表
        """
        return self._get_query_embedding(query)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表的列表
        """
        try:
            result = self._model.encode(
                texts,
                max_length=self.max_length,
                return_dense=True,
            )
            
            embeddings = result["dense_vecs"].tolist()
            # 按需手动 L2 归一化
            if self.normalize_embeddings:
                embeddings = [_l2_normalize(emb) for emb in embeddings]
            return embeddings
            
        except Exception as e:
            logger.error(f"批量获取嵌入向量失败: {e}")
            raise
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表的列表
        """
        return self._get_text_embeddings(texts)
    
    @classmethod
    def class_name(cls) -> str:
        """返回类名"""
        return "BGEEmbedding"
    
    def get_config(self) -> dict:
        """获取嵌入模型配置
        
        Returns:
            配置字典
        """
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings,
            "type": "BGE-M3",
        }
