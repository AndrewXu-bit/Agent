"""
重排序器 - 使用 BGE-Reranker-Large 对检索结果进行重排序

BGE-Reranker-Large 能够精确评估查询与文档的相关性，提升检索质量。

注意：直接使用 transformers 原生 API 实现 cross-encoder，
绕开 FlagEmbedding 与新版 transformers (>=5.x) 的兼容性问题
（FlagEmbedding 内部调用的 tokenizer.prepare_for_model 已被移除）。
"""

from typing import List, Dict, Any, Tuple
import os
import logging

logger = logging.getLogger(__name__)

# 默认 HuggingFace 镜像（国内访问加速），用户可通过环境变量 HF_ENDPOINT 覆盖
DEFAULT_HF_MIRROR = "https://hf-mirror.com"

def _ensure_hf_mirror():
    """如果未设置 HF_ENDPOINT，则使用国内镜像加速。"""
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = DEFAULT_HF_MIRROR
        logger.info(f"已设置 HF_ENDPOINT 镜像: {DEFAULT_HF_MIRROR}")


class BGEReranker:
    """基于 BGE-Reranker-Large 的重排序器
    
    对初步检索的结果进行相关性重排序，提高最终结果的质量。
    使用 transformers 原生 cross-encoder 实现。
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = None,
        max_length: int = 512,
    ):
        """初始化 BGE 重排序器
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备 (cuda/cpu/mps)，None 为自动检测
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.max_length = max_length
        
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # 如果是本地路径且存在，则不需要 HF 镜像；否则使用镜像加速
            from pathlib import Path as _Path
            if not _Path(model_name).exists():
                _ensure_hf_mirror()
            
            # 自动检测设备
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"加载 BGE Reranker 模型: {model_name} (device={device})")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            self.device = device
            
            logger.info("BGE Reranker 模型加载成功")
            
        except Exception as e:
            logger.error(f"加载 BGE Reranker 模型失败: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表，每个文档包含 text、score、metadata
            top_k: 返回前 K 个最相关的文档
            
        Returns:
            重排序后的文档列表，按相关性从高到低排序
        """
        if not documents:
            logger.warning("没有文档需要重排序")
            return []
        
        try:
            import torch
            logger.info(f"开始重排序: {len(documents)} 个文档")
            
            # 准备输入数据：每个 [query, doc_text] 对
            pairs = [[query, doc["text"]] for doc in documents]
            
            # 分批处理，避免显存/内存溢出
            batch_size = 16
            all_scores: List[float] = []
            
            with torch.no_grad():
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i:i + batch_size]
                    # tokenizer 自动处理 [text_a, text_b] 对
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    
                    logits = self.model(**inputs).logits.squeeze(-1)
                    # BGE-reranker 输出 logits，sigmoid 归一化到 [0,1]
                    scores = torch.sigmoid(logits).float().cpu().tolist()
                    
                    if isinstance(scores, float):
                        scores = [scores]
                    all_scores.extend(scores)
            
            # 将分数添加到文档中
            scored_docs = []
            for i, doc in enumerate(documents):
                doc_with_score = doc.copy()
                doc_with_score["rerank_score"] = float(all_scores[i])
                scored_docs.append(doc_with_score)
            
            # 按重排序分数降序排列
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # 返回 top_k 个结果
            result = scored_docs[:top_k]
            
            logger.info(f"重排序完成，返回 {len(result)} 个结果")
            return result
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 如果重排序失败，返回原始结果
            logger.warning("返回原始检索结果")
            return documents[:top_k]
    
    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[Dict[str, Any]]],
        top_k: int = 3,
    ) -> List[List[Dict[str, Any]]]:
        """批量重排序
        
        Args:
            queries: 查询文本列表
            document_lists: 对应的文档列表
            top_k: 每个查询返回的 top_k 结果
            
        Returns:
            重排序后的结果列表
        """
        results = []
        for query, docs in zip(queries, document_lists):
            ranked = self.rerank(query, docs, top_k)
            results.append(ranked)
        return results
    
    def get_config(self) -> dict:
        """获取重排序器配置
        
        Returns:
            配置字典
        """
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "type": "BGE-Reranker-Large",
        }
