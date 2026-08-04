"""
================================================================================
  Layer 3 - 检索层: 向量相似度检索器
  功能：
    - 余弦相似度匹配
    - Top8 召回
    - 低相似度过滤（threshold 过滤）
================================================================================
"""
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import RetrievalConfig, get_settings_cached
from utils.exceptions import RetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorSearcher:
    """
    向量相似度检索器。
    基于余弦相似度从向量库中召回最相关的文档块。
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        初始化向量检索器。

        Args:
            config: 检索配置（默认从全局配置读取）
        """
        self.config = config or get_settings_cached().retrieval
        logger.info(
            f"VectorSearcher 初始化完成 | "
            f"TopK: {self.config.vector_top_k} | "
            f"相似度阈值: {self.config.similarity_threshold}"
        )

    def search(
        self,
        query_embedding: List[float],
        vector_store,  # BaseVectorStore（由上层注入）
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        执行向量相似度检索。

        检索流程：
        1. 在向量库中执行余弦相似度搜索
        2. 过滤低于阈值的低质量结果
        3. 返回 TopK 结果

        Args:
            query_embedding: 查询向量
            vector_store:    向量存储实例（BaseVectorStore）
            top_k:           召回数量（默认从配置读取）

        Returns:
            List[Document]: 相似文档列表（按相似度降序）
        """
        if top_k is None:
            top_k = self.config.vector_top_k

        try:
            results = vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        except Exception as e:
            raise RetrievalError(f"向量检索失败: {str(e)}") from e

        # 低相似度过滤
        threshold = self.config.similarity_threshold
        filtered = [
            doc
            for doc in results
            if doc.metadata.get("similarity", 0) >= threshold
        ]

        logger.info(
            f"向量检索完成: 召回 {len(results)} → 过滤后 {len(filtered)} "
            f"(阈值: {threshold})"
        )

        return filtered
