"""
================================================================================
  Layer 3 - 检索层: 混合检索融合器
  功能（严禁简化）：
    - 向量检索结果 + BM25 检索结果融合
    - 权重分配：向量 0.6 + BM25 0.4
    - 结果去重降噪
    - 统一排序输出
================================================================================
"""
from typing import List, Dict, Optional

from langchain_core.documents import Document

from config.settings import RetrievalConfig, get_settings_cached
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridSearcher:
    """
    混合检索融合器。
    融合向量相似度检索和 BM25 关键词检索的结果，
    加权评分后去重排序，输出统一的检索结果列表。
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        初始化混合检索融合器。

        Args:
            config: 检索配置（默认从全局配置读取）
        """
        self.config = config or get_settings_cached().retrieval
        logger.info(
            f"HybridSearcher 初始化完成 | "
            f"向量权重: {self.config.vector_weight} | "
            f"BM25 权重: {self.config.bm25_weight}"
        )

    def merge(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
    ) -> List[Document]:
        """
        融合向量检索和 BM25 检索结果。

        融合策略：
        1. 归一化各来源分数
        2. 加权求和：score = α × vec_score + β × bm25_score
        3. 按融合分数降序排序
        4. 按 chunk_id 去重

        Args:
            vector_results: 向量检索结果列表
            bm25_results:   BM25 检索结果列表

        Returns:
            List[Document]: 融合后的排序文档列表
        """
        alpha = self.config.vector_weight
        beta = self.config.bm25_weight

        # 构建 chunk_id → 融合分数 的映射
        score_map: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Step 1: 处理向量检索结果
        for doc in vector_results:
            chunk_id = doc.metadata.get("chunk_id", "")
            vec_score = doc.metadata.get("similarity", 0)
            normalized_vec = self._normalize_score(vec_score, is_vector=True)
            score_map[chunk_id] = alpha * normalized_vec
            doc_map[chunk_id] = doc

        # Step 2: 处理 BM25 检索结果
        for doc in bm25_results:
            chunk_id = doc.metadata.get("chunk_id", "")
            bm25_score = doc.metadata.get("bm25_score", 0)
            normalized_bm25 = self._normalize_score(bm25_score, is_vector=False)

            if chunk_id in score_map:
                # 同一文档：累加 BM25 权重
                score_map[chunk_id] += beta * normalized_bm25
                # 合并元数据
                existing = doc_map[chunk_id]
                existing.metadata["bm25_score"] = bm25_score
            else:
                # 新文档：直接加入
                score_map[chunk_id] = beta * normalized_bm25
                doc.metadata["similarity"] = normalized_bm25  # 设置默认相似度
                doc_map[chunk_id] = doc

        # Step 3: 按融合分数降序排序
        sorted_items = sorted(
            score_map.items(), key=lambda x: x[1], reverse=True
        )

        # Step 4: 构建结果列表
        merged = []
        for chunk_id, fused_score in sorted_items:
            doc = doc_map[chunk_id]
            doc.metadata["hybrid_score"] = round(fused_score, 6)
            merged.append(doc)

        logger.info(
            f"混合检索融合完成: "
            f"向量 {len(vector_results)} + BM25 {len(bm25_results)} "
            f"→ 去重后 {len(merged)} 条"
        )

        return merged

    @staticmethod
    def _normalize_score(score: float, is_vector: bool) -> float:
        """
        归一化检索分数到 [0, 1] 区间。

        向量相似度（余弦）：原本就在 [0, 1]，直接使用。
        BM25 分数：无理论上界，使用 sigmoid 归一化。

        Args:
            score:    原始分数
            is_vector: 是否为向量相似度分数

        Returns:
            float: 归一化后的分数 [0, 1]
        """
        if is_vector:
            # 余弦相似度已在 [0, 1]，直接 clamp
            return max(0.0, min(1.0, score))
        else:
            # BM25 使用 sigmoid 归一化: 1 / (1 + exp(-score / k))
            # k=2 为缩放因子，使分数更平滑地映射到 [0, 1]
            import math
            k = 2.0
            try:
                return 1.0 / (1.0 + math.exp(-score / k))
            except OverflowError:
                return 1.0 if score > 0 else 0.0
