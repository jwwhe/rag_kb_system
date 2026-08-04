# 3_retrieval - 检索层 (Layer 3)
# 职责：查询改写 → 向量检索(MMR去重) → BM25检索 → 混合融合 → Rerank精排 → 多源融合
# 依赖：config + utils + Layer 2 (vector_store)
# 两阶段精排：MMR（多样性去重）+ BGE-Reranker（语义精排）
# 多源融合：论文原文 + 综述解读 + 实验笔记 按配额融合

from .query_rewrite import QueryRewriter
from .vector_search import VectorSearcher
from .bm25_search import BM25Searcher
from .hybrid_search import HybridSearcher
from .mmr_search import MMRReranker
from .rerank import Reranker
from .multi_source_fusion import MultiSourceFusioner

__all__ = [
    "QueryRewriter",
    "VectorSearcher",
    "BM25Searcher",
    "HybridSearcher",
    "MMRReranker",
    "Reranker",
    "MultiSourceFusioner",
]
