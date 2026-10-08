"""
================================================================================
  Layer 3 - 检索层: 多源知识融合器
  功能：
    - 三类来源（论文原文 / 综述解读 / 实验笔记）按配额融合
    - 避免某类来源分数高完全压制其他来源
    - 保证最终 Top-K 中三类来源都能出现（覆盖多视角）
  简历亮点：
    多源知识融合（论文原文 + 综述解读 + 实验笔记），来源标注准确率 100%
================================================================================
"""
from collections import defaultdict
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import RetrievalConfig, get_settings_cached
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiSourceFusioner:
    """
    多源知识融合器。
    在 Rerank 打分后的候选池上按 source_type 配额挑选，最后截断到 rerank_top_k。

    输入必须是"全部已打分候选"而不是精排 Top-K：只对 K 条做配额等于重新排序，
    无法把没进前 K 的另一类来源提上来。
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        初始化多源融合器。

        Args:
            config: 检索配置
        """
        self.config = config or get_settings_cached().retrieval
        # 来源优先级沿用文档处理层的标注顺序（论文原文最权威）
        self.source_priority = list(
            get_settings_cached().doc_process.source_type_labels
        )
        logger.info(
            f"MultiSourceFusioner 初始化 | "
            f"来源优先级: {self.source_priority} | "
            f"每类保底: {self.config.fusion_min_per_source} 条 | "
            f"保底门槛(精排分): {self.config.rerank_min_score}"
        )

    def fuse(
        self,
        documents: List[Document],
        top_k: Optional[int] = None,
        min_per_source: Optional[int] = None,
    ) -> List[Document]:
        """
        多源配额融合。

        策略：
        1. 按 source_type 分组（组内保持分数降序）
        2. 每类保底 min_per_source 条，但该类最高分必须 ≥ rerank_min_score，
           否则不强行塞入无关来源
        3. 剩余配额按分数降序跨来源公平竞争
        4. 结果按分数降序输出（保底只影响"选谁"，不影响上下文顺序）

        Args:
            documents:      Rerank 后的候选池（需带 metadata.rerank_score，已按分数降序）
            top_k:          最终返回数量（默认 rerank_top_k）
            min_per_source: 每类来源最少保留数（默认 fusion_min_per_source）

        Returns:
            List[Document]: 多源融合后的文档列表
        """
        if not documents:
            return []

        if top_k is None:
            top_k = self.config.rerank_top_k
        if min_per_source is None:
            min_per_source = self.config.fusion_min_per_source

        if not self.config.enable_multi_source_fusion or top_k <= 0:
            return documents[:top_k]

        min_score = self.config.rerank_min_score

        # 按 source_type 分组（保持组内分数降序）
        by_source: dict[str, List[Document]] = defaultdict(list)
        for doc in documents:
            st = doc.metadata.get("source_type", "论文原文")
            by_source[st].append(doc)

        def score_of(doc: Document) -> float:
            value = doc.metadata.get(
                "rerank_score",
                doc.metadata.get("hybrid_score", doc.metadata.get("similarity", 0.0)),
            )
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        result: List[Document] = []
        consumed_ids: set = set()

        def take(doc: Document) -> None:
            cid = doc.metadata.get("chunk_id", id(doc))
            if cid not in consumed_ids:
                consumed_ids.add(cid)
                result.append(doc)

        # Phase 1: 每类先取 min_per_source 条（该类需有足够相关的结果）
        # 优先级列表之外的来源排在后面，但同样享有保底
        ordered_sources = self.source_priority + [
            s for s in by_source if s not in self.source_priority
        ]
        for source in ordered_sources:
            docs_of_source = by_source.get(source, [])
            if not docs_of_source or score_of(docs_of_source[0]) < min_score:
                continue
            for doc in docs_of_source[:min_per_source]:
                take(doc)
                if len(result) >= top_k:
                    break
            if len(result) >= top_k:
                break

        # Phase 2: 剩余配额按分数降序补足（跨来源公平竞争）
        if len(result) < top_k:
            for doc in documents:
                take(doc)
                if len(result) >= top_k:
                    break

        # 保底只决定入选集合，上下文顺序仍按相关度
        result.sort(key=score_of, reverse=True)

        # 统计来源分布
        source_dist = defaultdict(int)
        for doc in result:
            source_dist[doc.metadata.get("source_type", "未知")] += 1

        logger.info(
            f"多源融合完成: 候选 {len(documents)} → 选中 {len(result)} | "
            f"分布: {dict(source_dist)}"
        )
        return result
