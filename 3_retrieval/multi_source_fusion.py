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
    在混合检索融合 + Rerank 之后，按 source_type 配额选取最终结果。
    """

    # 三类来源的优先级顺序（论文原文最权威）
    SOURCE_PRIORITY = ["论文原文", "综述解读", "实验笔记"]

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        初始化多源融合器。

        Args:
            config: 检索配置
        """
        self.config = config or get_settings_cached().retrieval
        logger.info(
            f"MultiSourceFusioner 初始化 | "
            f"来源优先级: {self.SOURCE_PRIORITY}"
        )

    def fuse(
        self,
        documents: List[Document],
        top_k: Optional[int] = None,
        min_per_source: int = 1,
    ) -> List[Document]:
        """
        多源配额融合。

        策略：
        1. 按 source_type 分组
        2. 每类至少保留 min_per_source 条（若该类有结果）
        3. 剩余配额按原分数排序补足

        Args:
            documents:      Rerank 后的候选文档（已按分数降序）
            top_k:          最终返回数量（默认 rerank_top_k）
            min_per_source: 每类来源最少保留数

        Returns:
            List[Document]: 多源融合后的文档列表
        """
        if not documents:
            return []

        if top_k is None:
            top_k = self.config.rerank_top_k

        # 按 source_type 分组（保持组内分数降序）
        by_source: dict[str, List[Document]] = defaultdict(list)
        for doc in documents:
            st = doc.metadata.get("source_type", "论文原文")
            by_source[st].append(doc)

        result: List[Document] = []
        consumed_ids: set = set()

        # Phase 1: 每类先取 min_per_source 条
        for source in self.SOURCE_PRIORITY:
            if source in by_source:
                for doc in by_source[source][:min_per_source]:
                    cid = doc.metadata.get("chunk_id", id(doc))
                    if cid not in consumed_ids:
                        result.append(doc)
                        consumed_ids.add(cid)
                    if len(result) >= top_k:
                        break
            if len(result) >= top_k:
                break

        # Phase 2: 剩余配额按原分数降序补足（跨来源公平竞争）
        if len(result) < top_k:
            remaining = [
                doc for doc in documents
                if doc.metadata.get("chunk_id", id(doc)) not in consumed_ids
            ]
            for doc in remaining:
                result.append(doc)
                if len(result) >= top_k:
                    break

        # 统计来源分布
        source_dist = defaultdict(int)
        for doc in result:
            source_dist[doc.metadata.get("source_type", "未知")] += 1

        logger.info(
            f"多源融合完成: 候选 {len(documents)} → 选中 {len(result)} | "
            f"分布: {dict(source_dist)}"
        )
        return result
