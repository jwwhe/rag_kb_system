"""
================================================================================
  Layer 5 - 增强层: 全文溯源追踪器
  功能（V1.0 必须全上线）：
    - 回答绑定文件名、页码、原文片段、相似度分数
    - 溯源信息结构化输出
    - 区分内网知识/外网知识来源
================================================================================
"""
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from config.settings import EnhanceConfig, get_settings_cached
from utils.logger import get_logger

logger = get_logger(__name__)


class CitationTracer:
    """
    全文溯源追踪器。
    为 AI 回答绑定完整的引用溯源信息，
    确保每个观点都能追溯到具体的源文档位置。
    """

    def __init__(self, config: Optional[EnhanceConfig] = None):
        """
        初始化溯源追踪器。

        Args:
            config: 增强配置（默认从全局配置读取）
        """
        self.config = config or get_settings_cached().enhance
        logger.info(
            f"CitationTracer 初始化完成 | "
            f"最低相似度: {self.config.citation_min_similarity}"
        )

    def build_citations(
        self,
        documents: List[Document],
        source_type: str = "internal",
    ) -> List[Dict[str, Any]]:
        """
        根据检索到的文档列表构建结构化引用信息。

        每条引用包含：
        - citation_id:   引用编号
        - file_name:     源文件名
        - page:          页码
        - original_text: 原文片段（前200字）
        - similarity:    相似度分数
        - source_type:   来源类型（internal / external）
        - doc_source:    文档来源类型（论文原文/综述解读/实验笔记，仅内部库）

        Args:
            documents:   检索结果文档列表
            source_type: 来源类型 "internal"(本地知识库) / "external"(网络搜索)

        Returns:
            List[Dict]: 结构化引用列表
        """
        if not self.config.enable_citation:
            return []

        citations = []

        for i, doc in enumerate(documents, 1):
            similarity = doc.metadata.get(
                "rerank_score",
                doc.metadata.get("hybrid_score",
                doc.metadata.get("similarity", 0)),
            )

            # 低相似度过滤
            if similarity < self.config.citation_min_similarity:
                continue

            citation = {
                "citation_id": i,
                "file_name": doc.metadata.get("file_name", "未知来源"),
                "page": doc.metadata.get("page", "N/A"),
                "original_text": doc.page_content[:200] + (
                    "..." if len(doc.page_content) > 200 else ""
                ),
                "similarity": round(similarity, 4),
                "source_type": source_type,
            }
            # 多源知识融合：透传文档自身的来源类型（论文原文/综述解读/实验笔记）
            if source_type == "internal":
                citation["doc_source"] = doc.metadata.get(
                    "source_type", "论文原文"
                )
            citations.append(citation)

        # 统计来源分布
        if source_type == "internal":
            from collections import Counter
            dist = Counter(c.get("doc_source", "未知") for c in citations)
            logger.info(
                f"溯源信息构建完成: {len(citations)} 条引用 "
                f"(分布: {dict(dist)})"
            )
        else:
            logger.info(
                f"溯源信息构建完成: {len(citations)} 条引用 "
                f"(来源: {source_type})"
            )
        return citations

    def format_citation_section(
        self,
        citations: List[Dict[str, Any]],
    ) -> str:
        """
        将引用列表格式化为可展示的溯源段落。

        Args:
            citations: 结构化引用列表

        Returns:
            str: 格式化的溯源文本
        """
        if not citations:
            return ""

        internal_citations = [
            c for c in citations if c["source_type"] == "internal"
        ]
        external_citations = [
            c for c in citations if c["source_type"] == "external"
        ]

        parts = []

        if internal_citations:
            parts.append("【本地知识库引用来源】")
            for c in internal_citations:
                parts.append(
                    f"  [{c['citation_id']}] {c['file_name']} "
                    f"(第{c['page']}页) | 相关度: {c['similarity']}\n"
                    f"      原文: \"{c['original_text']}\""
                )

        if external_citations:
            parts.append("\n【外部网络搜索来源】")
            for c in external_citations:
                parts.append(
                    f"  [{c['citation_id']}] {c['file_name']} | "
                    f"相关度: {c['similarity']}\n"
                    f"      原文: \"{c['original_text']}\""
                )

        return "\n\n".join(parts)

    def build_full_response(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
        is_from_web: bool = False,
    ) -> Dict[str, Any]:
        """
        构建完整的带溯源的结构化响应。

        Args:
            answer:       AI 生成的答案
            citations:    引用列表
            is_from_web:  答案是否来源于网络搜索

        Returns:
            Dict: 包含 answer, citations, source_type 的结构化响应
        """
        internal_count = sum(
            1 for c in citations if c["source_type"] == "internal"
        )
        external_count = sum(
            1 for c in citations if c["source_type"] == "external"
        )

        source_type = "external" if is_from_web else "internal"

        # 构建溯源文本
        citation_text = self.format_citation_section(citations)

        response = {
            "answer": answer,
            "citations": citations,
            "citation_text": citation_text,
            "source_type": source_type,
            "source_stats": {
                "internal_sources": internal_count,
                "external_sources": external_count,
                "total_sources": len(citations),
            },
        }

        return response
