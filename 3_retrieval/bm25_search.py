"""
================================================================================
  Layer 3 - 检索层: BM25 关键词检索器
  功能：
    - BM25 关键词精准匹配
    - 适用于名词、编号、专业术语等精确检索
    - 中文分词使用 jieba
================================================================================
"""
from typing import List, Optional, Dict

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config.settings import RetrievalConfig, get_settings_cached
from utils.exceptions import RetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)


class BM25Searcher:
    """
    BM25 关键词检索器。
    基于词频-逆文档频率的关键词精准匹配。
    适用于编号、专业术语、专有名词等精确查询场景。
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        初始化 BM25 检索器。

        Args:
            config: 检索配置（默认从全局配置读取）
        """
        self.config = config or get_settings_cached().retrieval
        self._index: Optional[BM25Okapi] = None
        self._corpus: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []
        self._dirty: bool = True  # 脏标记：文档变更后需重建索引

        logger.info(
            f"BM25Searcher 初始化完成 | TopK: {self.config.bm25_top_k}"
        )

    def build_index(self, documents: List[Document]) -> None:
        """
        从文档列表构建 BM25 索引。

        使用 jieba 进行中文分词，构建倒排索引。

        Args:
            documents: 所有文档块列表（来自向量库全量获取）
        """
        if not documents:
            logger.warning("文档列表为空，跳过 BM25 索引构建")
            self._index = None
            self._corpus = []
            self._tokenized_corpus = []
            return

        self._corpus = list(documents)

        # jieba 中文分词 + 去停用词
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in self._corpus
        ]

        # 构建 BM25 索引
        self._index = BM25Okapi(self._tokenized_corpus)
        self._dirty = False

        logger.info(
            f"BM25 索引构建完成: {len(self._corpus)} 篇文档"
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        执行 BM25 关键词检索。

        检索流程：
        1. 对查询文本进行分词
        2. 在 BM25 索引中计算匹配分数
        3. 返回 TopK 结果

        Args:
            query:  查询文本
            top_k:  返回数量（默认从配置读取）

        Returns:
            List[Document]: 匹配文档列表（按 BM25 分数降序）

        Raises:
            RetrievalError: 索引未构建或检索失败
        """
        if self._index is None or self._dirty:
            raise RetrievalError("BM25 索引未构建，请先调用 build_index()")

        if top_k is None:
            top_k = self.config.bm25_top_k

        try:
            tokenized_query = self._tokenize(query)
            scores = self._index.get_scores(tokenized_query)

            # 按分数降序排序，取 TopK
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            results = []
            for idx, score in indexed_scores[:top_k]:
                if score <= 0:
                    continue
                doc = self._corpus[idx]
                # 创建副本，添加 BM25 分数
                result_doc = Document(
                    page_content=doc.page_content,
                    metadata=dict(doc.metadata),
                )
                result_doc.metadata["bm25_score"] = float(score)
                results.append(result_doc)

            logger.info(
                f"BM25 检索完成: 查询 '{query[:30]}...' → {len(results)} 条结果"
            )
            return results

        except Exception as e:
            raise RetrievalError(f"BM25 检索失败: {str(e)}") from e

    def mark_dirty(self) -> None:
        """标记索引为脏（文档变更后调用）。"""
        self._dirty = True
        logger.info("BM25 索引已标记为脏")

    def _tokenize(self, text: str) -> List[str]:
        """
        对中文文本进行分词和清洗。

        Args:
            text: 输入文本

        Returns:
            List[str]: 分词结果
        """
        # jieba 精确模式分词
        tokens = jieba.lcut(text)

        # 基础清洗：去除空白字符和单字
        cleaned = [
            t.strip()
            for t in tokens
            if t.strip() and len(t.strip()) > 1
        ]
        return cleaned
