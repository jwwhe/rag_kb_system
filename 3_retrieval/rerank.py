"""
================================================================================
  Layer 3 - 检索层: Rerank 重排器
  功能：
    - 使用 bge-reranker-v2-m3 CrossEncoder 对候选文档精排打分
    - 保留 Top3 最优上下文送入 LLM 生成
  兼容性：
    - 使用 sentence_transformers.CrossEncoder（更稳定，避免 FlagEmbedding 版本冲突）
================================================================================
"""
import os
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import RetrievalConfig, get_settings_cached
from utils.exceptions import RetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Rerank 重排器。
    使用 sentence-transformers CrossEncoder 对检索结果进行精细排序，
    选出最优 Top3 上下文块送入生成层。
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or get_settings_cached().retrieval
        self._model = None  # 惰性加载

        logger.info(
            f"Reranker 初始化完成 | "
            f"模型: {self.config.rerank_model} | "
            f"TopK: {self.config.rerank_top_k} | "
            f"设备: {self.config.rerank_device}"
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        对候选文档进行重排序。

        Args:
            query:     用户查询文本
            documents: 候选文档列表
            top_k:     保留数量（默认 Top3）

        Returns:
            List[Document]: 重排后的最优文档列表
        """
        if not documents:
            logger.warning("候选文档列表为空，跳过重排")
            return []

        if top_k is None:
            top_k = self.config.rerank_top_k

        # 惰性加载模型
        if self._model is None:
            self._load_model()

        try:
            # 构建 (query, document) 对
            pairs = [[query, doc.page_content] for doc in documents]

            # CrossEncoder predict 打分
            scores = self._model.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
            )

            # 确保 scores 是列表格式
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            if not isinstance(scores, list):
                scores = [float(scores)]

        except Exception as e:
            raise RetrievalError(f"Rerank 模型推理失败: {str(e)}") from e

        # 绑定 rerank 分数并排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

        # 取 TopK
        top_docs = []
        for doc, score in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = round(float(score), 6)
            top_docs.append(doc)

        logger.info(
            f"Rerank 完成: {len(documents)} → Top{top_k} | "
            f"最高分: {top_docs[0].metadata.get('rerank_score', 'N/A') if top_docs else 'N/A'}"
        )

        return top_docs

    def _load_model(self):
        """
        惰性加载 bge-reranker 模型。
        使用 sentence_transformers.CrossEncoder，兼容性更好。
        """
        import os as _os
        if self.config.rerank_hf_endpoint:
            _os.environ.setdefault("HF_ENDPOINT", self.config.rerank_hf_endpoint)

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.config.rerank_model,
                device=self.config.rerank_device,
            )
            logger.info(
                f"bge-reranker CrossEncoder 加载完成: {self.config.rerank_model}"
            )
        except ImportError:
            raise RetrievalError(
                "sentence-transformers 未安装，请执行: pip install sentence-transformers"
            )
        except Exception as e:
            raise RetrievalError(
                f"bge-reranker 模型加载失败 [{self.config.rerank_model}]: {str(e)}"
            ) from e
