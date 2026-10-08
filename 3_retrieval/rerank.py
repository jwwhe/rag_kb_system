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
import re
import time
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import RetrievalConfig, get_settings_cached
from utils.exceptions import RetrievalError
from utils.logger import get_logger
from utils.runtime import apply_cpu_threads

logger = get_logger(__name__)

# 内容去重时忽略空白差异（PDF/Markdown 换行位置不稳定）
_WHITESPACE_RE = re.compile(r"\s+")


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

    def warmup(self):
        """
        预热：真正加载 CrossEncoder 权重。
        Reranker() 构造本身是惰性的，不调用此方法时模型加载开销会落在首个请求上。
        """
        if self._model is None:
            self._load_model()

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_all: bool = False,
    ) -> List[Document]:
        """
        对候选文档进行重排序。

        CPU 精排是本系统检索链路的主要耗时来源（约 2 秒/条），
        因此先按融合分数截断到 rerank_max_candidates，再送模型打分。

        Args:
            query:     用户查询文本
            documents: 候选文档列表
            top_k:     保留数量（默认 Top3）
            score_all: True 时返回全部已打分候选（不截断），
                       供多源融合在候选池上按来源配额挑选

        Returns:
            List[Document]: 重排后的文档列表（均带 metadata.rerank_score）
        """
        if not documents:
            logger.warning("候选文档列表为空，跳过重排")
            return []

        if top_k is None:
            top_k = self.config.rerank_top_k

        # 候选去重 + 截断：只保留检索分最高的 N 条
        candidates = self._cap_candidates(documents)
        if len(candidates) < len(documents):
            logger.info(
                f"Rerank 候选筛选: {len(documents)} → {len(candidates)} "
                f"(去重/截断，上限 {self.config.rerank_max_candidates})"
            )

        # 惰性加载模型
        if self._model is None:
            self._load_model()

        t0 = time.perf_counter()
        try:
            # 构建 (query, document) 对
            pairs = [[query, doc.page_content] for doc in candidates]

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
        predict_ms = (time.perf_counter() - t0) * 1000

        # 绑定 rerank 分数并排序（全部候选都带上 rerank_score，供溯源与配额挑选使用）
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: float(x[1]), reverse=True)
        for doc, score in scored_docs:
            doc.metadata["rerank_score"] = round(float(score), 6)

        if top_k is None:
            top_k = self.config.rerank_top_k
        keep = scored_docs if score_all else scored_docs[:top_k]
        result = [doc for doc, _ in keep]

        logger.info(
            f"Rerank 完成: {len(candidates)} → {len(result)} 条"
            f"{'（全量候选，未截断）' if score_all else f'（Top{top_k}）'} "
            f"({predict_ms:.0f}ms | {predict_ms / max(1, len(candidates)):.0f}ms/条) | "
            f"最高分: {result[0].metadata.get('rerank_score', 'N/A') if result else 'N/A'}"
        )

        return result

    def _cap_candidates(self, documents: List[Document]) -> List[Document]:
        """
        候选预筛：先按内容去重，再按检索分截断，控制 CrossEncoder 的推理条数。

        同一份文档重复入库时，向量检索与 BM25 会把内容完全相同的块一起送进精排，
        而相同文本的 rerank 分数必然相同——去重既省时间又不改变排序结果。

        分数优先级：hybrid_score > similarity；均缺失时保持原顺序。

        Args:
            documents: 全部候选文档

        Returns:
            List[Document]: 去重并截断后的候选（按分数降序）
        """
        max_candidates = self.config.rerank_max_candidates
        if not max_candidates or max_candidates <= 0:
            max_candidates = len(documents)

        def score_of(doc: Document) -> float:
            value = doc.metadata.get(
                "hybrid_score", doc.metadata.get("similarity")
            )
            try:
                return float(value)
            except (TypeError, ValueError):
                return -1.0

        picked: List[Document] = []
        seen_texts = set()
        for doc in sorted(documents, key=score_of, reverse=True):
            key = _WHITESPACE_RE.sub("", doc.page_content)
            if not key or key in seen_texts:
                continue
            seen_texts.add(key)
            picked.append(doc)
            if len(picked) >= max_candidates:
                break
        return picked

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

            t0 = time.perf_counter()
            # CPU 推理线程数调优（须在模型加载前生效）
            apply_cpu_threads()
            self._model = CrossEncoder(
                self.config.rerank_model,
                device=self.config.rerank_device,
                max_length=self.config.rerank_max_length,
            )
            logger.info(
                f"bge-reranker CrossEncoder 加载完成: {self.config.rerank_model} "
                f"({(time.perf_counter() - t0) * 1000:.0f}ms | "
                f"max_length={self.config.rerank_max_length})"
            )
        except ImportError:
            raise RetrievalError(
                "sentence-transformers 未安装，请执行: pip install sentence-transformers"
            )
        except Exception as e:
            raise RetrievalError(
                f"bge-reranker 模型加载失败 [{self.config.rerank_model}]: {str(e)}"
            ) from e
