"""
================================================================================
  Layer 1 - 文档处理层: 嵌入向量化器
  功能：
    - 使用 bge-m3 模型批量向量化
    - 失败自动重试、防卡死机制
    - 返回带嵌入向量的 Document 列表
================================================================================
"""
import os
import time
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from config.settings import DocProcessConfig, get_settings_cached
from utils.exceptions import DocumentProcessError
from utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    嵌入向量化器。
    基于 bge-m3 模型，支持批量编码和失败重试。
    """

    def __init__(self, config: Optional[DocProcessConfig] = None):
        """
        初始化嵌入模型。

        Args:
            config: 文档处理配置（默认从全局配置读取）

        Raises:
            DocumentProcessError: 模型加载失败
        """
        self.config = config or get_settings_cached().doc_process

        try:
            # 设置 HuggingFace 镜像（国内网络环境加速下载）
            if self.config.hf_endpoint:
                os.environ["HF_ENDPOINT"] = self.config.hf_endpoint
            if self.config.hf_home:
                os.environ["HF_HOME"] = self.config.hf_home

            # 加载 bge-m3 嵌入模型
            # encode_kwargs 设置 normalize_embeddings=True 以归一化输出
            self._model = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": self.config.embedding_device},
                encode_kwargs={
                    "normalize_embeddings": self.config.embedding_normalize,
                    "batch_size": self.config.embedding_batch_size,
                },
            )
            logger.info(
                f"Embedder 初始化完成 | "
                f"模型: {self.config.embedding_model} | "
                f"设备: {self.config.embedding_device} | "
                f"批次大小: {self.config.embedding_batch_size}"
            )
        except Exception as e:
            raise DocumentProcessError(
                f"嵌入模型加载失败 [{self.config.embedding_model}]: {str(e)}"
            ) from e

    def embed_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        为文档列表批量生成嵌入向量。

        内置重试机制：逐批编码，单批失败自动重试，防止整个批次因个别数据卡死。

        Args:
            documents: 待向量化的文档列表（已分块）

        Returns:
            List[Document]: 携带嵌入向量的文档列表（原 metadata 不变）

        Raises:
            DocumentProcessError: 全部重试失败
        """
        if not documents:
            logger.warning("输入文档列表为空，跳过向量化")
            return []

        batch_size = self.config.embedding_batch_size
        total = len(documents)
        success_count = 0
        fail_count = 0

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            batch_texts = [doc.page_content for doc in batch]
            batch_num = i // batch_size + 1

            # 带重试的批量编码
            embeddings = self._encode_with_retry(
                batch_texts, batch_num
            )

            if embeddings is None:
                fail_count += len(batch)
                logger.error(
                    f"第 {batch_num} 批向量化彻底失败，跳过 {len(batch)} 条"
                )
                continue

            # 将嵌入向量写回 Document.metadata（供向量库存取）
            for doc, emb in zip(batch, embeddings):
                doc.metadata["embedding"] = emb

            success_count += len(batch)

        logger.info(
            f"向量化完成: 成功 {success_count}/{total} | 失败 {fail_count}/{total}"
        )

        if success_count == 0:
            raise DocumentProcessError("所有文档向量化均失败")

        return documents

    def embed_query(self, query: str) -> List[float]:
        """
        为单条查询文本生成嵌入向量。

        Args:
            query: 查询文本

        Returns:
            List[float]: 嵌入向量
        """
        return self._model.embed_query(query)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        为多条文本生成嵌入向量。

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        return [self._model.embed_query(t) for t in texts]

    def _encode_with_retry(
        self, texts: List[str], batch_num: int
    ) -> Optional[List[List[float]]]:
        """
        带重试的批量编码，防止单批卡死导致整体失败。

        Args:
            texts:     待编码文本列表
            batch_num: 批次编号（日志用）

        Returns:
            Optional[List[List[float]]]: 嵌入向量列表，失败返回 None
        """
        max_retries = self.config.embedding_max_retries
        retry_delay = self.config.embedding_retry_delay

        for attempt in range(1, max_retries + 1):
            try:
                embeddings = self._model.embed_documents(texts)
                return embeddings
            except Exception as e:
                logger.warning(
                    f"第 {batch_num} 批向量化失败 (尝试 {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)  # 递增等待
                else:
                    logger.error(f"第 {batch_num} 批向量化已达最大重试次数")
        return None
