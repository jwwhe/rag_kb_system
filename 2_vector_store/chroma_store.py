"""
================================================================================
  Layer 2 - 向量存储层: Chroma 实现（开发环境）
  功能：
    - Chroma 本地持久化存储
    - 完整实现 BaseVectorStore 全部接口
    - 支持多 Collection 知识库隔离
================================================================================
"""
import os
import uuid
from typing import List, Dict, Optional, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from config.settings import VectorStoreConfig, get_settings_cached
from .base import BaseVectorStore
from utils.exceptions import VectorStoreError
from utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    基于 Chroma 的向量存储实现。
    用于开发环境，数据持久化到本地磁盘。
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        初始化 Chroma 向量存储。

        Args:
            config: 向量存储配置（默认从全局配置读取）
        """
        self.config = config or get_settings_cached().vector_store

        # 确保持久化目录存在
        persist_dir = os.path.abspath(self.config.chroma_persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

        # 初始化 Chroma 持久化客户端
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # 获取或创建 Collection
        collection_name = self.config.chroma_collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

        logger.info(
            f"ChromaVectorStore 初始化完成 | "
            f"持久化路径: {persist_dir} | "
            f"Collection: {collection_name} | "
            f"当前文档数: {self._collection.count()}"
        )

    # ==================== 入库 ====================

    def add_documents(self, documents: List[Document]) -> bool:
        """
        批量入库带嵌入向量的文档。

        Args:
            documents: 携带 embedding 的 Document 列表

        Returns:
            bool: 入库成功
        """
        if not documents:
            logger.warning("文档列表为空，跳过入库")
            return False

        ids = []
        embeddings = []
        metadatas = []
        documents_texts = []

        for doc in documents:
            # 嵌入向量从 metadata 中提取
            emb = doc.metadata.pop("embedding", None)
            if emb is None:
                logger.warning(f"文档 {doc.metadata.get('chunk_id', '?')} 缺少嵌入向量，跳过")
                continue

            doc_id = doc.metadata.get("chunk_id", str(uuid.uuid4()))
            ids.append(doc_id)
            embeddings.append(emb)
            metadatas.append(doc.metadata)
            documents_texts.append(doc.page_content)

        if not ids:
            raise VectorStoreError("没有有效的文档可入库（缺少嵌入向量）")

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_texts,
            )
            logger.info(f"Chroma 入库成功: {len(ids)} 条文档")
            return True
        except Exception as e:
            raise VectorStoreError(f"Chroma 入库失败: {str(e)}") from e

    # ==================== 删除 ====================

    def delete_by_file(self, file_name: str) -> int:
        """
        按文件名删除该文件的所有向量块。

        Args:
            file_name: 源文件名

        Returns:
            int: 删除的块数量
        """
        # 先查询该文件的所有 chunk_id
        results = self._collection.get(
            where={"file_name": file_name},
            include=["metadatas"],
        )

        ids_to_delete = results.get("ids", [])
        if not ids_to_delete:
            logger.info(f"未找到文件 '{file_name}' 的向量数据")
            return 0

        self._collection.delete(ids=ids_to_delete)
        logger.info(f"已删除文件 '{file_name}' 的 {len(ids_to_delete)} 个向量块")
        return len(ids_to_delete)

    def clear_collection(self) -> bool:
        """
        清空当前知识库所有数据（删除并重建 Collection）。
        """
        try:
            collection_name = self.config.chroma_collection_name
            self._client.delete_collection(name=collection_name)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"知识库已清空: {collection_name}")
            return True
        except Exception as e:
            raise VectorStoreError(f"清空知识库失败: {str(e)}") from e

    # ==================== 检索 ====================

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 8,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        向量相似度检索（余弦相似度）。

        Args:
            query_embedding: 查询向量
            top_k:           返回数量
            filter_dict:     元数据过滤条件

        Returns:
            List[Document]: 相似文档列表，metadata 中包含 similarity 分数
        """
        where_filter = filter_dict or {}
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        try:
            results = self._collection.query(**query_kwargs)
        except Exception as e:
            raise VectorStoreError(f"Chroma 检索失败: {str(e)}") from e

        return self._build_result_documents(results)

    def search_with_filter(
        self,
        query_embedding: List[float],
        metadata_filter: Dict[str, Any],
        top_k: int = 8,
    ) -> List[Document]:
        """
        带元数据过滤的向量检索。

        Args:
            query_embedding: 查询向量
            metadata_filter: 元数据过滤条件
            top_k:           返回数量

        Returns:
            List[Document]: 过滤后的相似文档列表
        """
        return self.search(query_embedding, top_k, metadata_filter)

    # ==================== 全量获取 ====================

    def get_all_documents(self) -> List[Document]:
        """
        获取知识库中全部文档（供 BM25 索引构建）。

        Returns:
            List[Document]: 所有文档列表
        """
        try:
            results = self._collection.get(
                include=["documents", "metadatas"],
            )
        except Exception as e:
            raise VectorStoreError(f"获取全部文档失败: {str(e)}") from e

        return self._build_result_documents_from_get(results)

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息。

        Returns:
            Dict: 包含总文档数、文件名列表
        """
        docs = self.get_all_documents()
        file_names = list(set(
            doc.metadata.get("file_name", "unknown") for doc in docs
        ))
        return {
            "store_type": "chroma",
            "collection_name": self.config.chroma_collection_name,
            "total_chunks": len(docs),
            "total_files": len(file_names),
            "file_names": sorted(file_names),
        }

    # ==================== 私有方法 ====================

    def _build_result_documents(self, results: Dict) -> List[Document]:
        """
        将 Chroma query 返回结果转换为 LangChain Document 列表。

        Chroma 返回结构：
          ids:       [[id1, id2, ...]]
          documents: [[text1, text2, ...]]
          metadatas: [[meta1, meta2, ...]]
          distances: [[dist1, dist2, ...]]

        Args:
            results: Chroma query 返回的原始结果

        Returns:
            List[Document]: 转换后的文档列表
        """
        documents = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for i in range(len(ids_list)):
            meta = metas_list[i] if i < len(metas_list) else {}
            # Chroma 返回 distance（越小越相似），转换为 similarity（越大越相似）
            distance = dists_list[i] if i < len(dists_list) else 1.0
            similarity = 1.0 - min(distance, 1.0)  # 余弦距离 → 余弦相似度

            meta["similarity"] = round(similarity, 6)
            meta["chunk_id"] = ids_list[i]

            doc = Document(
                page_content=docs_list[i] if i < len(docs_list) else "",
                metadata=meta,
            )
            documents.append(doc)

        return documents

    def _build_result_documents_from_get(self, results: Dict) -> List[Document]:
        """
        将 Chroma get 返回结果转换为 LangChain Document 列表。

        Args:
            results: Chroma get 返回的原始结果

        Returns:
            List[Document]: 转换后的文档列表
        """
        documents = []
        ids_list = results.get("ids", [])
        docs_list = results.get("documents", [])
        metas_list = results.get("metadatas", [])

        for i in range(len(ids_list)):
            meta = metas_list[i] if i < len(metas_list) else {}
            meta["chunk_id"] = ids_list[i]

            doc = Document(
                page_content=docs_list[i] if i < len(docs_list) else "",
                metadata=meta,
            )
            documents.append(doc)

        return documents
