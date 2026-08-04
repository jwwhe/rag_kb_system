"""
================================================================================
  Layer 2 - 向量存储层: Qdrant 实现（生产环境）
  功能：
    - Qdrant 向量数据库连接（Docker 容器挂载持久化）
    - 完整实现 BaseVectorStore 全部接口
    - 支持多 Collection 知识库隔离
================================================================================
"""
import uuid
from typing import List, Dict, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Record,
)
from langchain_core.documents import Document

from config.settings import VectorStoreConfig, get_settings_cached
from .base import BaseVectorStore
from utils.exceptions import VectorStoreError
from utils.logger import get_logger

logger = get_logger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """
    基于 Qdrant 的向量存储实现。
    用于生产环境，支持 Docker 容器挂载持久化。
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        初始化 Qdrant 向量存储。

        Args:
            config: 向量存储配置（默认从全局配置读取）

        Raises:
            VectorStoreError: Qdrant 服务连接失败
        """
        self.config = config or get_settings_cached().vector_store

        try:
            self._client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
            )
            # 验证连接
            self._client.get_collections()
        except Exception as e:
            raise VectorStoreError(
                f"Qdrant 连接失败 [{self.config.qdrant_host}:{self.config.qdrant_port}]: {str(e)}"
            ) from e

        self._collection_name = self.config.qdrant_collection_name

        # 确保 Collection 存在
        self._ensure_collection()

        collection_info = self._client.get_collection(self._collection_name)
        logger.info(
            f"QdrantVectorStore 初始化完成 | "
            f"地址: {self.config.qdrant_host}:{self.config.qdrant_port} | "
            f"Collection: {self._collection_name} | "
            f"向量数: {collection_info.vectors_count}"
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

        points = []
        for doc in documents:
            emb = doc.metadata.pop("embedding", None)
            if emb is None:
                logger.warning(f"文档 {doc.metadata.get('chunk_id', '?')} 缺少嵌入向量，跳过")
                continue

            point_id = doc.metadata.get("chunk_id", str(uuid.uuid4()))
            # Qdrant 不支持 str 作为 id，使用 UUID 或 hash
            point_uuid = self._str_to_uuid(point_id)

            # 构建 payload（元数据 + 文本内容）
            payload = dict(doc.metadata)
            payload["page_content"] = doc.page_content

            point = PointStruct(
                id=point_uuid,
                vector=emb,
                payload=payload,
            )
            points.append(point)

        if not points:
            raise VectorStoreError("没有有效的文档可入库（缺少嵌入向量）")

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            logger.info(f"Qdrant 入库成功: {len(points)} 条文档")
            return True
        except Exception as e:
            raise VectorStoreError(f"Qdrant 入库失败: {str(e)}") from e

    # ==================== 删除 ====================

    def delete_by_file(self, file_name: str) -> int:
        """
        按文件名删除该文件的所有向量块。

        Args:
            file_name: 源文件名

        Returns:
            int: 删除的块数量
        """
        # 先统计匹配数量
        count_filter = Filter(
            must=[
                FieldCondition(
                    key="file_name",
                    match=MatchValue(value=file_name),
                )
            ]
        )
        count_result = self._client.count(
            collection_name=self._collection_name,
            count_filter=count_filter,
        )
        total = count_result.count

        if total == 0:
            logger.info(f"未找到文件 '{file_name}' 的向量数据")
            return 0

        # 删除匹配的点
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=count_filter,
        )
        logger.info(f"已删除文件 '{file_name}' 的 {total} 个向量块")
        return total

    def clear_collection(self) -> bool:
        """
        清空当前知识库所有数据。
        """
        try:
            self._client.delete_collection(self._collection_name)
            self._ensure_collection()
            logger.info(f"知识库已清空: {self._collection_name}")
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
        向量相似度检索。

        Args:
            query_embedding: 查询向量
            top_k:           返回数量
            filter_dict:     元数据过滤条件

        Returns:
            List[Document]: 相似文档列表，metadata 中包含 similarity 分数
        """
        query_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            query_filter = Filter(must=conditions)

        try:
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as e:
            raise VectorStoreError(f"Qdrant 检索失败: {str(e)}") from e

        return self._build_result_documents(results)

    def search_with_filter(
        self,
        query_embedding: List[float],
        metadata_filter: Dict[str, Any],
        top_k: int = 8,
    ) -> List[Document]:
        """
        带元数据过滤的向量检索。
        """
        return self.search(query_embedding, top_k, metadata_filter)

    # ==================== 全量获取 ====================

    def get_all_documents(self) -> List[Document]:
        """
        获取知识库中全部文档（供 BM25 索引构建）。

        注意：Qdrant 不支持一键获取全部数据，使用 scroll 遍历。
        """
        try:
            all_records = []
            offset = None

            while True:
                records, next_offset = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=None,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                all_records.extend(records)
                if next_offset is None:
                    break
                offset = next_offset

            return self._build_documents_from_records(all_records)
        except Exception as e:
            raise VectorStoreError(f"获取全部文档失败: {str(e)}") from e

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息。
        """
        try:
            info = self._client.get_collection(self._collection_name)
            # 获取文件列表
            docs = self.get_all_documents()
            file_names = list(set(
                doc.metadata.get("file_name", "unknown") for doc in docs
            ))
            return {
                "store_type": "qdrant",
                "collection_name": self._collection_name,
                "total_chunks": info.vectors_count,
                "total_files": len(file_names),
                "file_names": sorted(file_names),
            }
        except Exception as e:
            raise VectorStoreError(f"获取统计信息失败: {str(e)}") from e

    # ==================== 私有方法 ====================

    def _ensure_collection(self):
        """确保 Collection 存在，不存在则创建。"""
        collections = [
            c.name for c in self._client.get_collections().collections
        ]
        if self._collection_name not in collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self.config.qdrant_vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"创建 Qdrant Collection: {self._collection_name}")

    def _build_result_documents(self, results: List) -> List[Document]:
        """
        将 Qdrant 搜索结果转换为 LangChain Document 列表。

        Args:
            results: Qdrant ScoredPoint 列表

        Returns:
            List[Document]: 转换后的文档列表
        """
        documents = []
        for point in results:
            payload = point.payload or {}
            page_content = payload.pop("page_content", "")
            # Qdrant 返回 score（越大越相似），直接作为 similarity
            payload["similarity"] = round(point.score, 6)
            payload["chunk_id"] = point.id

            doc = Document(page_content=page_content, metadata=payload)
            documents.append(doc)
        return documents

    def _build_documents_from_records(
        self, records: List[Record]
    ) -> List[Document]:
        """
        将 Qdrant Record 列表转换为 LangChain Document 列表。

        Args:
            records: Qdrant Record 列表

        Returns:
            List[Document]: 转换后的文档列表
        """
        documents = []
        for record in records:
            payload = record.payload or {}
            page_content = payload.pop("page_content", "")
            payload["chunk_id"] = record.id

            doc = Document(page_content=page_content, metadata=payload)
            documents.append(doc)
        return documents

    @staticmethod
    def _str_to_uuid(s: str) -> str:
        """
        将字符串 ID 转换为 UUID 格式（Qdrant ID 要求）。
        使用确定性 hash 保证同一字符串始终映射到同一 UUID。

        Args:
            s: 原始字符串 ID

        Returns:
            str: UUID 格式字符串
        """
        import hashlib
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
