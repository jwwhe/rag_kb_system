"""
================================================================================
  Layer 2 - 向量存储层: 抽象基类
  定义统一接口，Chroma / Qdrant 必须实现全部方法：
    - add_documents:      入库
    - delete_by_file:     删除单文件所有向量
    - clear_collection:   清空知识库
    - search:             向量相似度检索
    - search_with_filter: 带元数据过滤的检索
    - get_all_documents:  获取全部文档（供 BM25 索引构建）
================================================================================
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """
    向量存储抽象基类。
    所有向量库实现（Chroma / Qdrant）必须继承此类并实现全部方法。
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> bool:
        """
        将带嵌入向量的文档批量入库。

        Args:
            documents: 携带 embedding 的 Document 列表

        Returns:
            bool: 入库是否成功
        """
        ...

    @abstractmethod
    def delete_by_file(self, file_name: str) -> int:
        """
        按文件名删除该文件的所有向量块。

        Args:
            file_name: 源文件名

        Returns:
            int: 删除的块数量
        """
        ...

    @abstractmethod
    def clear_collection(self) -> bool:
        """
        清空当前知识库所有数据。

        Returns:
            bool: 清空是否成功
        """
        ...

    @abstractmethod
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
            List[Document]: 相似文档列表（含相似度分数）
        """
        ...

    @abstractmethod
    def search_with_filter(
        self,
        query_embedding: List[float],
        metadata_filter: Dict[str, Any],
        top_k: int = 8,
    ) -> List[Document]:
        """
        带元数据过滤的向量检索（用于知识库隔离）。

        Args:
            query_embedding:  查询向量
            metadata_filter:  元数据过滤条件
            top_k:            返回数量

        Returns:
            List[Document]: 过滤后的相似文档列表
        """
        ...

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        """
        获取知识库中全部文档（供 BM25 索引构建和元数据浏览）。

        Returns:
            List[Document]: 所有文档列表
        """
        ...

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息。

        Returns:
            Dict: 包含文档数、文件名列表等统计信息
        """
        ...
