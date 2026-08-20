"""
===============================================================================
  Layer 2 - 向量存储层: 工厂函数
  根据配置自动切换 PGvector（生产）/ Chroma（开发），业务代码无需感知差异。
===============================================================================
"""
from typing import Optional

from config.settings import Settings, get_settings_cached
from .base import BaseVectorStore
from .chroma_store import ChromaVectorStore
from .pgvector_store import PGVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreFactory:
    """
    向量存储工厂。
    根据配置中的 store_type 自动创建对应实例。
    """

    @staticmethod
    def create(settings: Optional[Settings] = None) -> BaseVectorStore:
        """
        根据配置创建向量存储实例。

        Args:
            settings: 全局配置（默认从全局配置读取）

        Returns:
            BaseVectorStore: PGvector / Chroma 实例

        Raises:
            ValueError: 不支持的 store_type
        """
        if settings is None:
            settings = get_settings_cached()

        store_type = settings.vector_store.store_type

        if store_type == "pgvector":
            logger.info("创建 PGvector 向量存储（生产环境）")
            return PGVectorStore()

        elif store_type == "chroma":
            logger.info("创建 Chroma 向量存储（开发环境 fallback）")
            return ChromaVectorStore()

        else:
            raise ValueError(
                f"不支持的向量库类型: {store_type}，可选值: pgvector / chroma"
            )


# ========== 全局单例 ==========

_vector_store_instance: Optional[BaseVectorStore] = None


def get_vector_store(settings: Optional[Settings] = None) -> BaseVectorStore:
    """
    获取向量存储全局单例。
    首次调用时根据配置自动创建，后续调用返回同一实例。

    Args:
        settings: 全局配置（仅首次创建时生效）

    Returns:
        BaseVectorStore: 向量存储实例
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreFactory.create(settings)
    return _vector_store_instance
