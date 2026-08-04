# 2_vector_store - 向量存储层 (Layer 2)
# 职责：统一抽象接口 → PGvector(生产) / Chroma(开发 fallback) 双库自动适配
# 依赖：config + utils + Layer 1 (embedder)

from .base import BaseVectorStore
from .factory import VectorStoreFactory, get_vector_store

__all__ = ["BaseVectorStore", "VectorStoreFactory", "get_vector_store"]
