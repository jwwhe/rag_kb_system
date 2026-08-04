"""
================================================================================
  统一异常体系
  每个架构层对应专用异常类，便于精确定位问题层级
================================================================================
"""

class RAGSystemException(Exception):
    """RAG 系统基础异常（所有自定义异常的基类）"""
    def __init__(self, message: str = "RAG系统异常", layer: str = "unknown"):
        self.layer = layer
        super().__init__(f"[{layer}] {message}")


class DocumentProcessError(RAGSystemException):
    """Layer 1 - 文档处理层异常"""
    def __init__(self, message: str = "文档处理失败"):
        super().__init__(message, layer="1_doc_process")


class VectorStoreError(RAGSystemException):
    """Layer 2 - 向量存储层异常"""
    def __init__(self, message: str = "向量存储操作失败"):
        super().__init__(message, layer="2_vector_store")


class RetrievalError(RAGSystemException):
    """Layer 3 - 检索层异常"""
    def __init__(self, message: str = "检索失败"):
        super().__init__(message, layer="3_retrieval")


class GenerationError(RAGSystemException):
    """Layer 4 - 生成层异常"""
    def __init__(self, message: str = "LLM生成失败"):
        super().__init__(message, layer="4_generate")


class EnhanceError(RAGSystemException):
    """Layer 5 - 增强层异常"""
    def __init__(self, message: str = "增强处理失败"):
        super().__init__(message, layer="5_enhance")


class NoKnowledgeFoundError(RetrievalError):
    """知识库中未找到相关内容（触发兜底策略）"""
    def __init__(self, message: str = "知识库中暂无该相关资料"):
        super().__init__(message)
