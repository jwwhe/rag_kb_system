"""
================================================================================
  API 接口层: Pydantic 请求/响应模型
  所有 API 接口的输入输出 Schema 定义
================================================================================
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ==================== 请求模型 ====================

class QuestionRequest(BaseModel):
    """智能问答请求"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户问题",
        example="这份合同的有效期是多久？",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="多轮对话会话 ID，相同 ID 会自动拼接历史消息用于指代消解",
        example="a1b2c3d4",
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="指定知识库（Collection），不指定则使用默认库",
    )


class FileDeleteRequest(BaseModel):
    """删除文件请求"""
    file_name: str = Field(
        ...,
        min_length=1,
        description="要删除的文件名（含扩展名）",
        example="合同模板.pdf",
    )


# ==================== 响应模型 ====================

class CitationItem(BaseModel):
    """引用条目"""
    citation_id: int = Field(..., description="引用编号")
    file_name: str = Field(..., description="源文件名")
    page: Any = Field(..., description="页码")
    original_text: str = Field(..., description="原文片段")
    similarity: float = Field(..., description="相似度分数")
    source_type: str = Field(..., description="来源类型: internal/external")


class SourceStats(BaseModel):
    """来源统计"""
    internal_sources: int = Field(0, description="本地知识库引用数")
    external_sources: int = Field(0, description="外部网络引用数")
    total_sources: int = Field(0, description="总引用数")


class QAResponse(BaseModel):
    """智能问答响应"""
    answer: str = Field(..., description="答案文本")
    citations: List[CitationItem] = Field(default_factory=list, description="引用列表")
    citation_text: str = Field("", description="格式化的溯源文本")
    source_type: str = Field("internal", description="答案来源: internal/external")
    source_stats: SourceStats = Field(default_factory=SourceStats, description="来源统计")


class FileUploadResponse(BaseModel):
    """文件上传响应"""
    file_name: str = Field(..., description="文件名")
    pages: int = Field(..., description="PDF 页数")
    chunks: int = Field(..., description="分块数量")
    message: str = Field("上传成功", description="状态描述")


class KBStatsResponse(BaseModel):
    """知识库统计响应"""
    store_type: str = Field(..., description="向量库类型: chroma/qdrant")
    collection_name: str = Field(..., description="集合名称")
    total_chunks: int = Field(0, description="总块数")
    total_files: int = Field(0, description="总文件数")
    file_names: List[str] = Field(default_factory=list, description="文件列表")


class DeleteResponse(BaseModel):
    """删除操作响应"""
    deleted_count: int = Field(0, description="删除数量")
    file_name: str = Field("", description="文件名")
    message: str = Field("操作成功", description="状态描述")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field("healthy", description="系统状态")
    version: str = Field(..., description="系统版本")
    env: str = Field(..., description="运行环境")
    vector_store: str = Field(..., description="向量库类型")
    llm: str = Field(..., description="当前 LLM")
