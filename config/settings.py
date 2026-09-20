"""
================================================================================
  RAG 知识库问答系统 - 统一配置中心
  所有参数必须从此文件读取，严禁任何模块硬编码配置值
  环境切换：设置环境变量 ENV=production 切换为生产配置
================================================================================
"""
import os
from typing import Literal, Optional
from dataclasses import dataclass, field


@dataclass
class DocProcessConfig:
    """Layer 1 - 文档处理层配置"""
    # 分块参数（固定值，不可修改）
    chunk_size: int = 800
    chunk_overlap: int = 150

    # 零宽断言分块（简历核心亮点）
    # keep_separator=False 修复 LangChain 默认陷阱：
    #   默认 True 会把分隔符放到下一块开头，导致句号出现在块首而非块尾
    #   改为 False + 零宽断言 (?<=。) 让句号正确留在上一块末尾
    keep_separator: bool = False
    # 是否启用零宽断言分隔符（关闭则回退到普通分隔符）
    use_zero_width_separator: bool = True

    # PDF 加载
    pdf_loader: str = "PyPDFLoader"  # 指定 PDF 加载器

    # OCR 识别（PDF 含图片 / 扫描件时启用）
    # 引擎：RapidOCR（PaddleOCR 的 ONNX 版，中英文开箱即用，pip 安装 rapidocr-onnxruntime）
    enable_ocr: bool = False  # 总开关（也可通过环境变量 OCR_ENABLE=1 开启）
    ocr_engine: str = "rapidocr"  # "rapidocr" | "none"
    ocr_min_chars_per_page: int = 50  # 单页文本低于此字符数视为扫描页，整页 OCR
    ocr_embedded_images: bool = True  # 是否识别文本页中的内嵌图片（图表/截图）
    ocr_confidence_threshold: float = 0.5  # 低于此置信度的识别结果丢弃
    ocr_render_dpi: int = 200  # 扫描页渲染分辨率（越高越清晰，越慢）

    # 文本清洗：过滤空白行、页眉页脚、无效冗余
    enable_cleaning: bool = True  # 清洗总开关（关闭则原文透传，供 A/B 对比评估）
    filter_blank_lines: bool = True
    filter_header_footer: bool = True  # 跨页首尾重复行统计识别（页眉页脚）
    min_line_length: int = 10  # 最短有效行长度（仅约束正文散行）
    clean_unicode_normalize: bool = True  # NFKC 归一 + 零宽/控制字符清理
    clean_fix_hyphenation: bool = True  # PDF 行尾连字符断词拼接
    clean_strip_page_artifacts: bool = True  # 页码/装饰线/乱码水印行剔除
    clean_header_footer_min_pages: int = 3  # 重复行判定为页眉页脚的最少跨页数
    clean_garbage_line_ratio: float = 0.6  # 非文字字符占比超过此值判为乱码行

    # 嵌入模型
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"  # "cpu" | "cuda"
    embedding_batch_size: int = 32  # 批量向量化大小
    embedding_normalize: bool = True  # 向量归一化

    # HuggingFace 镜像（国内用户建议设置）
    hf_endpoint: str = "https://hf-mirror.com"  # HuggingFace 镜像地址
    hf_home: str = ""  # 模型缓存目录（空=默认）

    # 失败重试
    embedding_max_retries: int = 5  # 最大重试次数
    embedding_retry_delay: float = 2.0  # 重试间隔（秒）


@dataclass
class VectorStoreConfig:
    """Layer 2 - 向量存储层配置"""
    # 向量库类型：chroma（默认，本地持久化） / pgvector（可选，需外部 PostgreSQL）
    store_type: Literal["pgvector", "chroma"] = "chroma"

    # PGvector 配置（可选，需要外部 PostgreSQL + pgvector 扩展）
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "rag_kb"
    pg_user: str = "rag"
    pg_password: str = "rag123"
    pg_table_name: str = "documents"  # 向量表名
    pg_vector_size: int = 1024  # bge-m3 输出维度 = 1024

    # Chroma 配置（默认向量库，本地持久化）
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "rag_kb_collection"

    # 多知识库隔离：不同知识库使用不同 Collection / 不同 source_type
    enable_multi_collection: bool = True


@dataclass
class RetrievalConfig:
    """Layer 3 - 检索层配置"""
    # 查询改写
    enable_query_rewrite: bool = True
    max_sub_questions: int = 3  # 最多生成同义子问题数量

    # 向量相似度检索
    vector_top_k: int = 8  # 向量检索最终召回数（MMR 后）
    similarity_threshold: float = 0.35  # 最低相似度阈值（低于此值过滤）

    # MMR 多样性重排（两阶段检索第一阶段）
    # 简历亮点：MMR + BGE-Reranker 两阶段精排，Top-1 命中率 45% → 78%
    enable_mmr: bool = True
    mmr_fetch_k: int = 30  # MMR 候选拉取数（从向量库先召回 30 条）
    mmr_lambda: float = 0.7  # MMR 多样性权衡：1.0=纯相关性，0.0=纯多样性

    # BM25 关键词检索
    bm25_top_k: int = 8

    # 混合检索融合
    vector_weight: float = 0.6  # 向量检索权重
    bm25_weight: float = 0.4   # BM25 检索权重

    # Rerank 重排
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 3  # 重排后保留最优 Top3
    rerank_device: str = "cpu"
    rerank_hf_endpoint: str = "https://hf-mirror.com"  # Rerank 模型镜像（国内）


@dataclass
class LLMConfig:
    """Layer 4 - 生成层 LLM 配置"""
    # 当前使用的 LLM：deepseek / qwen / ollama
    active_llm: Literal["deepseek", "qwen", "ollama"] = "deepseek"

    # DeepSeek 配置
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # Qwen (通义千问) 配置
    qwen_api_key: str = ""
    qwen_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    # Ollama 本地模型配置
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # 通用 LLM 参数
    temperature: float = 0.1  # 低温度，减少幻觉
    max_tokens: int = 2048
    request_timeout: int = 60  # 请求超时（秒）
    max_retries: int = 2


@dataclass
class EnhanceConfig:
    """Layer 5 - 增强层配置"""
    # 纠正型 RAG (CRAG)
    enable_corrective_rag: bool = True
    max_correction_iterations: int = 2  # 最多纠错迭代次数

    # 全文溯源
    enable_citation: bool = True
    citation_min_similarity: float = 0.3  # 溯源最低相似度

    # Web 搜索兜底
    enable_web_fallback: bool = True
    web_search_max_results: int = 5
    web_search_timeout: int = 30


@dataclass
class APIConfig:
    """API 接口配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "RAG 知识库问答系统"
    version: str = "1.0.0"

    # 文件上传
    max_upload_size_mb: int = 50
    allowed_extensions: list = field(default_factory=lambda: ["pdf", "docx", "md", "markdown"])

    # 请求超时
    api_timeout: int = 120


@dataclass
class Settings:
    """全局配置聚合类"""
    doc_process: DocProcessConfig = field(default_factory=DocProcessConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # 全局环境标识
    env: Literal["development", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"


# ========== 配置工厂函数 ==========

def get_settings() -> Settings:
    """
    获取全局配置实例。
    根据环境变量 ENV 自动切换开发/生产配置。
    生产环境可通过环境变量覆盖 LLM API Key 等敏感参数。

    Returns:
        Settings: 全局配置实例
    """
    settings = Settings()

    # 环境切换
    env = os.getenv("ENV", "development").lower()
    settings.env = "production" if env == "production" else "development"

    if settings.env == "production":
        settings.debug = False
        settings.log_level = "WARNING"
        settings.vector_store.store_type = os.getenv("STORE_TYPE", "chroma")

        # 生产环境：从环境变量读取敏感配置
        settings.llm.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        settings.llm.qwen_api_key = os.getenv("QWEN_API_KEY", "")
        settings.llm.ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        # PGvector 连接参数
        settings.vector_store.pg_host = os.getenv("PG_HOST", "pgvector")
        settings.vector_store.pg_port = int(os.getenv("PG_PORT", "5432"))
        settings.vector_store.pg_database = os.getenv("PG_DATABASE", "rag_kb")
        settings.vector_store.pg_user = os.getenv("PG_USER", "rag")
        settings.vector_store.pg_password = os.getenv("PG_PASSWORD", "rag123")

    else:
        # 开发环境：从环境变量读取（可选覆盖）
        settings.llm.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        settings.llm.qwen_api_key = os.getenv("QWEN_API_KEY", "")
        # 开发环境默认用 Chroma 本地向量库；可通过 STORE_TYPE=pgvector 切换
        settings.vector_store.store_type = os.getenv("STORE_TYPE", "chroma")
        # 允许环境变量覆盖 PG 连接
        settings.vector_store.pg_host = os.getenv("PG_HOST", "localhost")
        settings.vector_store.pg_port = int(os.getenv("PG_PORT", "5432"))
        settings.vector_store.pg_database = os.getenv("PG_DATABASE", "rag_kb")
        settings.vector_store.pg_user = os.getenv("PG_USER", "rag")
        settings.vector_store.pg_password = os.getenv("PG_PASSWORD", "rag123")

    # OCR 开关（两个环境共用）
    settings.doc_process.enable_ocr = os.getenv("OCR_ENABLE", "0").lower() in ("1", "true", "yes")

    # Chroma 持久化目录（两个环境共用）
    settings.vector_store.chroma_persist_dir = os.getenv(
        "CHROMA_PERSIST_DIR", settings.vector_store.chroma_persist_dir
    )

    return settings


# 全局单例（惰性初始化）
_settings_instance: Optional[Settings] = None


def get_settings_cached() -> Settings:
    """获取全局配置缓存单例"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = get_settings()
    return _settings_instance
