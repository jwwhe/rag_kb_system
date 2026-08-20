"""
================================================================================
  API 接口层: FastAPI 应用主入口
  功能：
    - 应用初始化、CORS 配置
    - 全局异常捕获
    - 生命周期管理（启动时预热模型）
================================================================================
"""
import sys
import os

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config.settings import get_settings_cached
from utils.response import error_response
from utils.exceptions import RAGSystemException
from utils.logger import get_logger

logger = get_logger(__name__)

# 获取配置
settings = get_settings_cached()

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    description="基于五层 RAG 架构的私有知识库智能问答系统",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置
# 注意：allow_origins=["*"] 时不能开启 allow_credentials（违反 CORS 规范，
# 浏览器会拒绝带凭证的跨域请求）；本 API 为无凭证公开接口，不启用凭证。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全局异常处理器 ====================

@app.exception_handler(RAGSystemException)
async def rag_exception_handler(request: Request, exc: RAGSystemException):
    """RAG 系统异常统一处理"""
    logger.error(f"RAG 异常 [{exc.layer}]: {exc}")
    return JSONResponse(
        status_code=500,
        content=error_response(
            code=500,
            message=str(exc),
        ).to_dict(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局未捕获异常处理"""
    logger.error(f"未捕获异常: {type(exc).__name__}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=error_response(
            code=500,
            message=f"服务器内部错误: {str(exc)}",
        ).to_dict(),
    )


# ==================== 生命周期事件 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("=" * 50)
    logger.info(f"  {settings.api.title} v{settings.api.version} 启动中...")
    logger.info(f"  环境: {settings.env}")
    logger.info(f"  向量库: {settings.vector_store.store_type}")
    logger.info(f"  LLM: {settings.llm.active_llm}")
    logger.info("=" * 50)

    # 预加载嵌入模型（首次启动会下载 bge-m3 ~2GB，请耐心等待）
    try:
        import importlib
        deps = importlib.import_module("api.dependencies")
        logger.info("正在预加载嵌入模型 bge-m3（首次需下载，约 2GB）...")
        deps.get_embedder()
        logger.info("嵌入模型加载完成")
    except Exception as e:
        logger.warning(f"嵌入模型预热失败（将在首次请求时自动加载）: {e}")

    # 预加载重排模型（首次启动会下载 bge-reranker ~1GB）
    try:
        logger.info("正在预加载重排模型 bge-reranker（首次需下载，约 1GB）...")
        deps = importlib.import_module("api.dependencies")
        deps.get_reranker()
        logger.info("重排模型加载完成")
    except Exception as e:
        logger.warning(f"重排模型预热失败（将在首次请求时自动加载）: {e}")

    # 预加载 BM25 索引（如果向量库中已有文档）
    try:
        deps = importlib.import_module("api.dependencies")
        deps.ensure_bm25_index()
    except Exception as e:
        logger.warning(f"BM25 索引预热失败（首次上传 PDF 后会自动构建）: {e}")

    logger.info("=" * 50)
    logger.info(f"  系统就绪，访问 http://localhost:{settings.api.port}/")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    logger.info(f"{settings.api.title} 正在关闭...")


# ==================== 注册路由 ====================

from api.routes import router
app.include_router(router, prefix="/api/v1")


# ==================== 健康检查 ====================

@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": settings.api.version,
        "env": settings.env,
        "vector_store": settings.vector_store.store_type,
        "llm": settings.llm.active_llm,
    }


# ==================== 前端入口 ====================

@app.get("/", tags=["页面"], include_in_schema=False)
async def root():
    """API 根路径 — 前端请使用 Streamlit (端口 8501)"""
    return JSONResponse(content={
        "message": "RAG 知识库问答系统 API",
        "version": settings.api.version,
        "docs": "/docs",
        "frontend": f"http://localhost:8501",
        "health": "/health",
    })
