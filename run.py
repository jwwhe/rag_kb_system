"""
================================================================================
  RAG 知识库问答系统 - 启动入口
  使用方式：
    python run.py                  # 默认启动（Chroma + 调试日志）
    ENV=production python run.py   # 生产环境启动（精简日志，STORE_TYPE 可切 pgvector）
================================================================================
"""
import sys
import os

# ============================================================================
# 【关键】在所有 import 之前设置 HuggingFace 镜像
# 必须在此处设置，因为 huggingface_hub 在首次 import 时就会读取端点配置
# 如果 import 之后才设置 os.environ，库已缓存旧地址，设置不会生效
# ============================================================================
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = _HF_MIRROR
# 兼容旧版 huggingface_hub 的变量名
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = _HF_MIRROR
# 禁用遥测，减少不必要的网络请求
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DISABLE_TELEMETRY"] = "YES"

print(f"[启动] HuggingFace 镜像: {_HF_MIRROR}")

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import uvicorn
from config.settings import get_settings


def main():
    """系统启动入口"""
    settings = get_settings()

    print("=" * 60)
    print(f"  {settings.api.title} v{settings.api.version}")
    print(f"  环境: {settings.env}")
    print(f"  向量库: {settings.vector_store.store_type}")
    print(f"  LLM: {settings.llm.active_llm}")
    print(f"  监听: {settings.api.host}:{settings.api.port}")
    print("=" * 60)

    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=settings.api.api_timeout,
    )


if __name__ == "__main__":
    main()
