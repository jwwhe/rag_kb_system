# api - API 接口层
# 职责：FastAPI 应用 → 路由 → 请求/响应模型 → 依赖注入
# 依赖：所有五层架构模块

from .main import app

__all__ = ["app"]
