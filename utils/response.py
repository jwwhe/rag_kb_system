"""
================================================================================
  统一 API 响应体
  所有 API 接口必须使用此模块构建返回体，禁止自定义返回格式
================================================================================
"""
from typing import Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class APIResponse:
    """统一 API 响应结构"""
    code: int = 200           # 业务状态码
    message: str = "success"   # 状态描述
    data: Any = None           # 响应数据
    timestamp: float = field(default_factory=time.time)  # 响应时间戳

    def to_dict(self) -> dict:
        """
        转换为字典格式（FastAPI 兼容）。

        Returns:
            dict: 可被 FastAPI 自动序列化的字典
        """
        return {
            "code": self.code,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }


def success_response(data: Any = None, message: str = "success") -> APIResponse:
    """
    构建成功响应。

    Args:
        data:    响应数据
        message: 状态描述

    Returns:
        APIResponse: 成功响应对象
    """
    return APIResponse(code=200, message=message, data=data)


def error_response(
    code: int = 500,
    message: str = "服务器内部错误",
    data: Any = None,
) -> APIResponse:
    """
    构建错误响应。

    Args:
        code:    业务错误码
        message: 错误描述
        data:    附加错误信息

    Returns:
        APIResponse: 错误响应对象
    """
    return APIResponse(code=code, message=message, data=data)
