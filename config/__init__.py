# config - 统一配置中心
# 所有系统参数从此处读取，禁止任何模块硬编码参数

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
