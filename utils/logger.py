"""
================================================================================
  统一日志工具
  开发环境输出详细 DEBUG 日志，生产环境仅输出 WARNING+
================================================================================
"""
import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: Optional[int] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    获取标准化的 logger 实例。

    Args:
        name:       日志名称（通常使用 __name__）
        level:      日志级别（默认 INFO）
        log_format: 自定义格式

    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    if level is None:
        level = logging.INFO

    if log_format is None:
        log_format = (
            "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
