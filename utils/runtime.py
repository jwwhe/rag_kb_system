"""
================================================================================
  运行时工具：CPU 推理线程编排
  背景：PyTorch 默认按物理核数开线程（本仓库实测 6），
        bge-m3 嵌入与 bge-reranker 精排都是 CPU 密集型，
        显式开到逻辑核数可提速约 20~30%。
================================================================================
"""
import os
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_applied = False


def resolve_cpu_threads(configured: Optional[int] = None) -> int:
    """
    计算实际使用的线程数。

    Args:
        configured: 配置值；0 或 None 表示自动

    Returns:
        int: 线程数（至少 1，至多 16）
    """
    if configured is None:
        try:
            from config.settings import get_settings_cached

            configured = get_settings_cached().cpu_threads
        except Exception:
            configured = 0

    if configured and configured > 0:
        return max(1, int(configured))

    logical = os.cpu_count() or 4
    return max(1, min(logical, 16))


def apply_cpu_threads(configured: Optional[int] = None) -> int:
    """
    把 torch / OMP 线程数调到配置值。幂等，只在首次调用时生效。

    Args:
        configured: 线程数，0/None 表示自动

    Returns:
        int: 生效的线程数
    """
    global _applied
    threads = resolve_cpu_threads(configured)

    if _applied:
        return threads

    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, threads // 4))
        except Exception:
            pass  # 已启动并行推理时该调用会报错，忽略
    except ImportError:
        pass

    _applied = True
    logger.info(f"CPU 推理线程数已设置: {threads}")
    return threads
