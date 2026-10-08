"""
================================================================================
  来源类型（source_type）标注工具
  功能：
    - 校验上传时指定的 source_type 是否属于配置的三类来源
    - 未指定时按文件名关键词推断，避免整批被套成同一个标签
  使用方：api/routes.py 上传接口 + app_streamlit.py 上传表单 + Layer 1 元数据
  背景：
    source_type 是"论文原文 / 综述解读 / 实验笔记"多源融合与溯源的唯一依据，
    一旦标错，MultiSourceFusioner 的配额与引用卡片都会失真。
  说明：放在 utils 而非 1_doc_process，是为了让前端不引入整个文档处理层
================================================================================
"""
from typing import List, Optional

from config.settings import get_settings_cached
from utils.logger import get_logger

logger = get_logger(__name__)

# 文件名关键词 → 来源类型（按声明顺序匹配，先命中者胜出）
_INFER_RULES = [
    (("综述", "解读", "review", "survey", "笔记解读"), "综述解读"),
    (("实验", "复现", "笔记", "note", "log", "debug", "踩坑"), "实验笔记"),
    (("论文", "原文", "paper", "article", "arxiv", "need"), "论文原文"),
]


def allowed_source_types() -> List[str]:
    """返回配置中登记的来源类型列表。"""
    return list(get_settings_cached().doc_process.source_type_labels)


def normalize_source_type(
    value: Optional[str], file_name: str = ""
) -> str:
    """
    校验并归一化单个文件的来源类型。

    - 值为空或不在配置列表内时，回退到文件名推断，再回退到默认类型
    - 大小写/空格差异容错（" 实验笔记 " → "实验笔记"）

    Args:
        value:     调用方传入的来源类型（可为空）
        file_name: 文件名，用于推断

    Returns:
        str: 合法的来源类型
    """
    labels = allowed_source_types()
    settings = get_settings_cached().doc_process

    cleaned = (value or "").strip()
    if cleaned in labels:
        return cleaned
    if cleaned:
        logger.warning(
            f"来源类型 '{cleaned}' 不在配置列表 {labels} 中，将自动推断: {file_name}"
        )

    if settings.infer_source_type and file_name:
        inferred = infer_source_type(file_name)
        if inferred:
            logger.info(f"按文件名推断来源类型: {file_name} → {inferred}")
            return inferred

    return settings.default_source_type


def infer_source_type(file_name: str) -> Optional[str]:
    """
    按文件名关键词推断来源类型。

    Args:
        file_name: 文件名（含扩展名）

    Returns:
        Optional[str]: 命中的来源类型，未命中返回 None
    """
    labels = allowed_source_types()
    lowered = file_name.lower()
    for keywords, label in _INFER_RULES:
        if label not in labels:
            continue
        if any(kw in lowered for kw in keywords):
            return label
    return None


def suggest_source_type(file_name: str) -> str:
    """
    给出文件名的推荐来源类型，推断不出时回退到配置的默认值。

    与 normalize_source_type 的区别：本函数供前端预填下拉框使用，
    不记录"非法取值"告警。
    """
    settings = get_settings_cached().doc_process
    if settings.infer_source_type:
        inferred = infer_source_type(file_name)
        if inferred:
            return inferred
    return settings.default_source_type


def resolve_source_types(
    file_names: List[str],
    batch_source_type: Optional[str] = None,
    per_file_source_types: Optional[List[str]] = None,
) -> List[str]:
    """
    为一批文件解析逐文件来源类型。

    优先级：per_file_source_types（与文件一一对应） > batch_source_type（整批同值）
    > 文件名推断 > 默认类型。

    Args:
        file_names:              文件名列表
        batch_source_type:       整批共用的来源类型（向后兼容旧客户端）
        per_file_source_types:   与 file_names 等长的来源类型列表

    Returns:
        List[str]: 每个文件对应的合法来源类型
    """
    if per_file_source_types is not None and len(per_file_source_types) != len(file_names):
        raise ValueError(
            f"source_type_list 长度({len(per_file_source_types)}) "
            f"与文件数量({len(file_names)})不一致"
        )

    resolved = []
    for idx, name in enumerate(file_names):
        value = (
            per_file_source_types[idx]
            if per_file_source_types is not None
            else batch_source_type
        )
        resolved.append(normalize_source_type(value, name))
    return resolved
