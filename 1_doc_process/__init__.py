# 1_doc_process - 文档处理层 (Layer 1)
# 职责：多格式加载(PDF/Word/MD) → 文本清洗 → 文本分割 → 向量化 → 元数据绑定（含多源标注）
# 依赖：仅依赖 config + utils

from .loader import DocumentLoader, PDFLoader  # PDFLoader 为兼容别名
from .cleaner import TextCleaner, CleanStats
from .splitter import TextSplitter
from .embedder import Embedder

__all__ = [
    "DocumentLoader",
    "PDFLoader",
    "TextCleaner",
    "CleanStats",
    "TextSplitter",
    "Embedder",
]
