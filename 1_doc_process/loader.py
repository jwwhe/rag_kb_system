"""
================================================================================
  Layer 1 - 文档处理层: 多格式文档加载器
  功能：
    - 支持 PDF / Word(.docx) / Markdown(.md/.markdown) 三种格式
    - 自动按扩展名分发到对应 LangChain Loader
    - 自动过滤空白行、页眉页脚、无效冗余内容
    - 支持多源类型标注（论文原文 / 综述解读 / 实验笔记）用于多源知识融合
    - 返回结构化的文档对象列表
================================================================================
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import DocProcessConfig, get_settings_cached
from utils.exceptions import DocumentProcessError
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """
    多格式文档加载器。
    根据 file 扩展名自动分发到对应 LangChain Loader，
    并统一执行文本清洗 + 元数据绑定。
    """

    # 支持的扩展名 → 加载器类型映射
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown"}

    # 多源类型白名单（用于多源知识融合）
    VALID_SOURCE_TYPES = {"论文原文", "综述解读", "实验笔记"}

    def __init__(self, config: Optional[DocProcessConfig] = None):
        """
        初始化文档加载器。

        Args:
            config: 文档处理配置（默认从全局配置读取）
        """
        self.config = config or get_settings_cached().doc_process
        logger.info(
            f"DocumentLoader 初始化完成 | "
            f"支持格式: {sorted(self.SUPPORTED_EXTENSIONS)} | "
            f"最短有效行: {self.config.min_line_length}"
        )

    def load_single(
        self,
        file_path: str,
        source_type: str = "论文原文",
        knowledge_base: str = "default",
    ) -> List[Document]:
        """
        加载单个文件并提取文本。

        Args:
            file_path:      文件绝对路径
            source_type:    来源类型（论文原文/综述解读/实验笔记），用于多源融合
            knowledge_base: 所属知识库标识（用于多库隔离）

        Returns:
            List[Document]: 按页/段分割的 LangChain Document 列表

        Raises:
            DocumentProcessError: 文件不存在、格式不支持或解析失败
        """
        if not os.path.isfile(file_path):
            raise DocumentProcessError(f"文件不存在: {file_path}")

        if source_type not in self.VALID_SOURCE_TYPES:
            raise DocumentProcessError(
                f"不支持的 source_type: {source_type}，可选: {self.VALID_SOURCE_TYPES}"
            )

        ext = Path(file_path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise DocumentProcessError(
                f"仅支持 {sorted(self.SUPPORTED_EXTENSIONS)} 格式文件: {file_path}"
            )

        # 按扩展名分发到对应 Loader
        try:
            loader = self._get_langchain_loader(file_path, ext)
            documents = loader.load()
            logger.info(
                f"文档加载成功: {file_path} | 格式: {ext} | 共 {len(documents)} 个块/页"
            )
        except Exception as e:
            raise DocumentProcessError(
                f"文档解析失败 [{file_path}]: {type(e).__name__}: {str(e)}"
            ) from e

        # 过滤无效内容 + 绑定多源元数据
        cleaned_docs = self._clean_documents(
            documents, file_path, source_type, knowledge_base
        )
        logger.info(
            f"文本清洗完成: {file_path} | "
            f"清洗前 {len(documents)} → 清洗后 {len(cleaned_docs)}"
        )
        return cleaned_docs

    def load_batch(
        self,
        file_paths: List[str],
        source_type: str = "论文原文",
        knowledge_base: str = "default",
    ) -> List[Document]:
        """
        批量加载文件（同一来源类型）。

        Args:
            file_paths:     文件路径列表
            source_type:    来源类型
            knowledge_base: 所属知识库

        Returns:
            List[Document]: 所有文档的合并列表
        """
        all_documents = []
        failed_files = []

        for file_path in file_paths:
            try:
                docs = self.load_single(file_path, source_type, knowledge_base)
                all_documents.extend(docs)
            except DocumentProcessError as e:
                logger.warning(f"跳过失败文件: {e}")
                failed_files.append(file_path)

        if failed_files:
            logger.warning(
                f"批量加载完成: 成功 {len(all_documents)} 段 | "
                f"失败 {len(failed_files)} 个文件: {failed_files}"
            )

        if not all_documents:
            raise DocumentProcessError("所有文件加载均失败")

        return all_documents

    # ==================== 私有方法 ====================

    def _get_langchain_loader(self, file_path: str, ext: str):
        """
        根据扩展名返回对应的 LangChain Loader 实例。
        延迟导入，避免未安装可选依赖时启动报错。

        Args:
            file_path: 文件路径
            ext:       扩展名（小写）

        Returns:
            LangChain Loader 实例
        """
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(file_path)

        elif ext == ".docx":
            try:
                from langchain_community.document_loaders import Docx2txtLoader
            except ImportError as e:
                raise DocumentProcessError(
                    "加载 Word 文档需要 python-docx，请执行: pip install python-docx"
                ) from e
            return Docx2txtLoader(file_path)

        elif ext in (".md", ".markdown"):
            # 优先用 UnstructuredMarkdownLoader（需 unstructured 包）
            # 未安装则 fallback 到 TextLoader（纯文本读取，零依赖）
            try:
                from langchain_community.document_loaders import UnstructuredMarkdownLoader
                return UnstructuredMarkdownLoader(file_path)
            except (ImportError, ModuleNotFoundError):
                from langchain_community.document_loaders import TextLoader
                logger.warning(
                    "unstructured 包未安装，Markdown 将按纯文本加载。"
                    "如需结构化解析请执行: pip install unstructured"
                )
                return TextLoader(file_path, encoding="utf-8")

        else:
            raise DocumentProcessError(f"未实现的加载器: {ext}")

    def _clean_documents(
        self,
        documents: List[Document],
        file_path: str,
        source_type: str,
        knowledge_base: str,
    ) -> List[Document]:
        """
        清洗文档内容：过滤空白行、无效短行、页眉页脚冗余。
        同时绑定多源元数据（source_type / knowledge_base）。

        Args:
            documents:      原始文档列表
            file_path:      源文件路径
            source_type:    来源类型
            knowledge_base: 知识库标识

        Returns:
            List[Document]: 清洗后的文档列表
        """
        file_name = os.path.basename(file_path)
        cleaned = []

        for doc in documents:
            page_num = doc.metadata.get("page", 0)
            raw_text = doc.page_content or ""

            # 按行过滤
            lines = raw_text.split("\n")
            valid_lines = []

            for line in lines:
                stripped = line.strip()
                if self.config.filter_blank_lines and not stripped:
                    continue
                if len(stripped) < self.config.min_line_length:
                    continue
                valid_lines.append(stripped)

            cleaned_text = "\n".join(valid_lines)

            if not cleaned_text.strip():
                continue

            # 绑定完整元数据（含多源融合字段）
            doc.page_content = cleaned_text
            doc.metadata.update({
                "file_name": file_name,
                "file_path": file_path,
                "page": page_num,
                "doc_id": str(uuid.uuid4())[:8],
                "source_type": source_type,        # 多源融合：论文/综述/笔记
                "knowledge_base": knowledge_base,  # 多知识库隔离
                "extension": Path(file_path).suffix.lower(),
            })
            cleaned.append(doc)

        return cleaned


# ==================== 兼容性保留 ====================
# 旧代码可能仍以 PDFLoader 名字导入，这里做向后兼容别名
PDFLoader = DocumentLoader
