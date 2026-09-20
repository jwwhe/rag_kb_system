"""
================================================================================
  Layer 1 - 文档处理层: 多格式文档加载器
  功能：
    - 支持 PDF / Word(.docx) / Markdown(.md/.markdown) 三种格式
    - 自动按扩展名分发到对应 LangChain Loader
    - 结构化文本清洗（TextCleaner）：字符归一化、断词拼接、页码/页眉页脚/乱码过滤
    - 保留 Markdown 结构与代码块，避免短行误删标题与围栏
    - PDF 图片/扫描件支持 OCR 识别（RapidOCR，需启用 enable_ocr 配置）
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

from .cleaner import TextCleaner

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
        self._cleaner = TextCleaner(self.config)
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

            # OCR：PDF 含图片/扫描件时识别文本
            # 必须在清洗前执行——扫描页提取出的文本为空，会被 _clean_documents 过滤掉
            if ext == ".pdf" and self.config.enable_ocr:
                documents = self._apply_ocr(documents, file_path)

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
            # TextLoader 纯文本加载（零依赖，兼容性好）
            # 如需结构化 MD 解析可换 UnstructuredMarkdownLoader（需 nltk punkt_tab）
            from langchain_community.document_loaders import TextLoader
            logger.info("Markdown 文件使用 TextLoader 加载")
            return TextLoader(file_path, encoding="utf-8")

        else:
            raise DocumentProcessError(f"未实现的加载器: {ext}")

    def _apply_ocr(
        self, documents: List[Document], file_path: str
    ) -> List[Document]:
        """
        对 PDF 应用 OCR：
          - 扫描页（文本稀疏/为空）→ 整页识别，替换/追加文本
          - 文本页 → 识别内嵌图片（图表/截图）
        引擎不可用时优雅降级，返回原文档列表。

        Args:
            documents: PyPDFLoader 按页产出的文档列表
            file_path: PDF 文件路径

        Returns:
            List[Document]: 追加了 OCR 文本的文档列表
        """
        if not documents:
            return documents

        # 延迟导入，避免 OCR 未启用时引入重型依赖
        from .ocr import get_ocr_engine

        engine = get_ocr_engine(self.config)
        if not engine.is_available():
            logger.warning("OCR 引擎不可用，跳过 OCR 流程")
            return documents

        try:
            pdf = engine.fitz.open(file_path)
        except Exception as e:
            logger.warning(f"PDF 打开失败，跳过 OCR: {type(e).__name__}: {e}")
            return documents

        logger.info(
            f"OCR 开始: {file_path} | 共 {len(pdf)} 页 | "
            f"扫描页判定阈值: {self.config.ocr_min_chars_per_page} 字符/页"
        )

        enriched = []
        ocr_pages = 0
        try:
            for doc in documents:
                page_num = doc.metadata.get("page", 0)
                if page_num >= len(pdf):  # 越界保护
                    enriched.append(doc)
                    continue

                raw_text = doc.page_content or ""
                ocr_text = engine.ocr_page(pdf, page_num, raw_text).strip()
                if not ocr_text:
                    enriched.append(doc)
                    continue

                # 扫描页无文本层 → 直接使用 OCR 结果；有残文 → 追加保留
                if raw_text.strip():
                    merged = f"{raw_text.strip()}\n\n[OCR识别]\n{ocr_text}"
                else:
                    merged = f"[OCR识别]\n{ocr_text}"

                new_doc = Document(
                    page_content=merged,
                    metadata=dict(doc.metadata),
                )
                new_doc.metadata["ocr_processed"] = True
                enriched.append(new_doc)
                ocr_pages += 1
                if ocr_pages % 10 == 0:
                    # 每 10 页打一条进度，便于长任务时确认后端仍在工作
                    logger.info(f"OCR 进行中: {file_path} | 已识别 {ocr_pages} 页")
        finally:
            pdf.close()

        if ocr_pages:
            logger.info(f"OCR 完成: {file_path} | 共识别 {ocr_pages} 页")
        return enriched

    def _clean_documents(
        self,
        documents: List[Document],
        file_path: str,
        source_type: str,
        knowledge_base: str,
    ) -> List[Document]:
        """
        清洗文档内容（委托 TextCleaner：字符归一 → 断词拼接 → 页眉页脚/版式噪声
        → 短行过滤，并保留 Markdown 结构与代码块），同时绑定多源元数据。

        Args:
            documents:      原始文档列表（按页/段）
            file_path:      源文件路径
            source_type:    来源类型
            knowledge_base: 知识库标识

        Returns:
            List[Document]: 清洗后的文档列表（无有效内容的页/段被丢弃）
        """
        file_name = os.path.basename(file_path)

        cleaned_texts, stats = self._cleaner.clean_pages(
            [doc.page_content or "" for doc in documents]
        )

        cleaned: List[Document] = []
        for doc, cleaned_text in zip(documents, cleaned_texts):
            if not cleaned_text:
                continue

            # 绑定完整元数据（含多源融合字段）
            doc.page_content = cleaned_text
            doc.metadata.update({
                "file_name": file_name,
                "file_path": file_path,
                "page": doc.metadata.get("page", 0),
                "doc_id": str(uuid.uuid4())[:8],
                "source_type": source_type,        # 多源融合：论文/综述/笔记
                "knowledge_base": knowledge_base,  # 多知识库隔离
                "extension": Path(file_path).suffix.lower(),
            })
            cleaned.append(doc)

        logger.info(f"文本清洗统计: {file_name} | {stats.summary()}")
        return cleaned


# ==================== 兼容性保留 ====================
# 旧代码可能仍以 PDFLoader 名字导入，这里做向后兼容别名
PDFLoader = DocumentLoader
