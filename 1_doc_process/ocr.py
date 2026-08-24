"""
===============================================================================
  Layer 1 - 文档处理层: PDF OCR 识别引擎
  功能：
    - 扫描件 PDF（无文本层）整页识别
    - 文本 PDF 内嵌图片（图表/截图）识别
    - 引擎：RapidOCR（PaddleOCR 的 ONNX 版，中英文开箱即用，pip install rapidocr）
    - 渲染：PyMuPDF（fitz），纯 pip 依赖，无需 poppler
    - 引擎缺失时优雅降级（跳过 OCR，不影响原有流程）
===============================================================================
"""
from io import BytesIO
from typing import Optional

from config.settings import DocProcessConfig, get_settings_cached
from utils.logger import get_logger

logger = get_logger(__name__)

# 图片过小视为水印/背景/图标，跳过 OCR（减少误识别与耗时）
_MIN_IMAGE_BYTES = 1024


class PDFOcrEngine:
    """
    PDF OCR 识别引擎（RapidOCR + PyMuPDF）。

    - RapidOCR 模型加载较重，做模块级惰性单例，整个进程只加载一次
    - PyMuPDF 仅用于把 PDF 页面渲染为图片 / 提取内嵌图片
    """

    # 全局惰性单例（模型只加载一次）
    _instance: Optional["PDFOcrEngine"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[DocProcessConfig] = None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.config = config or get_settings_cached().doc_process
        self._ocr = None
        self._ocr_warned = False
        self._fitz = None

    # ==================== 引擎状态 ====================

    @property
    def ocr(self):
        """惰性加载 RapidOCR（未安装时仅告警一次，不抛异常）。"""
        if self._ocr is None and not self._ocr_warned:
            try:
                try:
                    from rapidocr_onnxruntime import RapidOCR
                except ImportError:
                    # rapidocr 2.x 起包名改为 rapidocr
                    from rapidocr import RapidOCR
                self._ocr = RapidOCR()
                logger.info("RapidOCR 引擎加载完成")
            except Exception as e:
                self._ocr_warned = True
                logger.warning(
                    f"RapidOCR 不可用，跳过 OCR（可 pip install rapidocr）: {e}"
                )
        return self._ocr

    @property
    def fitz(self):
        """惰性加载 PyMuPDF（新版建议 import pymupdf，兼容旧 fitz 别名）。"""
        if self._fitz is None:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            self._fitz = fitz
        return self._fitz

    def is_available(self) -> bool:
        """OCR 引擎与渲染库是否都可用。"""
        if self.ocr is None:
            return False
        try:
            self.fitz
            return True
        except Exception:
            return False

    # ==================== OCR 核心 ====================

    def ocr_image_bytes(self, img_bytes: bytes) -> str:
        """
        对单张图片 bytes 执行 OCR，返回文本（空串=无识别结果）。

        Args:
            img_bytes: 图片原始字节（PNG/JPEG 等）

        Returns:
            str: 识别文本，多行文本用换行分隔；无结果返回空串
        """
        ocr = self.ocr
        if ocr is None or not img_bytes:
            return ""
        try:
            from PIL import Image

            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            raw = ocr(img)
            if not raw:
                return ""
            # rapidocr 2.x: RapidOCROutput 对象（.txts/.scores）
            if not isinstance(raw, (list, tuple)):
                try:
                    txts = raw.txts
                    scores = raw.scores
                except AttributeError:
                    return ""
                lines = []
                for i, text in enumerate(txts):
                    text = str(text).strip()
                    score = float(scores[i]) if i < len(scores) else 1.0
                    if text and score >= self.config.ocr_confidence_threshold:
                        lines.append(text)
                return "\n".join(lines)
            # rapidocr 1.x: (result, elapse)，result 为 [box, text, score] 列表或 None
            if isinstance(raw, tuple):
                raw = raw[0]
            if not raw:
                return ""
            lines = []
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text = str(item[1]).strip()
                    score = float(item[2]) if len(item) > 2 and item[2] is not None else 1.0
                    if text and score >= self.config.ocr_confidence_threshold:
                        lines.append(text)
                elif isinstance(item, str) and item.strip():
                    lines.append(item.strip())
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"图片 OCR 失败: {type(e).__name__}: {e}")
            return ""

    def render_page_bytes(self, pdf, page_index: int) -> bytes:
        """
        把 PDF 页渲染为 PNG 图片 bytes（供整页 OCR）。

        Args:
            pdf:        fitz 打开的文档对象
            page_index: 页码（0 起）

        Returns:
            bytes: PNG 图片字节
        """
        page = pdf[page_index]
        pix = page.get_pixmap(dpi=self.config.ocr_render_dpi)
        return pix.tobytes("png")

    def ocr_page_images(self, pdf, page_index: int) -> str:
        """
        提取页面内嵌图片并逐一 OCR（适用于有文本层的页中的图表/截图）。

        Args:
            pdf:        fitz 打开的文档对象
            page_index: 页码（0 起）

        Returns:
            str: 各图片识别文本，以"[图片N]"分组；无结果返回空串
        """
        if self.ocr is None:
            return ""
        groups = []
        try:
            page = pdf[page_index]
            for img_index, img_info in enumerate(page.get_images(full=True), 1):
                xref = img_info[0]
                try:
                    base = pdf.extract_image(xref)
                except Exception:
                    continue
                if not base:
                    continue
                img_bytes = base.get("image", b"")
                if len(img_bytes) < _MIN_IMAGE_BYTES:
                    continue  # 水印/背景小图跳过
                text = self.ocr_image_bytes(img_bytes)
                if text.strip():
                    groups.append(f"[图片{img_index}]\n{text}")
        except Exception as e:
            logger.warning(f"第 {page_index} 页内嵌图片 OCR 失败: {type(e).__name__}: {e}")
        return "\n\n".join(groups)

    def ocr_page(self, pdf, page_index: int, raw_text: str) -> str:
        """
        按页应用 OCR：
          - 扫描页（文本稀疏）→ 整页渲染识别
          - 文本页 → 仅识别内嵌图片

        Args:
            pdf:        fitz 打开的文档对象
            page_index: 页码（0 起）
            raw_text:   该页现有文本层内容

        Returns:
            str: OCR 补充文本（未启用/无结果返回空串）
        """
        if self.ocr is None:
            return ""
        if len(raw_text.strip()) < self.config.ocr_min_chars_per_page:
            # 扫描页：整页识别（内嵌图片已被渲染包含，无需重复提取）
            page_img = self.render_page_bytes(pdf, page_index)
            return self.ocr_image_bytes(page_img)
        if self.config.ocr_embedded_images:
            return self.ocr_page_images(pdf, page_index)
        return ""


# 模块级便捷入口（整个进程共享同一引擎实例）
def get_ocr_engine(config: Optional[DocProcessConfig] = None) -> PDFOcrEngine:
    """获取全局 OCR 引擎（惰性单例）。"""
    return PDFOcrEngine(config)