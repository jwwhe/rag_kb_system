"""
================================================================================
  Layer 1 - 文档处理层: 文本清洗器
  功能（按执行顺序）：
    1. 字符归一化：换行统一 → 全角字母数字转半角 → 零宽与控制字符清理
    2. 断词拼接：PDF 行尾连字符换行（inform-/ation）合并为完整单词
    3. 页眉页脚识别：统计每页首尾的跨页重复短行，纯统计驱动、无硬编码词典
    4. 版式噪声过滤：页码、装饰分隔线、乱码/水印行
    5. 结构保留：Markdown 标题/列表/表格/代码块豁免短行过滤，代码块缩进原样保留
    6. 短行过滤：仅作用于正文散行（表格碎片/孤字），阈值可配
  输出各规则命中计数，便于日志观测与清洗策略评估。
================================================================================
"""
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from config.settings import DocProcessConfig, get_settings_cached
from utils.logger import get_logger

logger = get_logger(__name__)


# ==================== 行级规则（模块级预编译，避免逐行重复编译）====================

# 版式噪声：页码 / 装饰线 / 纯符号碎片
_PAGE_ARTIFACT_PATTERNS = (
    re.compile(r"^\d{1,4}$"),  # 12
    re.compile(r"^[\(\[-]?\d{1,4}[\)\]\-\.]?$"),  # (12) / [12] / 12.
    re.compile(r"^\d{1,4}\s*[/｜|]\s*\d{1,4}$"),  # 3/20
    re.compile(r"^第\s*\d{1,4}\s*页(\s*[，,/]?\s*(共\s*)?\d+\s*页?)?$"),  # 第3页 / 第3页，共20页
    re.compile(r"^(page|p\.|no\.|nr\.|vol\.|lv\.)\s*\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I),
    re.compile(r"^[ivxlcdm]{1,8}$", re.I),  # 罗马数字页码
    re.compile(r"^[-_=*·—–…\.•●○◆■□\s]{3,}$"),  # 装饰分隔线
)

# Markdown / 文本结构行：标题、引用、列表、表格、代码围栏 → 豁免短行过滤
_STRUCTURAL_LINE = re.compile(
    r"^(#{1,6}\s|>\s?|\s{0,3}[-*+]\s|\s{0,3}\d+[.)]\s|\||```|~~~)"
)

# 代码围栏行（块内文本原样保留缩进，不做任何行级改写）
_CODE_FENCE = re.compile(r"^\s*(```|~~~)")

# 行尾断词连字符
_HYPHEN_TAIL = re.compile(r"\w-$")

# OCR 文本块标记：标记行本身丢弃，其后文本豁免短行过滤（识别结果常为短句/词组）
_OCR_MARKER = re.compile(r"^\[OCR[^\]]*\]$")

# 零宽字符 / BOM / 软连字符
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")

# 全角字母与数字（ＡＺ０９）→ 半角。
# 刻意不做整体 NFKC：全角标点（，：（）等 U+FF00 段）会被映射成 ASCII，
# 既破坏中文句号断点，也让溯源片段与原文对不上。
_FULLWIDTH_ALNUM = re.compile(r"[\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]")

# 字体替换/解码失败的占位符：PDF 常见 (cid:12) 与替换字符
_GARBAGE_TOKEN = re.compile(r"\(cid[:：]?\d+\)|[\ufffd\ufffc\u25a0\u25a1]+")

# 文字字符（中日韩 + 拉丁字母 + 数字），用于乱码行的"非文字占比"判定
_TEXT_CHARS = re.compile(r"[0-9A-Za-z一-鿿぀-ヿ]")

# 页眉页脚候选行最大长度
_BOUNDARY_LINE_MAX_LEN = 40

# 每页首尾各取几行参与页眉页脚统计
_BOUNDARY_LINE_WINDOW = 2


@dataclass
class CleanStats:
    """单次清洗过程的规则命中计数。"""

    pages: int = 0
    lines_in: int = 0
    lines_out: int = 0
    pages_kept: int = 0
    dropped_blank: int = 0
    dropped_artifact: int = 0  # 页码/装饰线/乱码/OCR 标记行
    dropped_header_footer: int = 0
    dropped_short: int = 0
    hyphen_joined: int = 0
    chars_normalized: int = 0  # 归一化阶段清理的字符数

    def summary(self) -> str:
        """一行文本形式的统计摘要（供日志）。"""
        return (
            f"有效页 {self.pages_kept}/{self.pages} | "
            f"行数 {self.lines_in} → {self.lines_out}（过滤 {self.lines_in - self.lines_out}）| "
            f"空白 {self.dropped_blank} | 版式噪声 {self.dropped_artifact} | "
            f"页眉页脚 {self.dropped_header_footer} | 短行 {self.dropped_short} | "
            f"断词拼接 {self.hyphen_joined} | 归一字符 {self.chars_normalized}"
        )


class TextCleaner:
    """
    文本清洗器。

    输入按页（或文档段落）切分的原始文本列表，输出清洗后的同序文本列表。
    跨页处理仅用于页眉页脚的重复行统计，不改变段落顺序与文本语义。
    """

    def __init__(self, config: Optional[DocProcessConfig] = None):
        """
        初始化清洗器。

        Args:
            config: 文档处理配置（默认从全局配置读取）
        """
        self.config = config or get_settings_cached().doc_process
        logger.info(
            f"TextCleaner 初始化完成 | "
            f"总开关: {self.config.enable_cleaning} | "
            f"最短有效行: {self.config.min_line_length} | "
            f"页眉页脚过滤: {self.config.filter_header_footer} | "
            f"断词拼接: {self.config.clean_fix_hyphenation} | "
            f"乱码占比阈值: {self.config.clean_garbage_line_ratio}"
        )

    # ==================== 对外接口 ====================

    def clean_pages(
        self, raw_texts: Sequence[str]
    ) -> Tuple[List[str], CleanStats]:
        """
        批量清洗按页切分的文本（含跨页页眉页脚检测）。

        Args:
            raw_texts: 每页/每段的原始文本，与返回结果同序

        Returns:
            Tuple[List[str], CleanStats]: 清洗后文本（空页为空串）与统计
        """
        stats = CleanStats(pages=len(raw_texts))
        if not raw_texts:
            return [], stats

        if not self.config.enable_cleaning:
            logger.warning("清洗总开关(enable_cleaning)已关闭，原文本直接透传")
            texts = [t or "" for t in raw_texts]
            stats.lines_in = stats.lines_out = sum(
                len(t.split("\n")) for t in texts
            )
            stats.pages_kept = sum(1 for t in texts if t.strip())
            return texts, stats

        # Step 1: 字符归一 + 断行 + 断词拼接
        pages_lines: List[List[str]] = []
        for raw in raw_texts:
            lines = self._join_hyphenated(
                self._normalize_chars(raw or "", stats).split("\n"), stats
            )
            pages_lines.append(lines)
            stats.lines_in += len(lines)

        # Step 2: 跨页页眉页脚识别（先统计后过滤，保证同一条页眉全篇一致清除）
        repeated = self._detect_boundary_repeats(pages_lines)

        # Step 3: 逐页行级过滤
        cleaned = [
            self._filter_page(lines, repeated, stats) for lines in pages_lines
        ]
        stats.pages_kept = sum(1 for t in cleaned if t)
        return cleaned, stats

    def clean_text(self, text: str) -> str:
        """
        清洗单段文本（无跨页统计），供查询改写等场景复用。

        Args:
            text: 原始文本

        Returns:
            str: 清洗后文本
        """
        cleaned, _ = self.clean_pages([text])
        return cleaned[0] if cleaned else ""

    # ==================== Step 1: 字符归一化 ====================

    def _normalize_chars(self, text: str, stats: CleanStats) -> str:
        """
        字符级归一化：换行统一 → 全角字母数字转半角 → 零宽/控制字符清理。

        Args:
            text:  原始文本
            stats: 统计对象（累加 chars_normalized）

        Returns:
            str: 归一化文本
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if not self.config.clean_unicode_normalize:
            return text

        # 全角字母数字（ＢＡＳＥ→BASE）归一半角，避免同形异码割裂检索与 BM25 分词
        normalized = _FULLWIDTH_ALNUM.sub(
            lambda m: unicodedata.normalize("NFKC", m.group()), text
        )

        # 零宽字符/BOM/软连字符 + 控制字符（保留 \n 与 \t）
        stripped = _ZERO_WIDTH.sub("", normalized)
        kept = "".join(
            ch for ch in stripped
            if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
        )

        removed = len(stripped) - len(kept)
        if removed:
            stats.chars_normalized += removed
            logger.debug(f"字符归一化清理 {removed} 个零宽/控制字符")
        return kept

    def _join_hyphenated(
        self, lines: List[str], stats: CleanStats
    ) -> List[str]:
        """
        PDF 断词拼接：行尾连字符 + 下行以小写字母开头 → 合并为同一单词。

        Args:
            lines: 单页行列表
            stats: 统计对象（累加 hyphen_joined）

        Returns:
            List[str]: 拼接后的行列表
        """
        if not self.config.clean_fix_hyphenation or len(lines) < 2:
            return lines

        merged: List[str] = []
        in_code_block = False
        i = 0
        while i < len(lines):
            cur = lines[i].rstrip()
            if _CODE_FENCE.match(cur):
                in_code_block = not in_code_block
                merged.append(lines[i])
                i += 1
                continue

            nxt = lines[i + 1].lstrip() if i + 1 < len(lines) else ""
            if (
                not in_code_block
                and _HYPHEN_TAIL.search(cur)
                and nxt[:1].isascii()
                and nxt[:1].islower()
            ):
                lines[i + 1] = cur[:-1] + nxt  # 去掉行尾连字符后续接到下行
                stats.hyphen_joined += 1
                i += 1
                continue

            merged.append(lines[i])
            i += 1
        return merged

    # ==================== Step 2: 页眉页脚识别 ====================

    def _detect_boundary_repeats(self, pages_lines: List[List[str]]) -> Set[str]:
        """
        统计每页首尾若干行中跨页重复出现的短行，判为页眉/页脚。

        判定阈值：出现在 >= max(clean_header_footer_min_pages, 总页数一半) 个不同页，
        且长度 <= 40、非 Markdown 结构行、非页码行。

        Args:
            pages_lines: 每页行列表

        Returns:
            Set[str]: 需整篇剔除的页眉页脚文本集合
        """
        if not self.config.filter_header_footer:
            return set()

        total_pages = len(pages_lines)
        min_pages = max(
            self.config.clean_header_footer_min_pages,
            math.ceil(total_pages / 2),
        )
        if total_pages < min_pages:
            return set()

        counter: Counter = Counter()
        for lines in pages_lines:
            content = [l.strip() for l in lines if l.strip()]
            boundary = content[:_BOUNDARY_LINE_WINDOW] + content[-_BOUNDARY_LINE_WINDOW:]
            # 同一页内重复只计一次，避免"页内重复"被误判为"跨页重复"
            for line in set(boundary):
                if (
                    len(line) <= _BOUNDARY_LINE_MAX_LEN
                    and not _STRUCTURAL_LINE.match(line)
                    and not self._is_page_artifact(line)
                ):
                    counter[line] += 1

        repeated = {line for line, n in counter.items() if n >= min_pages}
        if repeated:
            logger.info(
                f"识别页眉页脚 {len(repeated)} 种（跨 {min_pages}/{total_pages} 页重复）: "
                f"{sorted(repeated, key=len)[:3]}"
            )
        return repeated

    # ==================== Step 3: 逐行过滤 ====================

    def _filter_page(
        self, lines: List[str], repeated: Set[str], stats: CleanStats
    ) -> str:
        """
        对单页执行行级过滤。

        规则顺序：代码块原样保留 → 空白行 → OCR 标记 → 页眉页脚 → 版式噪声 → 短行。

        Args:
            lines:    单页行列表（已归一化）
            repeated: 页眉页脚文本集合
            stats:    统计对象

        Returns:
            str: 清洗后的页文本（无有效内容时为空串）
        """
        kept: List[str] = []
        in_code_block = False
        in_ocr_block = False

        for line in lines:
            if _CODE_FENCE.match(line):
                in_code_block = not in_code_block
                kept.append(line.strip())
                continue

            if in_code_block:
                kept.append(line.rstrip())  # 代码内容原样保留
                continue

            stripped = line.strip()

            if not stripped:
                if self.config.filter_blank_lines:
                    stats.dropped_blank += 1
                    continue
                kept.append("")
                continue

            if _OCR_MARKER.match(stripped):
                in_ocr_block = True  # 标记行丢弃，其后 OCR 文本豁免短行过滤
                stats.dropped_artifact += 1
                continue

            if self.config.filter_header_footer and stripped in repeated:
                stats.dropped_header_footer += 1
                continue

            # 结构行（标题/列表/表格/围栏）不做噪声与短行判定：
            # 表格分隔行 "| --- | --- |" 文字占比为 0，按散行规则会被误删
            is_structural = bool(_STRUCTURAL_LINE.match(stripped))

            if (
                not is_structural
                and self.config.clean_strip_page_artifacts
                and self._is_page_artifact(stripped)
            ):
                stats.dropped_artifact += 1
                continue

            if (
                not in_ocr_block
                and not is_structural
                and len(stripped) < self.config.min_line_length
            ):
                stats.dropped_short += 1
                continue

            kept.append(self._collapse_spaces(stripped))

        non_empty = sum(1 for l in kept if l.strip())
        stats.lines_out += non_empty
        return "\n".join(kept).strip()

    # ==================== 判定辅助 ====================

    def _is_page_artifact(self, line: str) -> bool:
        """
        是否为版式噪声行：页码、装饰线、乱码/水印。

        Args:
            line: 已 strip 的单行文本

        Returns:
            bool: 是否应作为噪声剔除
        """
        if any(p.match(line) for p in _PAGE_ARTIFACT_PATTERNS):
            return True
        return self._is_garbled(line)

    def _is_garbled(self, line: str) -> bool:
        """
        乱码判定：解码占位符占比过高，或文字字符占比过低。

        Args:
            line: 已 strip 的单行文本

        Returns:
            bool: 是否为乱码行
        """
        if len(line) < 8:
            return False

        placeholder_len = sum(len(m) for m in _GARBAGE_TOKEN.findall(line))
        if placeholder_len >= 4 and placeholder_len / len(line) > 0.3:
            return True

        ratio = 1 - len(_TEXT_CHARS.findall(line)) / len(line)
        return ratio > self.config.clean_garbage_line_ratio

    @staticmethod
    def _collapse_spaces(line: str) -> str:
        """
        行内空白规整：全角空格/制表符转空格，连续空格压缩，去首尾空白。

        Args:
            line: 单行文本

        Returns:
            str: 规整后的行
        """
        return re.sub(r"[ \u3000\t]+", " ", line).strip()
