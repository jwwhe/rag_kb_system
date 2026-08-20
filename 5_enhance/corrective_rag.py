"""
================================================================================
  Layer 5 - 增强层: 纠正型 RAG (CRAG)
  功能（V1.0 必须全上线）：
    - 答案自检：调用校验 Prompt 检查回答是否符合参考文档
    - 纠错：发现幻觉内容后自动修正或剔除
    - 迭代修正：最多 N 轮自检
================================================================================
"""
import importlib
import re
from typing import List, Optional, Callable

from langchain_core.documents import Document

from config.settings import EnhanceConfig, get_settings_cached
from utils.exceptions import EnhanceError
from utils.logger import get_logger

# Python 不允许直接 import 数字开头的包名，使用 importlib 动态导入
_pg = importlib.import_module("4_generate.prompts")
PromptTemplates = _pg.PromptTemplates

logger = get_logger(__name__)


class CorrectiveRAG:
    """
    纠正型 RAG (Corrective RAG)。
    对 LLM 生成的答案进行自检，发现并修正幻觉内容。
    """

    # 问题关键词（回退扫描用）
    # 注意：不能包含否定式短语（如"不存在""没有提到"），
    # 否则"不存在幻觉问题"等健康表述会被误判为有问题
    _problem_keywords = [
        "缺乏依据", "无出处", "找不到", "编造", "虚构",
        "矛盾", "不一致", "错误", "幻觉",
        "无原文支持", "未在文档中",
    ]

    # 否定语境前缀（出现时，紧跟的问题关键词不触发修正）
    # 例："不存在幻觉问题""未发现错误""没有编造内容"
    _NEGATION_PREFIXES = (
        "不存在", "未发现", "没有", "无需", "不是", "未出现", "不涉及",
    )

    def __init__(self, config: Optional[EnhanceConfig] = None):
        """
        初始化纠正型 RAG。

        Args:
            config: 增强配置（默认从全局配置读取）
        """
        self.config = config or get_settings_cached().enhance
        logger.info(
            f"CorrectiveRAG 初始化完成 | "
            f"最大迭代次数: {self.config.max_correction_iterations}"
        )

    def correct(
        self,
        answer: str,
        context_documents: List[Document],
        llm_call: Callable[[str], str],
    ) -> str:
        """
        对答案进行自检和纠错。

        流程：
        1. 使用校验 Prompt 让 LLM 审核答案
        2. 如果发现幻觉内容，要求 LLM 修正
        3. 最多迭代 self.config.max_correction_iterations 轮

        Args:
            answer:            原始答案
            context_documents: 参考文档列表
            llm_call:          LLM 调用函数

        Returns:
            str: 修正后的答案
        """
        if not self.config.enable_corrective_rag:
            return answer

        if not answer or not context_documents:
            return answer

        # 构建参考上下文
        context = self._build_context(context_documents)

        current_answer = answer

        for iteration in range(1, self.config.max_correction_iterations + 1):
            try:
                # 构建校验 Prompt
                system_prompt, user_prompt = PromptTemplates.build_verify_prompt(
                    context=context, answer=current_answer
                )

                # 调用 LLM 进行审核
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                review_result = llm_call(full_prompt)

                # 判断是否需要修正
                if self._is_clean(review_result):
                    logger.info(
                        f"CRAG 第 {iteration} 轮校验通过，答案无需修正"
                    )
                    break

                # 需要修正：要求 LLM 基于原文重写答案
                logger.info(
                    f"CRAG 第 {iteration} 轮发现问题，开始修正..."
                )
                current_answer = self._rewrite_answer(
                    context, current_answer, review_result, llm_call
                )

            except Exception as e:
                logger.warning(f"CRAG 第 {iteration} 轮校验异常: {e}")
                # 单轮异常不中断整体流程
                continue

        return current_answer

    def _is_clean(self, review_result: str) -> bool:
        """
        根据审核结果判断答案是否无幻觉。

        优先解析结构化结论行（"结论：通过 / 结论：需要修正"）；
        解析失败时回退到关键词扫描，且识别否定语境
        （如"不存在幻觉问题""未发现错误"不会被误判为有问题）。

        Args:
            review_result: LLM 审核输出

        Returns:
            bool: True 表示无问题，False 表示有问题需要修正
        """
        verdict = self._extract_verdict(review_result)
        if verdict is not None:
            return verdict

        # 回退：关键词扫描（排除否定语境，避免误判）
        result_lower = review_result.lower()
        for kw in self._problem_keywords:
            start = 0
            while True:
                idx = result_lower.find(kw, start)
                if idx == -1:
                    break
                prefix = result_lower[max(0, idx - 6):idx]
                if not any(neg in prefix for neg in self._NEGATION_PREFIXES):
                    return False
                start = idx + len(kw)
        return True

    def _extract_verdict(self, review_result: str):
        """
        从审核结果中解析结构化结论行。

        Returns:
            Optional[bool]: True=通过, False=需要修正, None=无法解析
        """
        for line in reversed(review_result.strip().split("\n")):
            line = line.strip()
            if not line:
                continue
            m = re.search(r"结论[:：]\s*(.+)", line)
            if not m:
                continue
            verdict_text = m.group(1)
            if any(k in verdict_text for k in
                   ("通过", "无需", "无问题", "没问题", "不需要修正", "无需修正")):
                return True
            if any(k in verdict_text for k in
                   ("修正", "有问题", "存在", "错误", "幻觉", "不符", "剔除", "删除", "缺乏")):
                return False
            return None  # 结论行无法识别，交由回退逻辑
        return None

    def _rewrite_answer(
        self,
        context: str,
        original_answer: str,
        review_result: str,
        llm_call: Callable[[str], str],
    ) -> str:
        """
        基于审核反馈修正答案。

        Args:
            context:         参考文档上下文
            original_answer: 原始答案
            review_result:   审核反馈
            llm_call:        LLM 调用函数

        Returns:
            str: 修正后的答案
        """
        correction_prompt = f"""你是一个严谨的答案修正助手。以下 AI 生成答案经审核发现了问题，请根据参考文档重新生成正确的答案。

【参考文档】
{context}

【原始答案】
{original_answer}

【审核反馈】
{review_result}

【修正要求】
1. 严格基于参考文档内容修正
2. 删除所有缺乏依据的陈述
3. 修正与文档矛盾的表述
4. 如果某个问题参考文档完全没有涉及，明确说明"文档中未提及"
5. 保持答案结构清晰，标注引用来源

修正后的答案："""

        try:
            corrected = llm_call(correction_prompt)
            return corrected.strip() if corrected else original_answer
        except Exception as e:
            logger.warning(f"答案修正失败，保留原答案: {e}")
            return original_answer

    @staticmethod
    def _build_context(documents: List[Document]) -> str:
        """构建参考文档上下文。"""
        parts = []
        for i, doc in enumerate(documents, 1):
            file_name = doc.metadata.get("file_name", "未知文件")
            page = doc.metadata.get("page", "N/A")
            parts.append(
                f"[文档-{i}] 来源: {file_name} (第{page}页)\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)
