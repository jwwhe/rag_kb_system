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

        简单判断逻辑：如果审核结果中没有明确的负面关键词，
        则认为审核通过。

        Args:
            review_result: LLM 审核输出

        Returns:
            bool: True 表示无问题，False 表示有问题需要修正
        """
        problem_keywords = [
            "缺乏依据", "无出处", "找不到", "编造", "虚构",
            "不存在", "矛盾", "不一致", "错误", "幻觉",
            "无原文支持", "未在文档中", "没有提到",
        ]
        result_lower = review_result.lower()
        return not any(kw in result_lower for kw in problem_keywords)

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
