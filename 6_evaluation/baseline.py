"""
================================================================================
  Layer 6 - 评估层: 纯 LLM 基线对比器
  功能：
    - 不走 RAG，直接用 LLM 回答同一批问题
    - 与 RAG 结果对比，量化"检索增强"带来的准确率提升
  简历亮点：
    RAG vs 纯LLM 准确率提升 25 个百分点
================================================================================
"""
from typing import List, Dict, Any, Optional

from utils.logger import get_logger
from .metrics import hallucination_rate, accuracy

logger = get_logger(__name__)


class PureLLMComparator:
    """
    纯 LLM 基线对比器。
    同一评测集分别跑：纯 LLM / RAG，输出对比报告。
    """

    # 纯 LLM Prompt（不提供参考文档）
    PURE_LLM_SYSTEM = "你是一个知识问答助手。请根据你自己的知识回答问题。"
    PURE_LLM_USER = "{question}\n\n请直接回答。"

    def __init__(self, llm_factory=None, rag_evaluator=None):
        """
        Args:
            llm_factory:    LLM 工厂
            rag_evaluator:  RAG 评估器实例（用于跑 RAG 侧）
        """
        from api.dependencies import get_llm_factory
        self.llm_factory = llm_factory or get_llm_factory()
        self.rag_evaluator = rag_evaluator

    def evaluate_pure_llm(
        self,
        dataset: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        用纯 LLM 跑评测集（无检索）。

        Args:
            dataset: 评测集

        Returns:
            Dict: 纯 LLM 指标
        """
        llm_call = self.llm_factory.get_llm_callable()
        answers: List[str] = []
        keyword_hits: List[bool] = []

        for idx, sample in enumerate(dataset, 1):
            question = sample["question"]
            expected_keywords = sample.get("expected_keywords", [])

            try:
                answer = llm_call(
                    self.PURE_LLM_USER.format(question=question)
                )
            except Exception as e:
                logger.error(f"纯 LLM 生成失败 (Q{idx}): {e}")
                answer = ""
            answers.append(answer)

            if expected_keywords:
                hit = any(kw in answer for kw in expected_keywords)
                keyword_hits.append(hit)

            if idx % 10 == 0:
                logger.info(f"纯 LLM 评估进度: {idx}/{len(dataset)}")

        result = {
            "samples": len(dataset),
            "hallucination_rate": hallucination_rate(answers),
            "keyword_accuracy": (
                accuracy(keyword_hits) if keyword_hits else None
            ),
            "answers": answers,
        }
        logger.info(f"纯 LLM 评估完成: 幻觉率={result['hallucination_rate']:.3f}")
        return result

    def compare(
        self,
        dataset: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        RAG vs 纯 LLM 对比评估。

        Args:
            dataset: 评测集

        Returns:
            Dict: 对比报告
        """
        if self.rag_evaluator is None:
            from api.dependencies import get_rag_evaluator
            self.rag_evaluator = get_rag_evaluator()

        # 纯 LLM
        pure_result = self.evaluate_pure_llm(dataset)

        # RAG
        rag_report = self.rag_evaluator.evaluate(
            dataset, enable_generation=True
        )

        # 对比
        pure_acc = pure_result.get("keyword_accuracy") or 0
        rag_acc = (rag_report.get("generation", {})
                   .get("keyword_accuracy") or 0)
        improvement = rag_acc - pure_acc

        report = {
            "samples": len(dataset),
            "pure_llm": {
                "keyword_accuracy": pure_acc,
                "hallucination_rate": pure_result["hallucination_rate"],
            },
            "rag": {
                "keyword_accuracy": rag_acc,
                "hallucination_rate": rag_report.get(
                    "generation", {}
                ).get("hallucination_rate", 0),
                "retrieval": rag_report.get("retrieval", {}),
            },
            "improvement": {
                "accuracy_gain": improvement,
                "accuracy_gain_percentage_points": round(
                    improvement * 100, 2
                ),
            },
        }
        logger.info(
            f"对比完成: RAG 准确率 {rag_acc*100:.1f}% vs 纯 LLM {pure_acc*100:.1f}% | "
            f"提升 {improvement*100:.1f} 个百分点"
        )
        return report
