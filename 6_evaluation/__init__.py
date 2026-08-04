# 6_evaluation - 评估层 (Layer 6)
# 职责：量化评估检索与生成质量
# 指标：Recall@K / MRR / 幻觉率 / RAG vs 纯LLM 对比
# 依赖：config + utils + Layer 2/3/4 (实际运行 RAG 链路)

from .metrics import (
    recall_at_k,
    mrr,
    hallucination_rate,
    precision_at_k,
    ndcg_at_k,
)
from .evaluator import RAGEvaluator
from .baseline import PureLLMComparator

__all__ = [
    "recall_at_k",
    "mrr",
    "hallucination_rate",
    "precision_at_k",
    "ndcg_at_k",
    "RAGEvaluator",
    "PureLLMComparator",
]
