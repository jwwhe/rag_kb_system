"""
================================================================================
  Layer 6 - 评估层: 检索与生成质量指标
  简历亮点：
    量化评估体系（Recall@K / MRR / 幻觉率），RAG vs 纯LLM 准确率提升 25 个百分点
================================================================================
"""
from typing import List, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Recall@K：前 K 条结果中覆盖了多少比例的相关文档。

    Args:
        retrieved_ids: 检索返回的文档 ID 列表（按分数降序）
        relevant_ids:  标注的相关文档 ID 列表
        k:             截断位置

    Returns:
        float: 0.0 ~ 1.0
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hit = sum(1 for rid in top_k if rid in set(relevant_ids))
    return hit / len(relevant_ids)


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Precision@K：前 K 条结果中相关文档的比例。"""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hit = sum(1 for rid in top_k if rid in set(relevant_ids))
    return hit / k


def mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """
    MRR（Mean Reciprocal Rank）：第一个相关文档的倒数排名。
    例：相关文档排在第 3 位 → 1/3 ≈ 0.333

    Returns:
        float: 0.0 ~ 1.0
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    NDCG@K：归一化折损累积增益。
    相关文档排得越靠前分数越高。
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, rid in enumerate(top_k, 1):
        if rid in relevant_set:
            dcg += 1.0 / (i if i <= 2 else (i / 2) + 0.5)

    # IDCG（理想排序）
    ideal_hits = min(len(relevant_ids), k)
    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / (i if i <= 2 else (i / 2) + 0.5)

    return dcg / idcg if idcg > 0 else 0.0


def hallucination_rate(
    answers: List[str],
    refusal_phrase: str = "知识库中暂无该相关资料",
) -> float:
    """
    幻觉率（启发式近似）：
    - 若回答中既未触发兜底拒绝、又包含无依据的"我/可能/或许"等不确定词，
      则视为潜在幻觉。
    - 严格判断需用 Verify Prompt + LLM，本函数提供快速启发式评估。

    Args:
        answers:        生成的回答列表
        refusal_phrase: 兜底拒绝文案

    Returns:
        float: 0.0 ~ 1.0，幻觉样本占比
    """
    if not answers:
        return 0.0
    suspicious_tokens = ["可能", "或许", "应该", "我猜测", "据我所知", "通常来说"]
    hallucinated = 0
    for ans in answers:
        if not ans or not ans.strip():
            continue
        # 触发兜底 → 非幻觉（明确承认不知道）
        if refusal_phrase in ans:
            continue
        # 包含不确定词 → 潜在幻觉
        if any(tok in ans for tok in suspicious_tokens):
            hallucinated += 1
    return hallucinated / len(answers)


def accuracy(
    judgments: List[bool],
) -> float:
    """简单准确率：判对数 / 总数。"""
    if not judgments:
        return 0.0
    return sum(1 for j in judgments if j) / len(judgments)


def aggregate(
    per_query_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    汇总多查询的平均指标。

    Args:
        per_query_metrics: 每条查询的指标字典

    Returns:
        Dict: 各指标的平均值
    """
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {
        k: sum(m.get(k, 0) for m in per_query_metrics) / len(per_query_metrics)
        for k in keys
    }
