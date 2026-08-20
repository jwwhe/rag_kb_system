"""
================================================================================
  Layer 3 - 检索层: 查询改写器
  功能（严禁简化）：
    - 口语转专业术语
    - 补全指代不明的表述
    - 生成多同义子问题（扩充检索覆盖面）
  实现方式：
    - 规则引擎 + LLM 增强（LLM callable 由 API 层注入，保持分层解耦）
================================================================================
"""
import importlib
import re
from typing import List, Optional, Callable

from config.settings import RetrievalConfig, get_settings_cached
from utils.logger import get_logger

# Python 不允许直接 import 数字开头的包名，使用 importlib 动态导入
_pg = importlib.import_module("4_generate.prompts")
PromptTemplates = _pg.PromptTemplates

logger = get_logger(__name__)


class QueryRewriter:
    """
    查询改写器。
    将用户口语化问题改写为适合检索的专业查询，
    并生成多个同义子问题以提高检索召回率。
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        初始化查询改写器。

        Args:
            config: 检索配置（默认从全局配置读取）
        """
        self.config = config or get_settings_cached().retrieval

        # 常见口语→书面语映射表（规则引擎）
        self._colloquial_map = {
            "啥": "什么",
            "咋": "怎么",
            "咋样": "怎么样",
            "咋办": "怎么办",
            "为啥": "为什么",
            "干嘛": "做什么",
            "行不行": "是否可行",
            "好不好": "是否合适",
            "方不方便": "是否方便",
            "麻不麻烦": "是否复杂",
        }

        # 常见指代词模式（补全用）
        # 注意：避免裸"这/那"替换（会误伤"这里/那边/这个"等正常词），
        # 只替换完整指代词；"它们"必须先于"它"匹配，否则变成"该内容们"
        self._pronoun_patterns = [
            (r"这个|那个|这些|那些", "该内容"),
            (r"它们|它", "该内容"),
            (r"上文|上述|前面提到", "前述内容"),
            (r"如下|以下", "下述内容"),
        ]

        logger.info(
            f"QueryRewriter 初始化完成 | "
            f"查询改写: {'启用' if self.config.enable_query_rewrite else '禁用'} | "
            f"最大子问题数: {self.config.max_sub_questions}"
        )

    def rewrite(
        self,
        query: str,
        llm_call: Optional[Callable[[str], str]] = None,
    ) -> List[str]:
        """
        改写查询，生成原始查询 + 改写查询 + 同义子问题列表。

        改写流程：
        1. 规则引擎：口语→书面语、指代消解
        2. LLM 增强（可选）：生成多同义子问题
        3. 去重合并

        Args:
            query:    用户原始查询
            llm_call: LLM 调用函数（签名: str -> str），由上层注入

        Returns:
            List[str]: 改写后的查询列表（原始查询排在首位）
        """
        if not query or not query.strip():
            return [query]

        query = query.strip()

        # Step 1: 规则引擎改写
        rule_rewritten = self._rule_based_rewrite(query)

        # Step 2: 构建查询列表（原始 + 规则改写）
        queries = [query]
        if rule_rewritten != query:
            queries.append(rule_rewritten)

        # Step 3: LLM 增强生成同义子问题
        if self.config.enable_query_rewrite and llm_call is not None:
            try:
                llm_queries = self._llm_rewrite(query, llm_call)
                queries.extend(llm_queries)
            except Exception as e:
                logger.warning(f"LLM 查询改写失败，降级为规则引擎结果: {e}")

        # Step 4: 去重（保持顺序）
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        logger.info(
            f"查询改写完成: '{query[:50]}...' → {len(unique_queries)} 个变体"
        )
        return unique_queries[: self.config.max_sub_questions + 1]

    def _rule_based_rewrite(self, query: str) -> str:
        """
        基于规则的查询改写：口语转书面语、指代消解。

        Args:
            query: 原始查询

        Returns:
            str: 改写后的查询
        """
        rewritten = query

        # 口语→书面语替换
        for colloquial, formal in self._colloquial_map.items():
            rewritten = rewritten.replace(colloquial, formal)

        # 基本指代消解（规则层面）
        for pattern, replacement in self._pronoun_patterns:
            rewritten = re.sub(pattern, replacement, rewritten)

        return rewritten

    def _llm_rewrite(
        self,
        query: str,
        llm_call: Callable[[str], str],
    ) -> List[str]:
        """
        使用 LLM 生成同义子问题。

        Args:
            query:    原始查询
            llm_call: LLM 调用函数

        Returns:
            List[str]: LLM 生成的同义查询列表
        """
        # 复用 prompts.py 的统一改写模板（避免两套 Prompt 漂移）
        system_prompt, user_prompt = PromptTemplates.build_rewrite_prompt(
            query, self.config.max_sub_questions
        )
        response = llm_call(f"{system_prompt}\n\n{user_prompt}")

        # 解析 LLM 返回的子问题列表
        sub_queries = self._parse_sub_queries(response)
        return sub_queries

    def _parse_sub_queries(self, response: str) -> List[str]:
        """
        解析 LLM 返回的子问题列表。

        Args:
            response: LLM 响应文本

        Returns:
            List[str]: 解析出的查询列表
        """
        queries = []
        for line in response.strip().split("\n"):
            # 去除编号前缀（如 "1. " "- " "· "）
            cleaned = re.sub(r"^[\d]+[\.\)、]\s*", "", line.strip())
            cleaned = re.sub(r"^[-·•]\s*", "", cleaned)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 2:
                queries.append(cleaned)
        return queries
