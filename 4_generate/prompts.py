"""
================================================================================
  Layer 4 - 生成层: 三套固定 Prompt 模板
  规范要求内置三套固定 Prompt：
    1. 问答 Prompt（RAG 生成）
    2. 改写 Prompt（查询优化）
    3. 校验 Prompt（防幻觉）
================================================================================
"""


class PromptTemplates:
    """
    三套固定 Prompt 模板。
    所有 Prompt 均在 code 中维护，禁止硬编码到业务逻辑中。
    """

    # ==================== 1. RAG 问答 Prompt ====================
    QA_SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请严格根据提供的参考文档内容回答问题。

【核心规则 - 必须遵守】
1. 只能根据「参考文档」中的内容回答，不得使用你自己的知识
2. 如果参考文档不足以回答问题，必须明确回复：知识库中暂无该相关资料，无法解答此问题
3. 禁止编造、脑补、扩写文档中不存在的信息
4. 禁止进行闲聊、科普、自由对话
5. 回答应结构化、条理清晰，关键内容标注引用来源

【回答格式】
- 先给出简洁的结论
- 再逐条展开详细说明（每条绑定引用编号）
- 最后附上引用列表"""

    QA_USER_PROMPT_TEMPLATE = """【参考文档】
{context}

【用户问题】
{question}

请根据以上参考文档回答问题。如果文档信息不足，请明确说明。"""

    # ==================== 2. 查询改写 Prompt ====================
    REWRITE_SYSTEM_PROMPT = """你是一个查询优化助手。任务是将用户的口语化问题改写为适合在知识库中检索的专业表达。

要求：
1. 将口语表达转换为书面/专业表达
2. 补全不清晰的指代
3. 生成多个不同角度的同义查询
4. 每个查询一行，不要编号
5. 不要添加解释或额外内容"""

    REWRITE_USER_PROMPT_TEMPLATE = """请改写以下查询，生成 {max_sub_questions} 个不同角度的同义查询（包含改写后的查询）：

{query}"""

    # ==================== 3. 答案校验 Prompt ====================
    VERIFY_SYSTEM_PROMPT = """你是一个答案质量审核员。请检查以下 AI 生成的回答是否严格基于参考文档。

审核标准：
1. 回答中的每条事实陈述是否都能在参考文档中找到依据？
2. 是否存在编造、脑补、扩写的内容？
3. 是否存在与参考文档矛盾的内容？

请逐条分析并给出审核结论。最后一行必须且只能输出审核结论，格式严格为以下两种之一（不要输出其他内容到该行）：
- 结论：通过
- 结论：需要修正"""

    VERIFY_USER_PROMPT_TEMPLATE = """【参考文档】
{context}

【AI 生成的回答】
{answer}

请审核以上回答是否严格基于参考文档。逐条分析每个事实陈述的出处。"""

    # ==================== 4. Web 兜底回答格式 ====================
    WEB_FALLBACK_SYSTEM_PROMPT = """你是一个知识问答助手。注意：以下信息来自外部网络搜索，不是本地私有知识库的内容。

请根据网络搜索结果回答问题，并明确标注所有信息的来源。"""

    WEB_FALLBACK_USER_PROMPT_TEMPLATE = """【网络搜索结果】
{web_results}

【用户问题】
{question}

请根据以上网络搜索结果回答问题。必须明确标注每条信息的来源 URL。"""

    # ==================== 5. 兜底拒绝文案 ====================
    NO_KNOWLEDGE_RESPONSE = "知识库中暂无该相关资料，无法解答此问题"
    EXTERNAL_SOURCE_PREFIX = "【以下内容来源于外部网络搜索，非本地知识库内容】"
    INTERNAL_SOURCE_PREFIX = "【以下内容来源于本地知识库】"

    @classmethod
    def build_qa_prompt(cls, context: str, question: str) -> tuple:
        """
        构建 RAG 问答 Prompt。

        Args:
            context:  检索到的参考文档内容
            question: 用户问题

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        user_prompt = cls.QA_USER_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        return cls.QA_SYSTEM_PROMPT, user_prompt

    @classmethod
    def build_rewrite_prompt(cls, query: str, max_sub_questions: int = 3) -> tuple:
        """
        构建查询改写 Prompt。

        Args:
            query:              原始查询
            max_sub_questions:  最大子问题数

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        user_prompt = cls.REWRITE_USER_PROMPT_TEMPLATE.format(
            query=query, max_sub_questions=max_sub_questions
        )
        return cls.REWRITE_SYSTEM_PROMPT, user_prompt

    @classmethod
    def build_verify_prompt(cls, context: str, answer: str) -> tuple:
        """
        构建答案校验 Prompt。

        Args:
            context: 参考文档内容
            answer:  AI 生成的回答

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        user_prompt = cls.VERIFY_USER_PROMPT_TEMPLATE.format(
            context=context, answer=answer
        )
        return cls.VERIFY_SYSTEM_PROMPT, user_prompt
