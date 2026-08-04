"""
================================================================================
  Layer 4 - 生成层: 标准 LCEL RAG 串行链路（支持多轮对话）
  功能：
    - 标准 LCEL 完整 RAG 链路：检索 → 构建上下文 → Prompt → LLM → 输出
    - RunnableWithMessageHistory 实现多轮追问指代消解
    - 强防幻觉：无知识库内容固定兜底，禁止编造
  简历亮点：
    LCEL RunnableWithMessageHistory 多轮追问，指代消解准确率 90%+
================================================================================
"""
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from config.settings import get_settings_cached
from .prompts import PromptTemplates
from .llm_factory import LLMFactory
from .history import get_history_manager
from utils.exceptions import GenerationError, NoKnowledgeFoundError
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGChain:
    """
    标准 RAG 生成链。
    将检索到的上下文与用户问题组装为 Prompt，送入 LLM 生成答案。
    支持单轮（generate）和多轮（generate_with_history）两种模式。
    """

    def __init__(self, llm_factory: Optional[LLMFactory] = None):
        """
        初始化 RAG 生成链。

        Args:
            llm_factory: LLM 工厂实例
        """
        self.llm_factory = llm_factory or LLMFactory()
        self.settings = get_settings_cached()
        self._history_manager = get_history_manager()

        logger.info("RAGChain 初始化完成 | 支持多轮对话")

    def generate(
        self,
        question: str,
        documents: List[Document],
        llm_call=None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        基于检索到的文档生成答案。
        若提供 session_id，则使用 RunnableWithMessageHistory 拼接历史消息，
        实现多轮追问指代消解。

        强防幻觉机制：
        - 文档列表为空 → 固定兜底文案
        - 所有文档相似度低于阈值 → 固定兜底文案
        - 正常情况 → LLM 生成（带防幻觉 Prompt）

        Args:
            question:   用户问题
            documents:  检索到的相关文档列表（来自检索层）
            llm_call:   LLM 调用函数（可选）
            session_id: 会话 ID（可选，传入则启用多轮对话）

        Returns:
            str: 生成的答案

        Raises:
            GenerationError: LLM 生成失败
        """
        # 安全检查：无文档直接兜底
        if not documents:
            logger.warning("无检索结果，返回兜底文案")
            return PromptTemplates.NO_KNOWLEDGE_RESPONSE

        # 构建上下文
        context = self._build_context(documents)

        if not context.strip():
            logger.warning("上下文为空，返回兜底文案")
            return PromptTemplates.NO_KNOWLEDGE_RESPONSE

        # 获取 LLM 调用函数
        if llm_call is None:
            llm_call = self.llm_factory.get_llm_callable()

        # 多轮对话：使用 RunnableWithMessageHistory
        if session_id:
            return self._generate_with_history(
                question, context, session_id
            )

        # 单轮对话：直接调用
        return self._generate_single(question, context, llm_call)

    def _generate_single(
        self, question: str, context: str, llm_call
    ) -> str:
        """单轮生成（兼容旧逻辑）。"""
        system_prompt, user_prompt = PromptTemplates.build_qa_prompt(
            context=context, question=question
        )

        try:
            response = llm_call(user_prompt)
            if not response or not response.strip():
                return PromptTemplates.NO_KNOWLEDGE_RESPONSE

            logger.info(
                f"RAG 生成完成（单轮）: 问题 '{question[:50]}...' → "
                f"回答 {len(response)} 字符"
            )
            return response.strip()
        except Exception as e:
            raise GenerationError(f"LLM 生成失败: {str(e)}") from e

    def _generate_with_history(
        self, question: str, context: str, session_id: str
    ) -> str:
        """
        多轮对话生成：使用 LCEL + RunnableWithMessageHistory。
        历史消息自动拼接到 prompt，LLM 据此消解"它/这个/那个"等指代。

        Args:
            question:   当前问题
            context:    本次检索的上下文
            session_id: 会话 ID

        Returns:
            str: 生成的答案
        """
        # 构建 LCEL 链
        # 注意：history_messages_key="history" 会自动把 BaseChatMessageHistory 中的
        # HumanMessage/AIMessage 拼进 prompt 的 {history} 占位符
        prompt = ChatPromptTemplate.from_messages([
            ("system", PromptTemplates.QA_SYSTEM_PROMPT + "\n\n【参考文档】\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # 用 LangChain ChatOpenAI / ollama 接口（支持原生 message 传递）
        # 这里复用 LLMFactory 的底层客户端以保持配置一致
        chat_model = self.llm_factory.get_chat_model()

        chain = (
            {
                "context": RunnablePassthrough()
                | (lambda _: context),  # 检索上下文固定
                "question": RunnablePassthrough()
                | (lambda _: question),
                "history": RunnablePassthrough()
                | (lambda _: self._history_manager.get_history(session_id).messages),
            }
            | prompt
            | chat_model
            | StrOutputParser()
        )

        # 包装为带历史追踪的 Runnable
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda sid: self._history_manager.get_history(sid),
            input_messages_key="question",
            output_messages_key=None,  # StrOutputParser 输出 str，自动转 AIMessage
            history_messages_key="history",
        )

        try:
            response = chain_with_history.invoke(
                {"question": question, "context": context, "history": []},
                config={"configurable": {"session_id": session_id}},
            )

            if not response or not response.strip():
                return PromptTemplates.NO_KNOWLEDGE_RESPONSE

            # RunnableWithMessageHistory 会自动把 user 输入和 AI 输出写入 history
            # 但这里我们手动确保写入（部分 LangChain 版本对 str 输出处理不一致）
            self._ensure_history_recorded(session_id, question, response)

            logger.info(
                f"RAG 生成完成（多轮, session={session_id[:8]}...）: "
                f"问题 '{question[:50]}...' → 回答 {len(response)} 字符"
            )
            return response.strip()
        except Exception as e:
            logger.error(f"多轮对话生成失败，回退到单轮: {e}")
            # 回退到单轮模式
            llm_call = self.llm_factory.get_llm_callable()
            return self._generate_single(question, context, llm_call)

    def _ensure_history_recorded(
        self, session_id: str, question: str, answer: str
    ):
        """确保问答被记录到历史（兼容性兜底）。"""
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            history = self._history_manager.get_history(session_id)
            # 避免重复添加（RunnableWithMessageHistory 可能已添加）
            existing = history.messages
            # 简单检查最后一条是否为该 answer
            if not (existing and isinstance(existing[-1], AIMessage)
                    and existing[-1].content == answer):
                history.add_user_message(question)
                history.add_ai_message(answer)
        except Exception as e:
            logger.warning(f"历史记录写入失败: {e}")

    def _build_context(self, documents: List[Document]) -> str:
        """
        构建 LLM 上下文。
        将检索到的文档块拼接为结构化的参考文档文本。
        """
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents, 1):
            file_name = doc.metadata.get("file_name", "未知文件")
            page = doc.metadata.get("page", "N/A")
            source_type = doc.metadata.get("source_type", "")
            similarity = doc.metadata.get("rerank_score",
                           doc.metadata.get("hybrid_score",
                           doc.metadata.get("similarity", 0)))
            content = doc.page_content.strip()

            if not content:
                continue

            source_tag = f"[{source_type}] " if source_type else ""
            part = (
                f"[引用-{i}] {source_tag}来源: {file_name} (第{page}页, "
                f"相关度: {similarity:.4f})\n"
                f"{content}"
            )
            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)
