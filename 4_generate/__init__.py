# 4_generate - 生成层 (Layer 4)
# 职责：多LLM统一封装 → 三套固定Prompt → LCEL RAG串行链路 → 多轮对话历史
# 依赖：config + utils + Layer 3 (retrieval results)

from .llm_factory import LLMFactory, get_llm
from .prompts import PromptTemplates
from .rag_chain import RAGChain
from .history import SessionHistoryManager, get_history_manager

__all__ = ["LLMFactory", "get_llm", "PromptTemplates", "RAGChain",
           "SessionHistoryManager", "get_history_manager"]
