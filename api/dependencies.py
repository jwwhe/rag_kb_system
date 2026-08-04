"""
================================================================================
  API 接口层: 依赖注入
  管理各层组件的初始化和注入，确保架构解耦
================================================================================
"""
import importlib
from functools import lru_cache
from typing import Optional

from config.settings import Settings, get_settings_cached

# 使用 importlib 导入数字开头的包
_dp = importlib.import_module("1_doc_process")
DocumentLoader = _dp.DocumentLoader
PDFLoader = _dp.PDFLoader  # 兼容别名
TextSplitter = _dp.TextSplitter
Embedder = _dp.Embedder

_vs = importlib.import_module("2_vector_store")
get_vector_store = _vs.get_vector_store

_rt = importlib.import_module("3_retrieval")
QueryRewriter = _rt.QueryRewriter
VectorSearcher = _rt.VectorSearcher
BM25Searcher = _rt.BM25Searcher
HybridSearcher = _rt.HybridSearcher
MMRReranker = _rt.MMRReranker
Reranker = _rt.Reranker
MultiSourceFusioner = _rt.MultiSourceFusioner

_gn = importlib.import_module("4_generate")
LLMFactory = _gn.LLMFactory
get_llm = _gn.get_llm
PromptTemplates = _gn.PromptTemplates
RAGChain = _gn.RAGChain

_en = importlib.import_module("5_enhance")
CorrectiveRAG = _en.CorrectiveRAG
CitationTracer = _en.CitationTracer
WebFallback = _en.WebFallback

_ev = importlib.import_module("6_evaluation")
RAGEvaluator = _ev.RAGEvaluator
PureLLMComparator = _ev.PureLLMComparator


# ==================== 全局组件缓存 ====================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置"""
    return get_settings_cached()


# ==================== 组件工厂（惰性初始化）====================

_embedder: Optional[Embedder] = None
_bm25_searcher: Optional[BM25Searcher] = None
_reranker: Optional[Reranker] = None
_mmr_reranker: Optional[MMRReranker] = None
_multi_source_fusioner: Optional[MultiSourceFusioner] = None
_llm_factory: Optional[LLMFactory] = None
_corrective_rag: Optional[CorrectiveRAG] = None
_citation_tracer: Optional[CitationTracer] = None
_web_fallback: Optional[WebFallback] = None
_rag_evaluator: Optional["RAGEvaluator"] = None
_pure_llm_comparator: Optional["PureLLMComparator"] = None


def get_embedder() -> Embedder:
    """获取嵌入器实例"""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_bm25_searcher() -> BM25Searcher:
    """获取 BM25 检索器实例"""
    global _bm25_searcher
    if _bm25_searcher is None:
        _bm25_searcher = BM25Searcher()
    return _bm25_searcher


def get_reranker() -> Reranker:
    """获取 Reranker 实例"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def get_mmr_reranker() -> MMRReranker:
    """获取 MMR 重排器实例"""
    global _mmr_reranker
    if _mmr_reranker is None:
        _mmr_reranker = MMRReranker()
    return _mmr_reranker


def get_multi_source_fusioner() -> MultiSourceFusioner:
    """获取多源融合器实例"""
    global _multi_source_fusioner
    if _multi_source_fusioner is None:
        _multi_source_fusioner = MultiSourceFusioner()
    return _multi_source_fusioner


def get_llm_factory() -> LLMFactory:
    """获取 LLM 工厂实例"""
    global _llm_factory
    if _llm_factory is None:
        _llm_factory = LLMFactory()
    return _llm_factory


def get_corrective_rag() -> CorrectiveRAG:
    """获取纠正型 RAG 实例"""
    global _corrective_rag
    if _corrective_rag is None:
        _corrective_rag = CorrectiveRAG()
    return _corrective_rag


def get_citation_tracer() -> CitationTracer:
    """获取溯源追踪器实例"""
    global _citation_tracer
    if _citation_tracer is None:
        _citation_tracer = CitationTracer()
    return _citation_tracer


def get_web_fallback() -> WebFallback:
    """获取 Web 搜索兜底实例"""
    global _web_fallback
    if _web_fallback is None:
        _web_fallback = WebFallback()
    return _web_fallback


def get_rag_evaluator() -> "RAGEvaluator":
    """获取 RAG 评估器实例"""
    global _rag_evaluator
    if _rag_evaluator is None:
        _rag_evaluator = RAGEvaluator()
    return _rag_evaluator


def get_pure_llm_comparator() -> "PureLLMComparator":
    """获取纯 LLM 对比器实例"""
    global _pure_llm_comparator
    if _pure_llm_comparator is None:
        _pure_llm_comparator = PureLLMComparator(rag_evaluator=get_rag_evaluator())
    return _pure_llm_comparator


def ensure_bm25_index():
    """确保 BM25 索引已构建（从向量库拉取全量文档）。"""
    bm25 = get_bm25_searcher()
    vector_store = get_vector_store()
    all_docs = vector_store.get_all_documents()
    if all_docs:
        bm25.build_index(all_docs)
