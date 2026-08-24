"""
================================================================================
  API 接口层: 路由定义
  对外提供的全部 API 端点：
    1. PDF 批量上传入库
    2. 智能问答（带引用、带溯源）
    3. 知识库管理：列表、删文件、清空库
================================================================================
"""
import os
import uuid
import tempfile
import asyncio
import threading
import dataclasses
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from api.schemas import (
    QuestionRequest,
    QAResponse,
    FileUploadResponse,
    KBStatsResponse,
    DeleteResponse,
    FileDeleteRequest,
)
from api.dependencies import (
    get_settings,
    get_embedder,
    get_bm25_searcher,
    get_reranker,
    get_mmr_reranker,
    get_multi_source_fusioner,
    get_llm_factory,
    get_corrective_rag,
    get_citation_tracer,
    get_web_fallback,
    get_rag_evaluator,
    get_pure_llm_comparator,
    ensure_bm25_index,
    DocumentLoader,
    TextSplitter,
    QueryRewriter,
    VectorSearcher,
    BM25Searcher,
    HybridSearcher,
    MMRReranker,
    MultiSourceFusioner,
    RAGChain,
    get_vector_store,
)
from utils.response import success_response, error_response
from utils.exceptions import (
    DocumentProcessError,
    NoKnowledgeFoundError,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# 允许的文件扩展名（与 settings.api.allowed_extensions 对齐）
ALLOWED_EXTENSIONS = (".pdf", ".docx", ".md", ".markdown")


# ==================== 1. 多格式文档批量上传入库 ====================

# 上传处理全局串行锁：
# 各层组件为进程级单例（PG 单连接 / OCR 引擎 / BM25 索引），
# 且 CPU 密集处理（OCR/分块/向量化）不应与问答并发抢占，
# 故所有上传串行执行（上传本身低频且耗时，串行可接受）
_UPLOAD_LOCK = threading.Lock()


def _process_upload_sync(
    contents: List[bytes],
    filenames: List[str],
    source_type: str,
    knowledge_base: str,
    enable_ocr: bool,
    _lock: threading.Lock = None,
) -> List[dict]:
    """
    同步执行完整上传流水线（在独立线程中运行，避免阻塞事件循环）。
    返回 FileUploadResponse 的 dict 列表；异常统一转为 HTTPException。
    """
    if _lock is not None:
        # 递归一次：持锁运行实际处理（避免整体重缩进）
        with _lock:
            return _process_upload_sync(
                contents, filenames, source_type, knowledge_base, enable_ocr
            )
    settings = get_settings()

    # 初始化各层组件（本次请求指定 OCR 时按请求覆盖配置）
    loader = DocumentLoader()
    if enable_ocr:
        loader = DocumentLoader(
            dataclasses.replace(
                get_settings().doc_process,
                enable_ocr=True,
            )
        )
    splitter = TextSplitter()
    embedder = get_embedder()
    vector_store = get_vector_store()

    results = []
    for filename, content in zip(filenames, contents):
        tmp_path = None
        try:
            logger.info(
                f"接收文件: name={filename}, size={len(content)} bytes, "
                f"source_type={source_type}"
            )
            if not content or len(content) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"上传的文件为空: {filename}",
                )

            # 写入临时文件（保留原始扩展名，便于 loader 分发）
            ext = Path(filename).suffix.lower()
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
            os.close(tmp_fd)
            with open(tmp_path, "wb") as f:
                f.write(content)

            logger.info(f"临时文件已写入: {tmp_path} ({len(content)} bytes)")

            # Layer 1: 文档处理（带多源标注，含可选 OCR）
            documents = loader.load_single(
                tmp_path, source_type=source_type, knowledge_base=knowledge_base
            )
            if not documents:
                raise DocumentProcessError(f"文档解析无有效内容: {filename}")

            # 文本分块
            chunks = splitter.split_documents(documents)

            # 修正元数据：用原始文件名覆盖 loader 的临时文件名
            for chunk in chunks:
                chunk.metadata["file_name"] = filename

            # 向量化
            chunks = embedder.embed_documents(chunks)

            # Layer 2: 入库
            vector_store.add_documents(chunks)

            # 重建 BM25 索引
            bm25 = get_bm25_searcher()
            bm25.mark_dirty()
            ensure_bm25_index()

            results.append(FileUploadResponse(
                file_name=filename,
                pages=len(documents),
                chunks=len(chunks),
                message=f"上传并入库成功（来源: {source_type}）",
            ).model_dump())

            logger.info(
                f"文件上传成功: {filename} | "
                f"{len(documents)} 段 → {len(chunks)} 块 | 来源: {source_type}"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"文件上传失败 [{filename}]: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"文件 {filename} 处理失败: {type(e).__name__}: {str(e)}",
            )

        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    return results


@router.post(
    "/documents/upload",
    response_model=dict,
    tags=["文档管理"],
    summary="上传 PDF/Word/Markdown 文件并入库",
)
async def upload_documents(
    files: List[UploadFile] = File(...),
    source_type: str = Form("论文原文", description="来源类型: 论文原文/综述解读/实验笔记"),
    knowledge_base: str = Form("default", description="所属知识库标识"),
    enable_ocr: bool = Form(False, description="启用 OCR 识别 PDF 图片/扫描件（仅本次上传生效）"),
):
    """
    批量上传文件，完成：
    1. 多格式文本提取（PDF / Word / Markdown，可选 OCR 识别 PDF 图片/扫描件）
    2. 零宽断言分块（chunk_size=800, overlap=150）
    3. bge-m3 向量化
    4. 存入 PGvector 向量数据库（带 source_type 多源标注）

    说明：OCR/分块/向量化耗时较长（扫描件每页约数秒），
    处理在后台线程执行，客户端需耐心等待；期间其他接口不受影响。
    """
    # 校验文件格式
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"仅支持 {ALLOWED_EXTENSIONS} 格式文件: {file.filename}",
            )

    # 先异步读完所有文件（网络 I/O，不占用处理线程）
    contents, filenames = [], []
    for file in files:
        await file.seek(0)
        contents.append(await file.read())
        filenames.append(file.filename)

    # 重活移入线程池，避免阻塞事件循环（其他接口仍可用）
    results = await asyncio.to_thread(
        _process_upload_sync,
        contents,
        filenames,
        source_type,
        knowledge_base,
        enable_ocr,
        _lock=_UPLOAD_LOCK,
    )

    return success_response(
        data=results,
        message=f"成功上传 {len(results)} 个文件",
    ).to_dict()


# ==================== 2. 智能问答接口 ====================

@router.post(
    "/qa/ask",
    response_model=dict,
    tags=["智能问答"],
    summary="智能问答（完整 RAG 链路）",
)
async def ask_question(request: QuestionRequest):
    """
    完整五层 RAG 链路问答：
    1. 查询改写
    2. 向量相似度检索 + BM25 关键词检索
    3. 混合检索融合
    4. Rerank 重排
    5. LLM 生成
    6. 纠正型 RAG 自检
    7. 全文溯源
    8. Web 搜索兜底（知识库无结果时）
    """
    settings = get_settings()
    question = request.question.strip()
    # 多轮对话 session_id（RunnableWithMessageHistory 使用）
    session_id = getattr(request, "session_id", None) or str(uuid.uuid4())

    # 初始化各层组件
    embedder = get_embedder()
    vector_store = get_vector_store()
    bm25 = get_bm25_searcher()
    reranker = get_reranker()
    mmr_reranker = get_mmr_reranker()
    llm_factory = get_llm_factory()
    corrective_rag = get_corrective_rag()
    citation_tracer = get_citation_tracer()
    web_fallback = get_web_fallback()

    llm_call = llm_factory.get_llm_callable()

    # ====== Layer 3: 检索（两阶段：MMR 去重 → Reranker 精排）======

    # Step 1: 查询改写
    rewriter = QueryRewriter()
    rewritten_queries = rewriter.rewrite(question, llm_call)

    # 使用改写后的主查询进行检索
    main_query = rewritten_queries[1] if len(rewritten_queries) > 1 else rewritten_queries[0]

    # Step 2: 向量语义检索（启用 MMR 时走两阶段，否则回退到普通检索）
    query_embedding = embedder.embed_query(main_query)
    if settings.retrieval.enable_mmr:
        # 两阶段检索第一阶段：MMR 多样性去重
        vector_results = mmr_reranker.mmr_search(
            query_embedding=query_embedding,
            vector_store=vector_store,
            top_k=settings.retrieval.vector_top_k,
            fetch_k=settings.retrieval.mmr_fetch_k,
            lambda_mult=settings.retrieval.mmr_lambda,
        )
    else:
        vector_searcher = VectorSearcher()
        vector_results = vector_searcher.search(query_embedding, vector_store)

    # Step 3: BM25 关键词检索
    # 对每个改写查询执行 BM25，合并结果
    all_bm25_results = []
    for q in rewritten_queries[:2]:  # 最多用 2 个查询变体
        try:
            bm25_results = bm25.search(q)
            all_bm25_results.extend(bm25_results)
        except Exception as e:
            logger.warning(f"BM25 检索跳过 '{q[:30]}...': {e}")

    # BM25 去重
    seen_ids = set()
    unique_bm25 = []
    for doc in all_bm25_results:
        cid = doc.metadata.get("chunk_id", "")
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_bm25.append(doc)

    # Step 4: 混合检索融合
    hybrid_searcher = HybridSearcher()
    merged_results = hybrid_searcher.merge(vector_results, unique_bm25)

    # Step 5: Rerank 重排
    reranked_results = reranker.rerank(main_query, merged_results)

    # Step 6: 多源知识融合（论文原文/综述解读/实验笔记 按配额融合）
    multi_source_fusioner = get_multi_source_fusioner()
    reranked_results = multi_source_fusioner.fuse(reranked_results)

    is_from_web = False
    final_answer = ""

    # 判断是否需要触发 Web 兜底：
    # 条件1：重排后无结果
    # 条件2：重排后最高分低于阈值（知识库无有效匹配）
    top_score = (
        reranked_results[0].metadata.get("rerank_score", 0)
        if reranked_results else 0
    )
    kb_has_valid_result = (
        len(reranked_results) > 0
        and top_score >= settings.retrieval.similarity_threshold
    )

    # ====== Layer 4 + Layer 5: 生成 + 增强 ======

    if kb_has_valid_result:
        # ====== 本地知识库回答 ======
        rag_chain = RAGChain(llm_factory)

        # 生成答案（传入 session_id 启用多轮对话）
        answer = rag_chain.generate(
            question, reranked_results, llm_call, session_id=session_id
        )

        # 纠正型 RAG 自检
        corrected_answer = corrective_rag.correct(
            answer, reranked_results, llm_call
        )

        final_answer = corrected_answer

        # 溯源
        citations = citation_tracer.build_citations(
            reranked_results, source_type="internal"
        )
        full_response = citation_tracer.build_full_response(
            final_answer, citations, is_from_web=False
        )

    else:
        # ====== Web 搜索兜底 ======
        reason = "无检索结果" if not reranked_results else f"最高相似度 {top_score:.3f} 低于阈值 {settings.retrieval.similarity_threshold}"
        logger.info(f"触发 Web 搜索兜底: {reason}")

        web_results = web_fallback.search(question)

        if web_results:
            # Web 生成回答
            final_answer = web_fallback.build_web_answer(
                question, web_results, llm_call
            )
            is_from_web = True

            # Web 结果溯源
            web_docs = web_fallback.web_results_to_documents(web_results)
            citations = citation_tracer.build_citations(
                web_docs, source_type="external"
            )
        else:
            # 完全无结果
            final_answer = "知识库中暂无该相关资料，无法解答此问题"
            citations = []

        full_response = citation_tracer.build_full_response(
            final_answer, citations, is_from_web=is_from_web
        )

    # 构建结构化响应
    return success_response(
        data=full_response,
        message="问答完成",
    ).to_dict()


# ==================== 3. 知识库管理 ====================

@router.get(
    "/kb/stats",
    response_model=dict,
    tags=["知识库管理"],
    summary="获取知识库统计信息",
)
async def get_kb_stats():
    """获取知识库统计信息：文档数、文件列表等。"""
    vector_store = get_vector_store()
    stats = vector_store.get_collection_stats()
    return success_response(data=stats).to_dict()


@router.delete(
    "/kb/files",
    response_model=dict,
    tags=["知识库管理"],
    summary="删除指定文件的向量数据",
)
async def delete_file(request: FileDeleteRequest):
    """按文件名删除该文件的所有向量块。"""
    vector_store = get_vector_store()
    bm25 = get_bm25_searcher()

    deleted_count = vector_store.delete_by_file(request.file_name)

    if deleted_count > 0:
        bm25.mark_dirty()
        ensure_bm25_index()

    return success_response(
        data=DeleteResponse(
            deleted_count=deleted_count,
            file_name=request.file_name,
            message=f"已删除 {deleted_count} 个向量块",
        ).model_dump(),
    ).to_dict()


@router.delete(
    "/kb/clear",
    response_model=dict,
    tags=["知识库管理"],
    summary="清空知识库",
)
async def clear_kb():
    """清空当前知识库所有数据（不可恢复）。"""
    vector_store = get_vector_store()
    bm25 = get_bm25_searcher()

    vector_store.clear_collection()
    bm25.mark_dirty()

    return success_response(
        message="知识库已清空",
    ).to_dict()


# ==================== 4. 评估体系 ====================

@router.post(
    "/eval/rag",
    response_model=dict,
    tags=["评估体系"],
    summary="RAG 检索与生成质量评估（Recall@K / MRR / 幻觉率）",
)
async def eval_rag(dataset: List[dict]):
    """
    对 RAG 链路跑量化评估。
    入参：评测集列表，每条形如
      {
        "question": "...",
        "relevant_doc_ids": ["chunk_id_1", ...],
        "expected_keywords": ["关键词1", ...]
      }
    """
    evaluator = get_rag_evaluator()
    report = evaluator.evaluate(dataset, enable_generation=True)
    return success_response(data=report, message="RAG 评估完成").to_dict()


@router.post(
    "/eval/compare",
    response_model=dict,
    tags=["评估体系"],
    summary="RAG vs 纯 LLM 对比评估",
)
async def eval_compare(dataset: List[dict]):
    """
    对同一评测集分别跑纯 LLM 和 RAG，输出准确率/幻觉率对比。
    """
    comparator = get_pure_llm_comparator()
    report = comparator.compare(dataset)
    return success_response(data=report, message="对比评估完成").to_dict()
