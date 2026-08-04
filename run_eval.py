"""
================================================================================
  一键评估脚本
  流程：
    1. 入库 data/samples/ 下的 3 份样例文档（论文/综述/笔记）
    2. 拉取所有 chunk_id，按关键词动态构建评测集
    3. 跑 RAG 评估（Recall@K / MRR / 幻觉率）
    4. 跑 RAG vs 纯 LLM 对比（准确率提升 X 个百分点）
    5. 保存报告到 data/eval_report.json
  用法：
    python run_eval.py                        # 用默认向量库
    python run_eval.py --store chroma         # 强制用 Chroma（开发环境）
    python run_eval.py --no-generation        # 只评检索，不跑生成（省时）
================================================================================
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Windows 控制台 GBK 编码兜底，避免 emoji 输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 把项目根目录加入 sys.path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# 优先使用本地模型缓存，避免联网校验失败
# （bge-m3 / bge-reranker 已下载到 ~/.cache/huggingface/hub）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def parse_args():
    p = argparse.ArgumentParser(description="RAG 系统一键评估")
    p.add_argument(
        "--store",
        choices=["pgvector", "chroma"],
        default=None,
        help="强制指定向量库类型（默认读 settings）",
    )
    p.add_argument(
        "--no-generation",
        action="store_true",
        help="只评估检索指标，不跑生成（节省 LLM 调用）",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="跳过入库步骤（已有数据时使用）",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="入库前先清空向量库（避免历史数据干扰）",
    )
    p.add_argument(
        "--samples-dir",
        default=str(ROOT / "data" / "samples"),
        help="样例文档目录",
    )
    return p.parse_args()


# ==================== 1. 入库样例文档 ====================
def upload_samples(samples_dir: str):
    """把样例文档按 source_type 入库。"""
    from api.dependencies import (
        DocumentLoader, TextSplitter, get_embedder,
        get_vector_store, ensure_bm25_index, get_bm25_searcher,
    )

    samples = Path(samples_dir)
    if not samples.exists():
        raise FileNotFoundError(f"样例目录不存在: {samples}")

    # 文件名 → source_type 映射
    source_map = {
        "论文原文": "论文原文",
        "综述解读": "综述解读",
        "实验笔记": "实验笔记",
    }

    loader = DocumentLoader()
    splitter = TextSplitter()
    embedder = get_embedder()
    vector_store = get_vector_store()

    total_chunks = 0
    for md_file in sorted(samples.glob("*.md")):
        # 推断 source_type
        st = "论文原文"
        for key, val in source_map.items():
            if key in md_file.name:
                st = val
                break

        print(f"  入库: {md_file.name} (source_type={st})")
        documents = loader.load_single(
            str(md_file), source_type=st, knowledge_base="eval"
        )
        if not documents:
            print(f"    ⚠️  解析无内容，跳过")
            continue

        chunks = splitter.split_documents(documents)
        chunks = embedder.embed_documents(chunks)
        vector_store.add_documents(chunks)
        total_chunks += len(chunks)
        print(f"    ✓ 生成 {len(chunks)} 个 chunk")

    # 重建 BM25 索引
    bm25 = get_bm25_searcher()
    bm25.mark_dirty()
    ensure_bm25_index()

    print(f"\n✓ 入库完成，共 {total_chunks} 个 chunk\n")
    return total_chunks


# ==================== 2. 动态构建评测集 ====================
# 评测问题模板：question + 匹配关键词（用于找 relevant_doc_ids）+ 预期答案关键词
EVAL_TEMPLATE = [
    {
        "question": "Transformer 模型使用了多少层编码器和解码器？",
        "match_keywords": ["6 层", "6层", "编码器", "解码器"],
        "expected_keywords": ["6", "编码器", "解码器"],
    },
    {
        "question": "缩放点积注意力公式中为什么要除以 sqrt(d_k)？",
        "match_keywords": ["sqrt", "饱和", "梯度消失", "方差"],
        "expected_keywords": ["梯度", "饱和", "softmax"],
    },
    {
        "question": "Transformer 在 WMT 2014 英德翻译上取得了多少 BLEU 分数？",
        "match_keywords": ["28.4", "BLEU", "英德"],
        "expected_keywords": ["28.4", "BLEU"],
    },
    {
        "question": "多头注意力机制使用了多少个头？",
        "match_keywords": ["h=8", "8 个头", "多头"],
        "expected_keywords": ["8", "多头"],
    },
    {
        "question": "复现实验中学习率 warmup_steps 设置为多少？",
        "match_keywords": ["warmup_steps=4000", "4000", "warmup"],
        "expected_keywords": ["4000", "warmup"],
    },
    {
        "question": "位置编码使用什么函数生成？",
        "match_keywords": ["正弦", "余弦", "sin", "cos", "位置编码"],
        "expected_keywords": ["正弦", "余弦", "sin", "cos"],
    },
    {
        "question": "复现实验中漏掉 sqrt(d_k) 会导致什么问题？",
        "match_keywords": ["收敛", "BLEU", "sqrt"],
        "expected_keywords": ["收敛", "BLEU"],
    },
    {
        "question": "Transformer 的训练成本相比之前模型降低了多少？",
        "match_keywords": ["1/4", "训练成本", "P100"],
        "expected_keywords": ["1/4", "四分之一", "成本"],
    },
    {
        "question": "位置编码的维度必须与什么一致？",
        "match_keywords": ["d_model", "维度", "512"],
        "expected_keywords": ["d_model", "512"],
    },
    {
        "question": "RNN 的核心问题是什么？",
        "match_keywords": ["并行", "梯度消失", "序列依赖", "RNN"],
        "expected_keywords": ["并行", "梯度", "依赖"],
    },
]


def build_dataset():
    """从向量库拉取所有 chunk，按关键词匹配构建评测集。"""
    from api.dependencies import get_vector_store

    vector_store = get_vector_store()
    all_docs = vector_store.get_all_documents()

    print(f"向量库共 {len(all_docs)} 个 chunk")

    dataset = []
    for item in EVAL_TEMPLATE:
        # 找出 content 包含任一 match_keyword 的 chunk_id 作为 relevant
        relevant_ids = []
        for doc in all_docs:
            content = doc.page_content
            if any(kw in content for kw in item["match_keywords"]):
                relevant_ids.append(
                    doc.metadata.get("chunk_id", str(id(doc)))
                )

        dataset.append({
            "question": item["question"],
            "relevant_doc_ids": relevant_ids,
            "expected_keywords": item["expected_keywords"],
            "match_count": len(relevant_ids),
        })

    # 打印预览
    print("\n评测集预览：")
    for i, d in enumerate(dataset, 1):
        print(f"  Q{i}: {d['question']}")
        print(f"     relevant chunk 数: {d['match_count']}")

    return dataset


# ==================== 3. 跑评估 ====================
def run_evaluation(dataset, enable_generation: bool):
    """跑 RAG 评估 + RAG vs 纯 LLM 对比。"""
    from api.dependencies import get_rag_evaluator, get_pure_llm_comparator

    reports = {}

    # 3.1 RAG 评估
    print("\n" + "=" * 60)
    print("【1】RAG 检索 + 生成质量评估")
    print("=" * 60)
    evaluator = get_rag_evaluator()
    reports["rag"] = evaluator.evaluate(
        dataset, enable_generation=enable_generation
    )
    print(f"\nRAG 评估结果:")
    print(json.dumps(reports["rag"], ensure_ascii=False, indent=2))

    # 3.2 RAG vs 纯 LLM 对比
    if enable_generation:
        print("\n" + "=" * 60)
        print("【2】RAG vs 纯 LLM 对比")
        print("=" * 60)
        comparator = get_pure_llm_comparator()
        reports["compare"] = comparator.compare(dataset)
        print(f"\n对比结果:")
        print(json.dumps(reports["compare"], ensure_ascii=False, indent=2))

    return reports


# ==================== 主流程 ====================
def main():
    args = parse_args()

    # 强制指定向量库
    if args.store:
        os.environ["STORE_TYPE"] = args.store
        print(f"⚠️  强制使用向量库: {args.store}")

    print("=" * 60)
    print("  RAG 系统一键评估")
    print("=" * 60)

    # Step 1: 入库
    if not args.skip_upload:
        if args.clear:
            print("\n【Step 0】清空向量库...")
            from api.dependencies import get_vector_store, get_bm25_searcher
            vs = get_vector_store()
            vs.clear_collection()
            bm25 = get_bm25_searcher()
            bm25.mark_dirty()
            print("✓ 向量库已清空\n")

        print("\n【Step 1】入库样例文档...")
        upload_samples(args.samples_dir)
    else:
        print("\n【Step 1】跳过入库（使用已有数据）")

    # Step 2: 构建评测集
    print("\n【Step 2】构建评测集...")
    dataset = build_dataset()
    if not dataset:
        print("❌ 评测集为空，退出")
        return

    # 保存评测集
    dataset_path = ROOT / "data" / "eval_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 评测集已保存: {dataset_path}")

    # Step 3: 跑评估
    enable_gen = not args.no_generation
    print(f"\n【Step 3】运行评估（生成评估: {'开启' if enable_gen else '关闭'}）...")
    reports = run_evaluation(dataset, enable_generation=enable_gen)

    # Step 4: 保存报告
    report_path = ROOT / "data" / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ 评估报告已保存: {report_path}")
    print("=" * 60)

    # 打印核心指标摘要
    if "compare" in reports:
        cmp = reports["compare"]
        if "improvement" in cmp:
            gain = cmp["improvement"].get("accuracy_gain_percentage_points", 0)
            print(f"\n🎯 核心结论：RAG 相比纯 LLM 准确率提升 {gain} 个百分点")
            print(f"   （简历可写：RAG vs 纯 LLM 准确率提升约 {gain} 个百分点）")

    if "rag" in reports and "retrieval" in reports["rag"]:
        r = reports["rag"]["retrieval"]
        print(f"\n📊 检索指标：")
        for k, v in r.items():
            print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    main()
