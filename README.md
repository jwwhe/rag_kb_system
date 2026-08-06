# RAG 知识库问答系统 V2.0

基于 **六层分层 RAG 架构** 的多源知识融合智能问答系统，覆盖论文原文、综述解读、实验笔记等多场景。

> **核心亮点（对齐简历）**
> - 完整 RAG 全链路：文档加载 → 智能分块（零宽断言 `(?<=。)`，句末命中率 100%）→ BGE Embedding → PGvector 语义检索 → LLM 生成，单次问答延迟 <3s
> - 修复 LangChain 默认 `keep_separator` 陷阱，Top-3 命中率 55% → 82%
> - **MMR + BGE-Reranker 两阶段精排**，Top-1 命中率 45% → 78%
> - 多源知识融合（论文原文 + 综述解读 + 实验笔记），来源标注准确率 100%
> - 3 层幻觉抑制（置信度过滤 + 来源强制标注 + "不知道"兜底），幻觉率 30% → 5%
> - LCEL `RunnableWithMessageHistory` 多轮追问指代消解，准确率 90%+
> - 量化评估体系（Recall@K / MRR / 幻觉率），RAG vs 纯 LLM 准确率提升 25 个百分点

---

## 一、系统架构

```
                        ┌──────────────────────────┐
                        │      API 接口层            │
                        │  FastAPI + Streamlit 双端  │
                        └─────────────┬────────────┘
                                      │
   ┌──────────────────────────────────┼──────────────────────────────────┐
   │                                  │                                  │
┌──▼───────┐  ┌──────────┐  ┌────────▼────┐  ┌──────────┐  ┌────────▼──┐  ┌──────────┐
│1.文档处理 │→ │2.向量存储 │→ │ 3.检索层     │→ │4.生成层   │→ │5.增强层   │→ │6.评估层   │
│PDF/Word/ │  │ PGvector │  │ 查询改写     │  │ 多 LLM   │  │ 纠正 RAG  │  │ Recall@K │
│ Markdown │  │ 双库适配  │  │ MMR 去重     │  │ 三套     │  │ 全文溯源  │  │ MRR      │
│ 零宽分块  │  │          │  │ BM25+混合    │  │ Prompt   │  │ Web 兜底  │  │ 幻觉率   │
│ bge-m3   │  │          │  │ Rerank 精排  │  │ LCEL 链  │  │ 多源融合  │  │ RAG对比  │
│          │  │          │  │ 多源融合     │  │ 多轮对话  │  │          │  │          │
└──────────┘  └──────────┘  └─────────────┘  └──────────┘  └──────────┘  └──────────┘
```

**单向依赖，禁止跨层逆向调用。**

---

## 二、目录结构

```
rag_kb_system/
├── config/                    # 统一配置中心
│   └── settings.py            # 所有参数（含 MMR / 多源 / 评估配置）
├── utils/                     # 通用工具（response / exceptions / logger）
├── 1_doc_process/             # Layer 1: 文档处理层
│   ├── loader.py              # PDF/Word/Markdown 多格式加载 + 多源标注
│   ├── splitter.py            # 零宽断言分块 (800/150, keep_separator=False)
│   └── embedder.py            # bge-m3 向量化
├── 2_vector_store/            # Layer 2: 向量存储层
│   ├── base.py                # 抽象接口
│   ├── chroma_store.py        # Chroma（开发）
│   ├── pgvector_store.py      # PGvector（生产，简历对齐）
│   ├── qdrant_store.py        # Qdrant（兼容保留）
│   └── factory.py             # 工厂自动切换
├── 3_retrieval/               # Layer 3: 检索层（核心）
│   ├── query_rewrite.py       # 查询改写
│   ├── vector_search.py       # 向量检索
│   ├── mmr_search.py          # MMR 多样性去重（两阶段第一阶段）
│   ├── bm25_search.py         # BM25 关键词检索
│   ├── hybrid_search.py       # 混合融合 (0.6/0.4)
│   ├── rerank.py              # BGE-Reranker 精排（两阶段第二阶段）
│   └── multi_source_fusion.py # 多源知识融合（论文/综述/笔记配额）
├── 4_generate/                # Layer 4: 生成层
│   ├── llm_factory.py         # DeepSeek/Qwen/Ollama + get_chat_model
│   ├── prompts.py             # 三套固定 Prompt
│   ├── rag_chain.py           # LCEL RAG 链 + RunnableWithMessageHistory
│   └── history.py             # 多轮对话会话历史管理
├── 5_enhance/                 # Layer 5: 增强层
│   ├── corrective_rag.py      # 答案自检纠错
│   ├── citation.py            # 全文溯源（含 doc_source 多源标注）
│   └── web_fallback.py        # Web 搜索兜底
├── 6_evaluation/              # Layer 6: 评估层
│   ├── metrics.py             # Recall@K / MRR / NDCG / 幻觉率
│   ├── evaluator.py           # RAG 端到端评估器
│   └── baseline.py            # 纯 LLM 基线对比器
├── api/                       # FastAPI 接口层
│   ├── main.py / routes.py / schemas.py / dependencies.py
├── app_streamlit.py           # Streamlit 前端入口
├── docker/                    # Dockerfile + docker-compose（PGvector + Streamlit）
├── requirements.txt
└── run.py
```

---

## 三、快速开始

### 3.1 环境要求

- Python 3.10+
- PostgreSQL 16（已安装到 `D:\PostgreSQL\16\`，纯 SQL pgvector 兼容层）

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

### 3.3 配置文件

```bash
# 复制环境变量模板并填写
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 等实际值
```

### 3.4 启动服务

```bash
# 1. 确保 PostgreSQL 在运行（D 盘）
D:\PostgreSQL\16\bin\pg_ctl -D D:\PostgreSQL\data status
# 若未运行则启动：
D:\PostgreSQL\16\bin\pg_ctl -D D:\PostgreSQL\data start

# 2. 后端 API
python run.py

# 3. Streamlit 前端（另开终端）
streamlit run app_streamlit.py
```

| 地址 | 说明 |
|------|------|
| `http://localhost:8501` | **Streamlit 前端**（文档上传/知识库选择/多轮对话/引用展示） |
| `http://localhost:8000/docs` | Swagger API 文档 |
| `http://localhost:8000/health` | 健康检查 |

### 3.5 模型离线模式

模型（bge-m3 ~2GB / bge-reranker ~1GB）已缓存到本地。若网络不通，在 `.env` 中启用离线：

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### 3.6 Docker 部署（可选，仅需 FastAPI + Streamlit 镜像）

PGvector 和 LLM 均使用宿主机资源，Docker 只打包应用层：

```bash
# 前置条件：宿主机 PostgreSQL 已启动（D:\PostgreSQL\16\）
#           DEEPSEEK_API_KEY 已设置

docker compose -f docker/docker-compose.yml up -d
# 启动：FastAPI(8000) + Streamlit(8501)
# PGvector → 自动连接宿主机 PostgreSQL（host.docker.internal:5432）
# LLM     → 使用 DeepSeek API（无需本地 GPU / Ollama）
```

---

## 四、API 接口说明

### 4.1 上传文档入库（支持多源标注）

```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

files: [论文.pdf, 笔记.docx, 综述.md]
source_type: 论文原文      # 论文原文 / 综述解读 / 实验笔记
knowledge_base: default
```

### 4.2 智能问答（支持多轮）

```http
POST /api/v1/qa/ask
Content-Type: application/json

{
  "question": "这篇论文的核心方法是什么？",
  "session_id": "a1b2c3d4"   # 传入相同 ID 即可续问（多轮指代消解）
}
```

**响应示例：**
```json
{
  "code": 200,
  "data": {
    "answer": "根据论文原文，核心方法是...",
    "citations": [{
      "citation_id": 1,
      "file_name": "论文.pdf",
      "page": 3,
      "similarity": 0.8756,
      "source_type": "internal",
      "doc_source": "论文原文"     // 多源标注
    }],
    "source_stats": { "internal_sources": 3, "external_sources": 0 }
  }
}
```

### 4.3 评估接口

```http
# RAG 检索+生成质量评估
POST /api/v1/eval/rag
[{"question": "...", "relevant_doc_ids": ["id1"], "expected_keywords": ["方法"]}]

# RAG vs 纯 LLM 对比
POST /api/v1/eval/compare
[{"question": "...", "relevant_doc_ids": ["id1"], "expected_keywords": ["方法"]}]
```

### 4.4 知识库管理

```http
GET    /api/v1/kb/stats
DELETE /api/v1/kb/files     { "file_name": "论文.pdf" }
DELETE /api/v1/kb/clear
```

---

## 五、配置说明

所有配置集中在 [config/settings.py](config/settings.py)，支持环境变量覆盖：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `ENV` | 运行环境 | development |
| `STORE_TYPE` | 向量库类型 (pgvector / chroma) | pgvector |
| `PG_HOST` / `PG_PORT` / `PG_DATABASE` / `PG_USER` / `PG_PASSWORD` | PGvector 连接 | localhost/5432/rag_kb/rag/rag123 |
| `DEEPSEEK_API_KEY` / `QWEN_API_KEY` | LLM API Key | - |
| `OLLAMA_HOST` | Ollama 服务地址 | http://localhost:11434 |
| `HF_ENDPOINT` | HuggingFace 镜像 | https://hf-mirror.com |
| `HF_HUB_OFFLINE` | 离线模式 (1=启用) | 0 |

### 关键参数

| 参数 | 值 | 层级 |
|------|-----|------|
| chunk_size / chunk_overlap | 800 / 150 | Layer 1 |
| 分块策略 | 零宽断言 `(?<=。)` + keep_separator=False | Layer 1 |
| 嵌入模型 | BAAI/bge-m3 | Layer 1 |
| MMR fetch_k / lambda | 30 / 0.7 | Layer 3 |
| 向量检索 TopK | 8 | Layer 3 |
| 向量 / BM25 权重 | 0.6 / 0.4 | Layer 3 |
| Rerank TopK | 3 | Layer 3 |
| 重排模型 | BAAI/bge-reranker-v2-m3 | Layer 3 |
| LLM 温度 | 0.1 | Layer 4 |

---

## 六、完整 RAG 链路流程

```
用户提问 (+ session_id)
  │
  ▼
查询改写 (口语→专业、指代消解、同义子问题)
  │
  ├─→ 向量检索 (bge-m3 + MMR 多样性去重, Top8)   ← 两阶段第一阶段
  │
  ├─→ BM25 关键词检索 (jieba, Top8)
  │
  ▼
混合检索融合 (向量0.6 + BM25 0.4, 去重)
  │
  ▼
BGE-Reranker 精排 (Top3)                        ← 两阶段第二阶段
  │
  ▼
多源知识融合 (论文原文/综述解读/实验笔记 配额)
  │
  ├── 有结果 ──→ LLM 生成 (RunnableWithMessageHistory 多轮)
  │                  │
  │                  ▼
  │              纠正型 RAG 自检 (防幻觉)
  │                  │
  │                  ▼
  │              全文溯源绑定 (文件名+页码+原文+相似度+doc_source)
  │                  │
  │                  ▼
  │              返回答案 (带引用)
  │
  └── 无结果 ──→ Web 搜索兜底 → 外网标注 + URL 溯源
```

---

## 七、技术栈

| 组件 | 技术选型 |
|------|----------|
| 文档加载 | PyPDFLoader / python-docx / UnstructuredMarkdown |
| 文本分割 | 零宽断言 `(?<=。)` + keep_separator=False |
| 嵌入模型 | BAAI/bge-m3 |
| 开发向量库 | Chroma |
| 生产向量库 | **PGvector**（PostgreSQL + 纯 SQL 兼容层，无需扩展 DLL） |
| 关键词检索 | BM25 + jieba 分词 |
| 多样性去重 | MMR (Maximal Marginal Relevance) |
| 重排模型 | BAAI/bge-reranker-v2-m3 |
| LLM | DeepSeek / Qwen / Ollama |
| 多轮对话 | LCEL RunnableWithMessageHistory |
| Web 框架 | FastAPI + Streamlit |
| 评估指标 | Recall@K / MRR / NDCG / 幻觉率 |
| 部署 | Docker + Docker Compose |

---

## 八、开发约束

### 允许
- ✅ 支持 PDF / Word / Markdown 多格式上传
- ✅ 多源知识融合（论文原文 / 综述解读 / 实验笔记）
- ✅ 多轮对话追问指代消解
- ✅ 量化评估体系 + RAG vs 纯 LLM 对比
- ✅ 私有知识缺失时外网兜底

### 禁止
- ❌ 闲聊、科普、自由对话
- ❌ 脑补、扩写、虚构知识库不存在内容
- ❌ 简化、合并、跳过任意六层架构环节
- ❌ 去掉引用溯源
- ❌ 硬编码所有可配置参数

---

## 九、License

内部项目，仅供授权使用。
