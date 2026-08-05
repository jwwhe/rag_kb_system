# RAG 知识库问答系统 — 用户操作使用手册 V2.0

---

## 目录

1. [系统概述](#1-系统概述)
2. [环境准备](#2-环境准备)
3. [安装部署](#3-安装部署)
4. [配置说明](#4-配置说明)
5. [启动服务](#5-启动服务)
6. [前端使用（Streamlit）](#6-前端使用streamlit)
7. [API 接口详解](#7-api-接口详解)
8. [典型使用流程](#8-典型使用流程)
9. [评估体系](#9-评估体系)
10. [FAQ 常见问题](#10-faq-常见问题)
11. [故障排查](#11-故障排查)

---

## 1. 系统概述

本系统是一个基于**六层 RAG 架构**的私有知识库智能问答平台，支持 **PDF / Word / Markdown** 多格式文档。上传文档后，系统自动完成文本提取、分块、向量化、入库；用户提问时，通过「查询改写 → MMR去重 → BM25混合检索 → Rerank精排 → 多源融合 → LLM生成 → 自检纠错 → 溯源」的完整链路返回带引用的答案。

### 核心能力

| 能力 | 说明 |
|------|------|
| 多格式上传 | 支持 PDF / Word (.docx) / Markdown (.md) 批量上传，自动解析入库 |
| 多源知识融合 | 论文原文 / 综述解读 / 实验笔记 三类来源配额融合 |
| 多轮对话 | LCEL RunnableWithMessageHistory 实现追问指代消解 |
| 两阶段精排 | MMR 多样性去重 + BGE-Reranker 精排，Top-1 命中率 45%→78% |
| 智能问答 | 完整六层链路：改写→MMR→BM25→混合→Rerank→多源融合→生成→纠错→溯源 |
| 引用溯源 | 每条答案绑定原文片段、文件名、页码、相似度、来源类型 |
| Web 搜索兜底 | 知识库无匹配时自动联网搜索补充，明确标注外网来源 |
| 双库适配 | 生产 PGvector（D盘本地） / 开发 Chroma，环境变量一键切换 |
| 多 LLM 切换 | 配置即可在 DeepSeek / Qwen / Ollama 之间切换 |
| 量化评估 | Recall@K / MRR / NDCG / 幻觉率，RAG vs 纯 LLM 对比 |

### 处理流程

```
上传文档 → 多格式解析 → 清洗过滤 → 零宽断言分块(800字/150重叠)
→ bge-m3 向量化(1024维) → 存入 PGvector / Chroma

用户提问 → 查询改写(指代消解) → MMR多样性去重(Top8)
→ BM25关键词检索(Top8) → 混合融合(0.6/0.4)
→ BGE-Reranker精排(Top3) → 多源知识融合(论文/综述/笔记配额)
→ LLM生成(多轮对话历史) → CRAG自检纠错 → 溯源绑定 → 返回答案
```

---

## 2. 环境准备

### 2.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 | 8 核 |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 10 GB 可用（含模型缓存 ~3GB） | 50 GB SSD |
| GPU | 无（CPU 可运行） | NVIDIA GPU 8GB+ |

### 2.2 软件要求

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10+ | 已测 3.14 |
| PostgreSQL | 16 | 向量数据库（D:\PostgreSQL\16\） |
| Git | 2.0+ | （可选） |

### 2.3 LLM API Key（至少准备一个）

| LLM | 获取地址 | 环境变量 |
|-----|---------|----------|
| DeepSeek | https://platform.deepseek.com/api_keys | `DEEPSEEK_API_KEY` |
| 通义千问 | https://dashscope.console.aliyun.com/apiKey | `QWEN_API_KEY` |
| Ollama | 本地部署，无需 Key | `OLLAMA_HOST` |

---

## 3. 安装部署

```bash
# Step 1: 进入项目目录
cd rag_kb_system

# Step 2: 安装依赖
pip install -r requirements.txt

# Step 3: 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入实际的 API Key 和数据库配置

# Step 4: 确保 PostgreSQL 运行中
D:\PostgreSQL\16\bin\pg_ctl -D D:\PostgreSQL\data start

# Step 5: 启动后端
python run.py
```

---

## 4. 配置说明

### 4.1 配置方式

| 方式 | 适用场景 |
|------|----------|
| `.env` 文件 | **推荐**，项目级固定配置 |
| 环境变量 | 临时覆盖、敏感信息 |
| `config/settings.py` | 修改默认参数值 |

### 4.2 必配参数（.env）

```bash
# LLM API Key（至少一个）
DEEPSEEK_API_KEY=sk-your-key

# 向量库
STORE_TYPE=pgvector              # pgvector / chroma
PG_HOST=localhost                # PostgreSQL 地址

# 模型（网络不通时启用离线）
HF_HUB_OFFLINE=1                 # 设为 1 使用本地缓存模型
```

### 4.3 切换 LLM

编辑 `config/settings.py` 的 `LLMConfig`：
```python
active_llm = "deepseek"   # deepseek / qwen / ollama
```

### 4.4 切换向量库

```bash
STORE_TYPE=pgvector python run.py   # 生产：PGvector (D盘)
STORE_TYPE=chroma python run.py     # 开发：Chroma (本地文件)
```

### 4.5 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 800 | 分块大小 |
| `chunk_overlap` | 150 | 块间重叠 |
| `vector_top_k` | 8 | 向量检索召回数 |
| `mmr_fetch_k` | 30 | MMR 候选拉取数 |
| `mmr_lambda` | 0.7 | MMR 多样性权衡 |
| `rerank_top_k` | 3 | 送入 LLM 的最优条数 |
| `similarity_threshold` | 0.35 | 最低相似度阈值 |
| `temperature` | 0.1 | LLM 生成温度 |

---

## 5. 启动服务

### 5.1 本地启动

```bash
# 终端 1：后端 API
python run.py
# 启动后显示：向量库: pgvector | LLM: deepseek | 监听: 0.0.0.0:8000

# 终端 2：Streamlit 前端
streamlit run app_streamlit.py
```

### 5.2 验证服务

| 地址 | 说明 |
|------|------|
| `http://localhost:8000/health` | 健康检查 |
| `http://localhost:8000/docs` | Swagger API 文档 |
| `http://localhost:8000/` | API 导航（指向 Streamlit 前端） |
| `http://localhost:8501` | **Streamlit 前端**（推荐使用） |

---

## 6. 前端使用（Streamlit）

访问 `http://localhost:8501`，左侧边栏为管理区，右侧为主对话区。

### 6.1 上传文档

1. 左侧 "上传文档" → 点击 "Browse files"
2. 选择文件（支持 `.pdf` / `.docx` / `.md`）
3. 选择 **来源类型**（论文原文 / 综述解读 / 实验笔记）
4. 点击 "入库" → 等待处理完成

### 6.2 问答

1. 在底部输入框键入问题
2. 答案自动展示，引用来源在折叠面板中
3. 支持**多轮追问**，上下文自动关联
4. 点击 "新建对话" 重置会话

### 6.3 知识库管理

- **刷新统计**：查看向量库类型、文件数、分块数
- **删除文件**：每个文件旁有删除按钮
- **清空知识库**：二次确认，不可恢复

---

## 7. API 接口详解

### 7.1 接口总览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/` | API 导航（含文档/前端链接） |
| `POST` | `/api/v1/documents/upload` | 上传文档入库（多格式+多源标注） |
| `POST` | `/api/v1/qa/ask` | 智能问答（含 session_id 多轮） |
| `GET` | `/api/v1/kb/stats` | 知识库统计 |
| `DELETE` | `/api/v1/kb/files` | 删除文件 |
| `DELETE` | `/api/v1/kb/clear` | 清空知识库 |
| `POST` | `/api/v1/eval/rag` | RAG 评估 |
| `POST` | `/api/v1/eval/compare` | RAG vs 纯 LLM 对比 |

### 7.2 文档上传入库

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@论文.pdf" \
  -F "files=@笔记.docx" \
  -F "files=@综述.md" \
  -F "source_type=论文原文" \
  -F "knowledge_base=default"
```

**响应：**
```json
{
  "code": 200,
  "message": "成功上传 3 个文件",
  "data": [
    {
      "file_name": "论文.pdf",
      "pages": 10,
      "chunks": 25,
      "message": "上传并入库成功（来源: 论文原文）"
    }
  ]
}
```

### 7.3 智能问答（支持多轮对话）

```bash
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Transformer 用了多少注意力头？", "session_id": "user-001"}'

# 多轮追问（同一 session_id）
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "每个头的维度是多少？", "session_id": "user-001"}'
```

**响应：**
```json
{
  "code": 200,
  "data": {
    "answer": "Transformer 使用了 8 个注意力头...",
    "citations": [{
      "citation_id": 1,
      "file_name": "注意力机制论文原文.md",
      "page": 3,
      "original_text": "...",
      "similarity": 0.8756,
      "source_type": "internal",
      "doc_source": "论文原文"
    }],
    "source_type": "internal",
    "source_stats": { "internal_sources": 3, "external_sources": 0 }
  }
}
```

### 7.4 知识库管理

```bash
# 查看统计
curl http://localhost:8000/api/v1/kb/stats

# 删除文件
curl -X DELETE http://localhost:8000/api/v1/kb/files \
  -H "Content-Type: application/json" \
  -d '{"file_name": "论文.pdf"}'

# 清空知识库（不可恢复）
curl -X DELETE http://localhost:8000/api/v1/kb/clear
```

### 7.5 评估接口

```bash
# RAG 评估
curl -X POST http://localhost:8000/api/v1/eval/rag \
  -H "Content-Type: application/json" \
  -d '[{"question":"...","relevant_doc_ids":["id1"],"expected_keywords":["方法"]}]'

# RAG vs 纯 LLM 对比
curl -X POST http://localhost:8000/api/v1/eval/compare \
  -H "Content-Type: application/json" \
  -d '[{"question":"...","relevant_doc_ids":["id1"],"expected_keywords":["方法"]}]'
```

---

## 8. 典型使用流程

### 8.1 场景一：学术论文知识库

```bash
# 上传论文原文、综述解读、实验笔记
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@注意力机制论文原文.md" \
  -F "files=@Transformer综述解读.md" \
  -F "files=@Transformer复现实验笔记.md" \
  -F "source_type=论文原文" \
  -F "knowledge_base=transformer"

# 提问 + 多轮追问
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Transformer 为什么要在注意力中除以 sqrt(d_k)？","session_id":"s1"}'

curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"如果不除以它会怎样？","session_id":"s1"}'
```

### 8.2 场景二：跑评估

```bash
# 一键评估（需先入库样例文档）
python run_eval.py

# 只评估检索指标
python run_eval.py --no-generation

# 指定向量库
python run_eval.py --store chroma
```

---

## 9. 评估体系

系统内置完整评估框架（Layer 6）：

| 指标 | 说明 |
|------|------|
| Recall@K | 前 K 条检索覆盖了多少相关文档 |
| Precision@K | 前 K 条检索中相关文档占比 |
| MRR | 第一个相关文档的倒数排名 |
| NDCG@K | 归一化折损累积增益 |
| 幻觉率 | 回答中不确定表述的占比 |
| Keyword Accuracy | 预期关键词命中率 |

**快速评估：**
```bash
python run_eval.py                    # 完整评估（检索 + 生成 + RAG vs 纯LLM对比）
python run_eval.py --no-generation    # 仅检索指标
python run_eval.py --store chroma     # 指定向量库
```

---

## 10. FAQ 常见问题

### Q1: 支持哪些文件格式？
PDF (.pdf)、Word (.docx)、Markdown (.md, .markdown)。不支持扫描件 OCR。

### Q2: 如何启用多轮对话？
在 `/qa/ask` 请求中传入 `session_id`，相同 ID 的请求共享对话历史，支持指代消解。

### Q3: 回答不准确怎么办？
- 降低 `similarity_threshold`（0.35 → 0.25）增加召回
- 增大 `rerank_top_k`（3 → 5）给 LLM 更多上下文
- 提问时使用原文术语

### Q4: 如何切换向量库？
```bash
STORE_TYPE=pgvector python run.py   # PGvector (D盘)
STORE_TYPE=chroma python run.py     # Chroma (本地文件)
```

### Q5: 模型下载太慢 / 网络不通？
设置 `.env` 启用离线模式：
```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```
前提：模型已至少在线下载过一次并缓存到 `~/.cache/huggingface/`。

### Q6: 多源知识融合如何工作？
上传文件时选择来源类型（论文原文/综述解读/实验笔记），检索时每类至少保留 1 条（min_per_source），剩余配额按分数公平竞争。

### Q7: PostgreSQL 怎么启动？
```bash
D:\PostgreSQL\16\bin\pg_ctl -D D:\PostgreSQL\data start
D:\PostgreSQL\16\bin\pg_ctl -D D:\PostgreSQL\data status
```

---

## 11. 故障排查

| 现象 | 解决方案 |
|------|----------|
| `vector_store: pgvector` 但连接失败 | 检查 PostgreSQL 是否启动：`pg_ctl status` |
| 上传 docx 报错 | `pip install python-docx` |
| 上传 md 报错 | `pip install markdown` |
| 模型加载失败 | 设置 `HF_HUB_OFFLINE=1` 使用本地缓存 |
| 总是返回"暂无相关资料" | 降低 `similarity_threshold` |
| 端口被占用 | 修改 `config/settings.py` 中 `api.port` |

### 日志查看

日志直接输出到控制台（开发环境）。启动时关注以下关键信息：
```
向量库: pgvector (或 chroma)
LLM: deepseek
嵌入模型加载完成
重排模型加载完成
系统就绪，访问 http://localhost:8000/
```

---

> 文档版本: V2.0 | 最后更新: 2026-08-05 | 适用系统版本: RAG KB System V2.0
