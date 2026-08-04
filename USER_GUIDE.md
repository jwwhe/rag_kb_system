# RAG 知识库问答系统 — 用户操作使用手册 V1.0

---

## 目录

1. [系统概述](#1-系统概述)
2. [环境准备](#2-环境准备)
3. [安装部署](#3-安装部署)
4. [配置说明](#4-配置说明)
5. [启动服务](#5-启动服务)
6. [API 接口详解](#6-api-接口详解)
7. [典型使用流程](#7-典型使用流程)
8. [FAQ 常见问题](#8-faq-常见问题)
9. [故障排查](#9-故障排查)

---

## 1. 系统概述

本系统是一个基于**五层 RAG 架构**的私有知识库智能问答平台，仅支持对用户上传的 **PDF 文件**进行知识问答。上传 PDF 后，系统自动完成文本提取、分块、向量化、入库；用户提问时，系统通过检索→重排→生成→纠错→溯源的完整链路返回带引用的答案。

### 核心能力

| 能力 | 说明 |
|------|------|
| PDF 批量上传 | 支持一次上传多个 PDF，自动解析入库 |
| 智能问答 | 完整的「查询改写→混合检索→重排→生成→纠错→溯源」链路 |
| 引用溯源 | 每条答案绑定原文片段、文件名、页码、相似度 |
| Web 搜索兜底 | 知识库无匹配时自动联网搜索补充 |
| 双环境支持 | 开发用 Chroma（本地文件），生产用 Qdrant（Docker 容器） |
| 多 LLM 切换 | 配置即可在 DeepSeek / Qwen / Ollama 之间切换 |

### 处理流程

```
上传 PDF → 文本提取 → 清洗过滤 → 分块(800字/150重叠)
→ bge-m3 向量化 → 存入向量库

用户提问 → 查询改写 → 向量检索(Top8) + BM25检索(Top8)
→ 混合融合(0.6/0.4) → Rerank重排(Top3) → LLM生成
→ 自检纠错 → 绑定溯源 → 返回答案
```

---

## 2. 环境准备

### 2.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 | 8 核 |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 10 GB 可用 | 50 GB SSD |
| GPU | 无（CPU 可运行） | NVIDIA GPU 8GB+（加速嵌入和重排模型） |

### 2.2 软件要求

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10 ~ 3.12 | 必须 |
| pip | 23.0+ | Python 包管理器 |
| Git | 2.0+ | （可选）版本管理 |
| Docker | 24.0+ | （可选）生产环境部署 |
| Docker Compose | 2.0+ | （可选）生产环境部署 |

### 2.3 LLM API Key（至少准备一个）

| LLM | 获取地址 | 环境变量 |
|-----|---------|----------|
| DeepSeek | https://platform.deepseek.com/api_keys | `DEEPSEEK_API_KEY` |
| 通义千问 | https://dashscope.console.aliyun.com/apiKey | `QWEN_API_KEY` |
| Ollama | 本地部署，无需 Key | `OLLAMA_HOST` |

> **注意**：至少配置一个 LLM 才能使用问答功能。推荐首次使用 DeepSeek（注册即送额度）。

---

## 3. 安装部署

### 3.1 方式一：本地开发部署（推荐入门）

```bash
# Step 1: 进入项目目录
cd rag_kb_system

# Step 2: 创建 Python 虚拟环境（推荐）
python -m venv venv

# Step 3: 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Step 4: 安装依赖
pip install -r requirements.txt

# Step 5: 验证安装
python -c "import langchain; import fastapi; print('依赖安装成功')"
```

### 3.2 方式二：Docker 生产部署

```bash
# Step 1: 进入项目目录
cd rag_kb_system

# Step 2: 创建 .env 文件并配置 API Key
cp .env.example .env
# 编辑 .env 文件，填入实际的 API Key

# Step 3: 构建并启动全部服务
docker-compose -f docker/docker-compose.yml up -d

# Step 4: 查看服务状态
docker-compose -f docker/docker-compose.yml ps

# 预期输出：rag-kb-api、rag-qdrant、rag-ollama 均为 Up 状态

# Step 5: 查看日志
docker-compose -f docker/docker-compose.yml logs -f rag-api
```

### 3.3 首次安装后的模型下载

首次启动时，系统会自动下载两个模型文件（约 2-3 GB），请耐心等待：

| 模型 | 大小 | 用途 | 层级 |
|------|------|------|------|
| `BAAI/bge-m3` | ~2 GB | 文本向量化 | Layer 1 |
| `BAAI/bge-reranker-v2-m3` | ~1 GB | 检索结果重排 | Layer 3 |

> **提示**：模型下载仅首次需要，后续启动直接加载本地缓存。国内用户可设置 HuggingFace 镜像加速：
> ```bash
> # Windows PowerShell
> $env:HF_ENDPOINT="https://hf-mirror.com"
> # Linux/macOS
> export HF_ENDPOINT="https://hf-mirror.com"
> ```

---

## 4. 配置说明

### 4.1 配置方式（三选一，优先级从高到低）

| 方式 | 适用场景 | 示例 |
|------|----------|------|
| 环境变量 | 生产部署、敏感信息 | `export DEEPSEEK_API_KEY=sk-xxx` |
| `.env` 文件 | 项目级固定配置 | `DEEPSEEK_API_KEY=sk-xxx` |
| `config/settings.py` | 修改默认参数 | 修改 dataclass 默认值 |

### 4.2 必配参数

```bash
# ====== LLM API Key（至少配置一个）======

# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-your-deepseek-api-key"

# Linux / macOS
export DEEPSEEK_API_KEY="sk-your-deepseek-api-key"

# 如果使用通义千问
export QWEN_API_KEY="your-qwen-api-key"

# 如果使用 Ollama 本地模型（先确保 Ollama 已启动并拉取模型）
ollama pull qwen2.5:7b
```

### 4.3 切换 LLM

编辑 `config/settings.py` 中的 `LLMConfig` 类，修改 `active_llm` 字段：

```python
# 使用 DeepSeek
active_llm: Literal["deepseek", "qwen", "ollama"] = "deepseek"

# 使用通义千问
active_llm: Literal["deepseek", "qwen", "ollama"] = "qwen"

# 使用 Ollama 本地模型
active_llm: Literal["deepseek", "qwen", "ollama"] = "ollama"
```

### 4.4 切换向量库（开发/生产）

```bash
# 开发环境：使用 Chroma（本地文件存储，默认）
ENV=development python run.py

# 生产环境：使用 Qdrant（需先启动 Qdrant 服务）
ENV=production python run.py
```

### 4.5 可调参数一览

以下参数均可在 `config/settings.py` 中调整：

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|----------|
| `chunk_size` | 800 | 文本分块大小 | 文档长段落多可增大到 1000 |
| `chunk_overlap` | 150 | 块间重叠字数 | 保持为 chunk_size 的 15-20% |
| `vector_top_k` | 8 | 向量检索召回数 | 知识库大时可增至 15 |
| `bm25_top_k` | 8 | BM25 检索召回数 | 精确查询多时可增大 |
| `vector_weight` | 0.6 | 融合时向量权重 | 语义查询为主保持 0.6 |
| `bm25_weight` | 0.4 | 融合时关键词权重 | 精确匹配多时可调至 0.5 |
| `rerank_top_k` | 3 | 重排后送入 LLM 的条数 | 3 是性价比最优值 |
| `similarity_threshold` | 0.35 | 最低相似度阈值 | 结果太少可降低到 0.25 |
| `temperature` | 0.1 | LLM 温度 | 知识问答保持低温 |

---

## 5. 启动服务

### 5.1 本地开发启动

```bash
cd rag_kb_system

# 方式一：直接启动
python run.py

# 方式二：通过 uvicorn 启动（可指定端口）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后会显示：

```
============================================================
  RAG 知识库问答系统 v1.0.0
  环境: development
  向量库: chroma
  LLM: deepseek
  监听: 0.0.0.0:8000
============================================================
```

### 5.2 验证服务状态

浏览器访问以下地址确认服务正常：

| 地址 | 说明 |
|------|------|
| `http://localhost:8000/health` | 健康检查，返回系统状态 |
| `http://localhost:8000/docs` | **Swagger 交互式 API 文档**（推荐在此测试） |
| `http://localhost:8000/redoc` | ReDoc API 文档 |

### 5.3 Docker 启动

```bash
cd rag_kb_system

# 启动全部服务（API + Qdrant + Ollama）
docker-compose -f docker/docker-compose.yml up -d

# 查看实时日志
docker-compose -f docker/docker-compose.yml logs -f

# 停止服务
docker-compose -f docker/docker-compose.yml down

# 停止并清除数据
docker-compose -f docker/docker-compose.yml down -v
```

---

## 6. API 接口详解

### 6.1 接口总览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/documents/upload` | 上传 PDF 入库 |
| `POST` | `/api/v1/qa/ask` | 智能问答 |
| `GET` | `/api/v1/kb/stats` | 知识库统计 |
| `DELETE` | `/api/v1/kb/files` | 删除文件 |
| `DELETE` | `/api/v1/kb/clear` | 清空知识库 |

---

### 6.2 健康检查

**请求：**
```bash
curl http://localhost:8000/health
```

**响应：**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "env": "development",
  "vector_store": "chroma",
  "llm": "deepseek"
}
```

---

### 6.3 PDF 上传入库

这是使用系统的**第一步**，上传 PDF 后系统自动完成全部处理流程。

**请求：**
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@合同模板.pdf" \
  -F "files=@员工手册.pdf"
```

**Swagger 操作步骤：**
1. 访问 `http://localhost:8000/docs`
2. 找到 `POST /api/v1/documents/upload`
3. 点击 "Try it out"
4. 点击 "Add string item" 添加 PDF 文件
5. 点击 "Execute"

**响应：**
```json
{
  "code": 200,
  "message": "成功上传 2 个文件",
  "data": [
    {
      "file_name": "合同模板.pdf",
      "pages": 5,
      "chunks": 12,
      "message": "上传并入库成功"
    },
    {
      "file_name": "员工手册.pdf",
      "pages": 20,
      "chunks": 48,
      "message": "上传并入库成功"
    }
  ],
  "timestamp": 1722492000.123456
}
```

**处理说明：**

| 步骤 | 说明 | 耗时（参考） |
|------|------|-------------|
| PDF 解析 | 提取每页文本，过滤空白/页眉页脚 | 1-2 秒/文件 |
| 文本分块 | 按 800 字/块分割，150 字重叠 | 毫秒级 |
| 向量化 | bge-m3 模型编码（首次加载模型约 30 秒） | 0.5-1 秒/块 |
| 入库 | 写入 Chroma/Qdrant | 毫秒级 |

> **限制**：单文件不超过 50MB（可在 config 中调整 `max_upload_size_mb`）。

---

### 6.4 智能问答（核心接口）

这是系统的**核心接口**，执行完整的五层 RAG 链路。

**请求：**
```bash
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "员工试用期是多久？"}'
```

**请求体：**
```json
{
  "question": "员工试用期是多久？",
  "collection_name": null
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | ✅ | 用户问题，1-2000 字符 |
| `collection_name` | string | ❌ | 指定知识库，null 使用默认库 |

**成功响应（知识库有匹配）：**
```json
{
  "code": 200,
  "message": "问答完成",
  "data": {
    "answer": "根据员工手册第三章第2条规定，新员工试用期为3个月。试用期内表现优秀者可提前转正，但最短不少于1个月。试用期工资按正式工资的80%发放。",
    "citations": [
      {
        "citation_id": 1,
        "file_name": "员工手册.pdf",
        "page": 12,
        "original_text": "第三章 试用期管理\n第2条 新员工试用期为3个月，自入职之日起计算。试用期内表现优秀者...",
        "similarity": 0.8756,
        "source_type": "internal"
      },
      {
        "citation_id": 2,
        "file_name": "员工手册.pdf",
        "page": 13,
        "original_text": "第5条 试用期工资按正式工资的80%发放，转正后恢复全额工资...",
        "similarity": 0.8123,
        "source_type": "internal"
      }
    ],
    "citation_text": "【本地知识库引用来源】\n  [1] 员工手册.pdf (第12页) | 相关度: 0.8756\n      原文: \"第三章 试用期管理\n第2条 新员工试用期为3个月...\"\n\n  [2] 员工手册.pdf (第13页) | 相关度: 0.8123\n      原文: \"第5条 试用期工资按正式工资的80%发放...\"",
    "source_type": "internal",
    "source_stats": {
      "internal_sources": 2,
      "external_sources": 0,
      "total_sources": 2
    }
  },
  "timestamp": 1722492100.654321
}
```

**知识库无匹配时的响应（自动 Web 兜底）：**
```json
{
  "code": 200,
  "message": "问答完成",
  "data": {
    "answer": "【以下内容来源于外部网络搜索，非本地知识库内容】\n\n根据公开信息，...",
    "source_type": "external",
    "source_stats": {
      "internal_sources": 0,
      "external_sources": 3,
      "total_sources": 3
    }
  }
}
```

**完全无结果响应：**
```json
{
  "code": 200,
  "message": "问答完成",
  "data": {
    "answer": "知识库中暂无该相关资料，无法解答此问题",
    "source_type": "internal",
    "source_stats": {
      "internal_sources": 0,
      "external_sources": 0,
      "total_sources": 0
    }
  }
}
```

### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | 最终答案文本 |
| `citations` | array | 引用条目列表 |
| `citations[].citation_id` | int | 引用编号（与 answer 中 [引用-N] 对应） |
| `citations[].file_name` | string | 源 PDF 文件名 |
| `citations[].page` | int/string | 源页码 |
| `citations[].original_text` | string | 原文片段（最多 200 字） |
| `citations[].similarity` | float | 相似度分数（0~1，越高越相关） |
| `citations[].source_type` | string | `internal` = 本地知识库 / `external` = 外网搜索 |
| `citation_text` | string | 格式化的溯源段落（可直接展示） |
| `source_type` | string | 答案来源类型 |
| `source_stats` | object | 来源统计汇总 |

---

### 6.5 知识库管理

#### 6.5.1 查看知识库统计

```bash
curl http://localhost:8000/api/v1/kb/stats
```

**响应：**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "store_type": "chroma",
    "collection_name": "rag_kb_collection",
    "total_chunks": 60,
    "total_files": 2,
    "file_names": ["合同模板.pdf", "员工手册.pdf"]
  }
}
```

#### 6.5.2 删除指定文件

```bash
curl -X DELETE http://localhost:8000/api/v1/kb/files \
  -H "Content-Type: application/json" \
  -d '{"file_name": "合同模板.pdf"}'
```

**响应：**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "deleted_count": 12,
    "file_name": "合同模板.pdf",
    "message": "已删除 12 个向量块"
  }
}
```

> **注意**：删除后该文件的所有分块从向量库中移除，BM25 索引自动重建。

#### 6.5.3 清空知识库

```bash
curl -X DELETE http://localhost:8000/api/v1/kb/clear
```

**响应：**
```json
{
  "code": 200,
  "message": "知识库已清空",
  "data": null
}
```

> **⚠️ 警告**：此操作不可恢复，请确认后再执行。

---

## 7. 典型使用流程

### 7.1 场景一：合同知识库问答

```bash
# ====== Step 1: 上传合同文件 ======
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@采购合同.pdf" \
  -F "files=@劳动合同.pdf" \
  -F "files=@租赁合同.pdf"

# 响应：3 个文件上传成功，共生成 85 个分块

# ====== Step 2: 确认入库情况 ======
curl http://localhost:8000/api/v1/kb/stats
# 响应：3 个文件，85 个分块

# ====== Step 3: 开始提问 ======
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "采购合同的违约金如何计算？"}'

# ====== Step 4: 追问具体细节 ======
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "违约金的上限是多少？"}'
```

### 7.2 场景二：规章制度查询

```bash
# 上传员工手册
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@员工手册2024版.pdf"

# 查询各类规定
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "年假天数如何计算？"}'

curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "加班费的计算标准是什么？"}'

curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "离职流程是怎样的？"}'
```

### 7.3 场景三：技术文档检索

```bash
# 上传技术文档
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@系统架构设计.pdf" \
  -F "files=@API接口文档.pdf" \
  -F "files=@数据库设计.pdf"

# 精确技术查询
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "用户认证模块使用了什么加密算法？"}'

# 编号/术语查询（BM25 优势场景）
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "错误码 E1001 表示什么？"}'
```

### 7.4 场景四：知识库维护

```bash
# 查看当前知识库状态
curl http://localhost:8000/api/v1/kb/stats

# 更新某个文件（先删后传）
curl -X DELETE http://localhost:8000/api/v1/kb/files \
  -H "Content-Type: application/json" \
  -d '{"file_name": "员工手册2024版.pdf"}'

curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@员工手册2025版.pdf"

# 确认更新结果
curl http://localhost:8000/api/v1/kb/stats
```

---

## 8. FAQ 常见问题

### Q1: 上传 PDF 后多久可以提问？

上传接口是同步的，接口返回成功即表示文档已处理完毕并入库，立刻可以提问。

### Q2: 为什么我的 PDF 上传后页数显示为 0？

可能原因：
- PDF 是扫描件（纯图片），PyPDFLoader 无法提取文本。**本系统 V1.0 不支持 OCR 识别。**
- PDF 被加密保护，需要先解密。
- 文件损坏，尝试用 PDF 阅读器打开确认。

### Q3: 回答不准确怎么办？

优化策略：
1. **提高检索质量**：将 `similarity_threshold` 从 0.35 降低到 0.25，召回更多候选
2. **增加上下文**：将 `rerank_top_k` 从 3 增加到 5，给 LLM 更多参考
3. **优化提问**：提问时使用原文中出现的术语和关键词
4. **检查 PDF 质量**：确认 PDF 文本是否被正确提取（查看上传返回的 chunks 数）

### Q4: 回答中出现"知识库中暂无该相关资料"

可能原因：
- 知识库为空（未上传任何 PDF）→ 先上传 PDF
- 问题与知识库内容不相关 → 这是正常的兜底行为
- 相似度阈值过高 → 调低 `similarity_threshold`
- PDF 文本提取失败 → 检查 PDF 是否为纯图片扫描件

### Q5: 模型下载太慢怎么办？

设置 HuggingFace 国内镜像：
```bash
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# Linux/macOS
export HF_ENDPOINT="https://hf-mirror.com"

# 然后再启动服务
python run.py
```

### Q6: 如何切换 LLM？

编辑 `config/settings.py`，修改 `LLMConfig.active_llm`：

```python
# 从 DeepSeek 切换到通义千问
active_llm = "qwen"

# 切换到 Ollama 本地模型
active_llm = "ollama"
```

切换后重启服务即可，无需修改任何代码。

### Q7: 内存占用太大怎么办？

内存占用主要来自两个模型：
- bge-m3（嵌入模型）：~2 GB
- bge-reranker-v2-m3（重排模型）：~1 GB

优化方案：
1. 使用 GPU 加速（设置 `embedding_device = "cuda"`），减少 CPU 内存占用
2. 使用 Ollama 本地 LLM 替代云端 API，减少网络延迟
3. Docker Compose 中限制容器内存：
   ```yaml
   rag-api:
     deploy:
       resources:
         limits:
           memory: 8G
   ```

### Q8: Web 搜索兜底如何工作？

当知识库中找不到相关内容时：
1. 系统自动使用 DuckDuckGo 搜索引擎搜索外网
2. 搜索结果送入 LLM 生成回答
3. 回答明确标注「外部网络搜索来源」
4. 私有知识库内容**永远优先**于外网结果

如需禁用 Web 搜索兜底，在 `config/settings.py` 中设置：
```python
enable_web_fallback: bool = False
```

---

## 9. 故障排查

### 9.1 服务无法启动

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'xxx'` | 依赖未安装 | `pip install -r requirements.txt` |
| `Address already in use` | 端口 8000 被占用 | 修改 `api.port` 或关闭占用进程 |
| `API Key 未配置` | 未设置 LLM Key | `export DEEPSEEK_API_KEY=sk-xxx` |
| Chroma 启动失败 | 数据目录权限不足 | 检查 `data/chroma_db/` 目录权限 |
| Qdrant 连接失败 | Qdrant 服务未启动 | `docker-compose up -d qdrant` |

### 9.2 上传失败

| 现象 | 解决方案 |
|------|----------|
| `仅支持 PDF 格式文件` | 确认文件扩展名为 `.pdf` |
| `文件超过 50MB` | 增大 `max_upload_size_mb` 配置值 |
| `PDF 解析失败` | 确认 PDF 未加密、未损坏 |
| `所有文档向量化均失败` | 检查 bge-m3 模型是否下载完整 |

### 9.3 问答无结果

| 现象 | 解决方案 |
|------|----------|
| 总是返回"暂无相关资料" | 降低 `similarity_threshold` 至 0.25 |
| 返回内容不相关 | 检查 PDF 文本是否已正确提取（查看上传返回的 chunks 数） |
| LLM 调用超时 | 增大 `request_timeout`，或切换更快的 LLM |

### 9.4 性能优化

| 场景 | 优化方案 |
|------|----------|
| 知识库文档量大（100+ PDF） | 增大 `vector_top_k` 和 `bm25_top_k` 至 15-20 |
| 首次提问慢 | 正常现象（模型加载），后续提问速度正常 |
| 向量化速度慢 | 增大 `embedding_batch_size` 至 64 |
| 长期运行后变慢 | 定期清理无用的 PDF 数据 |

### 9.5 日志查看

```bash
# 本地开发环境：日志直接输出到控制台
python run.py  # 观察控制台输出

# Docker 环境
docker-compose -f docker/docker-compose.yml logs -f rag-api

# 查看最近 100 行日志
docker-compose -f docker/docker-compose.yml logs --tail=100 rag-api
```

### 9.6 获取帮助

遇到问题时，请准备以下信息以便排查：
1. 操作系统和 Python 版本：`python --version`
2. 启动日志（包含错误信息）
3. 配置文件内容（隐藏 API Key）
4. 执行的命令和请求参数
5. PDF 文件是否可正常打开阅读

---

## 附录 A：Swagger 快速测试指南

访问 `http://localhost:8000/docs` 是测试系统最方便的方式：

```
1. 上传 PDF：
   POST /api/v1/documents/upload
   → Try it out → Choose File → 选择 PDF → Execute

2. 确认入库：
   GET /api/v1/kb/stats
   → Try it out → Execute
   → 确认 total_files > 0

3. 提问：
   POST /api/v1/qa/ask
   → Try it out → 输入 {"question": "你的问题"}
   → Execute → 查看 Response body

4. 管理数据：
   DELETE /api/v1/kb/files → 删除单个文件
   DELETE /api/v1/kb/clear  → 清空知识库
```

## 附录 B：Python 客户端示例

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# 1. 上传 PDF
with open("合同模板.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/documents/upload",
        files=[("files", ("合同模板.pdf", f, "application/pdf"))],
    )
print("上传结果:", response.json())

# 2. 查看知识库统计
response = requests.get(f"{BASE_URL}/kb/stats")
print("知识库状态:", response.json())

# 3. 智能问答
response = requests.post(
    f"{BASE_URL}/qa/ask",
    json={"question": "合同的有效期是多久？"},
)
result = response.json()
print("答案:", result["data"]["answer"])
print("引用来源:")
for cite in result["data"]["citations"]:
    print(f"  [{cite['citation_id']}] {cite['file_name']} "
          f"第{cite['page']}页 (相关度: {cite['similarity']})")

# 4. 删除文件
response = requests.delete(
    f"{BASE_URL}/kb/files",
    json={"file_name": "合同模板.pdf"},
)
print("删除结果:", response.json())
```

## 附录 C：cURL 命令速查表

```bash
# 健康检查
curl http://localhost:8000/health

# 上传单个 PDF
curl -X POST http://localhost:8000/api/v1/documents/upload -F "files=@文件.pdf"

# 上传多个 PDF
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@文件1.pdf" -F "files=@文件2.pdf"

# 知识库统计
curl http://localhost:8000/api/v1/kb/stats

# 提问（知识库内容）
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"你的问题"}'

# 提问（预计知识库无匹配，测试 Web 兜底）
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"今天天气怎么样"}'

# 删除指定文件
curl -X DELETE http://localhost:8000/api/v1/kb/files \
  -H "Content-Type: application/json" \
  -d '{"file_name":"文件名.pdf"}'

# 清空知识库（危险操作）
curl -X DELETE http://localhost:8000/api/v1/kb/clear
```

---

> 文档版本: V1.0 | 最后更新: 2026-07-31 | 适用系统版本: RAG KB System V1.0
