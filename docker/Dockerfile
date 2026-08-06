# ============================================================================
#  RAG 知识库问答系统 - Docker 镜像
#  构建：docker build -t rag-kb:latest -f docker/Dockerfile .
# ============================================================================
FROM python:3.10-slim

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# 时区
ENV TZ=Asia/Shanghai

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=production

COPY requirements.txt .

# 先装 CPU 版 PyTorch（体积小，避免 CUDA 大包超时）
RUN pip install --no-cache-dir --timeout 600 --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple \
    torch --extra-index-url https://mirrors.aliyun.com/pytorch-wheels/cpu

# 再装其他依赖（清华镜像，文件小不会超时）
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "run.py"]
