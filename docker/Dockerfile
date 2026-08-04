# ============================================================================
#  RAG 知识库问答系统 - Docker 镜像
#  构建方式: docker build -t rag-kb-system:1.0 -f docker/Dockerfile .
# ============================================================================
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=production \
    TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/chroma_db

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "run.py"]
