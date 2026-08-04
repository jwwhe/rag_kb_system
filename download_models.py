"""
================================================================================
  模型预下载脚本
  运行方式: python download_models.py
  在启动服务前手动下载所需模型，避免首次启动时等待
================================================================================
"""
import os
import sys

# 设置镜像（必须最先执行）
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = os.environ["HF_ENDPOINT"]

MODELS = [
    {
        "name": "BAAI/bge-m3",
        "size": "~2.0 GB",
        "purpose": "文本向量化（Layer 1 嵌入层）",
        "local_dir": None,  # None = 使用默认缓存目录
    },
    {
        "name": "BAAI/bge-reranker-v2-m3",
        "size": "~1.0 GB",
        "purpose": "检索结果重排序（Layer 3 检索层）",
        "local_dir": None,
    },
]

CACHE_BASE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def download_with_hub(model_name):
    """使用 huggingface_hub 下载模型"""
    from huggingface_hub import snapshot_download

    print(f"  正在下载: {model_name}")
    print(f"  镜像地址: {os.environ['HF_ENDPOINT']}")

    local_path = snapshot_download(
        repo_id=model_name,
        resume_download=True,
        max_workers=4,
    )
    return local_path


def download_with_sentence_transformers(model_name):
    """使用 sentence-transformers 加载模型（自动触发下载+缓存）"""
    from sentence_transformers import SentenceTransformer

    print(f"  正在加载: {model_name}")
    model = SentenceTransformer(model_name)
    # 模型加载成功即表示下载完成
    print(f"  模型已缓存")
    return None


def check_existing(model_name):
    """检查模型是否已缓存"""
    # 检查 huggingface hub 缓存
    safe_name = model_name.replace("/", "--")
    cache_path = os.path.join(CACHE_BASE, f"models--{safe_name}")
    if os.path.isdir(cache_path):
        # 计算大小
        total_size = 0
        for root, dirs, files in os.walk(cache_path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        size_mb = total_size / (1024 * 1024)
        return True, f"{size_mb:.0f} MB"
    return False, "未缓存"


def main():
    print("=" * 60)
    print("  RAG 知识库问答系统 — 模型预下载")
    print("=" * 60)
    print()
    print(f"  镜像地址: {os.environ['HF_ENDPOINT']}")
    print(f"  缓存目录: {CACHE_BASE}")
    print()

    for i, model in enumerate(MODELS, 1):
        print(f"[{i}/{len(MODELS)}] {model['name']}")
        print(f"  用途: {model['purpose']}")
        print(f"  大小: {model['size']}")

        # 检查是否已下载
        exists, info = check_existing(model["name"])
        if exists:
            print(f"  状态: 已缓存 ({info})，跳过下载")
            print()
            continue

        print(f"  状态: 未缓存，开始下载...")
        try:
            local_path = download_with_hub(model["name"])
            print(f"  结果: 下载完成 → {local_path}")
        except Exception as e:
            print(f"  错误: {e}")
            print(f"  提示: 请检查网络连接或尝试其他镜像")

        print()

    # 最终检查
    print("=" * 60)
    print("  模型缓存状态:")
    all_ok = True
    for model in MODELS:
        exists, info = check_existing(model["name"])
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {model['name']}: {info}")
        if not exists:
            all_ok = False

    if all_ok:
        print()
        print("  全部模型已就绪，可以启动服务：python run.py")
    else:
        print()
        print("  部分模型未下载成功，请检查网络后重试")

    print("=" * 60)


if __name__ == "__main__":
    main()
