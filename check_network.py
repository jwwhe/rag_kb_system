"""
================================================================================
  网络连通性检测脚本
  运行方式: python check_network.py
  用于检测 HuggingFace 官方和国内镜像的连通性
================================================================================
"""
import os
import sys
import time

# 测试目标列表
TARGETS = [
    ("HuggingFace 官方", "https://huggingface.co"),
    ("hf-mirror.com", "https://hf-mirror.com"),
    ("hf.xeduapi.com", "https://hf.xeduapi.com"),
    ("modelscope", "https://www.modelscope.cn"),
]

def test_http(url, timeout=8):
    """测试 HTTP 连通性"""
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=timeout)
        return True, f"HTTP {resp.status}"
    except Exception as e:
        return False, str(e)[:80]

def test_model_file(mirror_url, timeout=8):
    """测试能否访问模型文件"""
    test_url = f"{mirror_url}/BAAI/bge-m3/resolve/main/config.json"
    try:
        import urllib.request
        req = urllib.request.Request(test_url, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=timeout)
        return True, f"HTTP {resp.status}"
    except Exception as e:
        return False, str(e)[:80]

def main():
    print("=" * 60)
    print("  RAG 知识库问答系统 — 网络连通性检测")
    print("=" * 60)
    print()

    # 检测基本连通性
    print(">>> 1. 镜像站点连通性检测")
    print("-" * 40)
    for name, url in TARGETS:
        ok, msg = test_http(url)
        status = "[OK] 可达" if ok else "[FAIL] 不可达"
        print(f"  {status}  {name}")
        print(f"         {url} → {msg}")
    print()

    # 检测模型文件是否可下载
    print(">>> 2. bge-m3 模型文件下载测试")
    print("-" * 40)
    for name, url in TARGETS:
        ok, msg = test_model_file(url)
        status = "[OK] 可下载" if ok else "[FAIL] 失败"
        print(f"  {status}  {name}")
        print(f"         {msg}")
    print()

    # 当前环境变量
    print(">>> 3. 当前 HuggingFace 配置")
    print("-" * 40)
    hf_endpoint = os.environ.get("HF_ENDPOINT", "未设置")
    hf_home = os.environ.get("HF_HOME", "默认 (~/.cache/huggingface)")
    print(f"  HF_ENDPOINT = {hf_endpoint}")
    print(f"  HF_HOME     = {hf_home}")
    print()

    # 推荐配置
    print(">>> 4. 推荐操作")
    print("-" * 40)
    print("  如果 hf-mirror.com 可达，直接在终端中启动即可：")
    print("    python run.py")
    print()
    print("  如果所有镜像都不可达，请检查：")
    print("    1. 网络代理设置")
    print("    2. 防火墙是否阻止了 Python 的网络访问")
    print("    3. 尝试使用 VPN")
    print("=" * 60)


if __name__ == "__main__":
    main()
