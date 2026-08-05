"""
================================================================================
  RAG 知识库问答系统 - Streamlit 前端入口
  功能：
    - 多格式文档上传管理（PDF / Word / Markdown）
    - 多源类型选择（论文原文 / 综述解读 / 实验笔记）
    - 知识库统计与文件管理
    - 多轮对话（带历史消息 + 引用溯源展示）
    - 引用来源卡片展示（文件名、页码、相似度、来源类型）
  启动方式：
    streamlit run app_streamlit.py
  架构：
    Streamlit (前端) ──HTTP──> FastAPI (后端 /api/v1/*)
================================================================================
"""
import os
import sys
import uuid
import requests
import streamlit as st

# 确保项目根目录在 sys.path（便于直接读取 config）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import get_settings_cached

# ==================== 全局配置 ====================
_SETTINGS = get_settings_cached()
_API_BASE = os.getenv("RAG_API_BASE", f"http://localhost:{_SETTINGS.api.port}/api/v1")

# 支持的文件类型
SUPPORTED_TYPES = ["pdf", "docx", "md", "markdown"]
SOURCE_TYPES = ["论文原文", "综述解读", "实验笔记"]


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RAG 知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== 会话状态初始化 ====================
def init_session_state():
    """初始化 Streamlit 会话状态。"""
    defaults = {
        "session_id": str(uuid.uuid4()),       # 多轮对话 session_id
        "history": [],                          # 对话历史
        "api_base": _API_BASE,                  # 后端 API 地址
        "uploader_key": 0,                      # 上传组件 key（递增以清空）
        "kb_stats": None,                       # 知识库统计缓存
        "last_action": None,                    # 上一次操作描述
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ==================== API 调用封装 ====================
def call_upload(files, source_type: str, knowledge_base: str):
    """调用后端文档上传接口。"""
    url = f"{st.session_state.api_base}/documents/upload"
    multipart = [("files", (f.name, f.read(), "application/octet-stream")) for f in files]
    data = {"source_type": source_type, "knowledge_base": knowledge_base}
    try:
        resp = requests.post(url, files=multipart, data=data, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"上传失败: {e}")
        return None


def call_ask(question: str, session_id: str):
    """调用后端问答接口。"""
    url = f"{st.session_state.api_base}/qa/ask"
    payload = {"question": question, "session_id": session_id}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"问答失败: {e}")
        return None


def call_stats():
    """调用后端知识库统计接口。"""
    url = f"{st.session_state.api_base}/kb/stats"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def call_delete_file(file_name: str):
    url = f"{st.session_state.api_base}/kb/files"
    try:
        resp = requests.delete(url, json={"file_name": file_name}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"删除失败: {e}")
        return None


def call_clear_kb():
    url = f"{st.session_state.api_base}/kb/clear"
    try:
        resp = requests.delete(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"清空失败: {e}")
        return None


def refresh_kb_stats():
    """拉取知识库统计并缓存到 session_state。"""
    stats = call_stats()
    if stats and stats.get("code") == 200:
        st.session_state.kb_stats = stats.get("data", {})
    else:
        st.session_state.kb_stats = None


# ==================== 侧边栏：知识库管理 ====================
with st.sidebar:
    st.title("📚 知识库管理")

    # API 地址配置
    api_base_input = st.text_input(
        "后端 API 地址",
        value=st.session_state.api_base,
        help="FastAPI 后端地址，默认 http://localhost:8000/api/v1",
    )
    if api_base_input != st.session_state.api_base:
        st.session_state.api_base = api_base_input.rstrip("/")

    st.divider()

    # ---- 文档上传 ----
    st.subheader("📤 上传文档")
    uploaded_files = st.file_uploader(
        "选择文件（支持 PDF / Word / Markdown）",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    source_type = st.selectbox("来源类型", SOURCE_TYPES, help="用于多源知识融合")
    knowledge_base = st.text_input("知识库标识", value="default")

    if st.button("入库", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("请先选择文件")
        else:
            with st.spinner("处理中..."):
                result = call_upload(uploaded_files, source_type, knowledge_base)
            if result and result.get("code") == 200:
                st.success(result.get("message", "上传成功"))
                for item in result.get("data", []):
                    st.write(
                        f"📄 {item['file_name']} | "
                        f"{item['pages']} 段 → {item['chunks']} 块"
                    )
                # 上传成功后：清空文件选择器 + 自动刷新统计
                st.session_state.uploader_key += 1
                refresh_kb_stats()
                st.rerun()
            elif result:
                st.error(result.get("message", "上传失败"))

    st.divider()

    # ---- 知识库统计（持久化显示，不会因交互消失）----
    st.subheader("📊 知识库统计")

    # 首次加载自动拉取
    if st.session_state.kb_stats is None:
        refresh_kb_stats()

    col_refresh, _ = st.columns([1, 2])
    with col_refresh:
        if st.button("🔄 刷新", use_container_width=True):
            refresh_kb_stats()

    stats_data = st.session_state.kb_stats
    if stats_data:
        st.metric("向量库类型", stats_data.get("store_type", "-"))
        st.metric("总块数", stats_data.get("total_chunks", 0))
        st.metric("文件数", stats_data.get("total_files", 0))

        file_names = stats_data.get("file_names", [])
        if file_names:
            st.write("**文件列表：**")
            for fn in file_names:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {fn}")
                with col2:
                    if st.button("🗑️", key=f"del_{fn}", help=f"删除 {fn}"):
                        result = call_delete_file(fn)
                        if result and result.get("code") == 200:
                            st.success(f"已删除 {fn}")
                            refresh_kb_stats()
                            st.rerun()
                        else:
                            st.error(f"删除失败: {result}")
        else:
            st.caption("（知识库为空）")
    else:
        st.caption("（无法连接后端）")

    st.divider()

    # ---- 清空知识库 ----
    st.subheader("⚠️ 危险操作")
    if st.button("清空知识库", type="secondary", use_container_width=True):
        if st.session_state.get("confirm_clear"):
            result = call_clear_kb()
            if result and result.get("code") == 200:
                st.success("知识库已清空")
                st.session_state.confirm_clear = False
                refresh_kb_stats()
                st.rerun()
            else:
                st.error("清空失败")
        else:
            st.session_state.confirm_clear = True
            st.warning("再次点击确认清空（不可恢复）")
            st.rerun()

    st.divider()

    # ---- 重置对话 ----
    if st.button("🔄 新建对话", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()


# ==================== 主区域：多轮对话 ====================
st.title("💬 RAG 知识库智能问答")
st.caption(
    f"多源知识融合（论文原文 + 综述解读 + 实验笔记） | "
    f"Session: {st.session_state.session_id[:8]}"
)

# 渲染历史对话
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 引用溯源展示
        if msg.get("citations"):
            source_type_str = msg.get("source_type", "internal")
            source_label = "🌐 外网引用" if source_type_str == "external" else "📚 本地知识库"
            with st.expander(f"{source_label} - 共 {len(msg['citations'])} 条引用来源"):
                for cit in msg["citations"]:
                    similarity = cit.get("similarity", 0)
                    st.markdown(
                        f"**[{cit['citation_id']}] {cit['file_name']}** "
                        f"(第 {cit['page']} 页 | 相似度: {similarity:.4f} | "
                        f"来源: {cit.get('source_type', 'internal')})"
                    )
                    st.code(cit["original_text"][:300] + ("..." if len(cit["original_text"]) > 300 else ""))

# 输入框
if question := st.chat_input("请输入问题（支持多轮追问）"):
    # 显示用户问题
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 调用后端问答
    with st.chat_message("assistant"):
        with st.spinner("检索与生成中..."):
            result = call_ask(question, st.session_state.session_id)
        if not result:
            st.error("问答失败，请检查后端服务是否运行")
        elif result.get("code") != 200:
            st.error(result.get("message", "问答失败"))
        else:
            data = result.get("data", {})
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            source_type = data.get("source_type", "internal")

            st.markdown(answer)

            # 显示引用
            if citations:
                source_label = "🌐 外网引用" if source_type == "external" else "📚 本地知识库"
                with st.expander(f"{source_label} - 共 {len(citations)} 条引用来源"):
                    for cit in citations:
                        similarity = cit.get("similarity", 0)
                        st.markdown(
                            f"**[{cit['citation_id']}] {cit['file_name']}** "
                            f"(第 {cit['page']} 页 | 相似度: {similarity:.4f} | "
                            f"来源: {cit.get('source_type', 'internal')})"
                        )
                        st.code(cit["original_text"][:300] + ("..." if len(cit["original_text"]) > 300 else ""))

            # 写入历史
            st.session_state.history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "source_type": source_type,
            })
