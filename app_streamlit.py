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
    Streamlit (前端) --HTTP--> FastAPI (后端 /api/v1/*)
================================================================================
"""
import os
import sys
import uuid
import requests
import streamlit as st

# 确保项目根目录在 sys.path，便于直接读取 config
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
SUGGESTIONS = [
    "这篇论文的核心方法是什么？",
    "综述解读和论文原文的结论一致吗？",
    "实验笔记里记录了什么关键参数？",
]


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="知识库问答",
    page_icon=":material/menu_book:",
    layout="centered",
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
        st.error(f"上传失败: {e}", icon=":material/error:")
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
        st.error(f"问答失败: {e}", icon=":material/error:")
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
        st.error(f"删除失败: {e}", icon=":material/error:")
        return None


def call_clear_kb():
    url = f"{st.session_state.api_base}/kb/clear"
    try:
        resp = requests.delete(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"清空失败: {e}", icon=":material/error:")
        return None


def refresh_kb_stats():
    """拉取知识库统计并缓存到 session_state。"""
    stats = call_stats()
    if stats and stats.get("code") == 200:
        st.session_state.kb_stats = stats.get("data", {})
    else:
        st.session_state.kb_stats = None


# ==================== 引用与问答渲染 ====================
def render_source_stats(source_stats):
    """展示本地与外部来源的数量。"""
    if not source_stats:
        return
    parts = []
    if source_stats.get("internal_sources"):
        parts.append(f"本地来源 {source_stats['internal_sources']} 条")
    if source_stats.get("external_sources"):
        parts.append(f"外部来源 {source_stats['external_sources']} 条")
    if parts:
        st.caption("引用来源：" + "，".join(parts))


def render_citations(citations, source_type="internal"):
    """以紧凑卡片展示引用来源。"""
    if not citations:
        return
    label = "外部引用" if source_type == "external" else "本地引用"
    with st.expander(f"{label}，共 {len(citations)} 条", icon=":material/description:"):
        for cit in citations:
            with st.container(border=True):
                st.markdown(f"**{cit['citation_id']}** · {cit['file_name']}")
                page = cit.get("page", "-")
                similarity = cit.get("similarity", 0)
                kind = "外部" if cit.get("source_type") == "external" else "本地"
                st.caption(f"第 {page} 页 · 相似度 {similarity:.4f} · {kind}")
                snippet = cit.get("original_text", "")[:300]
                if len(cit.get("original_text", "")) > 300:
                    snippet += "..."
                st.code(snippet)


def ask_and_render(question: str):
    """追加用户问题，调用后端并渲染助手回答。"""
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/menu_book:"):
        with st.spinner("正在检索并生成回答"):
            result = call_ask(question, st.session_state.session_id)
        if not result:
            st.error("问答失败，请检查后端服务是否运行", icon=":material/error:")
            return
        if result.get("code") != 200:
            st.error(result.get("message", "问答失败"), icon=":material/error:")
            return

        data = result.get("data", {})
        answer = data.get("answer", "")
        citations = data.get("citations", [])
        source_type = data.get("source_type", "internal")

        st.markdown(answer)
        render_source_stats(data.get("source_stats"))
        render_citations(citations, source_type)

        st.session_state.history.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "source_type": source_type,
        })


@st.dialog("清空知识库", icon=":material/delete_forever:")
def clear_kb_dialog():
    """清空知识库前的二次确认弹窗。"""
    st.write("将删除全部文档与分块，此操作不可恢复。")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("确认清空", type="primary", icon=":material/check:", width="stretch"):
            result = call_clear_kb()
            if result and result.get("code") == 200:
                st.session_state.kb_stats = None
                st.toast("知识库已清空")
                st.rerun()
            else:
                st.error("清空失败，请检查后端连接", icon=":material/error:")
    with col2:
        if st.button("取消", icon=":material/close:", width="stretch"):
            st.rerun()


# ==================== 侧边栏：知识库管理 ====================
with st.sidebar:
    st.subheader("知识库管理")
    st.caption("维护文档、统计与后端连接")
    st.space("small")

    with st.container(border=True):
        api_base_input = st.text_input(
            "API 地址",
            value=st.session_state.api_base,
            help="FastAPI 后端地址，默认 http://localhost:8000/api/v1",
            label_visibility="collapsed",
            icon=":material/link:",
        )
        if api_base_input != st.session_state.api_base:
            st.session_state.api_base = api_base_input.rstrip("/")

    st.space("small")
    st.markdown("**上传文档**")

    with st.form("upload_form", border=False):
        uploaded_files = st.file_uploader(
            "选择文件",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            help="支持 PDF、Word、Markdown",
            label_visibility="collapsed",
        )
        source_type = st.segmented_control(
            "来源类型",
            SOURCE_TYPES,
            default=SOURCE_TYPES[0],
            key="source_type",
            help="用于多源知识融合",
        )
        knowledge_base = st.text_input(
            "知识库标识",
            value="default",
            help="文档所属知识库",
        )
        submitted = st.form_submit_button(
            "入库",
            type="primary",
            icon=":material/upload:",
            width="stretch",
        )

    if submitted:
        if not uploaded_files:
            st.warning("请先选择文件", icon=":material/upload_file:")
        else:
            with st.spinner("正在处理文档"):
                result = call_upload(uploaded_files, source_type, knowledge_base)
            if result and result.get("code") == 200:
                st.success(result.get("message", "上传成功"), icon=":material/check_circle:")
                with st.container(border=True):
                    for item in result.get("data", []):
                        st.caption(f"{item['file_name']}：{item['pages']} 段 → {item['chunks']} 块")
                # 上传成功后：清空文件选择器 + 自动刷新统计
                st.session_state.uploader_key += 1
                refresh_kb_stats()
                st.rerun()
            elif result:
                st.error(result.get("message", "上传失败"), icon=":material/error:")

    st.space("small")
    st.markdown("**知识库统计**")

    # 首次加载自动拉取
    if st.session_state.kb_stats is None:
        refresh_kb_stats()

    stats_data = st.session_state.kb_stats
    if stats_data:
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            st.badge("已连接", icon=":material/cloud_done:", color="green")
            if st.button("刷新", icon=":material/refresh:", key="refresh_stats", width="content"):
                refresh_kb_stats()
        st.metric("向量库", stats_data.get("store_type", "-"))
        st.metric("总块数", stats_data.get("total_chunks", 0))
        st.metric("文件数", stats_data.get("total_files", 0))

        file_names = stats_data.get("file_names", [])
        with st.expander(f"文件列表 · {len(file_names)}", icon=":material/description:"):
            if not file_names:
                st.caption("知识库为空")
            for fn in file_names:
                col1, col2 = st.columns([4, 1], vertical_alignment="center")
                with col1:
                    st.caption(fn)
                with col2:
                    if st.button(
                        "删除",
                        key=f"del_{fn}",
                        help=f"删除 {fn}",
                        icon=":material/delete:",
                        width="stretch",
                    ):
                        result = call_delete_file(fn)
                        if result and result.get("code") == 200:
                            st.toast(f"已删除 {fn}")
                            refresh_kb_stats()
                            st.rerun()
                        else:
                            st.error("删除失败", icon=":material/error:")
    else:
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            st.badge("未连接", icon=":material/cloud_off:", color="gray")
            if st.button("重试", icon=":material/refresh:", key="retry_stats", width="content"):
                refresh_kb_stats()
        st.caption("无法连接后端")

    st.space("small")
    st.markdown("**维护**")

    if st.button("清空知识库", icon=":material/delete_forever:", width="stretch"):
        clear_kb_dialog()

    if st.button("新建对话", icon=":material/add_comment:", width="stretch"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []
        st.toast("已新建对话")
        st.rerun()


# ==================== 主区域：多轮对话 ====================
st.title("知识库问答")
st.caption(f"多源检索与引用溯源，会话 {st.session_state.session_id[:8]}")

# 渲染历史对话
for msg in st.session_state.history:
    avatar = ":material/person:" if msg["role"] == "user" else ":material/menu_book:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        render_citations(msg.get("citations"), msg.get("source_type", "internal"))

# 空对话时提供建议问题
if not st.session_state.history:
    selected = st.pills(
        "建议问题",
        SUGGESTIONS,
        label_visibility="collapsed",
        key="starter_questions",
    )
    if selected:
        ask_and_render(selected)
        st.rerun()

# 输入框
if question := st.chat_input("输入问题", submit_mode="disable"):
    ask_and_render(question)
