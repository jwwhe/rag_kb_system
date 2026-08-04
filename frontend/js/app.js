/* ============================================================================
   RAG 知识库问答系统 — 前端交互逻辑
   双模式：员工问答端 + 管理后台端
   ============================================================================ */

// ==================== 全局状态 ====================
const API_BASE = '/api/v1';
let currentTab = 'qa';
let uploadQueue = [];  // { file, name, size }
let pendingModalAction = null;

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    if (currentTab === 'admin') refreshStats();
});

// ==================== 标签切换 ====================
function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
    document.getElementById('tabQa').style.display = tab === 'qa' ? '' : 'none';
    document.getElementById('tabAdmin').style.display = tab === 'admin' ? '' : 'none';
    if (tab === 'admin') refreshStats();
}

// ==================== 健康检查 ====================
async function checkHealth() {
    try {
        const res = await fetch('/health');
        const data = await res.json();
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');
        dot.className = 'status-dot online';
        text.textContent = `${data.llm || 'unknown'} · ${data.vector_store || ''}`;
    } catch (e) {
        document.getElementById('statusDot').className = 'status-dot offline';
        document.getElementById('statusText').textContent = '服务离线';
    }
}

// ==================== Toast 通知 ====================
function showToast(message, type = 'info', duration = 3500) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => { toast.remove(); }, duration);
}

// ==================== 模态对话框 ====================
function showModal(title, body, onConfirm) {
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalBody').innerHTML = body;
    document.getElementById('modalOverlay').style.display = 'flex';
    pendingModalAction = onConfirm;
}

function closeModal() {
    document.getElementById('modalOverlay').style.display = 'none';
    pendingModalAction = null;
}

function executeModalAction() {
    if (pendingModalAction) pendingModalAction();
    closeModal();
}

// ============================================================================
//  员工问答端
// ============================================================================

function handleInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
    }
}

function quickAsk(question) {
    document.getElementById('questionInput').value = question;
    askQuestion();
}

async function askQuestion() {
    const input = document.getElementById('questionInput');
    const btn = document.getElementById('btnSend');
    const question = input.value.trim();

    if (!question) return;
    if (btn.disabled) return;

    // UI 状态
    btn.disabled = true;
    btn.classList.add('loading');
    btn.innerHTML = '<span class="spinner"></span>';

    // 添加用户消息
    appendUserMessage(question);
    input.value = '';
    input.style.height = 'auto';

    // 添加加载消息
    const loadingMsg = appendLoadingMessage();

    try {
        const res = await fetch(`${API_BASE}/qa/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const result = await res.json();

        // 移除加载消息
        loadingMsg.remove();

        if (result.code === 200) {
            appendBotMessage(result.data);
        } else {
            appendErrorMessage(result.message || '问答请求失败');
        }
    } catch (e) {
        loadingMsg.remove();
        appendErrorMessage(`请求失败：${e.message}`);
    } finally {
        btn.disabled = false;
        btn.classList.remove('loading');
        btn.innerHTML = '<span>发送</span>';
        input.focus();
    }
}

// ---------- 消息渲染 ----------

function appendUserMessage(question) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'chat-msg user-msg';
    div.innerHTML = `
        <div class="msg-content">${escapeHtml(question)}</div>
        <div class="msg-avatar">👤</div>
    `;
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function appendBotMessage(data) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'chat-msg bot-msg';

    const sourceType = data.source_type || 'internal';
    const sourceTagClass = sourceType === 'external' ? 'source-tag-external' : 'source-tag-internal';
    const sourceTagText = sourceType === 'external' ? '🌐 外部网络搜索' : '📚 本地知识库';

    // 渲染引用卡片
    let citationsHtml = '';
    if (data.citations && data.citations.length > 0) {
        citationsHtml = '<div class="citation-cards">';
        data.citations.forEach(c => {
            const scoreClass = c.similarity >= 0.7 ? 'score-high'
                             : c.similarity >= 0.5 ? 'score-mid' : 'score-low';
            const similarityPercent = (c.similarity * 100).toFixed(1);
            citationsHtml += `
                <div class="citation-card">
                    <div class="citation-card-header">
                        <span class="citation-file">📄 ${escapeHtml(c.file_name)}</span>
                        <span class="citation-page">第 ${c.page} 页</span>
                        <span class="citation-similarity ${scoreClass}">相关度 ${similarityPercent}%</span>
                    </div>
                    <div class="citation-text">"${escapeHtml(c.original_text)}"</div>
                </div>
            `;
        });
        citationsHtml += '</div>';
    }

    div.innerHTML = `
        <div class="msg-avatar">🤖</div>
        <div class="msg-content">
            <span class="answer-source-tag ${sourceTagClass}">${sourceTagText}</span>
            <div class="answer-text">${formatAnswer(data.answer)}</div>
            ${citationsHtml}
        </div>
    `;

    container.appendChild(div);
    scrollToBottom();

    // 更新溯源面板
    updateSourcePanel(data);

    return div;
}

function appendLoadingMessage() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'chat-msg bot-msg';
    div.innerHTML = `
        <div class="msg-avatar">🤖</div>
        <div class="msg-content">
            <div style="display:flex;align-items:center;gap:12px;padding:12px 0;color:var(--text-muted);">
                <span class="spinner"></span> 正在检索知识库并生成回答...
            </div>
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function appendErrorMessage(message) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'chat-msg system-msg';
    div.innerHTML = `
        <div class="msg-avatar">⚠️</div>
        <div class="msg-content" style="color:var(--danger);">${escapeHtml(message)}</div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

// ---------- 溯源面板 ----------

function updateSourcePanel(data) {
    const panel = document.getElementById('sourceContent');

    if (!data.citations || data.citations.length === 0) {
        panel.innerHTML = `
            <div class="source-empty">
                <div class="empty-icon">📖</div>
                <p>本次回答无引用来源</p>
                <p class="empty-sub">${data.source_type === 'external' ? '答案来源于外部网络搜索' : '知识库中无匹配内容'}</p>
            </div>
        `;
        return;
    }

    let html = '';
    data.citations.forEach(c => {
        const scoreClass = c.similarity >= 0.7 ? 'score-high'
                         : c.similarity >= 0.5 ? 'score-mid' : 'score-low';
        const similarityPercent = (c.similarity * 100).toFixed(1);
        html += `
            <div class="source-item">
                <div class="source-item-header">
                    <span class="source-item-file">📄 ${escapeHtml(c.file_name)}</span>
                    <span class="source-item-page">第${c.page}页</span>
                </div>
                <div style="margin-bottom:4px;">
                    <span class="source-item-score ${scoreClass}">相关度 ${similarityPercent}%</span>
                </div>
                <div class="source-item-text">"${escapeHtml(c.original_text)}"</div>
            </div>
        `;
    });

    panel.innerHTML = html;
}

// ---------- 工具函数 ----------

function scrollToBottom() {
    const container = document.getElementById('chatMessages');
    setTimeout(() => { container.scrollTop = container.scrollHeight; }, 100);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatAnswer(text) {
    if (!text) return '';
    // 简单的 Markdown 转 HTML
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    return '<p>' + html + '</p>';
}

// ============================================================================
//  管理后台端
// ============================================================================

// ---------- 统计刷新 ----------

async function refreshStats() {
    try {
        // 知识库统计
        const statsRes = await fetch(`${API_BASE}/kb/stats`);
        const statsData = await statsRes.json();
        if (statsData.code === 200) {
            const s = statsData.data;
            document.getElementById('statFiles').textContent = s.total_files || 0;
            document.getElementById('statChunks').textContent = s.total_chunks || 0;
            document.getElementById('statStoreType').textContent = s.store_type || '-';
            renderFileTable(s.file_names || []);
        }

        // 健康检查
        const healthRes = await fetch('/health');
        const healthData = await healthRes.json();
        document.getElementById('statLLM').textContent = healthData.llm || '-';
    } catch (e) {
        console.error('刷新统计失败:', e);
    }
}

function renderFileTable(fileNames) {
    const tbody = document.getElementById('fileTableBody');
    if (fileNames.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="3" class="empty-cell">
                    <div class="table-empty">
                        <p>📭 知识库为空</p>
                        <p class="empty-sub">请先上传 PDF 文档</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = fileNames.map(name => `
        <tr>
            <td>📄 ${escapeHtml(name)}</td>
            <td style="color:var(--text-muted);">-</td>
            <td>
                <button class="btn btn-secondary" style="font-size:12px;padding:4px 12px;"
                        onclick="deleteFile('${escapeHtml(name)}')">
                    🗑 删除
                </button>
            </td>
        </tr>
    `).join('');
}

// ---------- 文件上传 ----------

function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.add('drag-over');
}

function handleDragLeave(e) {
    document.getElementById('uploadArea').classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.remove('drag-over');
    addFilesToQueue(e.dataTransfer.files);
}

function handleFileSelect(e) {
    addFilesToQueue(e.target.files);
    e.target.value = '';
}

function addFilesToQueue(fileList) {
    let added = 0;
    for (const file of fileList) {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showToast(`跳过非 PDF 文件: ${file.name}`, 'warning');
            continue;
        }
        if (uploadQueue.some(f => f.name === file.name)) {
            showToast(`文件已存在队列中: ${file.name}`, 'warning');
            continue;
        }
        uploadQueue.push({ file, name: file.name, size: file.size });
        added++;
    }

    if (added > 0) {
        showToast(`已添加 ${added} 个文件到上传队列`, 'info');
        renderUploadQueue();
    }
}

function renderUploadQueue() {
    const container = document.getElementById('uploadQueue');
    const list = document.getElementById('queueList');
    const progress = document.getElementById('queueProgress');

    if (uploadQueue.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = '';
    progress.textContent = `${uploadQueue.length} 个文件`;

    list.innerHTML = uploadQueue.map((item, index) => `
        <div class="queue-item">
            <span class="queue-item-name">📄 ${escapeHtml(item.name)}</span>
            <span class="queue-item-size">${formatFileSize(item.size)}</span>
            <span class="queue-item-status status-pending">待上传</span>
            <button class="queue-item-remove" onclick="removeFromQueue(${index})" title="移除">×</button>
        </div>
    `).join('');
}

function removeFromQueue(index) {
    uploadQueue.splice(index, 1);
    renderUploadQueue();
}

function clearQueue() {
    uploadQueue = [];
    renderUploadQueue();
}

async function startUpload() {
    if (uploadQueue.length === 0) {
        showToast('请先添加文件', 'warning');
        return;
    }

    const btn = document.getElementById('btnUpload');
    btn.disabled = true;
    btn.textContent = '上传中...';

    const formData = new FormData();
    uploadQueue.forEach(item => {
        formData.append('files', item.file);
    });

    try {
        const res = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData,
        });
        const result = await res.json();

        if (result.code === 200) {
            const dataList = result.data || [];
            const totalChunks = dataList.reduce((sum, d) => sum + (d.chunks || 0), 0);
            showToast(
                `上传成功！${dataList.length} 个文件，共 ${totalChunks} 个分块`,
                'success',
                5000
            );
            uploadQueue = [];
            renderUploadQueue();
            refreshStats();
        } else {
            showToast(result.message || '上传失败', 'error');
        }
    } catch (e) {
        showToast(`上传失败：${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '开始上传';
    }
}

// ---------- 文件删除 ----------

function deleteFile(fileName) {
    showModal(
        '确认删除',
        `<p>确定要删除文件 <strong>${escapeHtml(fileName)}</strong> 吗？</p>
         <p>该文件的所有分块及其向量数据将被移除，后续将无法检索到该文件的内容。</p>`,
        async () => {
            try {
                const res = await fetch(`${API_BASE}/kb/files`, {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_name: fileName }),
                });
                const result = await res.json();
                if (result.code === 200) {
                    showToast(`已删除: ${fileName}`, 'success');
                    refreshStats();
                } else {
                    showToast(result.message || '删除失败', 'error');
                }
            } catch (e) {
                showToast(`删除失败：${e.message}`, 'error');
            }
        }
    );
}

// ---------- 清空知识库 ----------

function confirmClearKB() {
    showModal(
        '⚠️ 确认清空知识库',
        `<p style="color:var(--danger);font-weight:600;">此操作将删除知识库中的所有文档数据！</p>
         <p>包括所有已上传 PDF 的分块、向量数据和索引。此操作不可恢复。</p>
         <p>建议清空前先备份原始 PDF 文件。确定要继续吗？</p>`,
        async () => {
            try {
                const res = await fetch(`${API_BASE}/kb/clear`, { method: 'DELETE' });
                const result = await res.json();
                if (result.code === 200) {
                    showToast('知识库已清空', 'success');
                    refreshStats();
                } else {
                    showToast(result.message || '清空失败', 'error');
                }
            } catch (e) {
                showToast(`清空失败：${e.message}`, 'error');
            }
        }
    );
}

// ---------- 工具函数 ----------

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
