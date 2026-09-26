/**
 * RAG 知识库 AI Agent 前端交互逻辑
 * 功能：文档上传、知识库列表管理、多轮问答（会话记忆）、检索调试视图
 */

// ---------------- DOM 元素引用 ----------------
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const docCountEl = document.getElementById('docCount');
const chunkCountEl = document.getElementById('chunkCount');
const docListEl = document.getElementById('docList');
const messageList = document.getElementById('messageList');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');

// 无相关资料的固定文案（与后端保持一致）
const NO_INFO_REPLY = '文档内无相关资料';

// 当前会话 ID：首轮为空，由后端创建后回填；"新对话"时重新置空
let sessionId = null;

// ---------------- 初始化 ----------------
document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
    bindEvents();
    checkModelReady();
});

/**
 * 轮询后端模型就绪状态。
 * 首次启动时后端需联网下载 Embedding 模型，期间禁用上传并给出提示。
 */
async function checkModelReady() {
    try {
        const resp = await fetch('/api/health');
        const data = await resp.json();
        if (data.model_ready) {
            uploadBox.style.pointerEvents = '';
            uploadBox.style.opacity = '';
            return; // 模型已就绪，停止轮询
        }
    } catch (err) {
        console.error('健康检查失败', err);
    }

    // 未就绪：禁用上传区域并提示，3 秒后重试
    uploadBox.style.pointerEvents = 'none';
    uploadBox.style.opacity = '0.6';
    uploadProgress.classList.remove('hidden');
    progressFill.style.width = '60%';
    progressText.textContent = '⏳ Embedding 模型加载中（首次启动需联网下载，请查看后端控制台进度）...';
    setTimeout(checkModelReady, 3000);
}

/**
 * 绑定上传、发送、新对话等交互事件
 */
function bindEvents() {
    // 点击上传区域触发文件选择
    uploadBox.addEventListener('click', () => fileInput.click());

    // 文件选择后逐个上传
    fileInput.addEventListener('change', () => {
        handleFiles(fileInput.files);
        fileInput.value = '';
    });

    // 拖拽上传支持
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });
    uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    // 发送按钮与回车快捷键
    sendBtn.addEventListener('click', sendQuestion);
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    // 新对话按钮：清空服务端历史并重置本地视图
    newChatBtn.addEventListener('click', startNewChat);

    // 输入框自动增高
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + 'px';
    });
}

// ---------------- 文档上传 ----------------

/**
 * 顺序上传选中的文件列表
 * @param {FileList} files 待上传文件
 */
async function handleFiles(files) {
    const allowed = ['.pdf', '.txt', '.md', '.markdown'];
    for (const file of files) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowed.includes(ext)) {
            alert(`不支持的文件类型：${file.name}（仅支持 PDF/TXT/Markdown）`);
            continue;
        }
        await uploadFile(file);
    }
}

/**
 * 上传单个文件并展示处理进度
 * @param {File} file 待上传文件
 */
async function uploadFile(file) {
    uploadProgress.classList.remove('hidden');
    progressFill.style.width = '30%';
    progressText.textContent = `正在解析并向量化：${file.name}`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        // 设置 120 秒超时，避免网络异常时请求无限挂起导致页面卡住
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 120000);

        const resp = await fetch('/api/documents/upload', {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });
        clearTimeout(timer);
        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.detail || '上传失败');
        }

        progressFill.style.width = '100%';
        progressText.textContent = `✅ ${file.name} 入库成功（${data.chunk_count} 个片段）`;
        await loadDocuments();
    } catch (err) {
        progressFill.style.width = '100%';
        progressFill.style.background = '#ef4444';
        // 超时中断时给出更友好的提示文案
        const msg = err.name === 'AbortError'
            ? '处理超时，请检查后端控制台日志后重试'
            : (err.message || '上传失败');
        progressText.textContent = `❌ ${msg}`;
        // 失败信息保留 6 秒便于阅读
        setTimeout(() => {
            uploadProgress.classList.add('hidden');
            progressFill.style.width = '0';
            progressFill.style.background = '';
        }, 6000);
        return;
    }

    // 2.5 秒后隐藏进度条并恢复样式
    setTimeout(() => {
        uploadProgress.classList.add('hidden');
        progressFill.style.width = '0';
        progressFill.style.background = '';
    }, 2500);
}

// ---------------- 知识库管理 ----------------

/**
 * 拉取知识库统计与文档列表并渲染
 */
async function loadDocuments() {
    try {
        const resp = await fetch('/api/documents');
        const data = await resp.json();

        docCountEl.textContent = data.document_count;
        chunkCountEl.textContent = data.chunk_count;
        renderDocList(data.documents);
    } catch (err) {
        console.error('加载文档列表失败', err);
    }
}

/**
 * 渲染文档列表
 * @param {Array} documents 文档元信息数组
 */
function renderDocList(documents) {
    if (!documents || documents.length === 0) {
        docListEl.innerHTML = '<li class="doc-empty">暂无文档，请先上传</li>';
        return;
    }

    docListEl.innerHTML = documents.map((doc) => `
        <li class="doc-item">
            <div class="doc-info">
                <div class="doc-name" title="${escapeHtml(doc.doc_name)}">${escapeHtml(doc.doc_name)}</div>
                <div class="doc-meta">${doc.chunk_count} 片段 · ${doc.upload_time}</div>
            </div>
            <button class="doc-delete" title="删除文档" onclick="deleteDocument('${doc.doc_id}')">🗑</button>
        </li>
    `).join('');
}

/**
 * 删除指定文档并刷新列表
 * @param {string} docId 文档 ID
 */
async function deleteDocument(docId) {
    if (!confirm('确定删除该文档及其全部知识片段？')) return;
    try {
        const resp = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
        if (!resp.ok) {
            const data = await resp.json();
            throw new Error(data.detail || '删除失败');
        }
        await loadDocuments();
    } catch (err) {
        alert(err.message);
    }
}

// ---------------- 会话管理（多轮对话） ----------------

/**
 * 开始新对话：通知后端清空历史（尽力而为）、重置本地会话 ID 与消息列表。
 */
async function startNewChat() {
    const oldSessionId = sessionId;
    sessionId = null;
    // 通知后端清空旧会话历史；失败不阻塞，本地置空已能切断上下文
    if (oldSessionId) {
        try {
            await fetch('/api/session/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: oldSessionId }),
            });
        } catch (err) {
            console.warn('清空服务端会话失败（已忽略）', err);
        }
    }
    renderWelcome();
    questionInput.focus();
}

/**
 * 用欢迎卡片重置消息区（新对话时调用）。
 */
function renderWelcome() {
    messageList.innerHTML = `
        <div class="welcome-card">
            <h3>👋 已开启新对话</h3>
            <ul>
                <li>上下文已清空，接下来的提问不会携带之前的对话内容</li>
            </ul>
        </div>
    `;
}

// ---------------- 问答对话 ----------------

/**
 * 发送用户问题并展示 Agent 回答（携带会话 ID 以支持多轮上下文）。
 */
async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question || sendBtn.disabled) return;

    // 展示用户消息并清空输入框
    appendMessage('user', question);
    questionInput.value = '';
    questionInput.style.height = 'auto';
    sendBtn.disabled = true;

    // 展示"检索中"占位消息
    const thinkingEl = appendThinking();

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // session_id 首轮为 null，后端创建后在响应中返回并回填
            body: JSON.stringify({ question, session_id: sessionId }),
        });
        const data = await resp.json();

        thinkingEl.remove();

        if (!resp.ok) {
            appendMessage('assistant', `⚠️ ${data.detail || '请求失败'}`);
            return;
        }

        // 回填会话 ID，后续提问携带同一 ID 形成多轮对话
        sessionId = data.session_id;

        // 渲染回答、引用来源与检索调试视图
        appendAssistantMessage(data);
    } catch (err) {
        thinkingEl.remove();
        appendMessage('assistant', `⚠️ 网络错误：${err.message}`);
    } finally {
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

/**
 * 渲染助手回答（含引用来源折叠面板与检索调试视图）
 * @param {Object} data 后端返回 {answer, sources, retrieval_count, debug}
 */
function appendAssistantMessage(data) {
    const isNoInfo = data.answer === NO_INFO_REPLY;

    const messageEl = document.createElement('div');
    messageEl.className = 'message assistant';

    messageEl.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="bubble ${isNoInfo ? 'no-info' : ''}">
            <div class="answer-text">${escapeHtml(data.answer)}</div>
            ${renderSources(data.sources)}
            ${renderDebugPanel(data.debug)}
        </div>
    `;
    messageList.appendChild(messageEl);
    scrollToBottom();
}

/**
 * 组装引用来源折叠面板 HTML，展示粗排余弦分与重排 logit。
 * @param {Array} sources 入选片段列表
 * @returns {string} 面板 HTML；无来源时返回空字符串
 */
function renderSources(sources) {
    if (!sources || sources.length === 0) return '';

    const items = sources.map((s) => `
        <div class="source-item">
            <div class="source-head">
                <span>[${s.index}] ${escapeHtml(s.doc_name)} · 第 ${s.chunk_index + 1} 段</span>
                <span class="source-scores">
                    余弦 ${s.retrieval_score}${s.rerank_score != null ? ` · 重排 ${s.rerank_score}` : ''}
                </span>
            </div>
            <div class="source-content">${escapeHtml(s.content)}</div>
        </div>
    `).join('');

    return `
        <details class="sources">
            <summary>📎 引用来源（${sources.length} 个片段）</summary>
            ${items}
        </details>
    `;
}

/**
 * 组装检索调试面板 HTML：检索管线信息 + 全部候选片段的双分数条与名次变化。
 * 用于调参时直观判断粗排/重排是否按预期工作。
 * @param {Object} debug 后端返回的检索调试信息
 * @returns {string} 面板 HTML；无调试信息时返回空字符串
 */
function renderDebugPanel(debug) {
    if (!debug) return '';

    // 重排状态文案：已生效 / 已降级 / 未启用
    const rerankStatus = !debug.rerank_enabled
        ? '未启用'
        : (debug.rerank_applied ? '已生效' : '⚠️ 已降级');

    // 多轮改写时展示实际用于检索的问题
    const rewriteLine = debug.rewritten
        ? `<div class="debug-meta-line">🔁 改写后检索问题：${escapeHtml(debug.retrieval_query)}</div>`
        : '';

    // 计算重排分归一化基准（仅取正分最大值），用于条形图宽度
    const rerankScores = debug.candidates
        .map((c) => c.rerank_score)
        .filter((v) => v != null && v > 0);
    const maxRerank = rerankScores.length ? Math.max(...rerankScores) : 1;

    // 逐条候选渲染
    const rows = debug.candidates.map((c) => {
        // 名次变化：▲上升 / ＝不变 / ▼下降 / 淘汰
        let moveHtml = '<span class="move-out">淘汰</span>';
        if (c.final_rank != null) {
            if (c.final_rank === c.coarse_rank) {
                moveHtml = '<span class="move-same">＝</span>';
            } else if (c.final_rank < c.coarse_rank) {
                moveHtml = `<span class="move-up">▲${c.coarse_rank - c.final_rank}</span>`;
            } else {
                moveHtml = `<span class="move-down">▼${c.final_rank - c.coarse_rank}</span>`;
            }
        }

        // 余弦条：分数大致在 0-1，直接换算百分比并钳制
        const cosinePct = Math.min(Math.max(c.retrieval_score * 100, 0), 100);
        // 重排条：logit 可正可负，按正分最大值归一化；负值不填充
        const rerankPct = c.rerank_score != null && c.rerank_score > 0
            ? (c.rerank_score / maxRerank) * 100
            : 0;
        const rerankVal = c.rerank_score != null ? c.rerank_score : '—';

        return `
            <div class="debug-row ${c.selected ? 'selected' : 'dropped'}">
                <div class="debug-rank">
                    <span class="rank-badge ${c.selected ? 'rank-in' : 'rank-none'}">
                        ${c.final_rank != null ? `#${c.final_rank}` : '—'}
                    </span>
                    ${moveHtml}
                    <span class="rank-coarse">粗排#${c.coarse_rank}</span>
                </div>
                <div class="debug-main">
                    <div class="debug-doc">${escapeHtml(c.doc_name)} · 第 ${c.chunk_index + 1} 段</div>
                    <div class="bar-line">
                        <span class="bar-label">余弦</span>
                        <div class="bar-track"><div class="bar-fill bar-cosine" style="width:${cosinePct.toFixed(1)}%"></div></div>
                        <span class="bar-val">${c.retrieval_score}</span>
                    </div>
                    <div class="bar-line">
                        <span class="bar-label">重排</span>
                        <div class="bar-track"><div class="bar-fill bar-rerank" style="width:${rerankPct.toFixed(1)}%"></div></div>
                        <span class="bar-val">${rerankVal}</span>
                    </div>
                    <details class="debug-snippet">
                        <summary>查看片段原文</summary>
                        <div class="debug-snippet-content">${escapeHtml(c.content)}</div>
                    </details>
                </div>
            </div>
        `;
    }).join('');

    return `
        <details class="debug-panel">
            <summary>🔍 检索调试（粗排 ${debug.coarse_count} → 重排${rerankStatus} → 入选 ${debug.final_count}）</summary>
            <div class="debug-meta">
                <div class="debug-meta-line">候选池上限：${debug.candidate_k} · 粗排阈值后剩余 ${debug.coarse_count} 个</div>
                <div class="debug-meta-line">重排模型：${debug.rerank_model ? escapeHtml(debug.rerank_model) : '—'} · 状态：${rerankStatus}</div>
                ${rewriteLine}
            </div>
            <div class="debug-list">${rows}</div>
        </details>
    `;
}

/**
 * 追加一条普通消息气泡
 * @param {string} role user 或 assistant
 * @param {string} text 消息文本
 */
function appendMessage(role, text) {
    const avatar = role === 'user' ? '🧑' : '🤖';
    const el = document.createElement('div');
    el.className = `message ${role}`;
    el.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="bubble"><div class="answer-text">${escapeHtml(text)}</div></div>
    `;
    messageList.appendChild(el);
    scrollToBottom();
    return el;
}

/**
 * 追加"检索思考中"占位消息
 * @returns {HTMLElement} 占位元素（用于后续移除）
 */
function appendThinking() {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="bubble">
            <div class="thinking"><span></span><span></span><span></span>&nbsp;正在检索与重排知识库片段...</div>
        </div>
    `;
    messageList.appendChild(el);
    scrollToBottom();
    return el;
}

// ---------------- 工具函数 ----------------

/**
 * 消息列表滚动到底部
 */
function scrollToBottom() {
    messageList.scrollTop = messageList.scrollHeight;
}

/**
 * HTML 转义，防止文档内容注入
 * @param {string} str 原始字符串
 * @returns {string} 转义后的字符串
 */
function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
