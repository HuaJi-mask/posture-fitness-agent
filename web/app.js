// ========== 配置 ==========
const API_BASE = "http://localhost:8000";
const STORAGE_KEY = "posture_fitness_sessions";

// ========== 状态 ==========
let sessions = [];
let currentSessionId = null;
let isLoading = false;
let historyLoadToken = 0;  // 切换会话时的令牌，防止历史串台
let abortController = null;   // 用于中断当前请求
let lastUserInput = "";       // 保存刚发送的用户输入，停止时回退到输入框

// ========== DOM ==========
const sessionList = document.getElementById("sessionList");
const sessionCount = document.getElementById("sessionCount");
const chatInner = document.getElementById("chatInner");
const chatContainer = document.querySelector(".chat-container");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");

// ========== 节点名称映射 ==========
const NODE_NAMES = {
    "triage": "导诊分析中",
    "posture_expert": "体态专家分析中",
    "fitness_expert": "健身专家分析中",
    "plan_generator": "方案生成中",
    "doctor_validator": "医生验证中",
    "coach_validator": "教练验证中",
    "merge_validation": "验证结果整合中",
    "plan_adjuster": "方案调整中",
    "enter_debate": "进入辩论中",
    "merge_expert_responses": "整合专家回复中"
};

// ========== 初始化 ==========
async function init() {
    bindEvents();
    await loadSessionsFromServer();
    if (sessions.length > 0) {
        switchSession(sessions[0].session_id);
    } else {
        showEmptyState();
    }
    messageInput.focus();
}

// ========== 事件绑定 ==========
function bindEvents() {
        sendBtn.addEventListener("click", function() {
        if (isLoading) {
            stopGeneration();  // 加载中点击 = 停止
        } else {
            sendMessage();     // 空闲时点击 = 发送
        }
    });

    messageInput.addEventListener("keydown", function(event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 120) + "px";
    });
}

// ========== 会话管理 ==========

function loadSessions() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
        sessions = JSON.parse(stored);
    }
    renderSessionList();
}

function saveSessions() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function renderSessionList() {
    sessionList.innerHTML = "";
    sessionCount.textContent = sessions.length;

    sessions.forEach(session => {
        const item = document.createElement("div");
        item.className = `session-item ${session.session_id === currentSessionId ? "active" : ""}`;
        item.onclick = () => switchSession(session.session_id);

        item.innerHTML = `
            <div class="session-title">${escapeHtml(session.title || "新对话")}</div>
            <div class="session-meta">
                <span>${session.created_at || ""}</span>
                <span class="session-delete" onclick="event.stopPropagation(); deleteSession('${session.session_id}')">删除</span>
            </div>
        `;
        sessionList.appendChild(item);
    });
}

function createNewSession() {
    if (isLoading) {
        alert("请等待当前回复完成后再新建对话");
        return;
    }

    const sessionId = "web_" + Date.now();
    const now = new Date();
    const timeStr = `${now.getMonth()+1}/${now.getDate()} ${now.getHours()}:${String(now.getMinutes()).padStart(2,'0')}`;

    const newSession = {
        session_id: sessionId,
        title: "新对话",
        created_at: timeStr,
        last_message: ""
    };

    sessions.unshift(newSession);
    saveSessions();
    switchSession(sessionId);
}

async function switchSession(sessionId) {
    if (isLoading) {
        alert("请等待当前回复完成后再切换对话");
        return;
    }

    // 令牌机制：快速连续切换时，丢弃过期请求的结果，避免历史串台
    const token = ++historyLoadToken;

    currentSessionId = sessionId;
    renderSessionList();
    chatInner.innerHTML = "";

    await loadSessionHistory(sessionId, token);
}

async function loadSessionHistory(sessionId, token) {
    try {
        const response = await fetch(`${API_BASE}/history?session_id=${sessionId}`);

        if (response.ok) {
            const data = await response.json();
            if (token !== historyLoadToken) return;  // 已切换到其它会话，丢弃本次结果

            if (data.messages && data.messages.length > 0) {
                data.messages.forEach(msg => {
                    addMessage(msg.role, msg.content, false);
                });
            } else {
                // 第一轮消息时 switchSession 与 sendMessage 并发：历史加载完成
                // 时消息可能已在渲染，此时覆盖会清掉正在输出的气泡
                if (!isLoading && chatInner.childElementCount === 0) {
                    showEmptyState();
                }
            }
        } else if (token === historyLoadToken) {
            if (!isLoading && chatInner.childElementCount === 0) {
                showEmptyState();
            }
        }
    } catch (error) {
        if (token !== historyLoadToken) return;
        console.error("加载历史失败:", error);
        if (!isLoading && chatInner.childElementCount === 0) {
            showEmptyState();
        }
    }
}

async function deleteSession(sessionId) {
    if (!confirm("确定要删除这个对话吗？")) return;

    try {
        // 调用后端删除接口
        const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
            method: "DELETE"
        });

        // 404 = 数据库里已不存在（如数据库被重置），按已删除处理，避免残留项无法删除
        if (!response.ok && response.status !== 404) {
            throw new Error("删除失败");
        }
    } catch (error) {
        alert("删除失败：" + error.message);
        return;
    }

    // 从本地列表移除
    sessions = sessions.filter(s => s.session_id !== sessionId);

    if (currentSessionId === sessionId) {
        if (sessions.length > 0) {
            switchSession(sessions[0].session_id);
        } else {
            currentSessionId = null;
            chatInner.innerHTML = "";
            showEmptyState();
        }
    }

    renderSessionList();
}

function updateSessionInfo(sessionId, title, lastMessage) {
    const session = sessions.find(s => s.session_id === sessionId);
    if (session) {
        if (session.title === "新对话" && title) {
            session.title = title.substring(0, 20);
        }
        session.last_message = lastMessage;
        saveSessions();
        renderSessionList();
    }
}

// ========== 聊天功能 ==========

function showEmptyState() {
    chatInner.innerHTML = `
        <div class="empty-state">
            <h2>👋 你好，我是你的健身助手</h2>
            <p>我可以帮你分析体态问题、制定健身计划<br>有什么可以帮你的吗？</p>
            <div class="features">
                <div class="feature">
                    <div class="feature-icon">🦴</div>
                    <div class="feature-title">体态评估</div>
                    <div class="feature-desc">多轮问诊，专业分析</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">💪</div>
                    <div class="feature-title">健身规划</div>
                    <div class="feature-desc">个性化训练方案</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">✅</div>
                    <div class="feature-title">双重验证</div>
                    <div class="feature-desc">医生+教练把关</div>
                </div>
            </div>
        </div>
    `;
}

function addMessage(role, content, animate = true) {
    const emptyState = chatInner.querySelector(".empty-state");
    if (emptyState) {
        chatInner.innerHTML = "";
    }

    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "👤" : "🤖";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content markdown-body";

    if (role === "assistant") {
        // AI 消息：渲染 Markdown（加粗、列表、标题等）。
        // marked/DOMPurify 已本地化到 vendor/（jsdelivr CDN 在国内经常加载失败，
        // 库缺失时这里会是 undefined，直接调用会抛 ReferenceError 导致整个
        // 回复气泡都渲染不出来）；万一本地文件也缺失，降级为纯文本显示，
        // 不能因为渲染库缺失就让页面报错
        if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
            // DOMPurify 防止 XSS（marked 输出里的 <script> 等会被过滤）
            contentDiv.innerHTML = DOMPurify.sanitize(marked.parse(content || ""));
        } else {
            contentDiv.textContent = content;
        }
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
    } else {
        // 用户消息：纯文本
        contentDiv.textContent = content;
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(avatar);
    }

    chatInner.appendChild(messageDiv);
    scrollToBottom();

    return contentDiv;
}

function showStatus(text) {
    const emptyState = chatInner.querySelector(".empty-state");
    if (emptyState) chatInner.innerHTML = "";

    const statusDiv = document.createElement("div");
    statusDiv.className = "status-indicator";
    statusDiv.id = "statusIndicator";
    statusDiv.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <span>${text}</span>
    `;
    chatInner.appendChild(statusDiv);
    scrollToBottom();
}

function updateStatus(text) {
    const statusDiv = document.getElementById("statusIndicator");
    if (statusDiv) {
        statusDiv.querySelector("span").textContent = text;
    }
}

function removeStatus() {
    const statusDiv = document.getElementById("statusIndicator");
    if (statusDiv) statusDiv.remove();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage() {
    const message = messageInput.value.trim();

    if (!message || isLoading) return;

    if (!currentSessionId) {
        createNewSession();
    }

    messageInput.value = "";
    messageInput.style.height = "auto";

    isLoading = true;
    // 按钮切换成"停止"（保持可用——禁用后点击事件不触发，停止功能就失效了）
    sendBtn.textContent = "停止";
    lastUserInput = message;           // 保存输入，停止时回退
    abortController = new AbortController();

    addMessage("user", message);
    updateSessionInfo(currentSessionId, message, message);
    showStatus("正在思考...");

    try {
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: currentSessionId,
                message: message
            }),
            signal: abortController.signal
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let lastContent = "";            // 最近一次回复内容（用于更新会话摘要）
        // 最终输出节点白名单：只有这些节点完成时才显示用户可见的回复气泡。
        // 中间节点（体态/健身专家、验证器、导诊）的 token 流不渲染成气泡，
        // 只通过 status 事件展示进度——多 Agent 流式输出的主流做法（中间过程
        // 折叠为状态，最终回复统一交付），避免 both 并行时出现多个对话框。
        // 注意：plan_generator 必须在白名单里——辩论链中方案生成后立即进入
        // 验证，若被当作中间节点，用户将永远看不到方案全文
        const FINAL_OUTPUT_NODES = new Set([
            "merge_expert_responses",  // 问诊汇聚（both 并行/单专家的最终回复）
            "plan_generator",          // 方案生成（辩论链的核心交付物）
            "plan_adjuster",           // 方案调整后的最终回复
            "merge_validation",        // 方案验证整合后的最终回复
        ]);

        // 处理一个完整的 SSE 事件（格式：data: {...}\n\n）
        function handleSSEEvent(eventStr) {
            const line = eventStr.trim();
            if (!line.startsWith("data: ")) return;

            let data;
            try {
                data = JSON.parse(line.substring(6));
            } catch (e) {
                console.error("解析 SSE 数据失败:", e);
                return;
            }

            if (data.type === "status") {
                // 节点开始：更新进度状态（"体态专家分析中..."→"整合专家回复中"）
                updateStatus(NODE_NAMES[data.node] || `${data.node} 处理中...`);
            } else if (data.type === "node") {
                // 节点完成：只有最终输出节点才显示回复气泡（addMessage 自带打字机效果）。
                // 中间节点的完整回复只记录摘要，不渲染成气泡
                if (FINAL_OUTPUT_NODES.has(data.node) && data.data?.response) {
                    removeStatus();
                    addMessage("assistant", data.data.response);
                    lastContent = data.data.response;
                } else if (data.data?.response) {
                    // 非最终节点（验证器、导诊等）：只记录摘要，不显示气泡
                    lastContent = data.data.response;
                }
            } else if (data.type === "token") {
                // Token 事件：忽略，不创建气泡。
                // 中间节点的 LLM token 流是"过程数据"，由最终输出节点统一交付。
                // 后端仍会发送 token 事件；若以后想给最终节点做真实流式（而非
                // 模拟打字机），可在此按 data.node 过滤，只渲染白名单节点的 token
            } else if (data.type === "done") {
                removeStatus();
                if (!lastContent) {
                    addMessage("assistant", "抱歉，没有生成回复。");
                } else {
                    updateSessionInfo(currentSessionId, null, lastContent);
                }
            } else if (data.type === "error") {
                removeStatus();
                addMessage("assistant", `出错了：${data.message}`);
            }
        }

        // 缓冲式解析：SSE 事件可能跨网络分块，按 "\n\n" 边界拼接完整后再处理
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // 把缓冲区中完整的事件（以 \n\n 结尾）切出来逐个处理
                let boundary;
                while ((boundary = buffer.indexOf("\n\n")) !== -1) {
                    const eventStr = buffer.slice(0, boundary);
                    buffer = buffer.slice(boundary + 2);
                    if (eventStr.trim()) handleSSEEvent(eventStr);
                }
            }
        } catch (e) {
            if (e && e.name === "AbortError") {
                // 用户主动点击"停止"：静默结束，不显示错误气泡（stopGeneration
                // 已删除用户气泡、回退输入框，这里只需安静收尾）
                console.log("用户停止了生成");
            } else {
                console.error("读取 SSE 流失败:", e);
                removeStatus();
                addMessage("assistant", `出错了：${e.message}`);
            }
        } finally {
            isLoading = false;
            sendBtn.textContent = "发送";
            sendBtn.disabled = false;
            abortController = null;
            removeStatus();
            if (lastContent) updateSessionInfo(currentSessionId, null, lastContent);
        }
    } catch (error) {
        if (error && error.name === "AbortError") {
            // 用户主动点击"停止"：静默结束
            console.log("用户停止了生成");
        } else {
            console.error("请求失败:", error);
            removeStatus();
            addMessage("assistant", `出错了：${error.message}`);
        }
    } finally {
        isLoading = false;
        sendBtn.textContent = "发送";
        sendBtn.disabled = false;
        abortController = null;
        messageInput.focus();
    }
}

function stopGeneration() {
    // 中断请求
    if (abortController) {
        abortController.abort();
        abortController = null;
    }

    // 回退用户输入到输入框
    if (lastUserInput) {
        messageInput.value = lastUserInput;
        messageInput.style.height = "auto";
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
        messageInput.focus();
        // 光标移到末尾
        messageInput.setSelectionRange(lastUserInput.length, lastUserInput.length);
    }

    // 删除刚发送的用户消息气泡（最后一条）
    const userMessages = chatInner.querySelectorAll(".message.user");
    if (userMessages.length > 0) {
        userMessages[userMessages.length - 1].remove();
    }

    // 移除状态指示器
    removeStatus();

    // 恢复按钮
    isLoading = false;
    sendBtn.textContent = "发送";
    sendBtn.disabled = false;

    lastUserInput = "";
}

// ========== 工具函数 ==========

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}


async function loadSessionsFromServer() {
    try {
        const response = await fetch(`${API_BASE}/sessions`);
        if (response.ok) {
            const data = await response.json();
            sessions = data.sessions || [];
            renderSessionList();
        }
    } catch (error) {
        console.error("从服务器加载会话失败:", error);
        // 失败了就用 localStorage 的
        loadSessionsFromLocal();
    }
}

function loadSessionsFromLocal() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
        sessions = JSON.parse(stored);
    }
    renderSessionList();
}


// 启动
init();

