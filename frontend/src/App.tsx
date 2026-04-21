import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

function normalizeMathDelimiters(src: string) {
  if (!src) return "";
  // MinerU / LaTeX sources frequently use \[ \] and \( \) which remark-math
  // does not recognize. Convert them to $$ $$ and $ $ so KaTeX picks them up.
  let s = src.replace(/\\\[([\s\S]+?)\\\]/g, (_m, body) => `\n$$${body}$$\n`);
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_m, body) => `$${body}$`);
  return s;
}

function MarkdownView({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
    >
      {normalizeMathDelimiters(text || "")}
    </ReactMarkdown>
  );
}

type TokenResponse = {
  access_token: string;
  token_type: string;
};

type DocumentItem = {
  id: number;
  filename: string;
  file_url: string;
  process_status: string;
  process_error: string;
  chunk_count: number;
  processed_at?: string;
  created_at: string;
};

type ReaderChunk = {
  chunk_index: number;
  page?: number | null;
  text: string;
};

type ChatCitation = {
  document_id: number;
  document_name: string;
  chunk_index: number;
  page?: number | null;
  score?: number;
};

type ChatMsg = { role: "user" | "assistant"; content: string; citations?: ChatCitation[] };
type RetrievalHit = { score: number; text: string; document_name: string; chunk_index: number; page?: number | null };
type ModelListResp = { provider: string; base_url: string; default_model: string; models: { name: string }[] };

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const DEFAULT_MODELS = ["MiniMax-M2.5", "DeepSeek-V3.2", "Qwen3-235B-A22B-Thinking-2507", "DeepSeek-R1-0528"];
const SIDEBAR_WIDTH = 280;
const SIDEBAR_COLLAPSED_WIDTH = 54;
const CHAT_MIN_RATIO = 0.22;
const CHAT_MAX_RATIO = 0.6;
const CHAT_DEFAULT_RATIO = 0.33;
const CHAT_PRESETS: { label: string; ratio: number }[] = [
  { label: "25%", ratio: 0.25 },
  { label: "33%", ratio: 0.33 },
  { label: "40%", ratio: 0.4 },
  { label: "50%", ratio: 0.5 }
];

function clampRatio(r: number) {
  return Math.max(CHAT_MIN_RATIO, Math.min(CHAT_MAX_RATIO, r));
}
function ratioToPx(r: number) {
  return Math.round(window.innerWidth * clampRatio(r));
}

function chatThreadKey(username: string, docId: number | undefined | null) {
  return `chat-thread:${username || "_"}:${docId ?? "global"}`;
}
function loadChatThread(username: string, docId: number | undefined | null): ChatMsg[] {
  try {
    const raw = localStorage.getItem(chatThreadKey(username, docId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (m: any): m is ChatMsg =>
        m && typeof m.content === "string" && (m.role === "user" || m.role === "assistant")
    );
  } catch {
    return [];
  }
}
function saveChatThread(username: string, docId: number | undefined | null, msgs: ChatMsg[]) {
  try {
    // Cap size to avoid localStorage blowout (last 200 messages / ~200KB).
    const capped = msgs.slice(-200);
    localStorage.setItem(chatThreadKey(username, docId), JSON.stringify(capped));
  } catch {
    /* quota exceeded etc. — non-fatal */
  }
}

function authHeaders(token: string) {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`
  };
}

async function parseResp<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.detail || "Request failed");
  }
  return (await resp.json()) as T;
}

function isTokenError(err: unknown) {
  const msg = (err as Error)?.message || "";
  const s = msg.toLowerCase();
  return s.includes("invalid token") || s.includes("unauthorized") || s.includes("401");
}

export function App() {
  const [token, setToken] = useState<string>(() => localStorage.getItem("token") ?? "");
  const [theme, setTheme] = useState<"light" | "dark">(
    () => (localStorage.getItem("theme") as "light" | "dark") || "light"
  );
  const [docsLoading, setDocsLoading] = useState(false);
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState<string>(() => localStorage.getItem("username") || "");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");

  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [uploading, setUploading] = useState(false);
  const [processingDoc, setProcessingDoc] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [chatRatio, setChatRatio] = useState<number>(() => {
    const saved = parseFloat(localStorage.getItem("chatRatio") || "");
    return Number.isFinite(saved) && saved > 0 ? clampRatio(saved) : CHAT_DEFAULT_RATIO;
  });
  const [chatWidth, setChatWidth] = useState<number>(() => ratioToPx(chatRatio));
  const [resizingChat, setResizingChat] = useState(false);

  const [readerMode, setReaderMode] = useState<"pdf" | "text">("pdf");
  const [pdfPage, setPdfPage] = useState(1);
  const [readerChunks, setReaderChunks] = useState<ReaderChunk[]>([]);

  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMsg[]>([]);
  const chatThreadOwnerRef = useRef<string>("");
  const [availableModels, setAvailableModels] = useState<string[]>(DEFAULT_MODELS);
  const [model, setModel] = useState(DEFAULT_MODELS[0]);
  const [useRetrievalInChat, setUseRetrievalInChat] = useState(true);

  const [retrievalQuery, setRetrievalQuery] = useState("");
  const [retrievalHits, setRetrievalHits] = useState<RetrievalHit[]>([]);
  const [searching, setSearching] = useState(false);
  const [showRetrievalPanel, setShowRetrievalPanel] = useState(true);
  const [retrievalTouched, setRetrievalTouched] = useState(false);
  const [retrievalError, setRetrievalError] = useState("");

  const [selectionMenu, setSelectionMenu] = useState<{ visible: boolean; x: number; y: number; text: string }>({
    visible: false,
    x: 0,
    y: 0,
    text: ""
  });

  const canLogin = username.trim().length >= 3 && password.trim().length >= 6;
  const fileViewerUrl = useMemo(() => {
    if (!selectedDoc) return "";
    const encodedToken = encodeURIComponent(token);
    return `${API_BASE}${selectedDoc.file_url}?token=${encodedToken}#page=${Math.max(1, pdfPage)}`;
  }, [selectedDoc, pdfPage, token]);

  const resizeStateRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const chatInputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!token) return;
    localStorage.setItem("token", token);
    fetchDocuments(token).catch((e: Error) => {
      if (isTokenError(e)) {
        setToken("");
        localStorage.removeItem("token");
        setAuthError("登录已过期，请重新登录");
        return;
      }
      setAuthError(e.message);
    });
    fetchModels(token).catch(() => {});
  }, [token]);

  useEffect(() => {
    if (!selectedDoc || !token) return;
    loadReaderChunks(selectedDoc.id).catch(() => setReaderChunks([]));
  }, [selectedDoc?.id, token]);

  // Load per-document chat thread from localStorage on doc/user switch.
  useEffect(() => {
    if (!token) return;
    const nextOwner = chatThreadKey(username, selectedDoc?.id);
    chatThreadOwnerRef.current = nextOwner;
    setChatMessages(loadChatThread(username, selectedDoc?.id));
  }, [selectedDoc?.id, username, token]);

  // Persist chat thread. Guarded by ownerRef so a pending load doesn't
  // overwrite the new doc's thread before it renders.
  useEffect(() => {
    if (!token) return;
    const owner = chatThreadKey(username, selectedDoc?.id);
    if (owner !== chatThreadOwnerRef.current) return;
    saveChatThread(username, selectedDoc?.id, chatMessages);
  }, [chatMessages, selectedDoc?.id, username, token]);

  useEffect(() => {
    const onClickOutside = () => {
      setSelectionMenu((prev) => ({ ...prev, visible: false }));
    };
    window.addEventListener("click", onClickOutside);
    return () => window.removeEventListener("click", onClickOutside);
  }, []);

  useEffect(() => {
    localStorage.setItem("chatRatio", String(chatRatio));
    setChatWidth(ratioToPx(chatRatio));
  }, [chatRatio]);

  useEffect(() => {
    const onResize = () => setChatWidth(ratioToPx(chatRatio));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [chatRatio]);

  useEffect(() => {
    if (!resizingChat) return;
    const onMove = (e: MouseEvent) => {
      const state = resizeStateRef.current;
      if (!state) return;
      const delta = state.startX - e.clientX;
      const nextPx = state.startWidth + delta;
      const nextRatio = clampRatio(nextPx / window.innerWidth);
      setChatRatio(nextRatio);
    };
    const onUp = () => {
      setResizingChat(false);
      resizeStateRef.current = null;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [resizingChat]);

  async function fetchDocuments(currentToken: string) {
    setDocsLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/api/documents`, {
        headers: { Authorization: `Bearer ${currentToken}` }
      });
      const data = await parseResp<DocumentItem[]>(resp);
      setDocs(data);
      if (!selectedDoc && data.length > 0) setSelectedDoc(data[0]);
      if (selectedDoc) {
        const latest = data.find((d) => d.id === selectedDoc.id);
        if (latest) setSelectedDoc(latest);
      }
    } finally {
      setDocsLoading(false);
    }
  }

  async function fetchModels(currentToken: string) {
    const resp = await fetch(`${API_BASE}/api/chat/models`, {
      headers: { Authorization: `Bearer ${currentToken}` }
    });
    const data = await parseResp<ModelListResp>(resp);
    const list = (data.models || []).map((m) => m.name);
    if (list.length > 0) {
      setAvailableModels(list);
      setModel(data.default_model || list[0]);
    }
  }

  async function loadReaderChunks(documentId: number) {
    const resp = await fetch(`${API_BASE}/api/retrieval/chunks/${documentId}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    const data = await parseResp<{ document_id: number; chunks: ReaderChunk[] }>(resp);
    setReaderChunks(data.chunks || []);
  }

  async function onAuthSubmit(e: FormEvent) {
    e.preventDefault();
    setAuthError("");
    const path = isRegister ? "/api/auth/register" : "/api/auth/login";
    try {
      const resp = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });
      const data = await parseResp<TokenResponse>(resp);
      localStorage.setItem("username", username.trim());
      setToken(data.access_token);
    } catch (err) {
      setAuthError((err as Error).message);
    }
  }

  function logout() {
    setToken("");
    localStorage.removeItem("token");
    setDocs([]);
    setSelectedDoc(null);
    setChatMessages([]);
    setRetrievalHits([]);
  }

  async function onUpload(file: File) {
    if (!token) return;
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const resp = await fetch(`${API_BASE}/api/documents/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: fd
      });
      const item = await parseResp<DocumentItem>(resp);
      setDocs((prev) => [item].concat(prev));
      setSelectedDoc(item);
      setPdfPage(1);
    } catch (err) {
      if (isTokenError(err)) {
        setToken("");
        localStorage.removeItem("token");
        setAuthError("登录已过期，请重新登录");
        return;
      }
      alert((err as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(doc: DocumentItem) {
    if (!token) return;
    if (!window.confirm(`确定删除文档 "${doc.filename}" 吗？此操作不可恢复。`)) return;
    try {
      const resp = await fetch(`${API_BASE}/api/documents/${doc.id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` }
      });
      await parseResp(resp);
      setDocs((prev) => {
        const next = prev.filter((d) => d.id !== doc.id);
        if (selectedDoc?.id === doc.id) {
          const fallback = next[0] ?? null;
          setSelectedDoc(fallback);
          setReaderChunks([]);
          setRetrievalHits([]);
          setRetrievalTouched(false);
          setRetrievalError("");
          setPdfPage(1);
        }
        return next;
      });
    } catch (err) {
      if (isTokenError(err)) {
        setToken("");
        localStorage.removeItem("token");
        setAuthError("登录已过期，请重新登录");
        return;
      }
      alert((err as Error).message);
    }
  }

  async function processSelectedDocument() {
    if (!token || !selectedDoc) return;
    setProcessingDoc(true);
    try {
      const resp = await fetch(`${API_BASE}/api/retrieval/process/${selectedDoc.id}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` }
      });
      await parseResp(resp);
      await fetchDocuments(token);
      await loadReaderChunks(selectedDoc.id);
    } catch (err) {
      alert((err as Error).message);
    } finally {
      setProcessingDoc(false);
    }
  }

  async function searchRetrieval() {
    if (!token || !retrievalQuery.trim()) return;
    setSearching(true);
    setRetrievalTouched(true);
    setRetrievalError("");
    try {
      const resp = await fetch(`${API_BASE}/api/retrieval/search`, {
        method: "POST",
        headers: authHeaders(token),
        body: JSON.stringify({
          query: retrievalQuery.trim(),
          top_k: 5,
          document_id: selectedDoc?.id
        })
      });
      const data = await parseResp<{ hits: RetrievalHit[] }>(resp);
      setRetrievalHits(data.hits || []);
      setShowRetrievalPanel(true);
    } catch (err) {
      setRetrievalHits([]);
      setRetrievalError((err as Error).message);
    } finally {
      setSearching(false);
    }
  }

  async function sendChat() {
    if (!token || !chatInput.trim()) return;
    const nextUser: ChatMsg = { role: "user", content: chatInput.trim() };
    const historyBeforeSend = [...chatMessages];
    setChatMessages((prev) => prev.concat(nextUser));
    setChatInput("");
    setChatLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: authHeaders(token),
        body: JSON.stringify({
          message: nextUser.content,
          model,
          history: historyBeforeSend.map((m) => ({ role: m.role, content: m.content })),
          use_retrieval: useRetrievalInChat,
          document_id: selectedDoc?.id ?? null,
          top_k: 5
        })
      });
      const data = await parseResp<{ answer: string; citations?: ChatCitation[] }>(resp);
      setChatMessages((prev) =>
        prev.concat({ role: "assistant", content: data.answer, citations: data.citations || [] })
      );
    } catch (err) {
      setChatMessages((prev) =>
        prev.concat({ role: "assistant", content: `请求失败: ${(err as Error).message}` })
      );
    } finally {
      setChatLoading(false);
    }
  }

  function startResizeChat(clientX: number) {
    resizeStateRef.current = { startX: clientX, startWidth: chatWidth };
    setResizingChat(true);
  }

  function askSelectedText() {
    if (!selectionMenu.text.trim()) return;
    setChatInput(`请基于文档解释以下划词内容：\n\n${selectionMenu.text}`);
    setSelectionMenu({ visible: false, x: 0, y: 0, text: "" });
    setTimeout(() => chatInputRef.current?.focus(), 0);
  }

  function onReaderContextMenu(e: React.MouseEvent) {
    const selectedText = window.getSelection()?.toString().trim() || "";
    if (!selectedText) return;
    e.preventDefault();
    setSelectionMenu({ visible: true, x: e.clientX, y: e.clientY, text: selectedText });
  }

  function jumpByCitation(c: ChatCitation) {
    if (c.page !== undefined && c.page !== null) {
      const page = Number(c.page);
      const safePage = page <= 0 ? 1 : page;
      setReaderMode("pdf");
      setPdfPage(safePage);
      return;
    }
    setReaderMode("text");
    const el = document.getElementById(`chunk-${c.chunk_index}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  if (!token) {
    return (
      <div className="auth-wrap">
        <form className="auth-card" onSubmit={onAuthSubmit}>
          <h1>AI Reader MVP</h1>
          <p>登录 / 上传PDF / 阅读 / 划词提问 / 引用跳转</p>
          <label>用户名</label>
          <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="请输入用户名" />
          <label>密码</label>
          <input
            value={password}
            type="password"
            onChange={(e) => setPassword(e.target.value)}
            placeholder="至少6位"
          />
          {authError && <div className="error-text">{authError}</div>}
          <button disabled={!canLogin} type="submit">
            {isRegister ? "注册并登录" : "登录"}
          </button>
          <button
            className="ghost-btn"
            type="button"
            onClick={() => {
              setIsRegister((v) => !v);
              setAuthError("");
            }}
          >
            {isRegister ? "已有账号? 去登录" : "没有账号? 去注册"}
          </button>
        </form>
      </div>
    );
  }

  const sidebarWidth = sidebarCollapsed ? SIDEBAR_COLLAPSED_WIDTH : SIDEBAR_WIDTH;
  const layoutStyle = { gridTemplateColumns: `${sidebarWidth}px 1fr 8px ${chatWidth}px` };

  return (
    <div className="layout" style={layoutStyle}>
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="sidebar-head">
          {!sidebarCollapsed && <h2>文档</h2>}
          <div className="sidebar-toolbar">
            {!sidebarCollapsed && (
              <button
                className="theme-toggle"
                title={theme === "light" ? "切换暗色" : "切换亮色"}
                aria-label="Toggle theme"
                onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
              >
                {theme === "light" ? "◐" : "◑"}
              </button>
            )}
            <button
              className="icon-btn"
              title={sidebarCollapsed ? "展开侧栏" : "收起侧栏"}
              aria-label="Toggle sidebar"
              onClick={() => setSidebarCollapsed((v) => !v)}
            >
              {sidebarCollapsed ? "›" : "‹"}
            </button>
          </div>
        </div>
        {!sidebarCollapsed && (
          <>
            <label className="upload-btn">
              {uploading ? "上传中…" : "＋ 上传 PDF"}
              <input
                type="file"
                accept=".pdf,application/pdf"
                onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
                disabled={uploading}
                hidden
              />
            </label>
            <div className="doc-list">
              {docsLoading && docs.length === 0 && (
                <>
                  {[0, 1, 2].map((i) => (
                    <div key={i} className="doc-skeleton">
                      <div className="skeleton-line" />
                      <div className="skeleton-line short" />
                    </div>
                  ))}
                </>
              )}
              {docs.map((d) => (
                <div key={d.id} className={`doc-item-row ${selectedDoc?.id === d.id ? "active" : ""}`}>
                  <button
                    className="doc-item"
                    onClick={() => {
                      setSelectedDoc(d);
                      setRetrievalHits([]);
                      setRetrievalTouched(false);
                      setRetrievalError("");
                      setPdfPage(1);
                    }}
                  >
                    <span>{d.filename}</span>
                    <small>
                      {d.process_status} | chunks: {d.chunk_count}
                    </small>
                  </button>
                  <button
                    className="doc-delete-btn"
                    title="删除此文档"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteDocument(d);
                    }}
                  >
                    删除
                  </button>
                </div>
              ))}
              {!docsLoading && docs.length === 0 && (
                <div className="empty">还没有文档，先上传一个 PDF</div>
              )}
            </div>
            <div className="sidebar-footer">
              <button className="ghost-btn small-btn" onClick={logout}>
                退出登录
              </button>
            </div>
          </>
        )}
      </aside>

      <main className="reader-panel">
        <div className="reader-head">
          <span>{selectedDoc ? selectedDoc.filename : "请选择文档"}</span>
          <div className="reader-actions">
            <div className="segmented" role="tablist" aria-label="阅读模式">
              <button
                className={`small-btn ${readerMode === "pdf" ? "active" : ""}`}
                onClick={() => setReaderMode("pdf")}
                role="tab"
                aria-selected={readerMode === "pdf"}
              >
                PDF
              </button>
              <button
                className={`small-btn ${readerMode === "text" ? "active" : ""}`}
                onClick={() => setReaderMode("text")}
                role="tab"
                aria-selected={readerMode === "text"}
              >
                文本
              </button>
            </div>
            {selectedDoc && (
              <button className="small-btn" onClick={processSelectedDocument} disabled={processingDoc}>
                {processingDoc ? "处理中…" : "处理并建索引"}
              </button>
            )}
          </div>
        </div>
        {!selectedDoc && <div className="empty">请选择文档后开始阅读</div>}
        {selectedDoc && readerMode === "pdf" && <iframe title="pdf-viewer" src={fileViewerUrl} className="pdf-frame" />}
        {selectedDoc && readerMode === "text" && (
          <div className="text-reader" onContextMenu={onReaderContextMenu}>
            {readerChunks.length === 0 && <div className="empty">暂无文本分块，请先点击“处理并建索引”</div>}
            {readerChunks.map((c) => (
              <div key={c.chunk_index} id={`chunk-${c.chunk_index}`} className="reader-chunk">
                <div className="reader-chunk-meta">
                  chunk #{c.chunk_index} {c.page ? `| 页码 ${c.page}` : ""}
                </div>
                <div className="reader-chunk-text md-content">
                  <MarkdownView text={c.text} />
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      <div
        className={`chat-resizer ${resizingChat ? "active" : ""}`}
        role="separator"
        aria-label="拖动调整聊天区宽度；双击还原"
        title="拖动调整宽度，双击还原"
        onMouseDown={(e) => startResizeChat(e.clientX)}
        onDoubleClick={() => setChatRatio(CHAT_DEFAULT_RATIO)}
      >
        <span className="chat-resizer-grip" aria-hidden />
      </div>

      <section className="chat-panel">
        <div className="chat-head">
          <div className="chat-head-left">
            <strong>AI 聊天</strong>
            <span className="chat-head-meta">
              {chatMessages.length > 0 ? `${chatMessages.length} 条` : "新对话"}
            </span>
          </div>
          <div className="chat-head-right">
            <div className="size-presets" role="group" aria-label="窗口宽度">
              {CHAT_PRESETS.map((p) => {
                const active = Math.abs(chatRatio - p.ratio) < 0.015;
                return (
                  <button
                    key={p.label}
                    className={`small-btn ${active ? "active" : "ghost-btn"}`}
                    title={`调整聊天区宽度到屏幕 ${p.label}`}
                    onClick={() => setChatRatio(p.ratio)}
                  >
                    {p.label}
                  </button>
                );
              })}
            </div>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {availableModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <button
              className="ghost-btn small-btn"
              title="清空当前文档的对话历史"
              disabled={chatMessages.length === 0}
              onClick={() => {
                if (window.confirm("清空当前文档的对话历史？此操作不可恢复。")) {
                  setChatMessages([]);
                }
              }}
            >
              清空
            </button>
          </div>
        </div>

        <div className="chat-controls">
          <label className="checkbox-line">
            <input
              type="checkbox"
              checked={useRetrievalInChat}
              onChange={(e) => setUseRetrievalInChat(e.target.checked)}
            />
            使用向量检索增强回答
          </label>
        </div>

        <div className="chat-log">
          {chatMessages.map((m, idx) => (
            <div key={idx} className={`bubble ${m.role}`}>
              {m.role === "assistant" ? (
                <>
                  <div className="md-content">
                    <MarkdownView text={m.content} />
                  </div>
                  {m.citations && m.citations.length > 0 && (
                    <div className="citation-list">
                      <strong>引用:</strong>
                      {m.citations.map((c, i) => (
                        <button key={`${i}-${c.chunk_index}`} className="citation-btn" onClick={() => jumpByCitation(c)}>
                          [{i + 1}] {c.document_name}#{c.chunk_index}
                          {c.page ? ` 第${c.page}页` : ""}
                        </button>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                m.content
              )}
            </div>
          ))}
          {chatLoading && (
            <div className="bubble assistant waiting">
              <span className="dot-loader">
                <span />
              </span>
              正在思考…
            </div>
          )}
        </div>

        <div className="chat-input">
          <textarea
            value={retrievalQuery}
            onChange={(e) => setRetrievalQuery(e.target.value)}
            placeholder="向量检索测试问题..."
            rows={2}
          />
          <button onClick={searchRetrieval} disabled={searching || !retrievalQuery.trim()}>
            {searching ? "检索中..." : "检索 TopK"}
          </button>

          {(retrievalTouched || retrievalHits.length > 0) && (
            <div className="retrieval-results-wrap">
              <div className="retrieval-results-head">
                <strong>检索结果（{retrievalHits.length}）</strong>
                <div className="retrieval-actions">
                  <button className="small-btn" onClick={() => setShowRetrievalPanel((v) => !v)}>
                    {showRetrievalPanel ? "收起结果" : "展开结果"}
                  </button>
                  <button
                    className="small-btn ghost-btn"
                    onClick={() => {
                      setRetrievalHits([]);
                      setRetrievalTouched(false);
                      setRetrievalError("");
                    }}
                  >
                    清空
                  </button>
                </div>
              </div>
              {showRetrievalPanel && (
                <div className="retrieval-results-scroll">
                  {retrievalError && <div className="retrieval-tip error-tip">检索失败: {retrievalError}</div>}
                  {!retrievalError && retrievalHits.length === 0 && (
                    <div className="retrieval-tip">
                      当前检索到 0 条结果。请先确认文档已“处理并建索引”，或尝试更具体的问题关键词。
                    </div>
                  )}
                  {retrievalHits.map((hit, idx) => (
                    <div key={`${idx}-${hit.chunk_index}`} className="bubble assistant">
                      [{hit.document_name}#{hit.chunk_index}] score={hit.score.toFixed(4)}
                      {hit.page ? ` | 第${hit.page}页` : ""}
                      {"\n"}
                      {hit.text}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="chat-input">
          <textarea
            ref={chatInputRef}
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="输入问题..."
            rows={4}
          />
          <button onClick={sendChat} disabled={chatLoading || !chatInput.trim()}>
            {chatLoading ? "发送中..." : "发送"}
          </button>
        </div>
      </section>

      {selectionMenu.visible && (
        <div
          className="selection-menu"
          style={{ left: selectionMenu.x, top: selectionMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <button className="small-btn" onClick={askSelectedText}>
            提问AI（划词）
          </button>
        </div>
      )}
    </div>
  );
}
