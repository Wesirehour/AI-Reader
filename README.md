# AI Reader

An end-to-end **PDF Retrieval-Augmented Generation (RAG)** workspace: upload a
PDF, and get an interactive reader plus an AI chat that answers questions
grounded in the document, with clickable citations back to the source pages.

Built with **FastAPI + React + LangChain + MinerU + FAISS/Milvus**.

---

## Features

- **PDF parsing**: MinerU preserves headings, tables, formulas, page numbers.
- **Structured chunking**: heading-aware, token-bounded (not character-bounded),
  preserves tables / display-math / code blocks atomically, with
  **Small-to-Big parent/child** windows (child chunks for precision, parent
  chunks for LLM context).
- **Hybrid retrieval**: BM25 (pure-Python) + dense embeddings, fused with
  **Reciprocal Rank Fusion (RRF)**, optional cross-encoder rerank.
  Switchable via `RETRIEVAL_MODE=dense|bm25|hybrid`.
- **Vector store**: FAISS by default, Milvus via one-line config switch.
- **SHA-256 file-hash index cache** — re-uploading the same document skips
  parsing + embedding entirely.
- **Streaming answers** over Server-Sent Events (`POST /api/chat/stream`)
  with citations delivered on the first frame.
- **Prompt-injection guardrails**: explicit evidence/instruction separator
  and light sanitization of retrieved chunks.
- **Offline retrieval evaluation**: `scripts/eval_retrieval.py` outputs
  Recall@k / MRR / nDCG@k for dense / bm25 / hybrid on a labeled JSONL.
- **Frontend**: React + Vite. Markdown + KaTeX rendering, per-document chat
  threads persisted locally, resizable chat pane with 25/33/40/50% presets,
  light/dark theme, responsive layout.

---

## Architecture

```
┌─────────────────┐      ┌────────────────────────────────┐
│  React frontend │◀────▶│  FastAPI backend               │
│  (Vite)         │ HTTP │  ├─ auth  (JWT)                │
└─────────────────┘ +SSE │  ├─ documents  (upload/delete) │
                         │  ├─ retrieval  (index/search)  │
                         │  └─ chat  (sync + stream)      │
                         └────────────┬───────────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
          MinerU parser        Structured chunker    LLM gateway
          (PDF → Markdown)     (heading-aware,       (OpenAI-compatible)
                                token-bounded,
                                parent/child)
                 │                    │
                 └──────────┬─────────┘
                            ▼
                ┌─────────────────────┐
                │  Vector store       │
                │  FAISS  |  Milvus   │
                └─────────────────────┘
```

---

## Prerequisites

- **Python** 3.10+ (3.11 recommended)
- **Node** 18+
- An **OpenAI-SDK-compatible** LLM endpoint (OpenAI, DeepSeek, Qwen, Doubao,
  or any proxy that speaks the OpenAI Chat Completions schema).

---

## Quick start

> **Important**
> Before starting the backend, create `backend/.env` from `backend/.env.example`
> and provide both `LLM_BASE_URL` and `LLM_API_KEY`.
> `.env` files are ignored by git and must not be committed.

### 1. Clone

```bash
git clone git@github.com:Wesirehour/AI-Reader.git
cd AI-Reader
```

### 2. Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# >>> Edit backend/.env and set LLM_BASE_URL and LLM_API_KEY (see below) <<<
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
# http://localhost:5173
```

Register a user in the UI, upload a PDF, click **"处理并建索引"**, then chat.

---

## Configuration — `backend/.env`

> `.env` is **gitignored**. Start from `backend/.env.example`.
>
> **You MUST set at least `LLM_BASE_URL` and `LLM_API_KEY`** — everything
> else has sensible defaults.

### Required

| Key | Description | Example |
|---|---|---|
| `LLM_BASE_URL` | OpenAI-SDK-compatible endpoint. Must speak `/v1/chat/completions`. The app auto-appends `/v1` if you omit it. | `https://api.deepseek.com` |
| `LLM_API_KEY`  | API key for the above endpoint. **Never commit this.** | `sk-...` |

### Commonly tuned

| Key | Default | Description |
|---|---|---|
| `SECRET_KEY` | `replace-with-a-strong-random-secret` | JWT signing key. Must be replaced for any non-local use. |
| `ALLOW_INSECURE_DEV` | `true` | Set `false` in production to enforce `SECRET_KEY` / `LLM_API_KEY` presence at startup. |
| `LLM_DEFAULT_MODEL` | `deepseek-v3.2` | Default model in the frontend dropdown. |
| `LLM_MODEL_LIST` | `deepseek-v3.2,doubao-seed-1-6-flash-250828,qwen3.5-plus` | Whitelist of models the frontend may select. |
| `RETRIEVAL_MODE` | `hybrid` | `dense` / `bm25` / `hybrid`. |
| `RERANKER_MODEL` | *(empty)* | Set to e.g. `BAAI/bge-reranker-base` to enable cross-encoder rerank. |
| `CHUNK_STRATEGY` | `structured` | `structured` (recommended) or `recursive` (legacy, for A/B). |
| `CHILD_MAX_TOKENS` | `220` | Child chunk size in tokens (retrieval unit). |
| `PARENT_MAX_TOKENS` | `900` | Parent window size (surfaced to LLM). |
| `EMBEDDINGS_MODEL_NAME` | `all-MiniLM-L6-v2` | Any sentence-transformers model works. For CN-heavy corpora try `BAAI/bge-base-zh-v1.5`. |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Mirror for HF model downloads (China-friendly). |

See [`backend/.env.example`](backend/.env.example) for the full list.

### Vector store switch — FAISS vs Milvus

Edit `backend/config/vector_store.json`:

```json
{ "backend": "faiss" }
```

or

```json
{
  "backend": "milvus",
  "milvus": {
    "uri": "http://localhost:19530",
    "collection_prefix": "rag_doc_",
    "metric_type": "COSINE"
  }
}
```

---

## API surface (main routes)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/auth/register` / `/api/auth/login` | JWT auth |
| `POST` | `/api/documents/upload` | Upload PDF |
| `GET`  | `/api/documents` | List user's documents |
| `DELETE` | `/api/documents/{id}` | Delete document + cached index |
| `GET`  | `/api/documents/file/{id}` | Stream the PDF inline (for the viewer iframe) |
| `POST` | `/api/retrieval/process/{id}` | Parse + chunk + index |
| `POST` | `/api/retrieval/search` | Hybrid search, returns top-k chunks |
| `GET`  | `/api/chat/models` | Model list the frontend will show |
| `POST` | `/api/chat` | Sync chat (with optional RAG context) |
| `POST` | `/api/chat/stream` | Streaming chat (SSE) |

---

## Evaluation

```bash
cd backend
python scripts/eval_retrieval.py \
    --labels scripts/eval_sample.jsonl \
    --owner-id <your_user_id> \
    --k 5 \
    --modes dense,bm25,hybrid \
    --verbose
```

`eval_sample.jsonl` is a JSONL of `{query, gold_chunk_indices, document_id}`
entries. The script prints diagnostics about which documents the given owner
owns and whether their FAISS indexes exist, so zero-result runs are easy to
debug.

---

## Repository layout

```
.
├── backend/
│   ├── app/
│   │   ├── api/              # auth / documents / retrieval / chat routers
│   │   ├── services/
│   │   │   ├── lc_mineru.py          # MinerU + FAISS / Milvus pipeline
│   │   │   ├── structured_chunking.py# heading-aware, token-level, parent/child
│   │   │   ├── hybrid.py             # BM25 + RRF fusion
│   │   │   ├── rerank.py             # cross-encoder rerank hook
│   │   │   ├── retrieval.py          # hybrid search orchestration
│   │   │   └── llm.py                # OpenAI-compat router + streaming
│   │   ├── models/ db/ schemas/ core/
│   │   └── main.py
│   ├── scripts/
│   │   ├── eval_retrieval.py
│   │   └── eval_sample.jsonl
│   ├── config/vector_store.json
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.tsx
    │   ├── styles.css
    │   └── main.tsx
    ├── package.json
    └── .env.example
```

---

## Security notes

- `.env` files are **never** committed. Copy `*.env.example` and fill in.
- Rotate `LLM_API_KEY` and `SECRET_KEY` immediately if they are ever exposed.
- In production: set `ALLOW_INSECURE_DEV=false`, use a strong `SECRET_KEY`,
  and front the backend with TLS + rate limiting.

---

## License

MIT
