"""BM25 sparse retrieval + RRF fusion. Pure-Python, no extra deps.

Design:
- Per document (keyed by file_hash), build an in-memory BM25 index from the
  same `chunks.json` produced by the dense pipeline. This keeps dense / sparse
  on IDENTICAL chunk boundaries, which is required for fusion to be meaningful.
- Tokenizer is intentionally simple (regex word split + lowercase). For Chinese
  we fall back to character-bigram tokens so BM25 still has signal without a
  jieba dependency.
- RRF fusion per the Cormack et al. 2009 definition: score = sum_i 1/(k+rank_i).
"""

import json
import math
import os
import re
import threading

_BM25_CACHE = {}
_BM25_LOCK = threading.Lock()

_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
_CJK_RE = re.compile(r"[一-鿿]")


def _tokenize(text):
    if not text:
        return []
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    # character bigrams for CJK runs so BM25 can match Chinese without jieba
    cjk_chars = _CJK_RE.findall(text)
    if len(cjk_chars) >= 2:
        tokens.extend(cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1))
    elif cjk_chars:
        tokens.extend(cjk_chars)
    return tokens


class BM25Index:
    """Minimal BM25 (Okapi) implementation, tuned for small/medium corpora."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs  # list of token lists, aligned to chunk_index
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        df = {}
        self.tf = []
        for toks in docs:
            freq = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            self.tf.append(freq)
            for t in freq:
                df[t] = df.get(t, 0) + 1

        self.idf = {}
        for t, n in df.items():
            # BM25+ style idf, floored at a small positive value to avoid negatives
            self.idf[t] = math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query_tokens, idx):
        score = 0.0
        freq = self.tf[idx]
        dl = self.doc_len[idx] or 1
        for qt in query_tokens:
            if qt not in freq:
                continue
            f = freq[qt]
            idf = self.idf.get(qt, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def topk(self, query, k=10):
        qt = _tokenize(query)
        if not qt or self.N == 0:
            return []
        scored = [(i, self.score(qt, i)) for i in range(self.N)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, k)]


def _load_chunks(chunks_json_path):
    if not os.path.exists(chunks_json_path):
        return []
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_bm25_for_file(file_hash, chunks_json_path):
    """Return (BM25Index, chunks). Cached per (file_hash, mtime)."""
    if not os.path.exists(chunks_json_path):
        return None, []
    mtime = os.path.getmtime(chunks_json_path)
    key = (file_hash, chunks_json_path, mtime)
    with _BM25_LOCK:
        cached = _BM25_CACHE.get(key)
        if cached is not None:
            return cached
        chunks = _load_chunks(chunks_json_path)
        tokens = [_tokenize(c.get("text", "")) for c in chunks]
        index = BM25Index(tokens)
        _BM25_CACHE[key] = (index, chunks)
        return index, chunks


def bm25_search(file_hash, chunks_json_path, query, top_k=10):
    """Return list of {chunk_index, text, metadata, score} sorted desc by score."""
    index, chunks = get_bm25_for_file(file_hash, chunks_json_path)
    if not index or not chunks:
        return []
    hits = index.topk(query, k=top_k)
    out = []
    for idx, score in hits:
        c = chunks[idx]
        out.append(
            {
                "chunk_index": int(c.get("chunk_index", idx)),
                "text": c.get("text", ""),
                "metadata": c.get("metadata", {}) or {},
                "score": float(score),
            }
        )
    return out


def rrf_fuse(rank_lists, k=60, top_k=10):
    """Reciprocal Rank Fusion.

    rank_lists: list of lists of items. Each item must have a stable key
    `(document_id, chunk_index)`. Within each list items are ordered best-first.
    """
    agg = {}
    payload = {}
    for lst in rank_lists:
        for rank, item in enumerate(lst):
            key = (item.get("document_id"), int(item.get("chunk_index", 0)))
            contrib = 1.0 / (k + rank + 1)  # rank is 0-indexed; +1 per paper
            agg[key] = agg.get(key, 0.0) + contrib
            # keep the first-seen payload (prefer dense hit if dense comes first)
            if key not in payload:
                payload[key] = item
    fused = []
    for key, s in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        item = dict(payload[key])
        item["rrf_score"] = s
        fused.append(item)
    return fused
