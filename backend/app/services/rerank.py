"""Rerank interface. Default is identity (no-op) so the system works without
heavy deps. Set RERANKER_MODEL env to enable a cross-encoder rerank (lazy-loaded).

Example:
    RERANKER_MODEL=BAAI/bge-reranker-base
"""

import os
import threading

_RERANKER_SINGLETON = {"model": None, "name": None}
_LOCK = threading.Lock()


def _load_cross_encoder(model_name):
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "RERANKER_MODEL is set but sentence-transformers is not installed. "
            "Install it or unset RERANKER_MODEL."
        ) from exc
    return CrossEncoder(model_name)


def _get_reranker():
    name = os.getenv("RERANKER_MODEL", "").strip()
    if not name:
        return None
    with _LOCK:
        if _RERANKER_SINGLETON["model"] is None or _RERANKER_SINGLETON["name"] != name:
            _RERANKER_SINGLETON["model"] = _load_cross_encoder(name)
            _RERANKER_SINGLETON["name"] = name
        return _RERANKER_SINGLETON["model"]


def rerank(query, items, top_k=None, text_key="text"):
    """Reorder items in-place by cross-encoder score. No-op if disabled or empty."""
    if not items:
        return items
    model = _get_reranker()
    if model is None:
        return items[: top_k] if top_k else items
    pairs = [(query, (it.get(text_key) or "")[:2000]) for it in items]
    try:
        scores = model.predict(pairs)
    except Exception:
        # On any failure, degrade to identity ordering rather than breaking search.
        return items[: top_k] if top_k else items
    scored = list(zip(items, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)
    out = []
    for it, s in scored:
        it = dict(it)
        it["rerank_score"] = float(s)
        out.append(it)
    return out[: top_k] if top_k else out
