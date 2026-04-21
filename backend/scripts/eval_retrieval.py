"""Offline retrieval evaluation with diagnostics.

Input: a JSONL file where each line is:
    {"query": "...", "gold_chunk_indices": [3, 7], "document_id": 1}

Output: Recall@k, MRR, nDCG@k for each mode in --modes.

Usage:
    python scripts/eval_retrieval.py \
        --labels scripts/eval_sample.jsonl \
        --owner-id 1 \
        --k 5 \
        --verbose

Diagnostics printed on startup:
    - which user owns what documents (so you can verify --owner-id is correct)
    - whether each label's document_id actually exists & is ready
    - for each query, top-k chunk indices retrieved vs gold (when --verbose)

Common reasons the previous run gave you all zeros:
    1. The owner_id has no documents — check the first diagnostic block below.
    2. document_id in your JSONL doesn't exist for this owner.
    3. The document exists but was never "processed" (no FAISS index on disk).
    4. The gold_chunk_indices in your JSONL don't match real chunk_index values
       (after re-processing with the new structured chunker, chunk_index space
       changes; you must relabel).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.services.retrieval import search_chunks  # noqa: E402


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_flags, k):
    rels = ranked_flags[:k]
    ideal = sorted(ranked_flags, reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(rels) / idcg) if idcg > 0 else 0.0


def diagnose_owner(db, owner_id, required_doc_ids):
    """Print who owns what so the user can spot owner/doc-id mismatches."""
    docs = db.query(Document).filter(Document.owner_id == owner_id).all()
    print("\n[diagnose] owner_id=%d owns %d document(s):" % (owner_id, len(docs)))
    if not docs:
        print("           (none) — search_chunks will return [] for this owner.")
        print("           -> verify the owner_id matches the user who uploaded the PDFs.")
        return {}
    by_id = {d.id: d for d in docs}
    for d in docs:
        status = d.process_status
        file_hash = (d.file_hash or "")[:12]
        index_dir = os.path.join(
            settings.mineru_output_dir, "lc_cache", d.file_hash or "", "faiss"
        )
        has_idx = os.path.exists(os.path.join(index_dir, "index.faiss"))
        print(
            "   - id=%d status=%s chunks=%d hash=%s… faiss=%s filename=%s"
            % (d.id, status, d.chunk_count, file_hash, "OK" if has_idx else "MISSING", d.filename)
        )
    missing = [did for did in required_doc_ids if did not in by_id]
    if missing:
        print(
            "[diagnose] ⚠ labels reference document_id(s) %s which this owner does NOT have."
            % missing
        )
    return by_id


def evaluate(labels, owner_id, k, mode, verbose=False, doc_meta=None):
    os.environ["RETRIEVAL_MODE"] = mode
    db = SessionLocal()
    recalls, mrrs, ndcgs = [], [], []
    n = 0
    empty_hits = 0
    try:
        for row in labels:
            query = row["query"]
            gold = set(int(x) for x in row.get("gold_chunk_indices", []))
            if not gold:
                continue
            hits = search_chunks(
                db=db,
                owner_id=owner_id,
                query=query,
                top_k=k,
                document_id=row.get("document_id"),
                mineru_output_dir=settings.mineru_output_dir,
                embeddings_model_name=settings.embeddings_model_name,
                embeddings_device=settings.embeddings_device,
            )
            ranked = [int(h.get("chunk_index", -1)) for h in hits]
            flags = [1 if c in gold else 0 for c in ranked]
            if not hits:
                empty_hits += 1
            recalls.append(1.0 if any(flags) else 0.0)
            rr = 0.0
            for i, f_ in enumerate(flags):
                if f_:
                    rr = 1.0 / (i + 1)
                    break
            mrrs.append(rr)
            ndcgs.append(ndcg_at_k(flags, k))
            n += 1
            if verbose:
                print(
                    "  [%s] q=%r doc=%s gold=%s ranked=%s hit=%s"
                    % (mode, query, row.get("document_id"), sorted(gold), ranked, bool(any(flags)))
                )
    finally:
        db.close()

    if n == 0:
        return {"mode": mode, "n": 0, "note": "no usable labels"}
    result = {
        "mode": mode,
        "n": n,
        "empty_result_queries": empty_hits,
        "recall@%d" % k: round(sum(recalls) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "ndcg@%d" % k: round(sum(ndcgs) / n, 4),
    }
    if empty_hits == n:
        result["note"] = (
            "all queries returned 0 hits — check owner_id, document_id, "
            "or that the document has been processed (FAISS index exists)."
        )
    return result


def load_labels(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--owner-id", type=int, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", default="dense,bm25,hybrid")
    ap.add_argument("--verbose", action="store_true", help="print per-query ranking")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    required_docs = sorted({row.get("document_id") for row in labels if row.get("document_id")})
    print("[eval] labels=%d required_document_ids=%s" % (len(labels), required_docs))

    db = SessionLocal()
    try:
        doc_meta = diagnose_owner(db, args.owner_id, required_docs)
    finally:
        db.close()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    for m in modes:
        print("\n[eval] running mode=%s" % m)
        results.append(evaluate(labels, args.owner_id, args.k, m, verbose=args.verbose, doc_meta=doc_meta))

    print("\n=== Results ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
