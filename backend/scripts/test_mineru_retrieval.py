import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os

from app.services.lc_mineru import build_faiss_from_pdf, load_faiss_index
from app.services.mineru_parser import compute_file_sha256


def ensure_index(args):
    file_hash = compute_file_sha256(args.pdf)
    index_dir = os.path.join(args.output_dir, "lc_cache", file_hash, "faiss")
    if args.rebuild or not os.path.exists(os.path.join(index_dir, "index.faiss")):
        build_faiss_from_pdf(
            pdf_path=args.pdf,
            output_dir=index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            mineru_mode=args.mineru_mode,
            embeddings_model=args.embeddings_model,
            embeddings_device=args.embeddings_device,
        )
    return index_dir


def main():
    parser = argparse.ArgumentParser(description="Test MinerU+LangChain retrieval (interactive query)")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output-dir", default="./mineru_output")
    parser.add_argument("--mineru-mode", default="flash")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--embeddings-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embeddings-device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true", help="Force re-parse and rebuild vector index")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit("PDF not found: %s" % args.pdf)
    os.makedirs(args.output_dir, exist_ok=True)

    index_dir = ensure_index(args)
    vs = load_faiss_index(
        index_dir=index_dir,
        embeddings_model=args.embeddings_model,
        embeddings_device=args.embeddings_device,
    )

    print("Index ready:", index_dir)
    print("Top-K:", args.top_k)
    print("-" * 80)
    while True:
        query = input("请输入 query（直接回车退出）: ").strip()
        if not query:
            break
        hits = vs.similarity_search_with_score(query, k=max(1, min(20, args.top_k)))
        print("Query:", query)
        for i, pair in enumerate(hits):
            doc, score = pair
            meta = doc.metadata or {}
            snippet = doc.page_content[:400].replace("\n", " ")
            print("[%d] score=%.6f chunk=%s" % (i + 1, float(score), str(meta.get("chunk_index", ""))))
            print(snippet)
            print("-" * 80)


if __name__ == "__main__":
    main()
