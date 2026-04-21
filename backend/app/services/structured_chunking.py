"""Structured Markdown chunking tailored for MinerU output.

Why a new splitter instead of RecursiveCharacterTextSplitter:
- We want to preserve document structure (heading path) in every chunk so that
  embeddings carry topical context like "§2.1 Model > ...".
- We want to NEVER split tables, display-math ($$...$$), or fenced code blocks
  across chunk boundaries, because those are typically the highest-signal
  content and splitting them mid-block destroys meaning.
- We want token-level (not character-level) budgets, so the embedding model
  does not silently truncate chunks that happen to be token-heavy.
- We want a small child / large parent hierarchy. Small children get better
  embedding precision; when a child hits, we surface the larger parent to the
  LLM for fuller context. This is the "Small-to-Big" pattern.

Contract:
    structured_chunks_from_markdown(md_text, ...) ->
        list[{"text": str, "metadata": {...}}]

Each returned chunk's `text` is what we embed; metadata includes:
    chunk_index:  global child id (assigned by caller across all pages)
    parent_id:    parent-block id (children with same parent_id share parent_text)
    parent_text:  the whole parent window text (for surfacing to the LLM)
    heading_path: "§2 Method > §2.1 Model" — prefixed onto `text` too for embed
    type:         "text" | "table" | "formula" | "code"
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# --- atomic block patterns ---------------------------------------------------
# Markdown pipe table: one header row, one separator row, then 1+ data rows.
_TABLE_BLOCK_RE = re.compile(
    r"(?:^\|[^\n]*\|\s*\n^\|[-:\s|]+\|\s*\n(?:^\|[^\n]*\|\s*\n?)+)",
    re.MULTILINE,
)
# Display math: $$ ... $$ (non-greedy, may span lines).
_DISPLAY_MATH_RE = re.compile(r"\$\$[\s\S]+?\$\$", re.MULTILINE)
# Fenced code: ``` ... ``` or ~~~ ... ~~~
_FENCED_CODE_RE = re.compile(r"(?:```[\s\S]+?```|~~~[\s\S]+?~~~)", re.MULTILINE)

_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?\.])\s+|\n{2,}")


def _extract_atomics(text: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    """Replace atomic blocks with sentinel placeholders so token/sentence
    splitters don't cut them in half.

    Returns (masked_text, atomics) where atomics = [(placeholder, raw, type)].
    Order matters: code fences first (they can contain $$...$$ literally),
    then display math, then tables.
    """
    atomics: List[Tuple[str, str, str]] = []

    def _make_sub(t: str):
        def _sub(m: re.Match) -> str:
            idx = len(atomics)
            placeholder = "\x00ATOM%d\x00" % idx
            atomics.append((placeholder, m.group(0), t))
            return "\n\n" + placeholder + "\n\n"
        return _sub

    masked = _FENCED_CODE_RE.sub(_make_sub("code"), text)
    masked = _DISPLAY_MATH_RE.sub(_make_sub("formula"), masked)
    masked = _TABLE_BLOCK_RE.sub(_make_sub("table"), masked)
    return masked, atomics


def _restore_atomics(text: str, atomics: List[Tuple[str, str, str]]) -> str:
    for ph, raw, _ in atomics:
        text = text.replace(ph, raw)
    return text


# --- heading path tracker ----------------------------------------------------
_HEAD_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_by_headers(md: str) -> List[Dict[str, Any]]:
    """Return list of {heading_path, body}. Heading path tracks the full
    ancestry ("Intro > Background > Related Work").
    """
    lines = md.split("\n")
    sections: List[Dict[str, Any]] = []
    path: List[str] = []
    body: List[str] = []

    def flush():
        text = "\n".join(body).strip()
        if text:
            sections.append({"heading_path": " > ".join([p for p in path if p]), "body": text})

    for line in lines:
        m = _HEAD_RE.match(line.rstrip())
        if m:
            flush()
            body = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # keep only ancestors above this level
            path = path[: level - 1]
            while len(path) < level - 1:
                path.append("")
            path.append(title)
        else:
            body.append(line)
    flush()
    return sections


# --- token counting ----------------------------------------------------------
_tokenizer_cache: Dict[str, Any] = {}


def _get_tokenizer(model_name: str):
    if not model_name:
        return None
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tok = None
    _tokenizer_cache[model_name] = tok
    return tok


def _token_len(tokenizer, text: str) -> int:
    if tokenizer is None:
        # Rough fallback: ~1.5 chars per token for mixed CJK+EN.
        return max(1, int(len(text) / 1.5))
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, int(len(text) / 1.5))


def _pack_sentences(
    sentences: List[str],
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """Greedy packer: accumulate sentences until token budget is hit, then
    emit a window and seed the next window with trailing sentences totalling
    up to `overlap_tokens`.
    """
    if not sentences:
        return []
    windows: List[str] = []
    cur: List[str] = []
    cur_tok = 0

    for s in sentences:
        st = _token_len(tokenizer, s)
        if st > max_tokens:
            # Sentence alone exceeds window. Flush what we have, then slice this
            # long sentence into fixed character slabs as a last resort.
            if cur:
                windows.append("\n".join(cur))
                cur, cur_tok = [], 0
            approx_chars = max(200, max_tokens * 3)
            for i in range(0, len(s), approx_chars):
                windows.append(s[i : i + approx_chars])
            continue
        if cur_tok + st > max_tokens and cur:
            windows.append("\n".join(cur))
            back: List[str] = []
            back_tok = 0
            for prev in reversed(cur):
                pt = _token_len(tokenizer, prev)
                if back_tok + pt > overlap_tokens:
                    break
                back.insert(0, prev)
                back_tok += pt
            cur, cur_tok = back, back_tok
        cur.append(s)
        cur_tok += st

    if cur:
        windows.append("\n".join(cur))
    return windows


def _split_prose(
    body: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    if not body.strip():
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(body) if s and s.strip()]
    if not sents:
        return [body.strip()]
    return _pack_sentences(sents, tokenizer, max_tokens, overlap_tokens)


# --- main entry --------------------------------------------------------------
def structured_chunks_from_markdown(
    md_text: str,
    tokenizer_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    child_max_tokens: int = 220,
    child_overlap_tokens: int = 40,
    parent_max_tokens: int = 900,
) -> List[Dict[str, Any]]:
    """Split a Markdown document into retrieval-ready CHILD chunks.

    Each child has an associated PARENT window (same heading section, up to
    `parent_max_tokens`). Atomic blocks (table / formula / code) become their
    own child chunks under the same parent so retrieval can hit them directly.
    """
    if not md_text or not md_text.strip():
        return []

    tokenizer = _get_tokenizer(tokenizer_model_name)
    masked, atomics = _extract_atomics(md_text)
    sections = _split_by_headers(masked)
    if not sections:
        sections = [{"heading_path": "", "body": masked}]

    children: List[Dict[str, Any]] = []
    parent_id = 0

    for sec in sections:
        heading = sec["heading_path"]
        # Build parent windows (non-overlapping; parent is for LLM context).
        parent_windows = _split_prose(sec["body"], tokenizer, parent_max_tokens, 0)
        if not parent_windows:
            continue
        for parent_window in parent_windows:
            parent_text = _restore_atomics(parent_window, atomics).strip()
            if not parent_text:
                continue

            # Children: split prose (placeholders stripped) + atomics present
            # in this window, each as its own child.
            contained = [a for a in atomics if a[0] in parent_window]
            prose_only = parent_window
            for ph, _, _ in contained:
                prose_only = prose_only.replace(ph, " ")

            prefix = ("[%s]\n" % heading) if heading else ""
            # prose children
            for ct in _split_prose(prose_only, tokenizer, child_max_tokens, child_overlap_tokens):
                restored = _restore_atomics(ct, atomics).strip()
                if not restored:
                    continue
                children.append(
                    {
                        "text": prefix + restored,
                        "metadata": {
                            "type": "text",
                            "heading_path": heading,
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                        },
                    }
                )
            # atomic children — kept whole
            for _, raw, kind in contained:
                raw = raw.strip()
                if not raw:
                    continue
                children.append(
                    {
                        "text": prefix + raw,
                        "metadata": {
                            "type": kind,
                            "heading_path": heading,
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                        },
                    }
                )
            parent_id += 1

    for i, c in enumerate(children):
        c["metadata"]["chunk_index"] = i

    return children
