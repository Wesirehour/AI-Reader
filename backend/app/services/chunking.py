import re


def _split_paragraphs(text):
    chunks = re.split(r"\n\s*\n", text)
    return [c.strip() for c in chunks if c and c.strip()]


def _split_long_text(text, size):
    items = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        items.append(text[start:end])
        start = end
    return items


def chunk_texts(texts, chunk_size=700, overlap=120):
    units = []
    for text in texts:
        units.extend(_split_paragraphs(text))

    prepared = []
    for u in units:
        if len(u) <= chunk_size:
            prepared.append(u)
        else:
            prepared.extend(_split_long_text(u, chunk_size))

    result = []
    cur = ""
    index = 0

    for p in prepared:
        if not cur:
            cur = p
            continue

        candidate = cur + "\n" + p
        if len(candidate) <= chunk_size:
            cur = candidate
            continue

        result.append({"chunk_index": index, "text": cur})
        index += 1
        tail = cur[-overlap:] if overlap > 0 else ""
        cur = (tail + "\n" + p).strip()

    if cur:
        result.append({"chunk_index": index, "text": cur})

    return result


def _split_markdown_sections(markdown_text):
    lines = (markdown_text or "").split("\n")
    sections = []
    current_title = "Document"
    current_level = 1
    current_body = []

    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    for line in lines:
        m = heading_re.match(line.strip())
        if m:
            if current_body:
                sections.append(
                    {"title": current_title, "level": current_level, "text": "\n".join(current_body).strip()}
                )
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append({"title": current_title, "level": current_level, "text": "\n".join(current_body).strip()})
    return [s for s in sections if s["text"]]


def chunk_markdown_by_heading(markdown_text, chunk_size=1000, overlap=150):
    sections = _split_markdown_sections(markdown_text)
    chunks = []
    idx = 0

    for sec in sections:
        prefix = "[H%d] %s\n" % (sec["level"], sec["title"])
        body = sec["text"]
        raw = prefix + body

        if len(raw) <= chunk_size:
            chunks.append(
                {
                    "chunk_index": idx,
                    "text": raw,
                    "section_title": sec["title"],
                    "section_level": sec["level"],
                }
            )
            idx += 1
            continue

        start = 0
        while start < len(body):
            end = min(len(body), start + max(200, chunk_size - len(prefix)))
            part = body[start:end]
            chunk_text = prefix + part
            chunks.append(
                {
                    "chunk_index": idx,
                    "text": chunk_text,
                    "section_title": sec["title"],
                    "section_level": sec["level"],
                }
            )
            idx += 1
            if end >= len(body):
                break
            start = max(0, end - overlap)

    return chunks
