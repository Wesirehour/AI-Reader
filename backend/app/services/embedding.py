import hashlib
import math
import re


def _tokenize(text):
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text)
    tokens = []
    for w in words:
        if re.match(r"^[\u4e00-\u9fff]+$", w):
            tokens.extend(list(w))
            if len(w) > 1:
                for i in range(len(w) - 1):
                    tokens.append(w[i : i + 2])
        else:
            tokens.append(w)
    return tokens


def embed_text(text, dim=384):
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for t in tokens:
        h = hashlib.sha1(t.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        vec[idx] += 1.0

    norm = math.sqrt(sum([x * x for x in vec])) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(vec_a, vec_b):
    n = min(len(vec_a), len(vec_b))
    s = 0.0
    for i in range(n):
        s += vec_a[i] * vec_b[i]
    return s
