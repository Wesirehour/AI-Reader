import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

import requests


def _clean_text(text):
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_response_text(resp):
    raw = resp.content or b""
    if not raw:
        return ""

    # Prefer UTF-8 for markdown; fallback to response encoding hints.
    try:
        return raw.decode("utf-8")
    except Exception:
        pass

    enc = resp.encoding or ""
    if enc:
        try:
            return raw.decode(enc, errors="replace")
        except Exception:
            pass

    apparent = getattr(resp, "apparent_encoding", "") or ""
    if apparent:
        try:
            return raw.decode(apparent, errors="replace")
        except Exception:
            pass

    return raw.decode("utf-8", errors="replace")


def _looks_like_mojibake(text):
    if not text:
        return False
    # Typical UTF-8->Latin1 mojibake footprint in Chinese markdown.
    bad = 0
    total = min(len(text), 5000)
    sample = text[:total]
    for ch in sample:
        code = ord(ch)
        if 192 <= code <= 255:
            bad += 1
    if total == 0:
        return False
    return (bad * 1.0 / total) > 0.03


def compute_file_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _ensure_mineru_success(result, stage):
    if not isinstance(result, dict):
        raise RuntimeError("MinerU API %s invalid response" % stage)
    code = result.get("code")
    if code in (0, "0", None):
        return
    msg = result.get("msg") or result.get("message") or "unknown"
    raise RuntimeError("MinerU API %s failed(code=%s): %s" % (stage, str(code), str(msg)))


def _parse_with_local_cli(pdf_path, out_dir, mineru_cmd):
    cmd = [mineru_cmd, "-p", pdf_path, "-o", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if proc.returncode != 0 and ("not recognized" in (proc.stderr or "").lower() or "no such file" in (proc.stderr or "").lower()):
        fallback_cmd = [sys.executable, "-m", "mineru", "-p", pdf_path, "-o", out_dir]
        proc = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError("MinerU local execution failed: %s" % (proc.stderr.strip() or proc.stdout.strip() or "unknown error"))

    md_files = []
    for root, _dirs, files in os.walk(out_dir):
        for name in files:
            if name.lower().endswith(".md"):
                md_files.append(os.path.join(root, name))

    texts = []
    for md in md_files:
        try:
            with open(md, "r", encoding="utf-8") as f:
                content = _clean_text(f.read())
            if content:
                texts.append(content)
        except Exception:
            continue
    return texts


def _parse_with_agent_api(
    pdf_path,
    out_dir,
    api_base_url,
    timeout_sec,
    poll_interval_sec,
    language,
    enable_table,
    enable_formula,
    is_ocr,
):
    create_url = api_base_url.rstrip("/") + "/parse/file"
    payload = {
        "enable_formula": bool(enable_formula),
        "enable_table": bool(enable_table),
        "language": language,
        "file_name": os.path.basename(pdf_path),
        "is_ocr": bool(is_ocr),
    }

    create_resp = requests.post(create_url, json=payload, timeout=30)
    create_resp.raise_for_status()
    result = create_resp.json()
    _ensure_mineru_success(result, "create_task")
    data = result.get("data") or {}

    upload_url = data.get("file_url") or data.get("upload_url")
    task_id = data.get("task_id") or data.get("id")
    if not upload_url or not task_id:
        raise RuntimeError("MinerU API response missing file_url/task_id: %s" % str(result)[:500])

    with open(pdf_path, "rb") as f:
        upload_resp = requests.put(upload_url, data=f, timeout=120)
    if upload_resp.status_code not in (200, 201):
        detail = upload_resp.text[:300] if upload_resp.text else ""
        raise RuntimeError("MinerU file upload failed, HTTP %s %s" % (upload_resp.status_code, detail))

    poll_url = api_base_url.rstrip("/") + "/parse/%s" % task_id
    deadline = time.time() + max(30, int(timeout_sec))
    final_data = None

    while time.time() < deadline:
        status_resp = requests.get(poll_url, timeout=30)
        status_resp.raise_for_status()
        status_result = status_resp.json()
        _ensure_mineru_success(status_result, "poll")
        status_data = status_result.get("data") or {}
        state = status_data.get("state")
        state_str = str(state).lower() if state is not None else ""

        if state_str in ("done", "success", "succeeded", "finished", "completed"):
            final_data = status_data
            break
        if state_str in ("failed", "error"):
            err = status_data.get("err_msg") or status_data.get("error_message") or "unknown"
            raise RuntimeError("MinerU parse failed: %s" % err)

        time.sleep(max(1, int(poll_interval_sec)))

    if final_data is None:
        raise RuntimeError("MinerU parse timeout after %s seconds, task_id=%s" % (timeout_sec, task_id))

    markdown_url = final_data.get("markdown_url")
    if not markdown_url:
        raise RuntimeError("MinerU parse done but markdown_url missing: %s" % str(final_data)[:500])

    md_resp = requests.get(markdown_url, timeout=60)
    md_resp.raise_for_status()
    markdown_text = _clean_text(_decode_response_text(md_resp))
    if not markdown_text:
        raise RuntimeError("MinerU markdown is empty, task_id=%s" % task_id)

    out_md = os.path.join(out_dir, "result.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    meta = {"task_id": task_id, "markdown_url": markdown_url}
    return markdown_text, meta


def _load_cache(cache_dir):
    meta_path = os.path.join(cache_dir, "meta.json")
    md_path = os.path.join(cache_dir, "result.md")
    if not os.path.exists(meta_path) or not os.path.exists(md_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_text = _clean_text(f.read())
        if not markdown_text or _looks_like_mojibake(markdown_text):
            return None
        return {"markdown_text": markdown_text, "meta": meta, "markdown_path": md_path}
    except Exception:
        return None


def _save_cache(cache_dir, markdown_text, meta):
    os.makedirs(cache_dir, exist_ok=True)
    md_path = os.path.join(cache_dir, "result.md")
    meta_path = os.path.join(cache_dir, "meta.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta or {}, f, ensure_ascii=False, indent=2)
    return md_path


def parse_with_mineru(
    pdf_path,
    output_root,
    mineru_cmd,
    parse_mode="api_file",
    api_base_url="https://mineru.net/api/v1/agent",
    timeout_sec=300,
    poll_interval_sec=3,
    language="ch",
    enable_table=True,
    enable_formula=True,
    is_ocr=False,
    file_hash=None,
    use_cache=True,
):
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    digest = file_hash or compute_file_sha256(pdf_path)
    cache_dir = os.path.join(output_root, "cache", digest)
    if use_cache:
        cached = _load_cache(cache_dir)
        if cached:
            return {
                "output_dir": cache_dir,
                "markdown_text": cached["markdown_text"],
                "markdown_path": cached["markdown_path"],
                "markdown_url": (cached["meta"] or {}).get("markdown_url", ""),
                "cached": True,
            }

    out_dir = os.path.join(output_root, doc_name + "_" + uuid.uuid4().hex[:8])
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if parse_mode == "local_cli":
        texts = _parse_with_local_cli(pdf_path=pdf_path, out_dir=out_dir, mineru_cmd=mineru_cmd)
        markdown_text = _clean_text("\n\n".join(texts))
        md_path = _save_cache(cache_dir=cache_dir, markdown_text=markdown_text, meta={})
        markdown_url = ""
    else:
        markdown_text, meta = _parse_with_agent_api(
            pdf_path=pdf_path,
            out_dir=out_dir,
            api_base_url=api_base_url,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            language=language,
            enable_table=enable_table,
            enable_formula=enable_formula,
            is_ocr=is_ocr,
        )
        md_path = _save_cache(cache_dir=cache_dir, markdown_text=markdown_text, meta=meta)
        markdown_url = (meta or {}).get("markdown_url", "")

    if not markdown_text:
        raise RuntimeError("MinerU finished but markdown content is empty")

    return {
        "output_dir": cache_dir,
        "markdown_text": markdown_text,
        "markdown_path": md_path,
        "markdown_url": markdown_url,
        "cached": False,
    }
