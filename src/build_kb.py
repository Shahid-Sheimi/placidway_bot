#!/usr/bin/env python3
"""
Build the PlacidWay knowledge base.

Splits each scraped page into overlapping chunks, embeds them with OpenAI,
and stores a JSON index (chunks + vectors) plus a simple TF-IDF fallback
so retrieval works even without an API key.

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 build_kb.py
"""

import json
import math
import os
import re
import sys
from collections import Counter
from typing import List, Dict, Any

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

# Load .env if present
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _fh:
        for _line in _fh:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
API_KEY = os.environ.get("OPENAI_API_KEY", "")

CHUNK_SIZE = 900      # chars per chunk
CHUNK_OVERLAP = 150   # chars of overlap between neighbours


# ---------------------------------------------------------------- chunking
def chunk_text(text: str, url: str, page_type: str, title: str) -> List[Dict]:
    """Split text on sentence-ish boundaries, keeping overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    n = len(text)
    idx = 0
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        # try to break on a sentence boundary
        if end < n:
            cut = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
                text.rfind("\n", start, end),
            )
            if cut > start + CHUNK_SIZE // 2:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "id": f"{idx}",
                    "url": url,
                    "page_type": page_type,
                    "title": title,
                    "text": chunk,
                }
            )
            idx += 1
        if end >= n:
            break
        start = end - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------- embeddings
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings API. Raises on failure."""
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = requests.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return [d["embedding"] for d in data["data"]]


# ---------------------------------------------------------------- tf-idf
def build_tfidf(chunks: List[Dict]):
    """Lightweight TF-IDF fallback for retrieval when no embeddings."""
    docs = [c["text"] for c in chunks]
    N = len(docs)
    df = []
    tokenized = []
    for d in docs:
        tokens = set(re.findall(r"[a-z0-9]+", d.lower()))
        tokenized.append(tokens)
        df.append({t: 1 for t in tokens})
    idf = {}
    for t in set(t for d in df for t in d):
        n = sum(1 for d in df if t in d)
        idf[t] = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
    vecs = []
    for i, d in enumerate(docs):
        tf = Counter(re.findall(r"[a-z0-9]+", d.lower()))
        length = sum(tf.values()) or 1
        vecs.append(
            {t: (c / length) * idf[t] for t, c in tf.items()}
        )
    return vecs, idf


def tfidf_score(query_tokens, vec):
    q_len = len(query_tokens) or 1
    q_tf = Counter(query_tokens)
    score = 0.0
    for t, c in q_tf.items():
        if t in vec:
            score += (c / q_len) * vec[t]
    return score


# ---------------------------------------------------------------- main
def main():
    kb_path = os.path.join(BASE_DIR, "data", "kb.json")
    if not os.path.exists(kb_path):
        print(f"[build_kb] {kb_path} not found. Run scraper.py first.")
        sys.exit(1)

    with open(kb_path, encoding="utf-8") as fh:
        records = json.load(fh)

    chunks = []
    for r in records:
        if not r.get("text"):
            continue
        chunks.extend(
            chunk_text(
                r["text"], r["url"], r["page_type"], r.get("title", "")
            )
        )
    print(f"[build_kb] {len(chunks)} chunks from {len(records)} pages")

    # TF-IDF fallback (always built)
    tfidf_vecs, idf = build_tfidf(chunks)
    for c, v in zip(chunks, tfidf_vecs):
        c["tfidf"] = v

    index: Dict[str, Any] = {
        "model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "chunks": chunks,
        "idf": idf,
        "embeddings": None,
    }

    # Optional OpenAI embeddings
    if API_KEY:
        try:
            print("[build_kb] embedding chunks with OpenAI ...")
            texts = [c["text"] for c in chunks]
            vecs = embed_texts(texts)
            index["embeddings"] = vecs
            print(f"[build_kb] embedded {len(vecs)} chunks")
        except Exception as exc:
            print(f"[build_kb] embedding failed: {exc}; using TF-IDF only")
    else:
        print("[build_kb] no OPENAI_API_KEY; using TF-IDF fallback only")

    out = os.path.join(BASE_DIR, "data", "kb_index.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False)
    print(f"[build_kb] wrote {out}")


if __name__ == "__main__":
    main()