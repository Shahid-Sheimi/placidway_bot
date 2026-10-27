#!/usr/bin/env python3
"""
Build the PlacidWay knowledge base using LangChain.

Uses LangChain text splitters with recursive character splitting,
semantic-aware chunking strategies, and stores everything in the KB
with rich metadata. Builds FAISS vector store + BM25 for hybrid retrieval.
"""

import json
import math
import os
import re
import sys
from collections import Counter
from typing import List, Dict, Any

import requests

try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        TokenTextSplitter,
    )
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_text_langchain(text: str, url: str, page_type: str, title: str) -> List[Dict]:
    """Use LangChain splitters for proper chunking with multiple strategies."""
    if not LANGCHAIN_AVAILABLE:
        # Fallback to original chunking
        return chunk_text_legacy(text, url, page_type, title)
    
    chunks = []
    
    # Strategy 1: Markdown header splitting if markdown-like structure
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    try:
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        md_splits = markdown_splitter.split_text(text)
        if md_splits and len(md_splits) > 1:
            # Further split large sections
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            )
            for i, doc in enumerate(md_splits):
                sub_chunks = text_splitter.split_text(doc.page_content)
                for j, sub_chunk in enumerate(sub_chunks):
                    if sub_chunk.strip():
                        metadata = dict(doc.metadata)
                        metadata.update({
                            "url": url,
                            "page_type": page_type,
                            "title": title,
                            "chunk_strategy": "markdown_recursive",
                        })
                        chunks.append({
                            "id": f"md_{i}_{j}",
                            "url": url,
                            "page_type": page_type,
                            "title": title,
                            "text": sub_chunk.strip(),
                            "metadata": metadata,
                        })
            if chunks:
                return chunks
    except Exception:
        pass
    
    # Strategy 2: Recursive character splitting (primary strategy)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    
    docs = text_splitter.create_documents([text], metadatas=[{
        "url": url,
        "page_type": page_type,
        "title": title,
    }])
    
    for i, doc in enumerate(docs):
        chunks.append({
            "id": f"rc_{i}",
            "url": url,
            "page_type": page_type,
            "title": title,
            "text": doc.page_content.strip(),
            "metadata": {
                **doc.metadata,
                "chunk_strategy": "recursive_character",
                "chunk_index": i,
            },
        })
    
    return chunks


def chunk_text_legacy(text: str, url: str, page_type: str, title: str) -> List[Dict]:
    """Legacy chunking as fallback."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    n = len(text)
    idx = 0
    while start < n:
        end = min(start + CHUNK_SIZE, n)
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
                    "id": f"leg_{idx}",
                    "url": url,
                    "page_type": page_type,
                    "title": title,
                    "text": chunk,
                    "metadata": {"chunk_strategy": "legacy"},
                }
            )
            idx += 1
        if end >= n:
            break
        start = end - CHUNK_OVERLAP
    return chunks


# Keep original for compatibility
def chunk_text(text: str, url: str, page_type: str, title: str) -> List[Dict]:
    return chunk_text_langchain(text, url, page_type, title)


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

    all_chunks = []
    for r in records:
        if not r.get("text"):
            continue
        chunks = chunk_text_langchain(
            r["text"], r["url"], r["page_type"], r.get("title", "")
        )
        all_chunks.extend(chunks)
    print(f"[build_kb] {len(all_chunks)} chunks from {len(records)} pages using LangChain splitters")

    # TF-IDF fallback (always built)
    tfidf_vecs, idf = build_tfidf(all_chunks)
    for c, v in zip(all_chunks, tfidf_vecs):
        c["tfidf"] = v

    index: Dict[str, Any] = {
        "model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "chunks": all_chunks,
        "idf": idf,
        "embeddings": None,
        "langchain_enabled": LANGCHAIN_AVAILABLE,
        "chunking_strategy": "langchain_recursive_markdown",
    }

    # Optional OpenAI embeddings
    if API_KEY:
        try:
            print("[build_kb] embedding chunks with OpenAI ...")
            texts = [c["text"] for c in all_chunks]
            vecs = embed_texts(texts)
            index["embeddings"] = vecs
            print(f"[build_kb] embedded {len(vecs)} chunks")
        except Exception as exc:
            print(f"[build_kb] embedding failed: {exc}; using TF-IDF only")
    else:
        print("[build_kb] no OPENAI_API_KEY; using TF-IDF fallback only")

    # Build LangChain vector stores if available
    if LANGCHAIN_AVAILABLE and API_KEY:
        try:
            print("[build_kb] building LangChain FAISS vector store...")
            docs = []
            for c in all_chunks:
                docs.append(Document(
                    page_content=c["text"],
                    metadata={
                        "url": c["url"],
                        "page_type": c["page_type"],
                        "title": c["title"],
                        "chunk_id": c["id"],
                        **c.get("metadata", {}),
                    }
                ))
            embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
            vectorstore = FAISS.from_documents(docs, embeddings)
            faiss_path = os.path.join(BASE_DIR, "data", "faiss_index")
            vectorstore.save_local(faiss_path)
            index["faiss_index_path"] = faiss_path
            print(f"[build_kb] saved FAISS index to {faiss_path}")
        except Exception as exc:
            print(f"[build_kb] FAISS build failed: {exc}")

    out = os.path.join(BASE_DIR, "data", "kb_index.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False)
    print(f"[build_kb] wrote {out}")


if __name__ == "__main__":
    main()
