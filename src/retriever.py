import os
import json
import re
from typing import List, Dict, Any

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _fh:
        for _line in _fh:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBED_MODEL = "text-embedding-3-small"

MIN_COSINE = 0.25  # Slightly lower threshold for better recall


def _tok(t):
    return re.findall(r"[a-z0-9]+", t.lower())


class Retriever:
    def __init__(self, index_path=None):
        index_path = index_path or os.path.join(BASE_DIR, "data", "kb_index.json")
        with open(index_path, encoding="utf-8") as fh:
            self.index = json.load(fh)
        self.chunks = self.index["chunks"]
        
        # Dense embeddings
        emb = self.index.get("embeddings")
        self.emb = None
        if emb:
            m = np.array(emb, dtype=np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.emb = m / norms
        
        # BM25 (legacy)
        if BM25Okapi:
            try:
                self.bm25 = BM25Okapi([_tok(c["text"] + " " + c.get("title", ""))
                                       for c in self.chunks])
            except Exception:
                self.bm25 = None
        else:
            self.bm25 = None
        
        # LangChain ensemble retriever
        self.ensemble_retriever = None
        if LANGCHAIN_AVAILABLE and API_KEY:
            try:
                faiss_path = self.index.get("faiss_index_path") or os.path.join(BASE_DIR, "data", "faiss_index")
                if os.path.exists(faiss_path):
                    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
                    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
                    
                    # BM25 retriever
                    docs = [Document(page_content=c["text"], metadata={
                        "url": c["url"], "page_type": c["page_type"], 
                        "title": c["title"], "chunk_id": c["id"]
                    }) for c in self.chunks]
                    bm25_retriever = BM25Retriever.from_documents(docs)
                    bm25_retriever.k = 20
                    
                    # Ensemble with weights
                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=[vector_retriever, bm25_retriever],
                        weights=[0.6, 0.4]  # Favor semantic search
                    )
                    print("[retriever] LangChain ensemble retriever initialized")
            except Exception as exc:
                print(f"[retriever] LangChain ensemble failed: {exc}")
                self.ensemble_retriever = None

    def search_langchain(self, query, top_k=6):
        """Use LangChain ensemble retriever."""
        if not self.ensemble_retriever:
            return None
        try:
            docs = self.ensemble_retriever.get_relevant_documents(query)
            results = []
            # Map back to chunks
            for doc in docs[:top_k]:
                chunk_id = doc.metadata.get("chunk_id")
                # Find matching chunk
                for chunk in self.chunks:
                    if chunk.get("id") == chunk_id or chunk["text"] == doc.page_content:
                        results.append((chunk, 1.0))
                        break
            return results if results else None
        except Exception as exc:
            print(f"[retriever] LangChain search failed: {exc}")
            return None

    def search(self, query, top_k=6):
        # Try LangChain first if available
        lc_results = self.search_langchain(query, top_k)
        if lc_results:
            return lc_results
        
        n = len(self.chunks)
        scores = []
        
        # BM25 scores
        bm_scores = np.zeros(n)
        if self.bm25:
            try:
                bm_scores = self.bm25.get_scores(_tok(query))
            except Exception:
                bm_scores = np.zeros(n)
        
        bm_rank = np.argsort(-bm_scores) if self.bm25 else np.arange(n)
        
        # Dense scores
        dense_rank = []
        cos_scores = None
        if self.emb is not None and API_KEY:
            try:
                q = np.array(self._embed(query), dtype=np.float32)
                q_norm = np.linalg.norm(q)
                if q_norm > 0:
                    q /= q_norm
                    cos_scores = self.emb @ q
                    dense_rank = np.argsort(-cos_scores)
            except Exception as exc:
                print(f"[retriever] dense failed: {exc}")
        
        # Reciprocal Rank Fusion with improved scoring
        fused = {}
        if len(bm_rank) > 0:
            for r, i in enumerate(bm_rank[:30]):
                fused[int(i)] = fused.get(int(i), 0) + 1.0 / (60 + r) * 0.4  # BM25 weight
        if len(dense_rank) > 0:
            for r, i in enumerate(dense_rank[:30]):
                fused[int(i)] = fused.get(int(i), 0) + 1.0 / (60 + r) * 0.6  # Dense weight
        
        if not fused:
            # Fallback to just BM25
            ranked = bm_rank[:top_k] if len(bm_rank) > 0 else list(range(min(top_k, n)))
        else:
            ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        
        # Relevance gate
        if cos_scores is not None and len(ranked) > 0:
            if cos_scores[ranked[0]] < MIN_COSINE and (cos_scores.max() if len(cos_scores) > 0 else 0) < MIN_COSINE:
                # Also check BM25
                if self.bm25 and bm_scores[ranked[0]] <= 0:
                    return []
        elif self.bm25 and len(ranked) > 0 and bm_scores[ranked[0]] <= 0:
            return []
        
        return [(self.chunks[i], fused.get(i, bm_scores[i])) for i in ranked]

    def _embed(self, text):
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBED_MODEL, "input": [text]},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
