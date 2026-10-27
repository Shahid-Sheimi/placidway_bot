#!/usr/bin/env python3
"""
Retrieval over the built PlacidWay knowledge base.

Chooses the best available signal:
  1. OpenAI embeddings + cosine similarity (if available)
  2. TF-IDF fallback (always available)

Returns ranked chunks so the LLM can be grounded strictly in them.
"""

import json
import math
import os
import re
from collections import Counter
from typing import List, Dict, Any, Tuple

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

OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
EMBED_MODEL = "text-embedding-3-small"
API_KEY = os.environ.get("OPENAI_API_KEY", "")


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class Retriever:
    def __init__(self, index_path: str = None):
        if index_path is None:
            index_path = os.path.join(BASE_DIR, "data", "kb_index.json")
        with open(index_path, encoding="utf-8") as fh:
            self.index = json.load(fh)
        self.chunks: List[Dict] = self.index["chunks"]
        self.embeddings = self.index.get("embeddings")
        self.idf = self.index.get("idf", {})
        self.has_embeddings = bool(self.embeddings)

    # ---------------------------------------------------------- embedding
    def _embed(self, text: str) -> List[float]:
        if not API_KEY:
            raise RuntimeError("no OPENAI_API_KEY")
        resp = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBED_MODEL, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    # ---------------------------------------------------------- tfidf
    def _tfidf_vec(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        tf = Counter(tokens)
        length = sum(tf.values()) or 1
        return {t: (c / length) * self.idf.get(t, 1.0) for t, c in tf.items()}

    def _tfidf_score(self, qvec: Dict[str, float], dvec: Dict[str, float]) -> float:
        score = 0.0
        for t, qw in qvec.items():
            if t in dvec:
                score += qw * dvec[t]
        return score

    # ---------------------------------------------------------- public
    def search(self, query: str, top_k: int = 6) -> List[Tuple[Dict, float]]:
        """Return [(chunk, score), ...] ranked best-first."""
        if self.has_embeddings and API_KEY:
            try:
                qv = self._embed(query)
                scored = [
                    (c, _cosine(qv, e))
                    for c, e in zip(self.chunks, self.embeddings)
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:top_k]
            except Exception as exc:
                print(f"[retriever] embedding search failed ({exc}); tfidf fallback")
        # TF-IDF fallback
        qv = self._tfidf_vec(query)
        scored = [
            (c, self._tfidf_score(qv, c.get("tfidf", {})))
            for c in self.chunks
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# small helper used by the module too
from collections import Counter  # noqa: E402


if __name__ == "__main__":
    r = Retriever()
    q = "How much does a dental implant cost in Mexico?"
    for chunk, score in r.search(q):
        print(f"{score:.3f}  {chunk['url']}")
        print("   ", chunk["text"][:160].replace("\n", " "))