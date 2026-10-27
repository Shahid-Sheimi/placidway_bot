import numpy as np
from rank_bm25 import BM25Okapi   # pip install rank-bm25

MIN_COSINE = 0.30   # tune on your test questions

def _tok(t): return re.findall(r"[a-z0-9]+", t.lower())

class Retriever:
    def __init__(self, index_path=None):
        index_path = index_path or os.path.join(BASE_DIR, "data", "kb_index.json")
        with open(index_path, encoding="utf-8") as fh:
            self.index = json.load(fh)
        self.chunks = self.index["chunks"]
        emb = self.index.get("embeddings")
        self.emb = None
        if emb:
            m = np.array(emb, dtype=np.float32)
            self.emb = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.bm25 = BM25Okapi([_tok(c["text"] + " " + c.get("title", ""))
                               for c in self.chunks])

    def search(self, query, top_k=6):
        n = len(self.chunks)
        bm = self.bm25.get_scores(_tok(query))
        bm_rank = np.argsort(-bm)

        dense_rank, cos = [], None
        if self.emb is not None and API_KEY:
            try:
                q = np.array(self._embed(query), dtype=np.float32)
                q /= np.linalg.norm(q)
                cos = self.emb @ q
                dense_rank = np.argsort(-cos)
            except Exception as exc:
                print(f"[retriever] dense failed: {exc}")

        # Reciprocal Rank Fusion
        fused = {}
        for ranking in (bm_rank[:20], dense_rank[:20]):
            for r, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0) + 1 / (60 + r)

        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]

        # Relevance gate: if dense is available, require at least one decent match
        if cos is not None and cos[ranked[0]] < MIN_COSINE and cos.max() < MIN_COSINE:
            return []
        if cos is None and bm[ranked[0]] <= 0:
            return []
        return [(self.chunks[i], fused[i]) for i in ranked]
