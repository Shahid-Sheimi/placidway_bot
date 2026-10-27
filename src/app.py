#!/usr/bin/env python3
"""
PlacidWay chatbot — FastAPI app.

- /            -> chat UI
- /api/chat    -> {answer, sources}
- /api/refresh -> re-scrape pages and rebuild the knowledge base
- /api/health  -> status

Grounded strictly on the 7 shared PlacidWay pages. Never invents facts.
"""

import json
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from typing import List, Optional
import hmac
from functools import lru_cache
_refresh_lock = threading.Lock()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
KB_INDEX = os.path.join(BASE_DIR, "data", "kb_index.json")

# Load .env if present (so the bot works whether started directly or via run.sh)
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _fh:
        for _line in _fh:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# make local modules importable as a package (works whether this file is
# run as `python src/app.py` or imported as `from src import app`)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src import llm as llm_mod  # noqa: E402
from src import retriever as rmod  # noqa: E402

app = FastAPI(title="PlacidWay Chatbot")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Simple in-memory conversation cache (bounded)
_conv_cache: "OrderedDict[str, List]" = OrderedDict()
_CACHE_MAX = 50

@lru_cache(maxsize=1)
def _load_retriever(mtime: float):
    return rmod.Retriever(KB_INDEX)

def get_retriever():
    if not os.path.exists(KB_INDEX):
        raise RuntimeError("Knowledge base not built yet.")
    return _load_retriever(os.path.getmtime(KB_INDEX))  # auto-reloads after rebuild
def get_retriever() -> rmod.Retriever:
    if not os.path.exists(KB_INDEX):
        raise RuntimeError(
            "Knowledge base not built yet. Run build_kb.py or hit /api/refresh."
        )
    return rmod.Retriever(KB_INDEX)


def _cache_get(session_id: str):
    return _conv_cache.get(session_id, [])


def _cache_put(session_id: str, msgs: List):
    _conv_cache[session_id] = msgs
    _conv_cache.move_to_end(session_id)
    while len(_conv_cache) > _CACHE_MAX:
        _conv_cache.popitem(last=False)


# ---------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    top_k: int = 6


class RefreshRequest(BaseModel):
    force: bool = False


@app.get("/api/health")
async def health():
    has_key = bool(os.environ.get("OPENAI_API_KEY", ""))
    kb_exists = os.path.exists(KB_INDEX)
    info = {
        "status": "ok",
        "openai_key_configured": has_key,
        "knowledge_base_built": kb_exists,
    }
    if kb_exists:
        try:
            with open(KB_INDEX, encoding="utf-8") as fh:
                idx = json.load(fh)
            info["chunks"] = len(idx.get("chunks", []))
            info["has_embeddings"] = bool(idx.get("embeddings"))
        except Exception:
            pass
    return JSONResponse(info)


@app.post("/api/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    top_k = max(1, min(req.top_k, 10))            # cap client input
    history = _cache_get(req.session_id or "default")

    search_q = llm_mod.rewrite_query(history, question)
    hits = get_retriever().search(search_q, top_k=top_k)
    chunks = [c for c, _ in hits]
    if not chunks:
        return JSONResponse({"answer": "I couldn't find this on our pages. "
                             "Please contact PlacidWay for a free quote.",
                             "sources": []})
    result = llm_mod.answer(chunks, question)

    history = _cache_get(session_id)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})
    _cache_put(session_id, history)

    return JSONResponse(
        {
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks_used": result["chunks_used"],
        }
    )


@app.post("/api/refresh")
def refresh(request: Request):
    token = os.environ.get("ADMIN_TOKEN", "")
    given = request.headers.get("x-admin-token", "")
    if not token or not hmac.compare_digest(token, given):
        raise HTTPException(status_code=403)
    if not _refresh_lock.acquire(blocking=False):
        return JSONResponse({"status": "already running"})
    def run():
        try: ...  # your existing subprocess calls
        finally: _refresh_lock.release()
    threading.Thread(target=run, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.get("/api/status")
async def status():
    return await health()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
