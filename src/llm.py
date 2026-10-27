#!/usr/bin/env python3
"""
Grounded answer generation.

Takes retrieved chunks and asks an LLM to answer using ONLY those chunks.
The prompt enforces:
  - cite source URLs
  - never invent prices / clinics / medical facts
  - if the chunks don't contain the answer, say so and point to a free quote
"""

import json
import os
import sys
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

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
CHAT_MODEL = "gpt-4o-mini"
API_KEY = os.environ.get("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """You are a careful, honest assistant for PlacidWay Medical Tourism.

STRICT RULES:
1. Answer the user's question using ONLY the facts in the retrieved passages below.
2. Every factual claim must be traceable to a passage. Cite the source URL(s) in parentheses.
3. If the retrieved passages do NOT contain enough information to answer, say so clearly and
   suggest contacting PlacidWay for a free, personalized quote. Never invent prices,
   clinics, treatments, locations, or medical facts.
4. For cost questions: include the currency (USD), the exact wording from the page
   (e.g. "starting from", "from", "package includes"), and any conditions mentioned.
5. Keep answers concise, warm, and helpful. Do not give medical advice or diagnoses.
6. If you are unsure about anything, say so rather than guessing.
"""

USER_PROMPT_TEMPLATE = """Retrieved passages (from PlacidWay pages only):
{passages}

User question: {question}

Now answer the user's question following the STRICT RULES above.
"""


def _format_passages(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {c['url']}\nTitle: {c.get('title','')}\n"
            f"Text: {c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def build_messages(chunks: List[Dict], question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                passages=_format_passages(chunks), question=question
            ),
        },
    ]


def answer(chunks: List[Dict], question: str) -> Dict[str, Any]:
    """Generate a grounded answer. Returns {answer, sources, used_chunks}."""
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    messages = build_messages(chunks, question)
    resp = requests.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 700,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]

    # collect unique source URLs actually used (from chunks we fed)
    sources = []
    seen = set()
    for c in chunks:
        u = c["url"]
        if u not in seen:
            seen.add(u)
            sources.append({"url": u, "title": c.get("title", "")})

    return {
        "answer": text,
        "sources": sources,
        "chunks_used": len(chunks),
    }


if __name__ == "__main__":
    sys.path.insert(0, BASE_DIR)
    from src import retriever as rmod

    rb = rmod.Retriever()
    q = "How much does alternative cancer treatment cost at ITC in Tijuana?"
    hits = rb.search(q)
    out = answer([c for c, _ in hits], q)
    print(out["answer"])
    print("\nSources:")
    for s in out["sources"]:
        print(" -", s["url"])