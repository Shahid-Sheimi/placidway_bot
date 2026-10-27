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
2. Every factual claim must end with a citation marker like [1] or [2][3],
   matching the passage numbers. Do NOT write URLs yourself.
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


def answer(chunks, question):
    ...
    text = data["choices"][0]["message"]["content"]

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)
                    if 1 <= int(n) <= len(chunks)})
    sources, seen = [], set()
    for n in cited:
        c = chunks[n - 1]
        if c["url"] not in seen:
            seen.add(c["url"])
            sources.append({"n": n, "url": c["url"], "title": c.get("title", "")})
    return {"answer": text, "sources": sources, "chunks_used": len(chunks)}


def rewrite_query(history, question):
    """Turn a follow-up into a standalone search query."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    resp = requests.post(OPENAI_API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": CHAT_MODEL, "temperature": 0, "max_tokens": 60,
              "messages": [{"role": "user", "content":
                f"Conversation:\n{convo}\n\nRewrite the last user question as a "
                f"standalone search query. Output only the query.\n"
                f"Question: {question}"}]},
        timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
  
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
