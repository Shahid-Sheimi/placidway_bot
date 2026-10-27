# PlacidWay Chatbot

A small, grounded chatbot for PlacidWay. It answers visitor questions using **only**
the seven shared PlacidWay pages, never inventing prices, clinics, or medical facts.
Every answer cites the source URL(s) it came from.

## Features

- **Content ingestion** — `src/scraper.py` fetches exactly the 7 allowed pages, respecting
  `robots.txt`, with a clear `User-Agent` and ~1 request per second.
- **Knowledge base** — `src/build_kb.py` chunks the pages and embeds them (OpenAI
  embeddings + TF-IDF fallback).
- **Grounded answers** — `src/llm.py` prompts an LLM to answer using *only* the retrieved
  passages and to cite source URLs.
- **Honest "I don't know"** — if the content doesn't contain the answer, the bot says so
  and suggests contacting PlacidWay for a free quote.
- **Cost questions** — prices come back with currency (USD) and the page's own wording
  ("starting from", package includes, etc.).
- **Simple chat UI** — a single-page chat interface.
- **Refresh** — `POST /api/refresh` re-scrapes and rebuilds the knowledge base.
- **Live deployment** — run with `./run.sh` (uvicorn) and open the public URL.

## Architecture

The system is a **retrieval-augmented generation (RAG)** pipeline. It never
crawls beyond the 7 hardcoded `ALLOWED_URLS` in `src/config.py`.

## Project layout

```
placidway_bot/
├── .env                 # your OpenAI key (copy from .env.example)
├── .env.example
├── .gitignore
├── README.md
├── run.sh               # start the server (uvicorn)
├── setup_venv.sh        # create .venv + install deps
├── requirements.txt
├── data/                # generated knowledge base
│   ├── kb.json          # raw scraped text (scraper.py)
│   └── kb_index.json    # chunks + embeddings/TF-IDF (build_kb.py)
├── static/              # frontend assets
│   ├── app.js
│   └── style.css
├── templates/           # Jinja2 chat UI
│   └── index.html
├── .venv/               # virtual environment (gitignored)
└── src/                 # Python package
    ├── __init__.py
    ├── config.py        # ALLOWED_URLS + boundary rules (single source of truth)
    ├── scraper.py       # fetches the 7 allowed pages
    ├── build_kb.py      # chunk + embed/TF-IDF -> kb_index.json
    ├── retriever.py     # search (embeddings or TF-IDF fallback)
    ├── llm.py           # grounded answer generation
    └── app.py           # FastAPI server (/ , /api/chat, /api/refresh, ...)
```

```
                        ┌─────────────────────────────────────────────┐
                        │                  VISITOR                     │
                        └──────────────────┬──────────────────────────┘
                                           │ HTTP
                                           ▼
                        ┌─────────────────────────────────────────────┐
                        │            FastAPI  (app.py)                 │
                        │                                             │
                        │  GET  /                 → Jinja2 chat UI    │
                        │  GET  /api/health       → status           │
                        │  POST /api/chat         → grounded answer  │
                        │  POST /api/refresh      → rebuild KB       │
                        │  GET  /api/status       → alias            │
                        │                                             │
                        │  /static  → static assets (CSS/JS)          │
                        └──────────────────┬──────────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────────┐
                        │             retriever.py                     │
                        │   Retriever.search(query, top_k)             │
                        │                                             │
                        │  ┌─────────────────────────┐                │
                        │  │ OpenAI embeddings       │                │
                        │  │ + cosine similarity     │                │
                        │  └───────────┬─────────────┘                │
                        │              │ failure                      │
                        │  ┌───────────▼─────────────┐                │
                        │  │ TF-IDF fallback         │                │
                        │  │ (always available)      │                │
                        │  └─────────────────────────┘                │
                        └──────────────────┬──────────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────────┐
                        │              llm.py                          │
                        │   answer(chunks, question)                   │
                        │                                             │
                        │  OpenAI Chat Completions (gpt-4o-mini)       │
                        │  temperature=0.0, max_tokens=700             │
                        │  System prompt enforces:                     │
                        │   • use ONLY retrieved passages              │
                        │   • cite source URLs                         │
                        │   • never invent prices/clinics/facts       │
                        │   • say "I don't know" + free-quote offer    │
                        └──────────────────┬──────────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────────┐
                        │   { answer, sources[], chunks_used }         │
                        │   + in-memory conversation cache             │
                        │   (OrderedDict, max 50 sessions)             │
                        └─────────────────────────────────────────────┘
```

### Knowledge-base build pipeline

```
7 allowed URLs
        │
        ▼
   src/scraper.py ──► data/kb.json ──► src/build_kb.py ──► data/kb_index.json
   (raw text,     (7 records,      (chunks +       (chunks + idf
    tables →       title,           embeddings/     + embeddings
    markdown rows) page_type)       TF-IDF)         or null)
```

`/api/refresh` runs `src/scraper.py` then `src/build_kb.py` in a background daemon
thread, so the server stays responsive during the rebuild.

### `data/kb_index.json` structure
OPENAI_PORT=8080
HOST=0.0.0.0API_KEY=PORT=8080
HOST=0.0.0.0sk-proj-PIUxf1KmHpkKdG_n4yEUo0PORT=8080
HOST=0.0.0.0DyVLXhSP_DEkNhlOjXcSp1f6Joo-G0I-XwuSL-OkkOonBgexPORT=8080
HOST=0.0.0.0eO_DT3BlbkFJ2Wz5ApttnChZXg-zN7WjRznyIkZawgWucFIYTcI_O9H8pyPORT=8080
HOST=0.0.0.01Zyj_4CcHZEQi8KBaBKWaQvfsFQA

```
data/kb_index.json
├── model: "text-embedding-3-small"
├── chat_model: "gpt-4o-mini"
├── chunks: [                    # one entry per chunk
│   ├── {
│   │   id: "0",
│   │   url: "https://...",
│   │   page_type: "treatment",
│   │   title: "...",
│   │   text: "...",            # ~900 chars, 150 overlap
│   │   tfidf: { token: weight } # always present
│   │ }
│   └── { ... }                  # ~N chunks
│ ]
├── idf: { token: idf_value }     # global IDF table
└── embeddings: [                 # OpenAI vectors (or null)
      [0.123, -0.456, ...]       # one per chunk
  ]
```

## Quick start

```bash
# 1. (once) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. set your OpenAI key
cp .env.example .env
# edit .env and add: OPENAI_API_KEY=sk-...

# 4. scrape + build the knowledge base
python src/scraper.py
python src/build_kb.py

# 5. run the server
bash ./run.sh
```


Then open `http://localhost:8080` (or the printed public URL).

## Environment variables

| Variable            | Required | Description                              |
| ------------------- | -------- | ---------------------------------------- |
| `OPENAI_API_KEY`    | yes      | OpenAI API key for embeddings + answers  |
| `PORT`              | no       | Port to listen on (default `8080`)       |
| `HOST`              | no       | Host to bind (default `0.0.0.0`)         |

## API

- `GET  /`                 — chat UI
- `GET  /api/health`       — server + knowledge-base status
- `POST /api/chat`         — `{ question, session_id?, top_k? }` → `{ answer, sources, chunks_used }`
- `POST /api/refresh`      — re-scrape pages and rebuild the knowledge base
- `GET  /api/status`       — alias of `/api/health`

## Refreshing the knowledge base

```bash
curl -X POST http://localhost:8080/api/refresh
```

The server re-scrapes the 7 pages and rebuilds the index in the background.

## Adding new pages

Edit `ALLOWED_URLS` (and `PAGE_TYPES`) in `src/config.py`, then run:

```bash
python src/scraper.py && python src/build_kb.py
```

or hit `/api/refresh`.

## Notes

- The bot **never** crawls beyond the URLs listed in `ALLOWED_URLS` (`src/config.py`).
- If the answer is not in the content, the bot says so and points the user to a free quote.
- All prices are reported with currency and the page's own context.