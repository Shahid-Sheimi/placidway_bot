# PlacidWay Chatbot

A grounded RAG chatbot for PlacidWay that answers questions using **only** the 7 authorized PlacidWay pages. It uses LangChain for advanced chunking and retrieval, maintains conversation context, and provides cited sources.

## Features

- **LangChain-Powered RAG**: Uses LangChain text splitters (RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter) for intelligent chunking
- **Hybrid Retrieval**: Combines FAISS vector search, BM25, and ensemble retrieval for better results
- **Grounded Answers**: Answers strictly from retrieved passages with citations [n]
- **Context Memory**: Maintains conversation history for follow-up questions
- **Proper Citations**: Returns deduplicated source links with titles
- **Strict Guardrails**: No medical advice, no booking, polite declines for out-of-scope queries
- **TF-IDF Fallback**: Works even without OpenAI API key

## Quick Start (Streamlit)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment
Copy `.env.example` to `.env` and add your OpenAI API key (optional - will use TF-IDF fallback without it):
```bash
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=sk-...
```

### 3. Build the knowledge base
```bash
python src/scraper.py      # Fetch the 7 allowed pages
python src/build_kb.py     # Chunk, embed with LangChain, build FAISS index
```

### 4. Run with Streamlit
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project Structure

```
placidway_bot/
├── streamlit_app.py    # Streamlit UI
├── requirements.txt    # Dependencies (includes LangChain)
├── src/
│   ├── config.py       # Allowed URLs and config
│   ├── scraper.py      # Fetches 7 authorized pages
│   ├── build_kb.py     # LangChain chunking + FAISS + embeddings
│   ├── retriever.py    # Hybrid retrieval (FAISS + BM25 + ensemble)
│   ├── llm.py          # Grounded answer generation with citations
│   └── app.py          # FastAPI server (optional)
├── data/
│   ├── kb.json         # Raw scraped content
│   └── kb_index.json   # Processed chunks with embeddings
└── static/             # Frontend assets (for FastAPI)
```

## How It Works

1. **Scraping**: Only the 7 hardcoded URLs from `config.py` are fetched
2. **Chunking**: LangChain's RecursiveCharacterTextSplitter creates overlapping chunks (1000 chars, 200 overlap)
3. **Embedding**: OpenAI text-embedding-3-small creates vectors, stored in FAISS
4. **Retrieval**: Hybrid search using FAISS + BM25 with Reciprocal Rank Fusion
5. **Generation**: LLM answers using only retrieved passages with strict guardrails
6. **Citations**: All factual claims cite source passages [n]

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repo, set `streamlit_app.py` as the main file
4. Add `OPENAI_API_KEY` in Secrets (optional)

### Local
```bash
streamlit run streamlit_app.py
```

## Environment Variables

- `OPENAI_API_KEY` (optional): For better embeddings and answers. Falls back to TF-IDF + extractive answers if not set.

## Notes

- All data stays in the KB - nothing is fetched at query time
- Never invents prices, clinics, or medical facts
- Strictly follows all guardrails for medical advice, booking, etc.
- Maintains conversation context across messages
