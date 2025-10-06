## Project: 02-rag-mini — RAG mini demo quick guide

Purpose
This project is a compact RAG pipeline demonstrating: PDF ingestion → token-aware chunking → OpenAI embeddings → FAISS (optional) or NumPy index → retrieval → chat answer in a Gradio UI.

Read these files first

- `app.py` — the complete flow and Gradio UI (embedding, index build, retrieve, answer).
- `rag-mini/config.py` — defaults for CHUNK_TOKENS, CHUNK_OVERLAP, EMBED_DIM, USE_FAISS.
- `rag-mini/text/chunking.py` — use `chunk_by_tokens()` (tiktoken-based) for splitting text.
- `rag-mini/io/pdf_reader.py` — `read_pdfs()` accepts Gradio file uploads and local paths.
- `rag-mini/utils/rate_limit.py` — `@with_backoff(...)` decorator for API calls.

Important patterns

- Token-based chunking (not char-based). Use `get_encoder()` in `rag-mini/text/chunking.py`.
- Embeddings are normalized (unit vectors) — code assumes dimension 1536 for `text-embedding-3-small`.
- Indexing: detect FAISS at import time; when FAISS is missing the code uses a normalized NumPy matrix and dot-product retrieval.
- OpenAI calls live in `embed_texts()` and `generate_answer()`; they should be decorated with `@with_backoff` in production.

Run locally

```bash
cd projects/02-rag-mini
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python app.py
```

Secrets

- This project previously had a committed `.env` under the project root. Never commit `.env` with real keys; use `.env.example` in repo instead.
