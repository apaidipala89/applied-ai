## Quick orientation

This repository contains small demo projects under `projects/` — each is a self-contained demo. The most relevant demo for new contributors / agents is `projects/02-rag-mini` which is a compact Retrieval-Augmented Generation (RAG) pipeline built with OpenAI embeddings + (FAISS or NumPy) and a Gradio UI.

Read these files first to get the shape of the codebase:

- `projects/02-rag-mini/app.py` — main demo / Gradio UI and the high-level flow (PDF -> chunks -> embeddings -> index -> retrieval -> chat answer).
- `projects/02-rag-mini/rag-mini/config.py` — runtime configuration and env variable defaults (chunk sizes, model names, USE_FAISS flag).
- `projects/02-rag-mini/rag-mini/io/pdf_reader.py` — PDF ingestion; accepts Gradio file objects and local files.
- `projects/02-rag-mini/rag-mini/text/chunking.py` — token-based chunking using `tiktoken` and the project's chunk/overlap conventions.
- `projects/02-rag-mini/rag-mini/utils/rate_limit.py` — generic `with_backoff` decorator used around OpenAI calls to handle RateLimitError/APIError with exponential backoff.

High-level architecture (follow the code):

1. PDF upload → `read_pdfs()` (pdf_reader.py) consolidates bytes and extracts text.
2. Text → `chunk_by_tokens()` (chunking.py) using `tiktoken` encoders; chunk size and overlap come from `config.py` or UI sliders.
3. Chunks → `embed_texts()` (in `app.py`) calling OpenAI embeddings (default model `text-embedding-3-small`, dim=1536).
4. Vectors → `build_index()` uses FAISS if available (controlled by `USE_FAISS`) otherwise a normalized NumPy matrix is used.
5. Query → `retrieve()` does similarity search (FAISS search or dot-product with NumPy) and returns top-k chunks.
6. `generate_answer()` sends system+user messages to OpenAI chat completions and returns the assistant output.

Key conventions & gotchas for agents

- Environment: `OPENAI_API_KEY` must be set. The project uses `python-dotenv` so a `.env` file can be present. If you find a committed `.env` with keys, rotate/remove the secret immediately.
- FAISS is optional: if `faiss` import fails the code falls back to a normalized NumPy matrix. Toggle with `USE_FAISS=0` in env to force NumPy.
- Token-based chunking is canonical here — always use `chunk_by_tokens(text, max_tokens, overlap)` instead of naive character splitting. Defaults in `config.py`.
- Embedding shape: embedding dim is 1536 (text-embedding-3-small). Many places assume this dim when normalizing vectors.
- OpenAI usage patterns: code calls `client.embeddings.create(...)` and `client.chat.completions.create(...)` (see `app.py`). Wrap such calls with the provided `@with_backoff(...)` to reduce transient failures.
- `with_backoff` works for sync and async functions and retries on `openai.RateLimitError` and `openai.APIError` by default. See `rag-mini/utils/rate_limit.py` for parameters (retries, backoff seconds, jitter).

How to run the demo locally (example)

1. Create a virtualenv and install dependencies for the demo folder:

```bash
cd projects/02-rag-mini
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set your OpenAI API key (or create a `.env` with OPENAI_API_KEY):

```bash
export OPENAI_API_KEY="sk-..."
# or use a .env with python-dotenv — repo already loads dotenv in some files
```

3. Run the Gradio demo:

```bash
python app.py
```

Patterns to mimic when editing / adding features

- Prefer token-aware operations: use `tiktoken` encoder via `get_encoder()` in `chunking.py`.
- Keep embedding and search code separated: `embed_texts()` and `build_index()` are good examples of small, testable units.
- Make OpenAI calls idempotent where possible and decorate with `@with_backoff(...)` for robust retries.
- PDF ingestion accepts multiple forms of file objects (Gradio uploads or local paths). Follow `read_pdfs()` patterns to be resilient to different input shapes.

Integration & external dependencies

- OpenAI python SDK (>=1.40.0) — used directly via `OpenAI(...)` and exceptions `RateLimitError`, `APIError`.
- FAISS (optional) — `faiss-cpu` if you want local vector indexing. The code auto-detects availability.
- `tiktoken` for tokenization and chunking.
- `gradio` for the UI in the demos.

Pro tips for agent edits and PRs

- If you change chunking defaults, update `config.py` and the UI sliders in `app.py` together.
- If you add new OpenAI usage, import and re-use `with_backoff` or add tests for retry behavior.
- Avoid committing secrets. If you discover a committed key (check `projects/02-rag-mini/.env`), call it out in the PR and rotate immediately.

If anything above is unclear or you want more detail on a specific file or flow, tell me which area to expand and I will iterate.
