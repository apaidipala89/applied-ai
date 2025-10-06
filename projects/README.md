# applied-ai

Building something real while reinforcing the math & theory.

> A curated set of small, focused AI engineering projects: chat, retrieval, agents, deployment experiments.

## Table of Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Project: 02-rag-mini](#project-02-rag-mini)
- [Environment variables](#environment-variables)
- [Architecture (RAG mini)](#architecture-rag-mini)
- [Journey log](#journey-log)
- [Roadmap](#roadmap)
- [Cost & safety tips](#cost--safety-tips)
- [Contributing](#contributing)
- [License](#license)

## Overview

Hands-on implementations to internalize core AI patterns:

- Minimal CLI chat
- Retrieval-Augmented Generation (RAG) with embeddings + FAISS/NumPy
- (Upcoming) multi-step research / tool-using agent
- (Upcoming) deployment + evaluation experiments

## Repository layout

```
projects/
  01-chatbot-hello-world/   # Minimal OpenAI CLI chatbot (baseline patterns)
  02-rag-mini/              # Local PDF Q&A (embeddings + retrieval + Gradio UI)
  03-agent-researcher/      # (planned) iterative research agent
```

## Quick start

Global (one project at a time):

```bash
cd projects/02-rag-mini
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit OPENAI_API_KEY
python app.py
```

Open: http://127.0.0.1:7860

## Project: 02-rag-mini

Status: ✅ Usable (embedding requires valid API billing)
Features:

- PDF ingestion (multiple files)
- Token-aware chunking (`tiktoken`)
- OpenAI embeddings (`text-embedding-3-small`) → FAISS or normalized NumPy
- Similarity search (inner product ~ cosine)
- Context-grounded answer generation (chat completions)
- Backoff + retry on transient API errors

When to use:

- Testing RAG fundamentals
- Benchmarking chunk size / overlap impacts
- Prototyping index swapping (FAISS vs in-memory)

## Environment variables

Define in `.env`:

- OPENAI_API_KEY (required; billing-enabled)
- USE_FAISS=1|0 (optional; defaults to 1 if faiss installed)
- EMBEDDING_MODEL (default: text-embedding-3-small)
- CHAT_MODEL (default: gpt-4o-mini)
- CHUNK_TOKENS / CHUNK_OVERLAP
- MAX_COMPLETION_TOKENS / TEMPERATURE / TOP_K

## Architecture (RAG mini)

Flow:
PDF(s) → extract text → token chunking → embeddings → index (FAISS|NumPy) → retrieve top-k → format context → LLM answer.

Key modules:

- `io/pdf_reader.py` (extraction)
- `text/chunking.py` (token slicing)
- `index/vector_store.py` (embedding + index build)
- `retriever/retriever.py` (similarity scoring)
- `llm/answer.py` (answer synthesis)
- `utils/rate_limit.py` (resilient API calls)

## Journey log

(Use this to track learning milestones — keep terse.)

- [ ] (YYYY-MM-DD) Baseline RAG latency measurement
- [ ] (YYYY-MM-DD) Compare overlap 0 vs 50 vs 100
- [ ] (YYYY-MM-DD) Add evaluation set & accuracy metric
- [ ] (YYYY-MM-DD) Cache embeddings to disk

## Roadmap

- [ ] Persist / load existing index
- [ ] Add dry-run (offline dummy vectors)
- [ ] Add answer citation highlighting
- [ ] Add simple eval harness (questions + expected answers)
- [ ] Switch to streaming responses for chat
- [ ] Add agent project scaffold (03-agent-researcher)
- [ ] Optional local embedding model path

## Cost & safety tips

- ChatGPT Plus ≠ API credits; add billing for API.
- Rotate leaked keys immediately (delete + recreate in dashboard).
- Cache embeddings to avoid re-paying for unchanged PDFs.
- Keep chunk size moderate (400–800 tokens) to reduce duplicate context.
- Log token usage (can wrap OpenAI calls & aggregate).

## Contributing

1. Fork / branch
2. Keep patches focused (one concern)
3. Run formatting / lint (TBD if adding tools)
4. Do not commit `.env` or keys

## License

TBD (add MIT/Apache-2.0 to clarify reuse).

---

A collection of applied AI engineering project, from simple chatbots to RAG pipelines, agents and deployments.
This repo documents my journey into AI engineering, with each project in its own folder.

- **01-chatbot-hello-world** - minimal OpenAI CLI chatbot
- **02-rag-mini** - local PDF Q&A with embeddings
- 03-agent-researcher - (coming soon) multi-step research agent
