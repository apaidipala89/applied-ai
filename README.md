# applied-ai

Building something real, while still reinforcing the math &amp; theory.

A collection of applied AI engineering project, from simple chatbots to RAG pipelines, agents and deployments.
This repo documents my journey into AI engineering, with each project in its own folder.

## 1. Elevator pitch

Hands-on AI engineering sandbox: chat, retrieval (RAG), emerging agents, and deployment experiments—built small, composable, and testable.

## 2. Project index

| Project                | Description                                                        | Status                    |
| ---------------------- | ------------------------------------------------------------------ | ------------------------- |
| 01-chatbot-hello-world | Minimal OpenAI CLI chat loop                                       | ✅                        |
| 02-rag-mini            | Local PDF Q&A (token chunking + embeddings + FAISS/NumPy + Gradio) | ✅ (requires API billing) |
| 03-agent-researcher    | Multi-step research / tool use scaffold                            | ⏳                        |
| (future) eval-harness  | Shared evaluation + regression tests                               | Planned                   |

## 3. Repository layout

```
applied-ai/
  README.md
  projects/
    01-chatbot-hello-world/
    02-rag-mini/
    03-agent-researcher/
  .github/
    copilot-instructions.md
```

## 4. Quick start (example: 02-rag-mini)

```bash
cd projects/02-rag-mini
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add a billing-enabled OPENAI_API_KEY
python app.py
```

Open: http://127.0.0.1:7860

### Quick start (example: 01-chatbot-hello-world)

```bash
cd projects/01-chatbot-hello-world
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # if present, else install openai + dotenv
cp .env.example .env  # ensure OPENAI_API_KEY set (billing-enabled)
python chat.py        # or the main script filename if different
```

Expected behavior:

- Prompts for user input in a loop
- Sends each turn to OpenAI Chat model
- Prints assistant reply until you type an exit command (e.g. Ctrl+C)

## 5. Core concepts practiced

- Token-aware chunking (tiktoken)
- Embedding + vector indexing (FAISS / normalized NumPy)
- Retrieval + context assembly
- Chat completion prompting
- Backoff / retry patterns
- Config via environment variables
- Secure key handling & rotation

## 5a. Project spotlight: 01-chatbot-hello-world

Purpose:

- Minimal baseline to learn API patterns before layering retrieval/agents.

Features (intentionally lean):

- Loads OPENAI_API_KEY from environment
- Simple REPL loop (stdin → model → stdout)
- Basic system prompt (optional)
- Graceful KeyboardInterrupt exit

Suggested enhancements (future):

- Token usage logging
- Streaming partial responses
- Command shortcuts (/reset, /model gpt-4o-mini, /exit)
- Persist conversation transcript to a timestamped file

High-level flow:
user input → build messages → openai chat completion → print assistant → repeat

## 6. Environment variables (common)

Set in a per-project `.env` (never commit real keys):

- OPENAI_API_KEY (required)
- USE_FAISS=1|0
- EMBEDDING_MODEL (default text-embedding-3-small)
- CHAT_MODEL (default gpt-4o-mini)
- CHUNK_TOKENS / CHUNK_OVERLAP
- MAX_COMPLETION_TOKENS / TEMPERATURE / TOP_K

## 7. Development workflow

```bash
# create & activate venv (root or per project)
python -m venv .venv && source .venv/bin/activate
# install project deps
pip install -r projects/02-rag-mini/requirements.txt
# (optional) add formatting/lint tools later: ruff, black, mypy
```

Keep PRs focused (single concern). Add or update README sections when introducing new patterns.

## 8. Security & key hygiene

- Do not commit `.env` (add to .gitignore).
- Rotate exposed keys immediately (delete + recreate in OpenAI dashboard).
- Consider adding a pre-commit hook to block `sk-` patterns.

## 9. Roadmap (high-level)

- [ ] Persist / load cached embeddings
- [ ] Add dry-run (no external API) mode
- [ ] Retrieval evaluation harness (precision @ k, groundedness prompts)
- [ ] Streaming chat responses
- [ ] Agent tool abstraction (search / summarize / plan)
- [ ] Cost logging & token accounting
- [ ] CI smoke tests (offline)
- [ ] Chatbot: add streaming + /commands
- [ ] Chatbot: add transcript save option

## 10. Learning log (fill as milestones)

Dates are optional but recommended — each contributor should add their own entry with the date they made the change (use ISO format YYYY-MM-DD).

- (2025-10-06) Initial RAG prototype — basic pipeline: PDF → token chunking → embeddings → FAISS/NumPy index
- (2025-10-06) Compared overlap strategies — measured retrieval quality vs latency and documented best-practice settings
- (YYYY-MM-DD) Added FAISS vs NumPy latency measurements

## 11. Contributing

1. Branch from main
2. Make focused change
3. Update relevant README section
4. Open PR (describe rationale + test notes)

## 12. License

Add a LICENSE file (e.g., MIT or Apache-2.0) before external contributions.

## 13. References

- OpenAI API docs
- FAISS wiki
- Retrieval best practices (chunk sizing, overlap trade-offs)
