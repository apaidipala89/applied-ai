## Project: 01-chatbot-hello-world — Quick agent guide

This is a tiny chat demo showing an OpenAI chat completion wired to a minimal UI.

Key files

- `app.py` — minimal Flask-like/console demo that constructs messages and calls OpenAI chat completions.
- `app_gradio.py` — Gradio-powered UI for the same demo.
- `requirements.txt` — libraries needed for the demo.

What to read first

- `app_gradio.py`: shows how the `OpenAI(api_key=...)` client is constructed and how responses are rendered.

Patterns & conventions

- Use `os.getenv("OPENAI_API_KEY")` and rely on a local `.env` (project-level) for development keys. Do not commit real keys.
- Keep UI and API interaction separate: the Gradio app builds messages and delegates to the `client.chat.completions.create(...)` call.

How to run

1. Create and activate a venv inside the project.
2. Install `pip install -r requirements.txt`.
3. Set `OPENAI_API_KEY` locally (or create a `.env`) and run `python app_gradio.py`.

Small edits to add

- If you add new OpenAI calls, wrap them with a retry/backoff helper (see `projects/02-rag-mini/rag-mini/utils/rate_limit.py`).
