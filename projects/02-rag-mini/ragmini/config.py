import os
from dotenv import load_dotenv

load_dotenv()

# Canonical constant (fixes ImportError in app.py)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Backwards compatibility (old mixed-case name)
OpenAI_API_KEY = OPENAI_API_KEY

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Chunking
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Embeddings
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 64))
EMBED_DIM = int(os.getenv("EMBED_DIM", 1536))  # for text-embedding-3-small default dim is 1536

# Indexing
USE_FAISS = os.getenv("USE_FAISS", "1") == "1"

# LLM
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", 500))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
TOP_K = int(os.getenv("TOP_K", 3))

__all__ = [
    "OPENAI_API_KEY",
    "OpenAI_API_KEY",
    "EMBEDDING_MODEL",
    "CHAT_MODEL",
    "CHUNK_TOKENS",
    "CHUNK_OVERLAP",
    "EMBED_BATCH_SIZE",
    "EMBED_DIM",
    "USE_FAISS",
    "MAX_COMPLETION_TOKENS",
    "TEMPERATURE",
    "TOP_K",
]