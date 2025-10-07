import os
import numpy as np
from typing import List
from openai import OpenAI

# use a relative import so the local package resolver finds the module
from ..utils.rate_limit import with_backoff
from ..text.chunking import chunk_by_tokens
from ..config import CHAT_MODEL, EMBEDDING_MODEL, OpenAI_API_KEY

_client = OpenAI(api_key=OpenAI_API_KEY)

@with_backoff
def _embed_batch(texts: List[str], model: str = None) -> List[np.ndarray]:
    model = model or EMBEDDING_MODEL
    # chunk_by_tokens handles token-aware splitting using project conventions
    text_chunks = [chunk for text in texts for chunk in chunk_by_tokens(text)]
    embeddings: List[np.ndarray] = []
    if not text_chunks:
        return embeddings
    response = _client.embeddings.create(
            model=model,
            input=text_chunks
        )
    embeddings = [np.array(item.embedding, dtype="float32") for item in response.data]
    return embeddings