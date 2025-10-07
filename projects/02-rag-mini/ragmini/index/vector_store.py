import numpy as np
from typing import List, Tuple, Union
from ..config import USE_FAISS, OPENAI_API_KEY, EMBEDDING_MODEL, EMBED_BATCH_SIZE
from ..utils.rate_limit import with_backoff

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    if USE_FAISS:
        raise ImportError("FAISS is not installed. Install faiss-cpu or set USE_FAISS=0.")

IndexLike = Union['faiss.IndexFlatIP', np.ndarray]

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set; cannot embed documents.")

from openai import OpenAI, RateLimitError
_client = OpenAI(api_key=OPENAI_API_KEY)

class EmbeddingQuotaError(Exception):
    """Raised when OpenAI reports insufficient quota."""

class EmbeddingError(Exception):
    """Generic embedding failure."""

@with_backoff()
def _embed_batch(text_batch: List[str]):
    try:
        resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=text_batch)
    except RateLimitError as e:
        msg = str(e)
        if "insufficient_quota" in msg or "quota" in msg.lower():
            raise EmbeddingQuotaError("OpenAI quota exceeded. Check billing or usage limits.") from e
        raise EmbeddingError(f"Rate limit error: {msg}") from e
    except Exception as e:
        raise EmbeddingError(f"Embedding request failed: {e}") from e
    return [np.array(d.embedding, dtype="float32") for d in resp.data]

def _embed_texts(texts: List[str]) -> List[np.ndarray]:
    embeddings: List[np.ndarray] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        embeddings.extend(_embed_batch(batch))
    return embeddings

def _to_normalized_matrix(vectors: List[np.ndarray]) -> np.ndarray:
    mat = np.vstack(vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms

def build_index(chunks: List[str]) -> Tuple[IndexLike, List[str]]:
    if not chunks:
        return (faiss.IndexFlatIP(1) if USE_FAISS and FAISS_AVAILABLE else np.empty((0, 0))), []
    try:
        embeddings = _embed_texts(chunks)
    except (EmbeddingQuotaError, EmbeddingError):
        # Re-raise so caller can present message
        raise
    if not embeddings:
        raise EmbeddingError("No embeddings returned.")
    dim = embeddings[0].shape[0]
    if USE_FAISS and FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(dim)
        normed = _to_normalized_matrix(embeddings)
        index.add(normed.astype("float32"))
        return index, chunks
    matrix = _to_normalized_matrix(embeddings)
    return matrix, chunks

def search(query: str, index: IndexLike, chunks: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    if not query.strip() or not chunks:
        return []
    q_vec = _embed_texts([query])[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    q_vec = q_vec.astype("float32").reshape(1, -1)
    if USE_FAISS and FAISS_AVAILABLE and isinstance(index, faiss.IndexFlatIP):
        D, I = index.search(q_vec, top_k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((chunks[int(idx)], float(score)))
        return results
    if isinstance(index, np.ndarray) and index.size:
        sims = (index @ q_vec.T).ravel()
        top_idx = np.argsort(-sims)[:top_k]
        return [(chunks[i], float(sims[i])) for i in top_idx]
    return []