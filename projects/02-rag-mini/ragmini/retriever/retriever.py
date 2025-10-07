from typing import List, Tuple, Any
from ..index.vector_store import search

def retrieve_similar_chunks(query: str, index: Any, chunks: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    return search(query, index, chunks, top_k)