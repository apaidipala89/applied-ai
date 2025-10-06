# We will chunk the text into smaller pieces(~500 tokens) 
# with some overlap(~50 tokens) so we keep the context.

from typing import List
from ..config import CHAT_MODEL
import tiktoken

_enc_cache = {}

def get_encoder(model: str = CHAT_MODEL):
    # using cl100k_base for all models as they are all based on gpt-4, if gpt-4o-mini not found
    if model not in _enc_cache:
        try:
            _enc_cache[model] = tiktoken.encoding_for_model(model)
        except Exception:
            _enc_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _enc_cache[model]

def chunk_by_tokens(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    # Validate inputs to prevent infinite loops
    if overlap >= max_tokens:
        raise ValueError(f"Overlap ({overlap}) must be less than max_tokens ({max_tokens})")
    
    enc = get_encoder()
    tokens = enc.encode(text)
    
    # Early return for small texts
    if len(tokens) <= max_tokens:
        return [text] if text.strip() else []
    
    chunks = []
    step = max_tokens - overlap
    max_chunks = 10000  # Safety limit to prevent runaway processing
    
    for i in range(0, len(tokens), step):
        if len(chunks) >= max_chunks:
            break  # Safety break to prevent infinite processing
            
        window = tokens[i:i + max_tokens]
        decode = enc.decode(window).strip()
        if decode and (not chunks or decode != chunks[-1]):
            chunks.append(decode)
            
        # Break if we've processed all tokens
        if i + max_tokens >= len(tokens):
            break
    
    return chunks