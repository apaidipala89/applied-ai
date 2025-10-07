"""
LLM answer generation utilities.

Generates an answer given a user question and retrieved context chunks.
"""

from __future__ import annotations
from typing import List, Optional

from openai import OpenAI
from openai import APIError, RateLimitError

from ..config import (
    CHAT_MODEL,
    MAX_COMPLETION_TOKENS,
    TEMPERATURE,
)
from ..utils.rate_limit import with_backoff

client = OpenAI()

def _build_system_prompt(context_chunks: List[str], base_preamble: Optional[str]) -> str:
    """
    Combine a base preamble with retrieved context chunks into a single system prompt.
    """
    preamble = (
        base_preamble
        or "You are a concise, helpful assistant. Use only the provided context to answer. If unsure, say you don't know."
    )
    if context_chunks:
        joined = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context_chunks))
        preamble += "\n\nRelevant context:\n" + joined
    return preamble

@with_backoff()  # retries on RateLimitError / APIError per project convention
def _chat_completion(messages, model: str, temperature: float, max_tokens: int):
    """
    Thin wrapper around OpenAI chat completion (kept small for retry decorator).
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def generate_answer(
    question: str,
    retrieved_chunks: List[str],
    system_preamble: Optional[str] = None,
    model: str = CHAT_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_COMPLETION_TOKENS,
) -> str:
    """
    Generate an answer using the chat model and retrieved context.

    Parameters
    ----------
    question : str
        The user's question.
    retrieved_chunks : List[str]
        Context chunks (already ranked) to ground the answer.
    system_preamble : Optional[str]
        Optional custom system prompt preamble.
    model : str
        Chat model name (defaults to config.CHAT_MODEL).
    temperature : float
        Sampling temperature.
    max_tokens : int
        Max tokens for completion.

    Returns
    -------
    str
        Assistant answer (may be empty string on failure).
    """
    if not question.strip():
        return ""

    system_prompt = _build_system_prompt(retrieved_chunks, system_preamble)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]

    try:
        resp = _chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except (RateLimitError, APIError) as e:
        return f"LLM error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

    try:
        content = resp.choices[0].message.content or ""
        return content.strip()
    except Exception:
        return ""

__all__ = ["generate_answer"]