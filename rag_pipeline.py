"""
rag_pipeline.py - The RAG brain: retrieve context, then ask Groq.

Steps for every question:
1. Search FAISS for the top 3 chunks.
2. Put those chunks + the question in a prompt.
3. Send it to the Groq LLM, which must answer ONLY from that context.
4. Return the answer and the source pages.
"""

from functools import lru_cache
from typing import Dict, List

from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

import config
from vector_database import VectorDatabase

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions about an uploaded PDF.

Rules:
1. Answer ONLY using the context provided by the user. Do not use outside knowledge.
2. If the answer is not clearly in the context, reply with exactly this sentence and nothing else:
"{config.FALLBACK_ANSWER}"
3. Be clear and concise. Use simple language.
4. Never invent facts, numbers, names or page numbers."""


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    """Create the Groq client once and reuse it."""
    return Groq(api_key=config.GROQ_API_KEY)


def _build_context(chunks: List[Dict]) -> str:
    """Join the retrieved chunks into one text block, labelled with file and page."""
    blocks = [f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(blocks)


def _ask_groq(question: str, context: str) -> str:
    """Send the context + question to Groq and return the answer text."""
    user_message = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = _get_client().chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


def answer_question(question: str, db: VectorDatabase) -> Dict:
    """
    Full RAG flow. Returns {"answer": str, "sources": [chunk dicts]}.
    Never raises for API problems - returns a friendly message instead.
    """
    if not config.is_api_key_configured():
        return {"answer": "⚠️ Groq API key is missing. Add it to the .env file and restart the app.", "sources": []}

    chunks = db.search(question, top_k=config.TOP_K)
    if not chunks:
        return {"answer": config.FALLBACK_ANSWER, "sources": []}

    try:
        answer = _ask_groq(question, _build_context(chunks))
    except AuthenticationError:
        return {"answer": "⚠️ Invalid Groq API key. Check GROQ_API_KEY in your .env file.", "sources": []}
    except RateLimitError:
        return {"answer": "⚠️ Groq rate limit reached. Please wait a few seconds and try again.", "sources": []}
    except APIConnectionError:
        return {"answer": "⚠️ Could not reach Groq. Check your internet connection.", "sources": []}
    except APIStatusError as exc:
        return {"answer": f"⚠️ Groq API error: {exc.message}. Check GROQ_MODEL in your .env file.", "sources": []}
    except Exception as exc:  # last safety net
        return {"answer": f"⚠️ Unexpected error: {exc}", "sources": []}

    if not answer or config.FALLBACK_ANSWER.lower() in answer.lower():
        return {"answer": config.FALLBACK_ANSWER, "sources": []}  # no misleading sources

    return {"answer": answer, "sources": chunks}
