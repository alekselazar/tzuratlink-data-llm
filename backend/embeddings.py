from __future__ import annotations

import math
from typing import List

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_BATCH_SIZE


def _safe_to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # Prevent lists/dicts from breaking the embeddings schema
    try:
        return str(x)
    except Exception:
        return ""


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Return embedding vectors for each text. Batches requests."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY required for embeddings.")
    if not isinstance(OPENAI_EMBEDDING_MODEL, str) or not OPENAI_EMBEDDING_MODEL.strip():
        raise RuntimeError(f"OPENAI_EMBEDDING_MODEL is empty/invalid: {OPENAI_EMBEDDING_MODEL!r}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    out: List[List[float]] = []

    # Keep stable alignment with inputs
    # We'll replace invalid/empty items with a small placeholder so counts match.
    PLACEHOLDER = " "  # safer than "" for many embedding models

    # Very rough char cap to avoid token-limit explosions (tune as needed)
    # Hebrew tends to be tokenized reasonably; 12k chars is a safe default for most embedding models.
    MAX_CHARS = 12000

    for i in range(0, len(texts), OPENAI_EMBEDDING_BATCH_SIZE):
        raw_batch = texts[i : i + OPENAI_EMBEDDING_BATCH_SIZE]

        batch: List[str] = []
        for t in raw_batch:
            s = _safe_to_text(t).strip()
            if not s:
                s = PLACEHOLDER
            if len(s) > MAX_CHARS:
                s = s[:MAX_CHARS]
            batch.append(s)

        try:
            resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
        except Exception as e:
            # Add a high-signal error message for debugging the exact offender
            lens = [(j, len(batch[j]), type(raw_batch[j]).__name__) for j in range(len(batch))]
            raise RuntimeError(
                f"Embeddings request failed: {e}. "
                f"Model={OPENAI_EMBEDDING_MODEL!r}. "
                f"Batch diagnostics (idx,len,type)={lens[:10]}{'...' if len(lens)>10 else ''}"
            ) from e

        # The API returns embeddings aligned to inputs
        for d in resp.data:
            out.append(d.embedding)

    return out


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)
