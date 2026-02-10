"""
OpenAI embeddings for commentary span matching and main-text alignment.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_BATCH_SIZE


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Return embedding vectors for each text. Batches requests."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY required for embeddings.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    out: List[List[float]] = []
    for i in range(0, len(texts), OPENAI_EMBEDDING_BATCH_SIZE):
        batch = [t or "" for t in texts[i : i + OPENAI_EMBEDDING_BATCH_SIZE]]
        resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
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
