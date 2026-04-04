"""Embedding module — wraps sentence-transformers for local embedding generation."""

import struct
import logging

from cortex.config import EMBEDDING_MODEL

log = logging.getLogger(__name__)

_model = None


def get_model():
    """Load and return the sentence-transformers model (lazy singleton)."""
    global _model
    if _model is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL}' (first time may download)...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _to_bytes(embedding) -> bytes:
    """Pack a numpy float array into bytes for sqlite-vec."""
    floats = embedding.tolist()
    return struct.pack(f"{len(floats)}f", *floats)


def embed(text: str) -> bytes:
    """Embed a single text string. Returns packed float32 bytes, L2-normalized."""
    if not text.strip():
        model = get_model()
        dim = model.get_sentence_embedding_dimension()
        return struct.pack(f"{dim}f", *([0.0] * dim))
    if len(text) > 5000:
        log.warning("Text length %d exceeds 5000 chars; truncation handled by model", len(text))
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return _to_bytes(vec)


def embed_batch(texts: list[str]) -> list[bytes]:
    """Embed multiple texts at once. Returns list of packed float32 bytes."""
    if not texts:
        return []
    for t in texts:
        if len(t) > 5000:
            log.warning("Batch text length %d exceeds 5000 chars", len(t))
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [_to_bytes(v) for v in vecs]


def embed_query(text: str) -> bytes:
    """Embed a query string. Alias for embed() for clarity at call sites."""
    return embed(text)
