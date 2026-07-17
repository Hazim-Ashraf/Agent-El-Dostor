"""Local, free multilingual embeddings (sentence-transformers).

Runs CPU-only inside Docker on macOS (no MPS passthrough in containers).
Default model is `intfloat/multilingual-e5-base`; e5 models expect the
"query:" / "passage:" instruction prefixes, which we add automatically.
"""
from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        # Imported lazily so tooling/imports don't pull torch until needed.
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model '%s' (CPU)...", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model, device="cpu")
    return _model


def _is_e5() -> bool:
    return "e5" in settings.embedding_model.lower()


def dimension() -> int:
    return int(_get_model().get_sentence_embedding_dimension())


def embed_passages(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    payload = [f"passage: {t}" for t in texts] if _is_e5() else texts
    vecs = model.encode(payload, normalize_embeddings=True, batch_size=8)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    model = _get_model()
    payload = f"query: {text}" if _is_e5() else text
    vec = model.encode([payload], normalize_embeddings=True)[0]
    return vec.tolist()
