"""Componentes de NLP de EduBot.

El resto del proyecto solo necesita ``create_engine``: recibe el nombre del
motor configurado en ``config.py`` y devuelve un objeto con la interfaz
``SimilarityEngine``. Por eso cambiar de TF-IDF a embeddings no requiere
modificar ``chatbot.py`` ni ``app.py``.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import (
    DETAIL_FIELDS,
    EduBotError,
    EngineError,
    ExactMatchEngine,
    KnowledgeBaseError,
    ModelLoadError,
    SimilarityEngine,
    Tramite,
)
from .embeddings import EmbeddingsEngine, is_model_cached
from .tfidf import TfidfEngine

ENGINE_CLASSES: dict[str, type[SimilarityEngine]] = {
    "embeddings": EmbeddingsEngine,
    "tfidf": TfidfEngine,
    "exact": ExactMatchEngine,
}

__all__ = [
    "DETAIL_FIELDS",
    "ENGINE_CLASSES",
    "EduBotError",
    "EmbeddingsEngine",
    "EngineError",
    "ExactMatchEngine",
    "KnowledgeBaseError",
    "ModelLoadError",
    "SimilarityEngine",
    "TfidfEngine",
    "Tramite",
    "create_engine",
    "is_model_cached",
]


def create_engine(
    name: str,
    *,
    embedding_model: str,
    notify: Callable[[str], None] | None = None,
) -> SimilarityEngine:
    """Crea el motor de NLP indicado (patrón fábrica).

    Args:
        name: ``"embeddings"``, ``"tfidf"`` o ``"exact"``.
        embedding_model: Modelo de sentence-transformers (solo embeddings).
        notify: Función para avisar al usuario (p. ej. descarga del modelo).

    Returns:
        Un motor sin entrenar; hay que llamar a ``fit`` con los documentos.

    Raises:
        EngineError: Si el motor no existe.
        ModelLoadError: Si el modelo de embeddings no se puede cargar.
    """
    key = name.strip().lower()
    if key == "embeddings":
        return EmbeddingsEngine(embedding_model, notify=notify)
    if key == "tfidf":
        return TfidfEngine()
    if key == "exact":
        return ExactMatchEngine()
    raise EngineError(f"Motor desconocido: '{name}'. Opciones: {', '.join(ENGINE_CLASSES)}.")
