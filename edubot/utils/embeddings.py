"""Motor de embeddings semánticos (versión 3.0 de EduBot).

Usa una red neuronal pre-entrenada (sentence-transformers) que convierte
cada frase en un vector de significado. Preguntas con palabras distintas
pero el mismo sentido ("perdí mi credencial" / "¿cómo repongo mi ID?")
quedan como vectores cercanos, sin necesidad de un diccionario de sinónimos.

La librería ``sentence-transformers`` (y PyTorch) se importa solo cuando se
crea el motor, para que TF-IDF funcione aunque no esté instalada. La primera
vez el modelo se descarga desde Hugging Face; el motor avisa al usuario
mediante la función ``notify``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from .base import EngineError, Matrix, ModelLoadError, SimilarityEngine

LOGGER = logging.getLogger(__name__)

MODEL_SIZE_HINT = "~470 MB"


class Encoder(Protocol):
    """Cualquier objeto con el método ``encode`` de sentence-transformers."""

    def encode(self, inputs: Sequence[str], **kwargs: Any) -> Any:
        """Convierte textos en embeddings.

        Args:
            inputs: Textos a codificar.
            **kwargs: Opciones de codificación.

        Returns:
            Matriz de embeddings.
        """


def resolve_repo_id(model_name: str) -> str:
    """Obtiene el identificador completo del modelo en Hugging Face.

    Args:
        model_name: Nombre corto (``paraphrase-...``) o completo (``org/modelo``).

    Returns:
        Identificador ``organización/modelo``.
    """
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


def is_model_cached(model_name: str) -> bool:
    """Indica si el modelo ya está descargado en el equipo.

    Args:
        model_name: Nombre del modelo o ruta a una carpeta local.

    Returns:
        ``True`` si no hace falta descargarlo.
    """
    if Path(model_name).expanduser().is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    cache_dirs = [None, os.environ.get("SENTENCE_TRANSFORMERS_HOME")]
    for cache_dir in cache_dirs:
        try:
            found = try_to_load_from_cache(
                resolve_repo_id(model_name), "modules.json", cache_dir=cache_dir
            )
        except Exception:
            # Cualquier fallo al leer la caché equivale a "no está en caché".
            found = None
        if isinstance(found, str):
            return True
    return False


class EmbeddingsEngine(SimilarityEngine):
    """Motor de similitud basado en embeddings de una red neuronal.

    Attributes:
        model_name: Nombre del modelo de sentence-transformers.
        batch_size: Número de frases por lote al codificar.
    """

    name: ClassVar[str] = "embeddings"

    def __init__(
        self,
        model_name: str,
        *,
        model: Encoder | None = None,
        notify: Callable[[str], None] | None = None,
        batch_size: int = 32,
    ) -> None:
        """Carga el modelo de embeddings.

        Args:
            model_name: Nombre del modelo de sentence-transformers.
            model: Codificador ya construido (útil en pruebas). Si es ``None``
                se carga ``model_name`` con sentence-transformers.
            notify: Función para avisar al usuario (p. ej. de la descarga).
            batch_size: Número de frases por lote al codificar.

        Raises:
            ModelLoadError: Si la librería no está instalada o el modelo no
                se puede cargar ni descargar.
        """
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self._notify = notify or LOGGER.info
        self._model: Encoder = model if model is not None else self._load_model()

    def _load_model(self) -> Encoder:
        """Importa sentence-transformers y carga (o descarga) el modelo.

        Returns:
            El modelo listo para codificar.

        Raises:
            ModelLoadError: Si la librería falta o el modelo no está disponible.
        """
        try:
            # Importación diferida: PyTorch tarda varios segundos en cargar
            # y no es necesario cuando se usa TF-IDF.
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelLoadError(
                "La librería 'sentence-transformers' no está instalada. "
                "Ejecuta: pip install -r requirements.txt"
            ) from exc

        if not is_model_cached(self.model_name):
            self._notify(
                f"Descargando el modelo de embeddings '{self.model_name}' "
                f"({MODEL_SIZE_HINT}). Solo ocurre la primera vez y puede "
                "tardar varios minutos según tu conexión."
            )
        try:
            return SentenceTransformer(self.model_name)
        except Exception as exc:
            # Hugging Face puede fallar por red, proxy, permisos o nombre
            # inválido; todas se reportan igual al usuario.
            raise ModelLoadError(
                f"No se pudo cargar el modelo '{self.model_name}'. "
                "Revisa tu conexión a internet (la primera vez se descarga "
                f"desde huggingface.co). Detalle: {exc}"
            ) from exc

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Codifica textos como embeddings normalizados (norma 1).

        Args:
            texts: Textos a codificar.

        Returns:
            Matriz 2D de forma (n_textos, dimensión).

        Raises:
            EngineError: Si el modelo falla al codificar.
        """
        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EngineError(f"Error al generar embeddings: {exc}") from exc
        return np.atleast_2d(np.asarray(vectors, dtype=np.float32))

    def _fit_vectors(self, documents: list[str]) -> Matrix:
        """Calcula los embeddings de la base de conocimiento.

        Args:
            documents: Textos de la base de conocimiento.

        Returns:
            Matriz de embeddings (un vector por documento).
        """
        return self._encode(documents)

    def transform(self, texts: Sequence[str]) -> Matrix:
        """Calcula los embeddings de nuevas frases.

        Args:
            texts: Textos a codificar.

        Returns:
            Matriz de embeddings.
        """
        return self._encode([text.strip() for text in texts])
