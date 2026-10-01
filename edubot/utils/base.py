"""Tipos base compartidos por todo EduBot.

Contiene:
    * La jerarquía de errores del proyecto.
    * ``Tramite``: una fila de la base de conocimiento (modelo del mundo).
    * ``SimilarityEngine``: la interfaz común de los motores de NLP. Gracias
      a ella TF-IDF, embeddings y coincidencia exacta son intercambiables.
    * ``ExactMatchEngine``: el motor de la versión 1.0 (coincidencia literal).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import basic_normalize

# Matriz de vectores: densa (numpy) o dispersa (scipy, en TF-IDF).
Matrix = Any


class EduBotError(Exception):
    """Error base de EduBot."""


class KnowledgeBaseError(EduBotError):
    """La base de conocimiento no existe o tiene un formato inválido."""


class EngineError(EduBotError):
    """El motor de NLP no se pudo crear o se usó de forma incorrecta."""


class ModelLoadError(EngineError):
    """El modelo de embeddings no está instalado o no se pudo descargar."""


DETAIL_FIELDS = (
    "requisitos",
    "documentos_necesarios",
    "costo",
    "plazo",
    "dependencia_responsable",
    "pasos_procedimiento",
)


@dataclass(frozen=True)
class Tramite:
    """Un trámite escolar de la base de conocimiento.

    Attributes:
        id: Identificador único.
        tramite: Nombre del trámite.
        preguntas: Formas de preguntar por el trámite. La primera es la
            pregunta de ejemplo principal.
        requisitos: Condiciones que debe cumplir el estudiante.
        documentos_necesarios: Documentos que hay que presentar.
        costo: Costo del trámite.
        plazo: Tiempo de respuesta o fechas.
        dependencia_responsable: Oficina que atiende el trámite.
        pasos_procedimiento: Pasos a seguir.
        respuesta_generica: Respuesta completa en lenguaje natural.
    """

    id: int
    tramite: str
    preguntas: tuple[str, ...]
    requisitos: str
    documentos_necesarios: str
    costo: str
    plazo: str
    dependencia_responsable: str
    pasos_procedimiento: str
    respuesta_generica: str

    @property
    def pregunta_ejemplo(self) -> str:
        """Devuelve la pregunta de ejemplo principal.

        Returns:
            La primera forma de preguntar por el trámite.
        """
        return self.preguntas[0]

    def get_field(self, name: str) -> str:
        """Obtiene el valor de un campo de detalle por su nombre de columna.

        Args:
            name: Nombre de la columna (p. ej. ``"costo"``).

        Returns:
            El texto del campo.

        Raises:
            KeyError: Si el campo no es un campo de detalle.
        """
        if name not in DETAIL_FIELDS:
            raise KeyError(f"Campo desconocido: {name}")
        return getattr(self, name)


class SimilarityEngine(ABC):
    """Interfaz común de los motores de NLP.

    Flujo de uso: ``fit(documentos)`` una vez y después
    ``similarities(pregunta)`` por cada pregunta del usuario.

    Las subclases solo implementan cómo convertir texto en vectores
    (``_fit_vectors`` y ``transform``); la similitud coseno es común.
    """

    name: ClassVar[str] = "base"

    def __init__(self) -> None:
        """Inicializa el motor sin documentos indexados."""
        self._doc_vectors: Matrix | None = None

    @property
    def is_fitted(self) -> bool:
        """Indica si el motor ya indexó documentos.

        Returns:
            ``True`` después de llamar a ``fit``.
        """
        return self._doc_vectors is not None

    def fit(self, documents: Sequence[str]) -> SimilarityEngine:
        """Vectoriza e indexa los documentos de la base de conocimiento.

        Args:
            documents: Textos contra los que se compararán las preguntas.

        Returns:
            El mismo motor, para encadenar llamadas.

        Raises:
            EngineError: Si no hay documentos o no se pueden vectorizar.
        """
        documents = list(documents)
        if not documents:
            raise EngineError("No hay documentos para indexar.")
        self._doc_vectors = self._fit_vectors(documents)
        return self

    def similarities(self, query: str) -> np.ndarray:
        """Calcula la similitud coseno entre la pregunta y cada documento.

        Args:
            query: Pregunta del usuario.

        Returns:
            Arreglo 1D con un valor en [0, 1] (aprox.) por documento.

        Raises:
            EngineError: Si el motor no se ha entrenado con ``fit``.
        """
        if self._doc_vectors is None:
            raise EngineError("El motor no tiene documentos: llama a fit() primero.")
        query_vector = self.transform([query])
        return cosine_similarity(query_vector, self._doc_vectors)[0]

    @abstractmethod
    def _fit_vectors(self, documents: list[str]) -> Matrix:
        """Aprende la representación y vectoriza los documentos.

        Args:
            documents: Textos de la base de conocimiento.

        Returns:
            Matriz con un vector por documento.
        """

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> Matrix:
        """Convierte textos en vectores con la representación aprendida.

        Args:
            texts: Textos a vectorizar.

        Returns:
            Matriz con un vector por texto.
        """


class ExactMatchEngine(SimilarityEngine):
    """Motor v1.0: solo reconoce preguntas idénticas a las de ejemplo.

    Ignora mayúsculas, acentos y signos de puntuación, pero cualquier
    palabra distinta hace que no haya coincidencia. Sirve como línea base
    para demostrar por qué se necesita NLP.
    """

    name: ClassVar[str] = "exact"

    def __init__(self) -> None:
        """Inicializa el índice de textos normalizados."""
        super().__init__()
        self._normalized_docs: list[str] = []

    def _fit_vectors(self, documents: list[str]) -> Matrix:
        """Guarda los textos normalizados y usa vectores one-hot.

        Args:
            documents: Textos de la base de conocimiento.

        Returns:
            Matriz identidad (un vector one-hot por documento).
        """
        self._normalized_docs = [basic_normalize(doc) for doc in documents]
        return np.eye(len(documents))

    def transform(self, texts: Sequence[str]) -> Matrix:
        """Representa cada texto como un vector one-hot de coincidencias.

        Args:
            texts: Textos a vectorizar.

        Returns:
            Matriz con 1 en las posiciones de documentos idénticos al texto.
        """
        vectors = np.zeros((len(texts), len(self._normalized_docs)))
        for row, text in enumerate(texts):
            normalized = basic_normalize(text)
            for column, doc in enumerate(self._normalized_docs):
                if normalized and doc == normalized:
                    vectors[row, column] = 1.0
        return vectors

    def similarities(self, query: str) -> np.ndarray:
        """Devuelve 1.0 para documentos idénticos a la pregunta y 0.0 al resto.

        Args:
            query: Pregunta del usuario.

        Returns:
            Arreglo 1D de unos y ceros.

        Raises:
            EngineError: Si el motor no se ha entrenado con ``fit``.
        """
        if self._doc_vectors is None:
            raise EngineError("El motor no tiene documentos: llama a fit() primero.")
        return self.transform([query])[0]
