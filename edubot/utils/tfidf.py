"""Motor TF-IDF + similitud coseno (versión 2.x de EduBot).

TF-IDF (Term Frequency - Inverse Document Frequency) da a cada palabra un
peso alto si aparece en la pregunta pero es poco común en el resto de la
base de conocimiento. Así, "kardex" pesa mucho más que "materia".

Para tolerar variaciones del lenguaje se combinan dos vectorizadores:

* Palabras (unigramas y bigramas) sobre el texto normalizado con
  sinónimos, que capturan el significado de cada término.
* N-gramas de caracteres (3 a 5), que reconocen palabras con la misma raíz
  ("inscribo", "inscripción") y toleran errores de ortografía.

Ambos vectores se normalizan y se concatenan con pesos ``sqrt(w)`` y
``sqrt(1 - w)``; el coseno del vector combinado equivale a
``w * coseno_palabras + (1 - w) * coseno_caracteres``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .base import EngineError, Matrix, SimilarityEngine
from .preprocessing import TextNormalizer

DEFAULT_WORD_WEIGHT = 0.6


class TfidfEngine(SimilarityEngine):
    """Motor de similitud basado en TF-IDF (algoritmo clásico de NLP).

    Attributes:
        word_weight: Peso del vectorizador de palabras (el resto es de caracteres).
    """

    name: ClassVar[str] = "tfidf"

    def __init__(
        self,
        word_weight: float = DEFAULT_WORD_WEIGHT,
        normalizer: TextNormalizer | None = None,
    ) -> None:
        """Configura los vectorizadores.

        Args:
            word_weight: Peso en [0, 1] de la similitud por palabras.
            normalizer: Normalizador de texto; si es ``None`` se crea uno.

        Raises:
            ValueError: Si ``word_weight`` está fuera de [0, 1].
        """
        super().__init__()
        if not 0.0 <= word_weight <= 1.0:
            raise ValueError("word_weight debe estar entre 0 y 1.")
        self.word_weight = word_weight
        self._normalizer = normalizer or TextNormalizer()
        self._vectorizer = FeatureUnion(
            [
                (
                    "palabras",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        token_pattern=r"(?u)\b\w+\b",
                        sublinear_tf=True,
                    ),
                ),
                (
                    "caracteres",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        sublinear_tf=True,
                    ),
                ),
            ],
            transformer_weights={
                "palabras": math.sqrt(word_weight),
                "caracteres": math.sqrt(1.0 - word_weight),
            },
        )

    def preprocess(self, texts: Sequence[str]) -> list[str]:
        """Normaliza textos (sinónimos, stopwords, plurales).

        Args:
            texts: Textos originales.

        Returns:
            Textos listos para vectorizar.
        """
        return [self._normalizer.normalize(text) for text in texts]

    def _fit_vectors(self, documents: list[str]) -> Matrix:
        """Aprende el vocabulario y los pesos IDF de la base de conocimiento.

        Args:
            documents: Textos de la base de conocimiento.

        Returns:
            Matriz dispersa TF-IDF (un vector por documento).

        Raises:
            EngineError: Si los documentos no contienen palabras útiles.
        """
        try:
            return self._vectorizer.fit_transform(self.preprocess(documents))
        except ValueError as exc:  # p. ej. "empty vocabulary"
            raise EngineError(f"No se pudo construir el vocabulario TF-IDF: {exc}") from exc

    def transform(self, texts: Sequence[str]) -> Matrix:
        """Vectoriza textos con el vocabulario aprendido.

        Las palabras desconocidas se ignoran; una pregunta sin palabras
        conocidas produce un vector de ceros (similitud 0).

        Args:
            texts: Textos a vectorizar.

        Returns:
            Matriz dispersa TF-IDF.

        Raises:
            EngineError: Si el motor no se ha entrenado con ``fit``.
        """
        if not self.is_fitted:
            raise EngineError("El motor TF-IDF no tiene documentos: llama a fit() primero.")
        return self._vectorizer.transform(self.preprocess(texts))
