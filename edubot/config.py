"""Configuración central de EduBot.

Todos los parámetros ajustables del agente viven aquí: el motor de NLP,
los umbrales de similitud, la ruta de la base de conocimiento y el modelo
de embeddings. Ningún otro módulo define valores "mágicos".

Los valores se pueden sobrescribir sin tocar el código mediante variables
de entorno (útil para pruebas o despliegues):

    EDUBOT_ENGINE=tfidf streamlit run app.py

Variables reconocidas:
    EDUBOT_ENGINE: ``embeddings``, ``tfidf`` o ``exact``.
    EDUBOT_THRESHOLD: umbral para TF-IDF y coincidencia exacta.
    EDUBOT_EMBEDDINGS_THRESHOLD: umbral para el motor de embeddings.
    EDUBOT_DATA_PATH: ruta al CSV de la base de conocimiento.
    EDUBOT_MODEL: nombre del modelo de sentence-transformers.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# --- Motor de NLP -----------------------------------------------------------
# "embeddings" (v3.0, red neuronal), "tfidf" (v2.x, algoritmo clásico)
# o "exact" (v1.0, coincidencia literal).
ENGINE = "embeddings"

# Si el motor principal no se puede cargar (p. ej. no hay internet para
# descargar el modelo), EduBot usa este motor y avisa al usuario.
FALLBACK_ENGINE = "tfidf"

ENGINE_CHOICES = ("embeddings", "tfidf", "exact")

# --- Heurística: umbrales de similitud coseno ------------------------------
# Si la mejor coincidencia no alcanza el umbral, EduBot responde
# "No entiendo" en lugar de arriesgarse a dar información incorrecta.
SIMILARITY_THRESHOLD = 0.3

# Los embeddings producen similitudes más altas incluso entre frases sin
# relación (comparten estructura de pregunta), por eso su umbral es mayor.
EMBEDDINGS_THRESHOLD = 0.5

# --- Modelo de red neuronal -------------------------------------------------
# Modelo multilingüe (incluye español), ~470 MB, se descarga una sola vez.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Base de conocimiento ---------------------------------------------------
DATA_PATH = PROJECT_DIR / "datos.csv"
EVALUATION_PATH = PROJECT_DIR / "evaluacion.csv"
# "utf-8-sig" acepta archivos guardados desde Excel (con BOM).
CSV_ENCODING = "utf-8-sig"

# --- Interacción -------------------------------------------------------------
# Número de trámites alternativos que se sugieren cuando no hay respuesta.
TOP_K_SUGGESTIONS = 3

# Temas que se sugieren cuando EduBot no entiende la pregunta.
TOPIC_HINTS = (
    "inscripción",
    "becas",
    "horarios",
    "pagos",
    "biblioteca",
    "kardex",
    "certificados",
    "credencial",
    "servicio social",
    "titulación",
    "transporte",
    "servicios médicos",
)


# --- Información del equipo -------------------------------------------------
UNIVERSITY = "Universidad Autónoma de Nuevo León"
FACULTY = "Facultad de Ingeniería Mecánica y Eléctrica"
COURSE = "Laboratorio Temas Selectos de Sistemas Inteligentes"
SEMESTER = "Agosto - Diciembre 2026"
PROFESSOR = "Raquel Martinez Martinez"

# (nombre, matrícula, brigada)
TEAM_MEMBERS = (
    ("Jose David Meza Flores", "2069864", "305"),
    ("Jesús Arturo Corpus Zavala", "2132346", "No está inscrito"),
    ("Brian Horacio Reyna Diaz de León", "1995434", "305"),
    ("Victor Hugo Bernal Ríos", "2056233", "305"),
    ("Samuel Alejandro Manrique Mujica", "2132342", "305"),
)


@dataclass(frozen=True)
class Settings:
    """Parámetros de ejecución de EduBot.

    Es inmutable para que una misma configuración pueda compartirse entre
    la consola, la interfaz web y las pruebas sin efectos secundarios.

    Attributes:
        engine: Motor de NLP principal (``embeddings``, ``tfidf`` o ``exact``).
        fallback_engine: Motor de respaldo si el principal falla al cargar.
            ``None`` desactiva el respaldo.
        threshold: Umbral de similitud para TF-IDF y coincidencia exacta.
        embeddings_threshold: Umbral de similitud para embeddings.
        embedding_model: Nombre del modelo de sentence-transformers.
        data_path: Ruta al CSV de la base de conocimiento.
        evaluation_path: Ruta al CSV con preguntas de evaluación.
        csv_encoding: Codificación de los archivos CSV.
        top_k: Número de sugerencias alternativas a mostrar.
        topic_hints: Temas que se ofrecen cuando no hay respuesta.
    """

    engine: str = ENGINE
    fallback_engine: str | None = FALLBACK_ENGINE
    threshold: float = SIMILARITY_THRESHOLD
    embeddings_threshold: float = EMBEDDINGS_THRESHOLD
    embedding_model: str = EMBEDDING_MODEL
    data_path: Path = DATA_PATH
    evaluation_path: Path = EVALUATION_PATH
    csv_encoding: str = CSV_ENCODING
    top_k: int = TOP_K_SUGGESTIONS
    topic_hints: tuple[str, ...] = TOPIC_HINTS

    def __post_init__(self) -> None:
        """Valida los valores para detectar errores de configuración pronto.

        Raises:
            ValueError: Si el motor no existe o un umbral está fuera de [0, 1].
        """
        for name in (self.engine, self.fallback_engine):
            if name is not None and name not in ENGINE_CHOICES:
                raise ValueError(
                    f"Motor desconocido: '{name}'. Opciones válidas: {', '.join(ENGINE_CHOICES)}."
                )
        for label, value in (
            ("threshold", self.threshold),
            ("embeddings_threshold", self.embeddings_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} debe estar entre 0 y 1; se recibió {value}.")
        if self.top_k < 0:
            raise ValueError("top_k no puede ser negativo.")

    def threshold_for(self, engine: str) -> float:
        """Devuelve el umbral de similitud que corresponde a un motor.

        Args:
            engine: Nombre del motor activo.

        Returns:
            El umbral de similitud coseno a aplicar.
        """
        if engine == "embeddings":
            return self.embeddings_threshold
        return self.threshold

    def with_engine(self, engine: str) -> Settings:
        """Crea una copia de la configuración con otro motor principal.

        Args:
            engine: Nombre del nuevo motor.

        Returns:
            Una nueva instancia de ``Settings``.
        """
        return replace(self, engine=engine)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Construye la configuración aplicando variables de entorno.

        Args:
            environ: Diccionario de variables; por defecto ``os.environ``.

        Returns:
            Una instancia de ``Settings`` con los valores sobrescritos.

        Raises:
            ValueError: Si alguna variable tiene un valor inválido.
        """
        env = os.environ if environ is None else environ
        overrides: dict[str, object] = {}
        if env.get("EDUBOT_ENGINE"):
            overrides["engine"] = env["EDUBOT_ENGINE"].strip().lower()
        if env.get("EDUBOT_THRESHOLD"):
            overrides["threshold"] = _parse_float("EDUBOT_THRESHOLD", env)
        if env.get("EDUBOT_EMBEDDINGS_THRESHOLD"):
            overrides["embeddings_threshold"] = _parse_float("EDUBOT_EMBEDDINGS_THRESHOLD", env)
        if env.get("EDUBOT_DATA_PATH"):
            overrides["data_path"] = Path(env["EDUBOT_DATA_PATH"]).expanduser()
        if env.get("EDUBOT_MODEL"):
            overrides["embedding_model"] = env["EDUBOT_MODEL"].strip()
        return cls(**overrides)


def _parse_float(key: str, env: Mapping[str, str]) -> float:
    """Convierte una variable de entorno a ``float`` con un error claro.

    Args:
        key: Nombre de la variable.
        env: Diccionario de variables de entorno.

    Returns:
        El valor numérico.

    Raises:
        ValueError: Si el valor no es numérico.
    """
    try:
        return float(env[key])
    except ValueError as exc:
        raise ValueError(f"{key} debe ser un número; se recibió '{env[key]}'.") from exc
