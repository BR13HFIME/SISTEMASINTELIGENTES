"""Pruebas de coincidencia: preprocesamiento, motores, umbral y campos."""

from __future__ import annotations

import sys
import zlib
from dataclasses import replace

import numpy as np
import pytest

from chatbot import EduBot, compare_engines, evaluate, load_evaluation_cases
from config import EMBEDDING_MODEL, Settings
from utils import (
    EmbeddingsEngine,
    EngineError,
    ExactMatchEngine,
    ModelLoadError,
    TfidfEngine,
    create_engine,
    is_model_cached,
)
from utils.preprocessing import TextNormalizer, basic_normalize, light_stem, strip_accents
from utils.template import detect_requested_fields, is_follow_up, split_steps

# Preguntas que NO se usaron para ajustar sinónimos ni umbral (conjunto de
# validación). Sirven para medir la generalización real del motor TF-IDF.
HELD_OUT = [
    ("dónde me reinscribo", 1),
    ("qué días puedo inscribir materias", 1),
    ("forma de pago de la cuota de rectoría", 2),
    ("en qué aula es mi clase", 3),
    ("dónde consulto mi boleta", 4),
    ("beca para estudiantes con promedio alto", 5),
    ("ayuda económica para alumnos de escasos recursos", 6),
    ("ocupo mi kardex", 7),
    ("comprobante de estudios para mi trabajo", 8),
    ("certificado total de estudios", 9),
    ("credencial nueva para alumnos de primer ingreso", 10),
    ("extravié mi credencial", 11),
    ("préstamo de libros a domicilio", 12),
    ("alquilar libros en la biblioteca central", 13),
    ("cuándo puedo iniciar mi servicio social", 14),
    ("requisitos para prácticas profesionales", 15),
    ("tengo que presentar el egel para titularme", 16),
    ("darme de baja de cálculo", 18),
    ("examen extra", 19),
    ("tengo una materia en tercera oportunidad", 20),
    ("cambio de carrera dentro de la facultad", 21),
    ("becas para intercambio en el extranjero", 22),
    ("seguro médico gratis para estudiantes", 23),
    ("dónde está la enfermería", 24),
    ("a qué hora sale el tigrebus de fime", 25),
    ("descuento en el metro para estudiantes", 26),
    ("no me deja entrar a teams", 27),
    ("recuperar mi clave de siase", 28),
    ("venden comida en la facultad", 29),
    ("materias en verano", 30),
    ("dónde salen los avisos importantes", 31),
]

NONSENSE = [
    "asdfghjkl",
    "qwertyuiop zxcvbnm",
    "¿Quién ganó el mundial de fútbol?",
    "receta de pastel de chocolate",
    "¿Cuál es la capital de Francia?",
    "dime un chiste",
    "precio del dólar hoy",
    "¿Quién es el presidente de México?",
]


@pytest.fixture(scope="module")
def tfidf_bot() -> EduBot:
    """Bot TF-IDF sobre la base de conocimiento real."""
    return EduBot(Settings(engine="tfidf", fallback_engine=None))


@pytest.fixture(scope="module")
def exact_bot() -> EduBot:
    """Bot v1.0 de coincidencia exacta."""
    return EduBot(Settings(engine="exact", fallback_engine=None))


def best_id(bot: EduBot, question: str) -> int | None:
    """Id del trámite elegido, o ``None`` si la pregunta fue rechazada."""
    match = bot.find_best_match(question)
    return match.tramite.id if match.accepted else None


# --- Preprocesamiento ---------------------------------------------------------


def test_basic_normalize_removes_case_accents_and_punctuation():
    assert strip_accents("Inscripción, año, pingüino") == "Inscripcion, ano, pinguino"
    assert basic_normalize("¿Cómo   me INSCRIBO?") == "como me inscribo"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("¿Cómo me inscribo a materias?", ["inscripcion", "materia"]),
        ("Quiero dar de alta mis clases", ["inscripcion", "materia"]),
        ("perdí mi credencial", ["reposicion", "credencial"]),
        ("historial académico", ["kardex"]),
        ("¿Cuánto cuesta el servicio social?", ["serviciosocial"]),
    ],
)
def test_normalizer_maps_synonyms_and_drops_stopwords(text, expected):
    assert TextNormalizer().tokenize(text) == expected


def test_light_stem_removes_plurals():
    assert light_stem("calificaciones") == "calificacion"
    assert light_stem("libros") == "libro"
    assert light_stem("mes") == "mes"


# --- Detección de campos (preguntas multi-campo) ----------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "¿Qué necesito y cuánto tarda el kardex?",
            {"requisitos", "documentos_necesarios", "plazo"},
        ),
        ("¿Cuánto cuesta la credencial?", {"costo"}),
        ("¿Cuáles son los pasos para la baja?", {"pasos_procedimiento"}),
        (
            "¿Qué documentos llevo y a qué oficina voy?",
            {"documentos_necesarios", "dependencia_responsable"},
        ),
        ("Háblame de la beca por promedio", set()),
    ],
)
def test_detect_requested_fields(question, expected):
    assert set(detect_requested_fields(question).fields) == expected


def test_ambiguous_words_do_not_hide_generic_answer():
    request = detect_requested_fields("¿Dónde veo mis calificaciones?")
    assert request.fields == ("dependencia_responsable",)
    assert request.include_generic
    assert not detect_requested_fields("¿Cuánto cuesta?").include_generic


def test_follow_up_detection():
    normalizer = TextNormalizer()
    assert is_follow_up("¿y cuánto cuesta?", normalizer)
    assert is_follow_up("¿Dónde se hace?", normalizer)
    assert not is_follow_up("¿Cuánto cuesta la credencial?", normalizer)
    assert not is_follow_up("hola", normalizer)


def test_split_steps():
    assert split_steps("1. Paga. 2. Entra a SIASE. 10. Listo.") == [
        "Paga.",
        "Entra a SIASE.",
        "Listo.",
    ]


# --- TF-IDF: sinónimos, reordenamiento y umbral ------------------------------


@pytest.mark.parametrize(
    ("question", "expected_id"),
    [
        ("¿Cómo me inscribo a materias?", 1),  # pregunta de ejemplo
        ("materias inscribo como me", 1),  # palabras reordenadas
        ("quiero registrar mis materias", 1),  # sinónimo
        ("se me extravió la credencial", 11),  # sinónimo
        ("me quiero salir de una materia", 18),  # paráfrasis
        ("estoy enfermo, ¿hay doctor?", 24),  # paráfrasis
        ("QUÉ  NECESITO PARA EL SERVICIO SOCIAL", 14),  # mayúsculas y espacios
    ],
)
def test_tfidf_matches_variations(tfidf_bot, question, expected_id):
    assert best_id(tfidf_bot, question) == expected_id


def test_tfidf_generalizes_to_held_out_questions(tfidf_bot):
    hits = sum(best_id(tfidf_bot, question) == expected for question, expected in HELD_OUT)
    assert hits / len(HELD_OUT) >= 0.9


@pytest.mark.parametrize("question", NONSENSE)
def test_threshold_rejects_unrelated_questions(tfidf_bot, question):
    match = tfidf_bot.find_best_match(question)
    assert not match.accepted
    assert match.score < tfidf_bot.threshold


def test_threshold_controls_acceptance():
    question = "quiero registrar mis materias para el siguiente semestre"
    strict = EduBot(Settings(engine="tfidf", threshold=0.99))
    lenient = EduBot(Settings(engine="tfidf", threshold=0.0))
    assert not strict.find_best_match(question).accepted
    assert lenient.find_best_match(question).accepted
    # Con umbral 0 tampoco se responde sin evidencia (similitud 0).
    assert not lenient.find_best_match("zzzz").accepted


def test_ranking_is_sorted(tfidf_bot):
    match = tfidf_bot.find_best_match("¿Cuánto cuesta el duplicado de credencial?")
    scores = [score for _, score in match.ranking]
    assert scores == sorted(scores, reverse=True)
    assert match.ranking[0][0] == match.tramite


def test_vectorize_question_returns_one_row(tfidf_bot):
    assert tfidf_bot.vectorize_question("¿Cómo saco mi kardex?").shape[0] == 1


# --- v1.0 coincidencia exacta y comparación ----------------------------------


def test_exact_engine_only_matches_identical_questions(exact_bot):
    assert best_id(exact_bot, "como me inscribo a materias") == 1
    assert best_id(exact_bot, "¿Cómo me inscribo en materias?") is None


def test_tfidf_beats_exact_match_on_evaluation_set(tfidf_bot, exact_bot):
    cases = load_evaluation_cases(Settings().evaluation_path)
    tfidf_accuracy = evaluate(tfidf_bot, cases).accuracy
    exact_accuracy = evaluate(exact_bot, cases).accuracy
    assert tfidf_accuracy >= 0.9
    assert tfidf_accuracy > exact_accuracy


def test_compare_engines_skips_unavailable_engines(monkeypatch):
    def fail(self):
        raise ModelLoadError("sin conexión (simulado)")

    monkeypatch.setattr(EmbeddingsEngine, "_load_model", fail)
    reports = compare_engines(Settings())
    assert [report.engine for report in reports] == ["tfidf", "exact"]


# --- Interfaz común de motores y embeddings ----------------------------------


class FakeEncoder:
    """Codificador determinista que imita a sentence-transformers.

    Proyecta las palabras normalizadas en un vector de 64 dimensiones, lo
    que basta para probar la integración sin descargar la red neuronal.
    """

    def encode(self, inputs, normalize_embeddings=False, **kwargs):
        """Devuelve un vector por texto, igual que ``SentenceTransformer.encode``."""
        vectors = np.zeros((len(inputs), 64), dtype=np.float32)
        for row, text in enumerate(inputs):
            for token in basic_normalize(text).split():
                vectors[row, zlib.crc32(token.encode()) % 64] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1, norms)
        return vectors


def test_embeddings_engine_is_interchangeable():
    engine = EmbeddingsEngine("modelo-falso", model=FakeEncoder())
    bot = EduBot(Settings(engine="embeddings"), engine=engine)
    assert bot.engine_name == "embeddings"
    assert bot.threshold == Settings().embeddings_threshold
    response = bot.respond("¿Cómo me inscribo a materias?")
    assert response.answered and response.tramite.id == 1
    assert bot.vectorize_question("hola").shape == (1, 64)


def test_embeddings_reports_missing_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ModelLoadError, match="no está instalada"):
        EmbeddingsEngine(EMBEDDING_MODEL)


def test_engine_must_be_fitted_before_use():
    for engine in (TfidfEngine(), ExactMatchEngine()):
        with pytest.raises(EngineError):
            engine.similarities("hola")


def test_factory_rejects_unknown_engine():
    with pytest.raises(EngineError, match="desconocido"):
        create_engine("gpt", embedding_model=EMBEDDING_MODEL)
    assert isinstance(create_engine("tfidf", embedding_model=EMBEDDING_MODEL), TfidfEngine)


@pytest.mark.skipif(
    not is_model_cached(EMBEDDING_MODEL),
    reason="El modelo de embeddings no está descargado (requiere internet).",
)
def test_real_embeddings_model_understands_paraphrases():
    bot = EduBot(replace(Settings(), engine="embeddings", fallback_engine=None))
    assert best_id(bot, "se me perdió la credencial de la uanl") == 11
    assert best_id(bot, "receta de pastel de chocolate") is None
