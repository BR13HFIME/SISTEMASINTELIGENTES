"""Pruebas del agente: base de conocimiento, respuestas, contexto y errores."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

import chatbot
from chatbot import REQUIRED_COLUMNS, ConversationContext, EduBot, load_knowledge_base
from config import DATA_PATH, Settings
from utils import DETAIL_FIELDS, KnowledgeBaseError, ModelLoadError
from utils import embeddings as embeddings_module
from utils.template import NOT_UNDERSTOOD_MESSAGE

VALID_ROW = {
    "id": "1",
    "tramite": "Kardex oficial",
    "pregunta_ejemplo": "¿Cómo saco mi kardex? | Necesito mi historial académico",
    "requisitos": "Ser alumno; Sin adeudos",
    "documentos_necesarios": "Matrícula",
    "costo": "$150 MXN",
    "plazo": "1 día hábil",
    "dependencia_responsable": "Departamento Escolar",
    "pasos_procedimiento": "1. Solicita. 2. Paga. 3. Recoge.",
    "respuesta_generica": "El kardex se solicita en Tesorería.",
}


def write_csv(path: Path, rows: list[dict[str, str]], columns=REQUIRED_COLUMNS) -> Path:
    """Escribe un CSV de prueba con las columnas indicadas."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture(scope="module")
def bot() -> EduBot:
    """Bot con TF-IDF sobre la base de conocimiento real."""
    return EduBot(Settings(engine="tfidf", fallback_engine=None))


@pytest.fixture
def broken_model_loader(monkeypatch):
    """Simula que el modelo de embeddings no se puede descargar."""

    def fail(self):
        raise ModelLoadError("sin conexión (simulado)")

    monkeypatch.setattr(embeddings_module.EmbeddingsEngine, "_load_model", fail)


# --- Base de conocimiento ---------------------------------------------------


def test_knowledge_base_has_at_least_25_complete_tramites():
    tramites = load_knowledge_base(DATA_PATH)
    assert len(tramites) >= 25
    assert len({t.id for t in tramites}) == len(tramites)
    for tramite in tramites:
        assert tramite.preguntas and tramite.respuesta_generica
        assert all(tramite.get_field(name) for name in DETAIL_FIELDS)


def test_question_variants_are_split(tmp_path):
    path = write_csv(tmp_path / "datos.csv", [VALID_ROW])
    (tramite,) = load_knowledge_base(path)
    assert tramite.preguntas == ("¿Cómo saco mi kardex?", "Necesito mi historial académico")
    assert tramite.pregunta_ejemplo == "¿Cómo saco mi kardex?"


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="No se encontró"):
        load_knowledge_base(tmp_path / "no_existe.csv")


def test_missing_columns_raise(tmp_path):
    columns = [c for c in REQUIRED_COLUMNS if c != "costo"]
    path = write_csv(tmp_path / "datos.csv", [VALID_ROW], columns)
    with pytest.raises(KnowledgeBaseError, match="costo"):
        load_knowledge_base(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": "uno"}, "no es un número"),
        ({"tramite": ""}, "faltan valores"),
        ({"pregunta_ejemplo": " | "}, "vacía"),
    ],
)
def test_invalid_rows_raise(tmp_path, changes, message):
    path = write_csv(tmp_path / "datos.csv", [{**VALID_ROW, **changes}])
    with pytest.raises(KnowledgeBaseError, match=message):
        load_knowledge_base(path)


def test_duplicate_ids_raise(tmp_path):
    path = write_csv(tmp_path / "datos.csv", [VALID_ROW, VALID_ROW])
    with pytest.raises(KnowledgeBaseError, match="repetido"):
        load_knowledge_base(path)


def test_empty_knowledge_base_raises(tmp_path):
    path = write_csv(tmp_path / "datos.csv", [])
    with pytest.raises(KnowledgeBaseError, match="no tiene trámites"):
        load_knowledge_base(path)


def test_row_with_unquoted_commas_raises(tmp_path):
    path = tmp_path / "datos.csv"
    path.write_text(",".join(REQUIRED_COLUMNS) + "\n1,A,¿b?,c,d,e,f,g,h,i,j,k\n", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="más columnas"):
        load_knowledge_base(path)


def test_non_utf8_file_raises(tmp_path):
    path = tmp_path / "datos.csv"
    path.write_bytes(",".join(REQUIRED_COLUMNS).encode() + b"\n1,Tr\xe1mite\n")
    with pytest.raises(KnowledgeBaseError, match="UTF-8"):
        load_knowledge_base(path)


# --- Respuestas ---------------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_is_handled(bot, question):
    response = bot.respond(question)
    assert response.status == "empty"
    assert not response.answered


def test_example_question_gets_generic_answer(bot):
    response = bot.respond("¿Cómo me inscribo a materias?")
    assert response.status == "answered"
    assert response.tramite.tramite == "Inscripción a materias"
    assert response.tramite.respuesta_generica in response.text


def test_multi_field_question_extracts_each_field(bot):
    response = bot.respond("¿Qué necesito y cuánto tarda para sacar mi kardex?")
    assert response.tramite.tramite == "Kardex oficial"
    assert {"requisitos", "plazo"} <= set(response.fields)
    assert "Requisitos" in response.text and "Plazo" in response.text
    assert "Costo" not in response.text
    assert response.tramite.respuesta_generica not in response.text


def test_ambiguous_field_keeps_generic_answer(bot):
    response = bot.respond("¿Dónde veo mi horario de clases?")
    assert response.tramite.tramite == "Consulta de horario de clases"
    assert "SIASE" in response.text  # la respuesta general dice dónde verlo
    assert "Dependencia responsable" in response.text


def test_steps_are_rendered_as_numbered_list(bot):
    response = bot.respond("¿Cuáles son los pasos para el duplicado de credencial?")
    assert "1. Entra a SIASE" in response.text
    assert "2. Elige Duplicado" in response.text


def test_nonsense_question_is_rejected_with_topics(bot):
    response = bot.respond("¿Quién ganó el mundial de fútbol?")
    assert response.status == "not_understood"
    assert NOT_UNDERSTOOD_MESSAGE in response.text
    assert "Puedes preguntar sobre:" in response.text
    assert response.score < response.threshold


# --- Modelo del mundo: contexto de la conversación ---------------------------


def test_follow_up_question_uses_conversation_context(bot):
    context = ConversationContext()
    first = bot.respond("Perdí mi credencial", context)
    assert first.tramite.tramite == "Duplicado de credencial"
    follow_up = bot.respond("¿y cuánto cuesta?", context)
    assert follow_up.status == "follow_up"
    assert follow_up.tramite == first.tramite
    assert "$170" in follow_up.text


def test_follow_up_without_context_is_not_understood(bot):
    assert bot.respond("¿y cuánto cuesta?", ConversationContext()).status == "not_understood"


def test_new_topic_replaces_context(bot):
    context = ConversationContext()
    bot.respond("Perdí mi credencial", context)
    bot.respond("¿Cómo saco mi kardex?", context)
    assert bot.respond("¿cuánto tarda?", context).tramite.tramite == "Kardex oficial"
    context.reset()
    assert context.last_tramite_id is None and context.turns == 0


# --- Configuración y motor de respaldo ---------------------------------------


def test_fallback_engine_is_used_when_model_cannot_load(broken_model_loader):
    bot = EduBot(Settings(engine="embeddings", fallback_engine="tfidf"))
    assert bot.engine_name == "tfidf"
    assert bot.threshold == Settings().threshold
    assert any("embeddings" in warning for warning in bot.warnings)
    assert bot.respond("¿Cómo saco mi kardex?").answered


def test_without_fallback_the_error_is_raised(broken_model_loader):
    with pytest.raises(ModelLoadError):
        EduBot(Settings(engine="embeddings", fallback_engine=None))


@pytest.mark.parametrize(
    "kwargs", [{"engine": "gpt"}, {"threshold": 1.5}, {"embeddings_threshold": -0.1}]
)
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_settings_from_environment(tmp_path):
    env = {
        "EDUBOT_ENGINE": "TFIDF",
        "EDUBOT_THRESHOLD": "0.4",
        "EDUBOT_DATA_PATH": str(tmp_path / "otro.csv"),
    }
    settings = Settings.from_env(env)
    assert settings.engine == "tfidf"
    assert settings.threshold_for("tfidf") == 0.4
    assert settings.threshold_for("embeddings") == settings.embeddings_threshold
    assert settings.data_path == tmp_path / "otro.csv"
    with pytest.raises(ValueError, match="número"):
        Settings.from_env({"EDUBOT_THRESHOLD": "alto"})


# --- Consola -------------------------------------------------------------------


def test_console_answers_single_question(capsys):
    exit_code = chatbot.main(["--engine", "tfidf", "-q", "¿Cuánto cuesta el kardex?"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Kardex oficial" in output and "$150" in output
    assert "**" not in output  # sin Markdown en la terminal


def test_console_reports_missing_data(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EDUBOT_DATA_PATH", str(tmp_path / "no_existe.csv"))
    assert chatbot.main(["--engine", "tfidf", "-q", "hola"]) == 1
    assert "No se encontró" in capsys.readouterr().err


def test_bot_with_custom_knowledge_base(tmp_path):
    library = {
        **VALID_ROW,
        "id": "2",
        "tramite": "Biblioteca",
        "pregunta_ejemplo": "¿A qué hora abre la biblioteca?",
    }
    path = write_csv(tmp_path / "datos.csv", [VALID_ROW, library])
    bot = EduBot(replace(Settings(engine="tfidf"), data_path=path))
    assert len(bot.tramites) == 2
    assert bot.respond("horario de la biblioteca").tramite.tramite == "Biblioteca"
