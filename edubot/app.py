"""Interfaz web de EduBot con Streamlit (actuadores del agente).

Ejecutar con::

    streamlit run app.py

Toda la lógica de NLP vive en ``chatbot.py``; este archivo solo se encarga
de mostrar la conversación. El motor se elige en ``config.py`` (o desde la
barra lateral) sin modificar este código.
"""

from __future__ import annotations

import streamlit as st

from chatbot import BotResponse, ConversationContext, EduBot
from config import ENGINE_CHOICES, Settings
from utils import EngineError, KnowledgeBaseError, is_model_cached, template
from utils.embeddings import MODEL_SIZE_HINT

UANL_BLUE = "#003366"
UANL_GOLD = "#F2B705"

ENGINE_LABELS = {
    "embeddings": "Embeddings (red neuronal) · v3.0",
    "tfidf": "TF-IDF + similitud coseno · v2.1",
    "exact": "Coincidencia exacta · v1.0",
}
AVATARS = {"user": "🧑‍🎓", "assistant": "🎓"}

CUSTOM_CSS = f"""
<style>
  .edubot-header {{
    background: linear-gradient(135deg, {UANL_BLUE} 0%, #0B4F8A 100%);
    border-bottom: 4px solid {UANL_GOLD};
    border-radius: 12px;
    color: #FFFFFF;
    margin-bottom: 1.25rem;
    padding: 1.1rem 1.4rem;
  }}
  .edubot-header h1 {{
    color: #FFFFFF;
    font-size: 1.9rem;
    margin: 0;
    padding: 0;
  }}
  .edubot-header p {{
    color: #DCE6F2;
    margin: 0.25rem 0 0 0;
  }}
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {{
    color: {UANL_BLUE};
  }}
  div[data-testid="stForm"] {{
    border: 2px solid {UANL_BLUE}22;
    border-radius: 12px;
  }}
  .edubot-footer {{
    color: #6B7280;
    font-size: 0.8rem;
    margin-top: 1.5rem;
    text-align: center;
  }}
</style>
"""


@st.cache_resource(show_spinner=False)
def load_bot(engine: str) -> EduBot:
    """Crea (una sola vez por motor) la instancia de EduBot.

    El bot no guarda el estado de la conversación, por lo que puede
    compartirse entre sesiones; el contexto vive en ``st.session_state``.

    Args:
        engine: Motor de NLP a usar.

    Returns:
        El bot con la base de conocimiento indexada.
    """
    return EduBot(Settings.from_env().with_engine(engine))


def init_state(default_engine: str) -> None:
    """Inicializa las variables de sesión de Streamlit.

    Args:
        default_engine: Motor configurado en ``config.py``.
    """
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("context", ConversationContext())
    st.session_state.setdefault("engine", default_engine)


def get_bot(engine: str, settings: Settings) -> EduBot:
    """Carga el bot mostrando un aviso mientras tanto.

    Si el modelo de embeddings no está descargado se avisa al usuario de
    que la primera carga tardará más.

    Args:
        engine: Motor solicitado.
        settings: Configuración base (para conocer el modelo).

    Returns:
        El bot listo; detiene la app con un error si no se puede crear.
    """
    message = "Cargando EduBot..."
    if engine == "embeddings" and not is_model_cached(settings.embedding_model):
        message = (
            f"Descargando el modelo de embeddings ({MODEL_SIZE_HINT}). "
            "Solo ocurre la primera vez y puede tardar varios minutos..."
        )
    try:
        with st.spinner(message):
            return load_bot(engine)
    except KnowledgeBaseError as exc:
        st.error(f"❌ Error en la base de conocimiento: {exc}")
    except EngineError as exc:
        st.error(f"❌ No se pudo iniciar el motor de NLP: {exc}")
    st.stop()


def render_sidebar(settings: Settings) -> str:
    """Dibuja la barra lateral con información del proyecto y opciones.

    Args:
        settings: Configuración base.

    Returns:
        El motor seleccionado por el usuario.
    """
    with st.sidebar:
        st.markdown("## 🎓 EduBot")
        st.caption(
            "Asistente virtual para trámites escolares · Laboratorio de Temas "
            "Selectos de Sistemas Inteligentes · FIME-UANL"
        )
        st.markdown("### ⚙️ Motor de NLP")
        engine = st.radio(
            "Motor de NLP",
            options=ENGINE_CHOICES,
            index=ENGINE_CHOICES.index(st.session_state.engine),
            format_func=ENGINE_LABELS.get,
            label_visibility="collapsed",
        )
        st.session_state.engine = engine
        if st.button("🧹 Limpiar conversación"):
            st.session_state.messages = []
            st.session_state.context.reset()
            st.rerun()
        st.markdown("### 💡 Temas")
        st.markdown(" · ".join(settings.topic_hints))
    return engine


def render_engine_status(bot: EduBot) -> None:
    """Muestra en la barra lateral el motor activo y sus avisos.

    Args:
        bot: Bot cargado.
    """
    with st.sidebar:
        st.markdown("### 📊 Estado del agente")
        col_engine, col_threshold = st.columns(2)
        col_engine.metric("Motor activo", bot.engine_name)
        col_threshold.metric("Umbral", f"{bot.threshold:.2f}")
        st.caption(f"{len(bot.tramites)} trámites en la base de conocimiento.")
        for warning in bot.warnings:
            st.warning(warning, icon="⚠️")
        st.info(
            "La información es de referencia (ciclo 2025-2026). Verifica costos "
            "y fechas en uanl.mx y fime.uanl.mx.",
            icon="ℹ️",
        )


def render_details(response: BotResponse) -> None:
    """Muestra el análisis del agente debajo de una respuesta.

    Args:
        response: Respuesta generada por EduBot.
    """
    if response.status == "follow_up" and response.tramite is not None:
        st.caption(f"🔁 Pregunta de seguimiento sobre **{response.tramite.tramite}**")
    elif response.score is not None:
        verdict = "✅ supera" if response.answered else "⛔ no alcanza"
        st.caption(
            f"🎯 Similitud {response.score:.2f} · {verdict} el umbral "
            f"{response.threshold:.2f} · motor {response.engine}"
        )
    if response.tramite is not None:
        with st.expander("📑 Ficha completa del trámite"):
            for spec in template.FIELD_SPECS:
                value = response.tramite.get_field(spec.name)
                st.markdown(template.format_field(spec, value))
    if response.ranking:
        with st.expander("🔎 Ranking de similitud"):
            for name, score in response.ranking:
                st.progress(min(max(score, 0.0), 1.0), text=f"{name} — {score:.2f}")


def render_history() -> None:
    """Dibuja el historial de la conversación (o la bienvenida)."""
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.markdown(template.build_welcome(Settings.from_env().topic_hints))
        return
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
            st.markdown(message["content"])
            if message.get("response") is not None:
                render_details(message["response"])


def handle_question(bot: EduBot, question: str) -> None:
    """Procesa una pregunta y la agrega al historial.

    Args:
        bot: Bot cargado.
        question: Texto escrito por el usuario.
    """
    response = bot.respond(question, st.session_state.context)
    st.session_state.messages.append({"role": "user", "content": question.strip()})
    st.session_state.messages.append(
        {"role": "assistant", "content": response.text, "response": response}
    )


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(page_title="EduBot · FIME-UANL", page_icon="🎓", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    settings = Settings.from_env()
    init_state(settings.engine)
    engine = render_sidebar(settings)
    bot = get_bot(engine, settings)
    render_engine_status(bot)

    st.markdown(
        '<div class="edubot-header"><h1>🎓 EduBot</h1>'
        "<p>Asistente virtual para trámites escolares de la FIME-UANL</p></div>",
        unsafe_allow_html=True,
    )
    render_history()

    with st.form("pregunta", clear_on_submit=True):
        question = st.text_input(
            "Escribe tu pregunta",
            placeholder="Ej. ¿Qué necesito y cuánto cuesta el duplicado de credencial?",
        )
        submitted = st.form_submit_button("Preguntar", type="primary")
    if submitted:
        if question.strip():
            handle_question(bot, question)
            st.rerun()
        st.warning(template.build_empty_message())

    st.markdown(
        '<p class="edubot-footer">Proyecto académico · Agente reactivo basado en '
        "modelos · NLP con TF-IDF y sentence-transformers</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
