"""Generación de respuestas en lenguaje natural mediante plantillas.

Este módulo decide QUÉ información mostrar de un trámite:

* Si la pregunta pide datos concretos ("¿qué necesito y cuánto tarda?"),
  se detectan los campos (requisitos, plazo, ...) y se arma una respuesta
  solo con ellos (soporte multi-campo).
* Si la pregunta es general ("háblame de la beca"), se usa la respuesta
  genérica del trámite.

Las respuestas se generan en Markdown, que Streamlit muestra con formato;
``markdown_to_plain`` las adapta para la consola.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .base import Tramite
from .preprocessing import TextNormalizer, basic_normalize

NOT_UNDERSTOOD_MESSAGE = "No entiendo tu pregunta. Intenta reformularla."


@dataclass(frozen=True)
class FieldSpec:
    """Describe cómo detectar y presentar un campo de un trámite.

    Attributes:
        name: Nombre de la columna en ``datos.csv``.
        label: Título que se muestra al usuario.
        icon: Emoji que acompaña al título.
        style: ``"list"`` (viñetas), ``"steps"`` (numerado) o ``"text"``.
        patterns: Expresiones regulares (sobre texto normalizado) que indican
            claramente que el usuario pide este campo ("¿cuánto cuesta?").
        weak_patterns: Indicios ambiguos ("¿dónde...?", "¿cuándo...?"): el
            campo se agrega, pero sin omitir la respuesta general.
    """

    name: str
    label: str
    icon: str
    style: str
    patterns: tuple[re.Pattern[str], ...]
    weak_patterns: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class FieldRequest:
    """Campos que solicita una pregunta.

    Attributes:
        strong: Campos pedidos de forma explícita.
        weak: Campos con indicios ambiguos (no incluidos en ``strong``).
    """

    strong: tuple[str, ...] = ()
    weak: tuple[str, ...] = ()

    @property
    def fields(self) -> tuple[str, ...]:
        """Todos los campos solicitados, en el orden de ``FIELD_SPECS``.

        Returns:
            Nombres de columna.
        """
        requested = set(self.strong) | set(self.weak)
        return tuple(spec.name for spec in FIELD_SPECS if spec.name in requested)

    @property
    def include_generic(self) -> bool:
        """Indica si conviene mostrar también la respuesta general.

        Returns:
            ``True`` cuando no se pidió ningún campo de forma explícita.
        """
        return not self.strong


def _patterns(*regexes: str) -> tuple[re.Pattern[str], ...]:
    """Compila expresiones regulares delimitadas por palabras completas.

    Args:
        *regexes: Expresiones regulares sin delimitadores.

    Returns:
        Tupla de patrones compilados.
    """
    return tuple(re.compile(rf"\b(?:{regex})\b") for regex in regexes)


_NEED = r"que (?:se )?(?:necesita|necesito|ocupa|ocupo|requiero|requiere)"

FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="requisitos",
        label="Requisitos",
        icon="📋",
        style="list",
        patterns=_patterns(
            r"requisitos?",
            r"requiere\w*",
            _NEED,
            r"que (?:me )?piden",
            r"que tengo que cumplir",
            r"quien(?:es)? pueden?",
            r"condiciones",
        ),
    ),
    FieldSpec(
        name="documentos_necesarios",
        label="Documentos necesarios",
        icon="📄",
        style="list",
        patterns=_patterns(
            r"documentos?",
            r"papeles",
            r"papeleria",
            r"que (?:debo|tengo que|hay que) llevar",
            r"que llevo",
            r"expediente",
            _NEED,
        ),
    ),
    FieldSpec(
        name="costo",
        label="Costo",
        icon="💰",
        style="text",
        patterns=_patterns(
            r"cuanto (?:cuesta|cuestan|cobran|vale|sale|es|hay que pagar|pago|se paga)",
            r"costos?",
            r"precios?",
            r"cobran?",
            r"tiene costo",
        ),
        weak_patterns=_patterns(r"gratis", r"gratuit[oa]"),
    ),
    FieldSpec(
        name="plazo",
        label="Plazo",
        icon="⏱️",
        style="text",
        patterns=_patterns(
            r"cuanto (?:tiempo|tarda|tardan|demora|dura)",
            r"cuant[oa]s (?:dias|horas|semanas|meses)",
            r"tardan?",
            r"demora",
            r"plazos?",
            r"vigencia",
            r"duracion",
        ),
        weak_patterns=_patterns(r"cuando", r"fechas?", r"a que hora", r"horas?"),
    ),
    FieldSpec(
        name="dependencia_responsable",
        label="Dependencia responsable",
        icon="🏢",
        style="text",
        patterns=_patterns(
            r"a quien",
            r"con quien",
            r"quien (?:me )?(?:atiende|lo da|lo tramita|lo hace|lo entrega)",
            r"oficinas?",
            r"dependencia",
            r"departamento",
            r"ventanilla",
            r"responsable",
        ),
        weak_patterns=_patterns(r"donde", r"a donde"),
    ),
    FieldSpec(
        name="pasos_procedimiento",
        label="Pasos a seguir",
        icon="🧭",
        style="steps",
        patterns=_patterns(
            r"pasos?",
            r"procedimiento",
            r"proceso",
            r"que (?:hago|tengo que hacer|debo hacer|hay que hacer)",
            r"como le hago",
        ),
    ),
)

_STEP_SPLIT = re.compile(r"(?:^|\s+)(?=\d{1,2}\.\s)")
_STEP_NUMBER = re.compile(r"^\d{1,2}\.\s*")


def detect_requested_fields(question: str) -> FieldRequest:
    """Detecta qué campos de un trámite solicita la pregunta.

    Ejemplo: "¿Qué necesito y cuánto tarda el kardex?" pide requisitos,
    documentos y plazo.

    Args:
        question: Pregunta original del usuario.

    Returns:
        Los campos solicitados de forma explícita y los ambiguos.
    """
    normalized = basic_normalize(question)
    strong = tuple(
        spec.name
        for spec in FIELD_SPECS
        if any(pattern.search(normalized) for pattern in spec.patterns)
    )
    weak = tuple(
        spec.name
        for spec in FIELD_SPECS
        if spec.name not in strong
        and any(pattern.search(normalized) for pattern in spec.weak_patterns)
    )
    return FieldRequest(strong=strong, weak=weak)


def is_follow_up(question: str, normalizer: TextNormalizer) -> bool:
    """Indica si la pregunta es de seguimiento ("¿y cuánto cuesta?").

    Una pregunta de seguimiento pide un dato concreto pero no menciona
    ningún tema: tras quitar stopwords y palabras de intención no queda nada.

    Args:
        question: Pregunta original del usuario.
        normalizer: Normalizador que elimina stopwords y palabras de intención.

    Returns:
        ``True`` si debe responderse con el trámite de la conversación.
    """
    requested = detect_requested_fields(question).fields
    return bool(requested) and not normalizer.tokenize(question)


def split_items(value: str) -> list[str]:
    """Divide un campo de lista separado por punto y coma.

    Args:
        value: Texto como ``"CURP; Kardex; Fotografía"``.

    Returns:
        Lista de elementos sin espacios sobrantes.
    """
    return [item.strip() for item in value.split(";") if item.strip()]


def split_steps(value: str) -> list[str]:
    """Divide un procedimiento numerado ("1. ... 2. ...") en pasos.

    Args:
        value: Texto con pasos numerados.

    Returns:
        Lista de pasos sin su número.
    """
    parts = (_STEP_NUMBER.sub("", part).strip() for part in _STEP_SPLIT.split(value))
    return [part for part in parts if part]


def format_field(spec: FieldSpec, value: str) -> str:
    """Da formato Markdown a un campo según su estilo.

    Args:
        spec: Especificación del campo.
        value: Valor del campo en la base de conocimiento.

    Returns:
        Sección en Markdown con título y contenido.
    """
    header = f"{spec.icon} **{spec.label}:**"
    value = value.strip() or "No especificado."
    if spec.style == "steps":
        steps = split_steps(value)
        if len(steps) > 1:
            body = "\n".join(f"{number}. {step}" for number, step in enumerate(steps, 1))
            return f"{header}\n{body}"
    if spec.style == "list":
        items = split_items(value)
        if len(items) > 1:
            body = "\n".join(f"- {item}" for item in items)
            return f"{header}\n{body}"
    return f"{header} {value}"


def build_answer(
    tramite: Tramite, fields: Sequence[str] = (), include_generic: bool | None = None
) -> str:
    """Arma la respuesta para un trámite identificado.

    Args:
        tramite: Trámite que mejor responde la pregunta.
        fields: Campos solicitados.
        include_generic: Si se antepone la respuesta genérica. Por defecto
            solo se usa cuando no se pidió ningún campo.

    Returns:
        Respuesta en Markdown.
    """
    if include_generic is None:
        include_generic = not fields
    specs = [spec for spec in FIELD_SPECS if spec.name in fields]
    sections = [format_field(spec, tramite.get_field(spec.name)) for spec in specs]
    if include_generic:
        parts = [f"**{tramite.tramite}**", tramite.respuesta_generica, *sections]
        if not sections:
            parts.append(
                "_¿Quieres más detalle? Pregúntame por los requisitos, documentos, "
                "costo, plazo, dónde se tramita o los pasos._"
            )
        return "\n\n".join(parts)
    intro = f"Sobre **{tramite.tramite}**, esto es lo que necesitas saber:"
    return "\n\n".join([intro, *sections])


def build_not_understood(suggestions: Iterable[str] = (), topic_hints: Iterable[str] = ()) -> str:
    """Arma la respuesta cuando ninguna coincidencia supera el umbral.

    Args:
        suggestions: Nombres de trámites parecidos a la pregunta.
        topic_hints: Temas generales sobre los que se puede preguntar.

    Returns:
        Mensaje en Markdown que pide reformular la pregunta.
    """
    parts = [f"🤔 {NOT_UNDERSTOOD_MESSAGE}"]
    suggestions = list(suggestions)
    if suggestions:
        names = ", ".join(f"**{name}**" for name in suggestions)
        parts.append(f"¿Quizás quisiste preguntar por: {names}?")
    hints = list(topic_hints)
    if hints:
        parts.append(f"Puedes preguntar sobre: {', '.join(hints)}, etc.")
    return "\n\n".join(parts)


def build_empty_message() -> str:
    """Mensaje para cuando el usuario envía una pregunta vacía.

    Returns:
        Mensaje en Markdown con un ejemplo de pregunta.
    """
    return (
        "✏️ Escribe una pregunta sobre algún trámite escolar. Por ejemplo: «¿Cómo saco mi kardex?»"
    )


def build_welcome(topic_hints: Iterable[str] = ()) -> str:
    """Mensaje de bienvenida con instrucciones de uso.

    Args:
        topic_hints: Temas sobre los que se puede preguntar.

    Returns:
        Mensaje en Markdown.
    """
    hints = ", ".join(topic_hints)
    lines = [
        "👋 ¡Hola! Soy **EduBot**, tu asistente para trámites escolares de la FIME-UANL.",
        "Escríbeme tu pregunta con tus propias palabras, por ejemplo:",
        "- «¿Qué necesito para sacar mi credencial?»\n"
        "- «¿Cuánto cuesta y cuánto tarda el kardex?»\n"
        "- «Perdí mi credencial, ¿qué hago?»",
        "Después puedes preguntar «¿y cuánto cuesta?» y recordaré el trámite del que hablamos.",
    ]
    if hints:
        lines.append(f"Temas disponibles: {hints}.")
    return "\n\n".join(lines)


def markdown_to_plain(text: str) -> str:
    """Quita marcas de Markdown (negritas y cursivas) para la consola.

    Args:
        text: Texto en Markdown.

    Returns:
        Texto plano.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
