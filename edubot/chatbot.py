"""Motor principal de EduBot y versión de consola.

EduBot es un agente reactivo basado en modelos (Russell & Norvig):

1. Percepción: recibe la pregunta del estudiante.
2. Estado actual: la pregunta se convierte en vector (TF-IDF o embeddings).
3. Modelo del mundo: la base de conocimiento (``datos.csv``) y el contexto
   de la conversación (último trámite consultado).
4. Reglas condición-acción: si la similitud coseno supera el umbral se
   responde; si no, se dice "No entiendo".
5. Acción: se genera la respuesta en lenguaje natural con plantillas.

Uso desde la terminal::

    python chatbot.py                       # conversación interactiva
    python chatbot.py --engine tfidf        # elegir motor
    python chatbot.py -q "¿Cómo saco mi kardex?"
    python chatbot.py --compare             # comparar motores
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from config import ENGINE_CHOICES, Settings
from utils import (
    DETAIL_FIELDS,
    EngineError,
    KnowledgeBaseError,
    SimilarityEngine,
    Tramite,
    create_engine,
    template,
)
from utils.base import Matrix
from utils.preprocessing import TextNormalizer

LOGGER = logging.getLogger("edubot")

REQUIRED_COLUMNS = (
    "id",
    "tramite",
    "pregunta_ejemplo",
    *DETAIL_FIELDS,
    "respuesta_generica",
)
# Columnas que no pueden quedar vacías en ninguna fila.
MANDATORY_VALUES = ("id", "tramite", "pregunta_ejemplo", "respuesta_generica")
# Separador de variantes dentro de "pregunta_ejemplo".
QUESTION_SEPARATOR = "|"


@dataclass(frozen=True)
class MatchResult:
    """Resultado de comparar una pregunta contra la base de conocimiento.

    Attributes:
        tramite: Trámite con mayor similitud (aunque no supere el umbral).
        score: Similitud coseno del mejor trámite.
        threshold: Umbral aplicado.
        accepted: ``True`` si ``score`` alcanza el umbral.
        ranking: Los trámites más parecidos con su similitud, de mayor a menor.
    """

    tramite: Tramite | None
    score: float
    threshold: float
    accepted: bool
    ranking: tuple[tuple[Tramite, float], ...]


@dataclass
class ConversationContext:
    """Estado interno del agente durante una conversación.

    Permite responder preguntas de seguimiento como "¿y cuánto cuesta?".

    Attributes:
        last_tramite_id: Último trámite respondido.
        turns: Número de preguntas procesadas.
    """

    last_tramite_id: int | None = None
    turns: int = 0

    def reset(self) -> None:
        """Olvida la conversación actual."""
        self.last_tramite_id = None
        self.turns = 0


@dataclass(frozen=True)
class BotResponse:
    """Respuesta completa del agente a una pregunta.

    Attributes:
        text: Respuesta en Markdown.
        status: ``"answered"``, ``"follow_up"``, ``"not_understood"`` o ``"empty"``.
        engine: Motor de NLP que procesó la pregunta.
        tramite: Trámite usado en la respuesta, si lo hay.
        score: Similitud de la mejor coincidencia (``None`` si no se buscó).
        threshold: Umbral aplicado.
        fields: Campos incluidos en la respuesta.
        ranking: Nombres de los trámites más parecidos con su similitud.
    """

    text: str
    status: str
    engine: str
    tramite: Tramite | None = None
    score: float | None = None
    threshold: float | None = None
    fields: tuple[str, ...] = ()
    ranking: tuple[tuple[str, float], ...] = ()

    @property
    def answered(self) -> bool:
        """Indica si EduBot respondió con información de un trámite.

        Returns:
            ``True`` para respuestas directas o de seguimiento.
        """
        return self.status in ("answered", "follow_up")


def _split_questions(raw: str) -> tuple[str, ...]:
    """Separa las variantes de pregunta de una celda.

    Args:
        raw: Texto de la columna ``pregunta_ejemplo``.

    Returns:
        Variantes no vacías.
    """
    return tuple(part.strip() for part in raw.split(QUESTION_SEPARATOR) if part.strip())


def _row_to_tramite(row: dict[str, str | None], line: int) -> Tramite:
    """Convierte una fila del CSV en un ``Tramite`` validado.

    Args:
        row: Fila leída por ``csv.DictReader``.
        line: Número de línea en el archivo (para mensajes de error).

    Returns:
        El trámite construido.

    Raises:
        KnowledgeBaseError: Si faltan valores obligatorios o el id no es entero.
    """
    values = {column: (row.get(column) or "").strip() for column in REQUIRED_COLUMNS}
    empty = [column for column in MANDATORY_VALUES if not values[column]]
    if empty:
        raise KnowledgeBaseError(f"Línea {line}: faltan valores en {', '.join(empty)}.")
    try:
        tramite_id = int(values["id"])
    except ValueError as exc:
        raise KnowledgeBaseError(
            f"Línea {line}: el id '{values['id']}' no es un número entero."
        ) from exc
    questions = _split_questions(values["pregunta_ejemplo"])
    if not questions:
        raise KnowledgeBaseError(f"Línea {line}: 'pregunta_ejemplo' está vacía.")
    return Tramite(
        id=tramite_id,
        tramite=values["tramite"],
        preguntas=questions,
        **{field: values[field] for field in DETAIL_FIELDS},
        respuesta_generica=values["respuesta_generica"],
    )


def load_knowledge_base(path: Path, encoding: str = "utf-8-sig") -> list[Tramite]:
    """Lee y valida la base de conocimiento.

    Args:
        path: Ruta al archivo CSV.
        encoding: Codificación del archivo.

    Returns:
        Lista de trámites en el orden del archivo.

    Raises:
        KnowledgeBaseError: Si el archivo no existe, no es CSV válido, le
            faltan columnas, está vacío o tiene ids repetidos.
    """
    path = Path(path)
    if not path.is_file():
        raise KnowledgeBaseError(f"No se encontró la base de conocimiento: {path}")
    try:
        with path.open(encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            header = [name.strip() for name in (reader.fieldnames or [])]
            missing = [column for column in REQUIRED_COLUMNS if column not in header]
            if missing:
                raise KnowledgeBaseError(
                    f"El archivo {path.name} no tiene las columnas: {', '.join(missing)}."
                )
            reader.fieldnames = header
            tramites = []
            for row in reader:
                if None in row:  # más celdas que encabezados
                    raise KnowledgeBaseError(
                        f"Línea {reader.line_num}: la fila tiene más columnas "
                        "que el encabezado (¿faltan comillas?)."
                    )
                if not any((value or "").strip() for value in row.values()):
                    continue  # fila en blanco
                tramites.append(_row_to_tramite(row, reader.line_num))
    except UnicodeDecodeError as exc:
        raise KnowledgeBaseError(
            f"No se pudo leer {path.name} como {encoding}. Guárdalo como CSV UTF-8."
        ) from exc
    except csv.Error as exc:
        raise KnowledgeBaseError(f"Formato CSV inválido en {path.name}: {exc}") from exc

    if not tramites:
        raise KnowledgeBaseError(f"La base de conocimiento {path.name} no tiene trámites.")
    seen: set[int] = set()
    for tramite in tramites:
        if tramite.id in seen:
            raise KnowledgeBaseError(f"El id {tramite.id} está repetido en {path.name}.")
        seen.add(tramite.id)
    return tramites


class EduBot:
    """Agente conversacional que responde preguntas sobre trámites escolares.

    Attributes:
        settings: Configuración en uso.
        tramites: Trámites de la base de conocimiento.
        engine: Motor de NLP activo.
        threshold: Umbral de similitud del motor activo.
        warnings: Avisos para el usuario (p. ej. uso del motor de respaldo).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        engine: SimilarityEngine | None = None,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        """Carga la base de conocimiento y prepara el motor de NLP.

        Args:
            settings: Configuración; por defecto ``Settings.from_env()``.
            engine: Motor ya creado (útil en pruebas). Si es ``None`` se crea
                el indicado en la configuración.
            notify: Función para avisar al usuario durante la carga.

        Raises:
            KnowledgeBaseError: Si ``datos.csv`` no existe o es inválido.
            EngineError: Si no se puede crear ningún motor.
        """
        self.settings = settings or Settings.from_env()
        self.tramites = load_knowledge_base(self.settings.data_path, self.settings.csv_encoding)
        self._by_id = {tramite.id: tramite for tramite in self.tramites}
        self._normalizer = TextNormalizer()
        self.warnings: list[str] = []
        self.engine = engine if engine is not None else self._create_engine(notify)
        self.threshold = self.settings.threshold_for(self.engine.name)

        documents, owners = self._build_corpus()
        self._doc_owner = np.asarray(owners, dtype=int)
        self.engine.fit(documents)

    @property
    def engine_name(self) -> str:
        """Nombre del motor de NLP activo.

        Returns:
            ``"embeddings"``, ``"tfidf"`` o ``"exact"``.
        """
        return self.engine.name

    def _create_engine(self, notify: Callable[[str], None] | None) -> SimilarityEngine:
        """Crea el motor configurado o, si falla, el de respaldo.

        Args:
            notify: Función para avisar al usuario.

        Returns:
            El motor de NLP listo para entrenar.

        Raises:
            EngineError: Si fallan tanto el motor principal como el de respaldo.
        """
        primary = self.settings.engine
        try:
            return create_engine(
                primary, embedding_model=self.settings.embedding_model, notify=notify
            )
        except EngineError as exc:
            fallback = self.settings.fallback_engine
            if not fallback or fallback == primary:
                raise
            message = (
                f"No se pudo usar el motor '{primary}' ({exc}). Se usará '{fallback}' en su lugar."
            )
            LOGGER.warning(message)
            self.warnings.append(message)
            return create_engine(
                fallback, embedding_model=self.settings.embedding_model, notify=notify
            )

    def _build_corpus(self) -> tuple[list[str], list[int]]:
        """Construye los documentos a indexar.

        Cada trámite aporta su nombre y todas sus preguntas de ejemplo.

        Returns:
            Los textos y, para cada uno, la posición de su trámite en
            ``self.tramites``.
        """
        documents: list[str] = []
        owners: list[int] = []
        for position, tramite in enumerate(self.tramites):
            for text in (*tramite.preguntas, tramite.tramite):
                documents.append(text)
                owners.append(position)
        return documents, owners

    def get_tramite(self, tramite_id: int) -> Tramite | None:
        """Busca un trámite por su id.

        Args:
            tramite_id: Identificador del trámite.

        Returns:
            El trámite o ``None`` si no existe.
        """
        return self._by_id.get(tramite_id)

    def vectorize_question(self, question: str) -> Matrix:
        """Convierte la pregunta en vector con el motor activo.

        Args:
            question: Pregunta del usuario.

        Returns:
            Matriz de una fila (TF-IDF disperso o embedding denso).
        """
        return self.engine.transform([question])

    def find_best_match(self, question: str) -> MatchResult:
        """Encuentra el trámite más parecido a la pregunta.

        Algoritmo: vectorizar → comparar (similitud coseno) → rankear →
        aplicar el umbral. La similitud de un trámite es la máxima entre
        sus preguntas de ejemplo.

        Args:
            question: Pregunta del usuario.

        Returns:
            El mejor trámite, su similitud y si supera el umbral.
        """
        similarities = np.asarray(self.engine.similarities(question), dtype=float)
        scores = np.zeros(len(self.tramites))
        np.maximum.at(scores, self._doc_owner, np.clip(similarities, 0.0, 1.0))
        order = np.argsort(-scores, kind="stable")
        best = int(order[0])
        best_score = float(scores[best])
        top_k = max(self.settings.top_k, 1)
        ranking = tuple(
            (self.tramites[int(index)], float(scores[index])) for index in order[:top_k]
        )
        # Heurística: nunca responder sin evidencia (similitud 0).
        accepted = best_score > 0.0 and best_score >= self.threshold
        return MatchResult(
            tramite=self.tramites[best],
            score=best_score,
            threshold=self.threshold,
            accepted=accepted,
            ranking=ranking,
        )

    def generate_response(self, match: MatchResult, question: str) -> str:
        """Genera la respuesta en lenguaje natural.

        Args:
            match: Resultado de ``find_best_match``.
            question: Pregunta original (para detectar los campos pedidos).

        Returns:
            Respuesta en Markdown.
        """
        if match.accepted and match.tramite is not None:
            request = template.detect_requested_fields(question)
            return template.build_answer(
                match.tramite, request.fields, include_generic=request.include_generic
            )
        # Sugerir trámites con algo de parecido (al menos la mitad del umbral).
        suggestions = [
            tramite.tramite
            for tramite, score in match.ranking
            if score > 0.0 and score >= match.threshold / 2
        ]
        return template.build_not_understood(suggestions, self.settings.topic_hints)

    def respond(self, question: str, context: ConversationContext | None = None) -> BotResponse:
        """Ejecuta un ciclo completo del agente: percibir, decidir y actuar.

        Args:
            question: Pregunta del usuario (percepción).
            context: Estado de la conversación; se actualiza con el trámite
                respondido. Si es ``None`` la pregunta se trata de forma aislada.

        Returns:
            La respuesta y los datos del análisis.
        """
        context = context if context is not None else ConversationContext()
        question = (question or "").strip()
        context.turns += 1
        if not question:
            return BotResponse(
                text=template.build_empty_message(), status="empty", engine=self.engine_name
            )

        previous = (
            self.get_tramite(context.last_tramite_id)
            if context.last_tramite_id is not None
            else None
        )
        if previous is not None and template.is_follow_up(question, self._normalizer):
            fields = template.detect_requested_fields(question).fields
            return BotResponse(
                text=template.build_answer(previous, fields, include_generic=False),
                status="follow_up",
                engine=self.engine_name,
                tramite=previous,
                fields=fields,
            )

        match = self.find_best_match(question)
        text = self.generate_response(match, question)
        ranking = tuple((tramite.tramite, score) for tramite, score in match.ranking)
        if not match.accepted:
            return BotResponse(
                text=text,
                status="not_understood",
                engine=self.engine_name,
                score=match.score,
                threshold=match.threshold,
                ranking=ranking,
            )
        context.last_tramite_id = match.tramite.id
        return BotResponse(
            text=text,
            status="answered",
            engine=self.engine_name,
            tramite=match.tramite,
            score=match.score,
            threshold=match.threshold,
            fields=template.detect_requested_fields(question).fields,
            ranking=ranking,
        )


# --- Evaluación y comparación de motores -----------------------------------


@dataclass(frozen=True)
class EvaluationCase:
    """Pregunta de prueba con el trámite que debería responderla.

    Attributes:
        question: Pregunta tal como la escribiría un estudiante.
        expected_id: Id del trámite correcto, o ``None`` si EduBot debe
            responder "No entiendo".
    """

    question: str
    expected_id: int | None


@dataclass(frozen=True)
class EvaluationReport:
    """Resultados de evaluar un motor.

    Attributes:
        engine: Nombre del motor evaluado.
        results: Por caso: (caso, id obtenido o ``None``, similitud, acierto).
    """

    engine: str
    results: tuple[tuple[EvaluationCase, int | None, float, bool], ...]

    @property
    def accuracy(self) -> float:
        """Proporción de casos resueltos correctamente.

        Returns:
            Valor entre 0 y 1.
        """
        if not self.results:
            return 0.0
        return sum(ok for *_, ok in self.results) / len(self.results)


def load_evaluation_cases(path: Path, encoding: str = "utf-8-sig") -> list[EvaluationCase]:
    """Lee las preguntas de evaluación (columnas ``pregunta`` e ``id_esperado``).

    Args:
        path: Ruta al CSV de evaluación.
        encoding: Codificación del archivo.

    Returns:
        Lista de casos de prueba.

    Raises:
        KnowledgeBaseError: Si el archivo no existe o tiene un formato inválido.
    """
    path = Path(path)
    if not path.is_file():
        raise KnowledgeBaseError(f"No se encontró el archivo de evaluación: {path}")
    cases = []
    with path.open(encoding=encoding, newline="") as handle:
        for row in csv.DictReader(handle):
            question = (row.get("pregunta") or "").strip()
            expected = (row.get("id_esperado") or "").strip()
            if not question:
                continue
            try:
                expected_id = int(expected) if expected else None
            except ValueError as exc:
                raise KnowledgeBaseError(f"id_esperado inválido: '{expected}'") from exc
            cases.append(EvaluationCase(question, expected_id))
    return cases


def evaluate(bot: EduBot, cases: Sequence[EvaluationCase]) -> EvaluationReport:
    """Mide qué tan bien responde un bot a un conjunto de preguntas.

    Args:
        bot: Instancia de EduBot con el motor a evaluar.
        cases: Casos de prueba.

    Returns:
        El reporte con el resultado de cada caso.
    """
    results = []
    for case in cases:
        match = bot.find_best_match(case.question)
        obtained = match.tramite.id if match.accepted and match.tramite else None
        results.append((case, obtained, match.score, obtained == case.expected_id))
    return EvaluationReport(engine=bot.engine_name, results=tuple(results))


def compare_engines(
    settings: Settings, engines: Sequence[str] = ENGINE_CHOICES
) -> list[EvaluationReport]:
    """Evalúa varios motores con el mismo conjunto de preguntas.

    Args:
        settings: Configuración base (ruta de datos y de evaluación).
        engines: Motores a comparar.

    Returns:
        Un reporte por cada motor que se pudo cargar.
    """
    cases = load_evaluation_cases(settings.evaluation_path, settings.csv_encoding)
    reports = []
    for name in engines:
        # Sin respaldo: si un motor falla debe reportarse, no sustituirse.
        engine_settings = replace(settings, engine=name, fallback_engine=None)
        try:
            bot = EduBot(engine_settings, notify=print)
        except EngineError as exc:
            print(f"⚠️  Motor '{name}' no disponible: {exc}")
            continue
        reports.append(evaluate(bot, cases))
    return reports


def print_comparison(reports: Sequence[EvaluationReport]) -> None:
    """Imprime una tabla comparativa de los motores evaluados.

    Args:
        reports: Reportes generados por ``compare_engines``.
    """
    if not reports:
        print("No se pudo evaluar ningún motor.")
        return
    width = 52
    header = f"{'Pregunta':<{width}}  {'Esperado':>8}  " + "  ".join(
        f"{report.engine:>14}" for report in reports
    )
    print(header)
    print("-" * len(header))
    for index, (case, *_) in enumerate(reports[0].results):
        question = case.question
        if len(question) > width:
            question = question[: width - 1] + "…"
        expected = "rechazo" if case.expected_id is None else str(case.expected_id)
        cells = []
        for report in reports:
            _, obtained, score, ok = report.results[index]
            label = "—" if obtained is None else str(obtained)
            cells.append(f"{('✔' if ok else '✘')} {label:>4} ({score:.2f})")
        print(f"{question:<{width}}  {expected:>8}  " + "  ".join(f"{c:>14}" for c in cells))
    print("-" * len(header))
    print(
        f"{'Exactitud':<{width}}  {'':>8}  "
        + "  ".join(f"{report.accuracy:>14.0%}" for report in reports)
    )


# --- Interfaz de consola -----------------------------------------------------


def chat_loop(bot: EduBot) -> int:
    """Conversación interactiva en la terminal.

    Args:
        bot: Instancia de EduBot lista para responder.

    Returns:
        Código de salida (0).
    """
    print(template.markdown_to_plain(template.build_welcome(bot.settings.topic_hints)))
    print(
        f"\n(Motor: {bot.engine_name}, umbral: {bot.threshold}). Escribe 'salir' para terminar.\n"
    )
    context = ConversationContext()
    while True:
        try:
            question = input("Tú: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.strip().lower() in {"salir", "exit", "quit", "adios", "adiós"}:
            break
        response = bot.respond(question, context)
        print(f"\nEduBot: {template.markdown_to_plain(response.text)}\n")
    print("¡Hasta luego! 👋")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define los argumentos de la línea de comandos.

    Returns:
        El analizador de argumentos.
    """
    parser = argparse.ArgumentParser(
        description="EduBot: asistente virtual para trámites escolares (consola)."
    )
    parser.add_argument("--engine", choices=ENGINE_CHOICES, help="Motor de NLP a usar.")
    parser.add_argument("--threshold", type=float, help="Umbral de similitud (0 a 1).")
    parser.add_argument("-q", "--pregunta", help="Responde una sola pregunta y termina.")
    parser.add_argument(
        "--compare", action="store_true", help="Compara los motores con evaluacion.csv."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Muestra mensajes de depuración."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de la consola.

    Args:
        argv: Argumentos de la línea de comandos (por defecto ``sys.argv``).

    Returns:
        Código de salida: 0 si todo salió bien, 1 si hubo un error.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        settings = Settings.from_env()
        if args.engine:
            settings = settings.with_engine(args.engine)
        if args.threshold is not None:
            settings = replace(
                settings, threshold=args.threshold, embeddings_threshold=args.threshold
            )
        if args.compare:
            print_comparison(compare_engines(settings))
            return 0
        bot = EduBot(settings, notify=print)
    except (KnowledgeBaseError, EngineError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for warning in bot.warnings:
        print(f"⚠️  {warning}")
    if args.pregunta is not None:
        print(template.markdown_to_plain(bot.respond(args.pregunta).text))
        return 0
    return chat_loop(bot)


if __name__ == "__main__":
    sys.exit(main())
