"""Preprocesamiento de texto en español para los motores de NLP.

TF-IDF solo compara palabras, así que necesita ayuda para entender que
"inscribirme", "inscripción" y "dar de alta materias" hablan de lo mismo.
Este módulo normaliza el texto en cuatro pasos:

1. Minúsculas, sin acentos ni signos de puntuación.
2. Sustitución de sinónimos y frases por un término canónico.
3. Eliminación de palabras vacías (stopwords) y de palabras de intención
   ("cuánto", "requisitos", "dónde"...), que indican QUÉ dato se pide pero
   no SOBRE QUÉ trámite se pregunta.
4. Lematización ligera (quitar plurales) para unificar variantes.

Todas las constantes son inmutables (``frozenset`` / tuplas).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

SPANISH_STOPWORDS = frozenset(
    """
    a al algo algun alguna alguno algunos ante antes aqui asi aun bien cada
    como con contra cual cuales de del desde e el ella ellas ellos en entre
    era eres es esa esas ese eso esos esta estan estar estas este esto estos
    estoy favor fue ha han hay he la las le les lo los mas me mi mis mucho muy
    nada ni no nos nosotros o oye otra otro para pero poco por porque pues
    puede pueden puedo que quiero quisiera se sea ser si sin sobre soy su sus
    tambien te tengo tiene tienen todo toda todos tu tus un una uno unos unas
    va vas voy y ya yo hola buenas buenos gracias tardes noches dias porfa
    porfavor hace hacer hago haga hacen debo debe tenemos tener toca alguien
    hacerlo sigue sale salen dan son mientras ahora luego entonces despues ahi
    alli solo tan tanto cosa cosas mismo misma
    """.split()
)

# Palabras que expresan el dato solicitado (costo, plazo, lugar...). Se
# eliminan para que la búsqueda se centre en el tema; ``template.py`` usa
# sus propios patrones para detectar qué campos responder.
INTENT_WORDS = frozenset(
    """
    requisito requisitos requiere requieren requerimientos necesito necesita
    necesitan necesario necesarios ocupo ocupa ocupan documento documentos
    papeles papeleria llevar llevo cuanto cuanta cuantos cuantas cuesta
    cuestan costo costos precio precios cobran cobra gratis gratuito gratuita
    cuando tiempo tarda tardan demora demoran plazo plazos fecha fechas donde
    adonde quien quienes oficina dependencia pasos paso procedimiento proceso
    tramite tramites tramitar tramito solicito solicitar solicitud pido pedir
    saco sacar obtengo obtener consigo conseguir info informacion dato datos
    saber manera forma
    """.split()
)

# (término canónico, variantes). Las variantes de varias palabras se
# sustituyen primero, de la más larga a la más corta.
# fmt: off
SYNONYM_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("inscripcion", (
        "inscripcion", "inscripciones", "inscribo", "inscribir", "inscribirme",
        "inscribirse", "inscribe", "inscribi", "inscrito", "inscrita",
        "reinscripcion", "reinscripciones", "reinscribir", "reinscribirme",
        "reinscribo", "dar de alta", "alta de materias", "registrar",
        "registrarme", "registrarse",
    )),
    ("beca", (
        "beca", "becas", "becado", "becada", "apoyo economico",
        "ayuda economica", "estimulo economico", "descuento", "apoyo", "apoyos",
    )),
    ("credencial", (
        "credencial", "credenciales", "carnet", "gafete", "identificacion escolar",
        "credencial de estudiante",
    )),
    ("reposicion", (
        "reposicion", "duplicado", "reponer", "repongo", "reponerla", "perdi",
        "perdio", "perdida", "perdido", "extravie", "extravio", "extraviada",
        "robaron", "robo", "rompio", "otra credencial",
    )),
    ("kardex", (
        "kardex", "historial academico", "trayectoria academica",
        "record academico", "historial de calificaciones",
    )),
    ("calificacion", (
        "calificacion", "calificaciones", "nota", "notas", "boleta",
        "pase la materia", "pasar la materia", "aprobe", "aprobar", "aprobado",
        "aprobada",
    )),
    ("horario", ("horario", "horarios")),
    ("salon", ("salon", "salones", "aula", "aulas")),
    ("constancia", (
        "constancia", "constancias", "comprobante de estudios",
        "carta de alumno", "estoy estudiando", "que soy alumno", "que soy estudiante",
        "que estoy inscrito", "comprobar", "compruebe", "comprueba",
    )),
    ("certificado", ("certificado", "certificados")),
    ("titulacion", (
        "titulacion", "titularme", "titularse", "titularte", "titulo",
        "titulo profesional", "recibirme", "recibirse", "me titulo",
    )),
    ("pasante", (
        "pasante", "pasantes", "termine la carrera", "terminar la carrera",
        "termino la carrera", "acabe la carrera",
    )),
    ("baja", (
        "baja", "bajas", "darme de baja", "dar de baja", "doy de baja",
        "salirme de", "desinscribir", "dejar la materia",
    )),
    ("reprobar", (
        "reprobe", "reprobar", "repruebo", "reprobada", "reprobado",
        "reprobaron", "trone", "tronar", "no pase", "no acredite", "no aprobe",
        "no apruebo",
    )),
    ("extraordinario", (
        "extraordinario", "extraordinarios", "extra", "extras",
        "segunda oportunidad", "examen extraordinario",
    )),
    ("recursar", (
        "recursar", "recursando", "tercera oportunidad",
        "cuarta oportunidad", "quinta oportunidad", "sexta oportunidad",
        "volver a cursar", "repetir la materia", "oportunidades",
    )),
    ("serviciosocial", ("servicio social",)),
    ("practicas", (
        "practicas profesionales", "practica profesional", "practicas",
        "practicante", "estadia",
    )),
    ("imss", (
        "imss", "seguro social", "seguro facultativo", "seguro medico", "nss",
        "numero de seguro social", "seguridad social",
    )),
    ("medico", (
        "medico", "medica", "servicio medico", "servicios medicos", "doctor",
        "doctora", "enfermeria", "enfermera", "consultorio", "consulta medica",
        "me siento mal", "enfermo", "enferma", "clinica",
    )),
    ("cuota", ("cuota", "cuotas", "colegiatura", "mensualidad", "cuota interna")),
    ("pago", (
        "pago", "pagos", "pagar", "paga", "pagan", "pague", "recibo de pago",
        "banco", "bancos", "banorte", "transferencia", "spei",
    )),
    ("biblioteca", ("biblioteca", "bibliotecas")),
    ("libro", ("libro", "libros")),
    ("prestamo", ("prestamo", "prestado", "prestados", "prestan", "prestar")),
    ("renta", ("renta", "rentar", "rento", "alquilar", "alquilado", "alquiler")),
    ("tigrebus", ("tigrebus", "tigre bus", "tigrevan", "tigre van")),
    ("transporte", (
        "transporte", "camion", "camiones", "autobus", "autobuses", "bus",
        "metro", "ruta", "rutas",
    )),
    ("tarifa", ("tarifa preferencial", "tarifa", "10 pesos", "me muevo")),
    ("cafeteria", (
        "cafeteria", "cafeterias", "comedor", "comida", "comer", "almorzar",
        "desayunar", "lonche",
    )),
    ("correo", ("correo", "correo institucional", "email", "mail", "outlook")),
    ("office", (
        "office", "office 365", "microsoft", "microsoft 365", "teams", "word",
        "excel", "powerpoint",
    )),
    ("contrasena", ("contrasena", "password", "clave", "nip", "pin")),
    ("intercambio", (
        "intercambio", "movilidad", "movilidad academica", "extranjero",
        "otro pais", "estudiar fuera",
    )),
    ("cambiocarrera", (
        "cambio de carrera", "cambiarme de carrera", "cambiar de carrera",
        "cambio carrera", "otra ingenieria", "cambiarme a", "cambiar a",
        "pasarme a otra carrera",
    )),
    ("carrera", (
        "carrera", "carreras", "ingenieria", "mecatronica", "mecanica",
        "electrica", "electronica", "aeronautica", "biomedica", "manufactura",
        "materiales", "computacion",
    )),
    ("revalidacion", (
        "revalidacion", "revalidar", "revalido", "equivalencia",
        "equivalencias",
    )),
    ("intersemestral", (
        "intersemestral", "verano", "curso de verano", "cursos de verano",
    )),
    ("convocatoria", ("convocatoria", "convocatorias", "aviso", "avisos", "anuncios")),
    ("ingles", ("ingles", "idioma", "segundo idioma", "toefl")),
    ("egel", ("egel", "ceneval")),
    ("materia", (
        "materia", "materias", "clase", "clases", "asignatura", "asignaturas",
        "unidad de aprendizaje", "unidades de aprendizaje",
    )),
    ("facultad", ("facultad", "fime", "escuela", "prepa")),
    ("universidad", ("universidad", "uanl", "uni")),
)
# fmt: on


def strip_accents(text: str) -> str:
    """Elimina acentos y diacríticos (á → a, ñ → n, ü → u).

    Args:
        text: Texto original.

    Returns:
        El texto sin marcas diacríticas.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def basic_normalize(text: str) -> str:
    """Normalización mínima: minúsculas, sin acentos ni puntuación.

    Es la que usa la coincidencia exacta (v1.0): "¿Cómo me inscribo?" y
    "como me inscribo" se consideran el mismo texto.

    Args:
        text: Texto original.

    Returns:
        Texto en minúsculas, sin acentos y con espacios simples.
    """
    lowered = strip_accents(text.lower())
    without_punctuation = re.sub(r"[^\w\s]|_", " ", lowered)
    return " ".join(without_punctuation.split())


def light_stem(token: str) -> str:
    """Lematización ligera: elimina plurales regulares del español.

    No es un lematizador completo; solo unifica variantes frecuentes
    (``calificaciones`` → ``calificacion``, ``libros`` → ``libro``).

    Args:
        token: Palabra normalizada.

    Returns:
        La palabra sin la terminación plural.
    """
    if len(token) > 5 and token.endswith("es") and token[-3] in "nrld":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


class TextNormalizer:
    """Normalizador de preguntas en español para TF-IDF.

    Compila una sola vez las expresiones regulares de sinónimos, por lo que
    conviene crear una instancia y reutilizarla.

    Attributes:
        remove_intent_words: Si es ``True`` elimina las palabras de intención
            (costo, plazo, dónde...) además de las stopwords.
    """

    def __init__(
        self,
        synonym_groups: Iterable[tuple[str, Iterable[str]]] = SYNONYM_GROUPS,
        stopwords: frozenset[str] = SPANISH_STOPWORDS,
        intent_words: frozenset[str] = INTENT_WORDS,
        remove_intent_words: bool = True,
    ) -> None:
        """Construye el normalizador.

        Args:
            synonym_groups: Pares (canónico, variantes) ya normalizados.
            stopwords: Palabras vacías a eliminar.
            intent_words: Palabras de intención a eliminar.
            remove_intent_words: Si se eliminan también las palabras de intención.
        """
        self.remove_intent_words = remove_intent_words
        self._stopwords = stopwords
        self._intent_words = intent_words
        self._word_map: dict[str, str] = {}
        phrase_map: dict[str, str] = {}
        for canonical, variants in synonym_groups:
            for variant in variants:
                key = basic_normalize(variant)
                target = phrase_map if " " in key else self._word_map
                target[key] = canonical
        self._canonical_terms = frozenset(canonical for canonical, _ in synonym_groups)
        self._phrase_map = phrase_map
        self._phrase_pattern = (
            re.compile(
                r"\b(?:"
                + "|".join(
                    re.escape(phrase) for phrase in sorted(phrase_map, key=len, reverse=True)
                )
                + r")\b"
            )
            if phrase_map
            else None
        )

    def _replace_phrases(self, text: str) -> str:
        """Sustituye frases sinónimas por su término canónico.

        Args:
            text: Texto ya normalizado con ``basic_normalize``.

        Returns:
            Texto con las frases reemplazadas.
        """
        if self._phrase_pattern is None:
            return text
        return self._phrase_pattern.sub(lambda match: f" {self._phrase_map[match.group(0)]} ", text)

    def _map_token(self, token: str) -> str | None:
        """Convierte una palabra a su forma canónica o la descarta.

        Args:
            token: Palabra normalizada.

        Returns:
            El término canónico o lematizado, o ``None`` si es una palabra vacía.
        """
        if token in self._canonical_terms:
            return token
        if token in self._word_map:
            return self._word_map[token]
        if token in self._stopwords:
            return None
        if self.remove_intent_words and token in self._intent_words:
            return None
        stem = light_stem(token)
        return self._word_map.get(stem, stem)

    def tokenize(self, text: str) -> list[str]:
        """Normaliza un texto y lo divide en términos significativos.

        Args:
            text: Pregunta o documento original.

        Returns:
            Lista de términos canónicos, sin stopwords.
        """
        normalized = self._replace_phrases(basic_normalize(text))
        tokens = (self._map_token(token) for token in normalized.split())
        return [token for token in tokens if token]

    def normalize(self, text: str) -> str:
        """Normaliza un texto y lo devuelve como cadena.

        Args:
            text: Pregunta o documento original.

        Returns:
            Los términos significativos separados por espacios. Una cadena
            vacía indica que el texto no menciona ningún tema.
        """
        return " ".join(self.tokenize(text))
