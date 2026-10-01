# 🎓 EduBot: Asistente Virtual para Trámites Escolares

Proyecto académico de la materia **Laboratorio de Temas Selectos de Sistemas
Inteligentes** — Facultad de Ingeniería Mecánica y Eléctrica (FIME), UANL.

EduBot es un chatbot que responde preguntas sobre trámites escolares
(inscripción, becas, kardex, credencial, servicio social, titulación,
transporte, servicios médicos, etc.) escritas **en lenguaje natural**. No
necesita que la pregunta coincida palabra por palabra con su base de
conocimiento: entiende sinónimos, palabras reordenadas y paráfrasis, y cuando
no está seguro dice "No entiendo" en lugar de inventar una respuesta.

## 👥 Información del equipo

**Universidad Autónoma de Nuevo León**
Facultad de Ingeniería Mecánica y Eléctrica
Laboratorio Temas Selectos de Sistemas Inteligentes

- **Semestre:** Agosto - Diciembre 2026
- **Docente:** Raquel Martinez Martinez
- **Proyecto 1:** Chatbot de preguntas frecuentes con procesamiento de lenguaje natural

| Nombre | Matrícula | Brigada |
|---|---|---|
| Jose David Meza Flores | 2069864 | 305 |
| Jesús Arturo Corpus Zavala | 2132346 | No está inscrito |
| Brian Horacio Reyna Diaz de León | 1995434 | 305 |
| Victor Hugo Bernal Ríos | 2056233 | 305 |
| Samuel Alejandro Manrique Mujica | 2132342 | 305 |

## Propuesta del proyecto

- **Problemática:** Los estudiantes de nuevo ingreso pierden tiempo buscando
  respuestas a preguntas repetitivas sobre trámites escolares, porque la
  información está dispersa.
- **Propuesta:** Un chatbot de consola o interfaz web sencilla que recibe la
  pregunta del estudiante en texto libre y responde con la información
  correspondiente, aunque la pregunta no esté escrita exactamente igual que en
  la base de datos de respuestas.
- **Justificación como sistema inteligente:** Usa técnicas de procesamiento de
  lenguaje natural (similitud de texto con TF-IDF y embeddings semánticos con
  sentence-transformers) para interpretar el significado de la pregunta en
  lugar de buscar coincidencias exactas de palabras. Esa comprensión
  semántica, y no una simple búsqueda por palabra clave, es lo que lo
  convierte en un sistema inteligente.

---

## Técnicas de Inteligencia Artificial

| Técnica | Dónde se aplica | Archivo |
|---|---|---|
| **Redes neuronales artificiales** | `sentence-transformers` convierte cada pregunta en un vector de significado (embedding). Frases distintas con el mismo sentido quedan cerca. | `utils/embeddings.py` |
| **Algoritmos de NLP** | TF-IDF (palabras + n-gramas de caracteres) y similitud coseno: vectorizar → comparar → rankear → aplicar umbral → responder. | `utils/tfidf.py`, `chatbot.py` |
| **Heurística** | Umbral de similitud coseno (0.3 en TF-IDF): por debajo de él, EduBot no responde. Nunca responde con similitud 0. | `config.py`, `chatbot.py` |

## Arquitectura del agente

EduBot es un **agente reactivo basado en modelos** (Russell & Norvig).

- **Entorno:** parcialmente observable, estocástico, episódico, estático,
  discreto y de agente individual.
- **Estado interno:** recuerda el último trámite consultado, así que responde
  preguntas de seguimiento como "¿y cuánto cuesta?".

```
 Estudiante ──► Percepción ──► Estado actual ──► Reglas condición-acción ──► Acción ──► Actuadores
               (pregunta)     (vector TF-IDF    (similitud coseno ≥ umbral?)  (respuesta   (Streamlit /
                               o embedding)              ▲                    o "no        consola)
                                                         │                    entiendo")
                                              Modelo del mundo
                                     (datos.csv + contexto de la conversación)
```

| Paso | Implementación |
|---|---|
| 1. Percepción | `EduBot.respond()` recibe el texto escrito en la interfaz |
| 2. Estado actual | `EduBot.vectorize_question()` → TF-IDF o embeddings |
| 3. Modelo del mundo | `datos.csv` (31 trámites) + `ConversationContext` |
| 4. Reglas condición-acción | `EduBot.find_best_match()` compara contra el umbral |
| 5. Acción | `EduBot.generate_response()` arma la respuesta con plantillas |
| 6. Actuadores | `app.py` (web) o `chatbot.py` (terminal) |

### Preguntas multi-campo

Si la pregunta pide datos concretos, EduBot extrae solo esos campos del trámite:

> **Tú:** ¿Qué necesito y cuánto tarda para sacar mi kardex?
>
> **EduBot:** Sobre **Kardex oficial**, esto es lo que necesitas saber:
> 📋 **Requisitos:** … 📄 **Documentos necesarios:** … ⏱️ **Plazo:** Se entrega al siguiente día hábil…

Si la pregunta es general, responde con la respuesta completa del trámite.
Las palabras ambiguas ("¿dónde…?", "¿cuándo…?") agregan el campo
correspondiente sin ocultar la respuesta general.

---

## Estructura del proyecto

```
edubot/
├── README.md
├── requirements.txt        ← Dependencias con versiones exactas
├── datos.csv               ← Base de conocimiento (31 trámites)
├── evaluacion.csv          ← Preguntas para comparar los motores
├── config.py               ← Configuración (motor, umbrales, rutas, modelo)
├── chatbot.py              ← Motor del agente + consola + comparación
├── app.py                  ← Interfaz web con Streamlit
├── utils/
│   ├── __init__.py         ← Fábrica de motores (create_engine)
│   ├── base.py             ← Tramite, interfaz SimilarityEngine, motor exacto (v1.0)
│   ├── preprocessing.py    ← Normalización, sinónimos y stopwords en español
│   ├── tfidf.py            ← Motor TF-IDF (v2.x)
│   ├── embeddings.py       ← Motor sentence-transformers (v3.0)
│   └── template.py         ← Detección de campos y plantillas de respuesta
├── tests/
│   ├── test_chatbot.py     ← Base de conocimiento, respuestas, contexto, errores
│   └── test_matcher.py     ← Preprocesamiento, motores, umbral, campos
├── .streamlit/config.toml  ← Tema con colores institucionales
├── pytest.ini / ruff.toml  ← Configuración de pruebas y estilo
└── CLAUDE.md               ← Contexto del proyecto para asistentes de IA
```

## Instalación

Requisitos: **Python 3.10 o superior** y conexión a internet la primera vez.

```bash
cd edubot
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> `sentence-transformers` instala PyTorch (~1 GB). La **primera** vez que se
> usa el motor de embeddings se descarga el modelo
> `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) desde Hugging Face; EduBot
> lo avisa en pantalla. Si no hay internet, EduBot usa TF-IDF automáticamente
> y muestra una advertencia.

## Uso

### Interfaz web

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. Escribe tu pregunta y presiona
**Preguntar**. En la barra lateral puedes cambiar de motor, ver el umbral
activo y limpiar la conversación. Debajo de cada respuesta aparecen la
similitud obtenida, la ficha completa del trámite y el ranking de similitud.

### Consola

```bash
python chatbot.py                                   # conversación interactiva
python chatbot.py --engine tfidf                    # elegir motor
python chatbot.py -q "¿Cuánto cuesta el duplicado de credencial?"
python chatbot.py --compare                         # comparar los tres motores
```

## Configuración

Todo se ajusta en `config.py`:

```python
ENGINE = "embeddings"          # "embeddings" | "tfidf" | "exact"
FALLBACK_ENGINE = "tfidf"      # se usa si el principal no carga
SIMILARITY_THRESHOLD = 0.3     # TF-IDF y coincidencia exacta
EMBEDDINGS_THRESHOLD = 0.5     # embeddings
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
```

El motor es **intercambiable**: `chatbot.py` y `app.py` solo usan la interfaz
`SimilarityEngine`, así que cambiar `ENGINE` no requiere tocar más código.
También se puede sobrescribir sin editar archivos:

```bash
EDUBOT_ENGINE=tfidf streamlit run app.py
```

Variables disponibles: `EDUBOT_ENGINE`, `EDUBOT_THRESHOLD`,
`EDUBOT_EMBEDDINGS_THRESHOLD`, `EDUBOT_DATA_PATH`, `EDUBOT_MODEL`.

## Base de conocimiento (`datos.csv`)

Columnas: `id, tramite, pregunta_ejemplo, requisitos, documentos_necesarios,
costo, plazo, dependencia_responsable, pasos_procedimiento, respuesta_generica`.

Convenciones:

- `pregunta_ejemplo` admite **varias formas de preguntar** separadas por `|`;
  la primera es la principal. Más variantes = mejor reconocimiento.
- `requisitos` y `documentos_necesarios` separan elementos con `;`
  (se muestran como viñetas).
- `pasos_procedimiento` usa el formato `1. … 2. … 3. …` (lista numerada).
- Guardar como **CSV UTF-8**. Si el archivo falta, está dañado, tiene ids
  repetidos o columnas faltantes, EduBot muestra un error claro.

> ⚠️ Los datos se recopilaron de los sitios oficiales de la UANL y la FIME
> (ciclo 2025-2026). Los costos y fechas cambian cada semestre: verifícalos en
> [uanl.mx](https://www.uanl.mx) y [fime.uanl.mx](https://www.fime.uanl.mx)
> antes de usarlos.

## Pruebas

```bash
python -m pytest tests/ -v
```

72 pruebas cubren: carga y validación del CSV (archivo inexistente, columnas
faltantes, ids repetidos, codificación inválida), coincidencia correcta con
sinónimos y palabras reordenadas, rechazo por umbral de preguntas sin
sentido, preguntas vacías, preguntas multi-campo, contexto de la
conversación, motor de respaldo y la interfaz común de motores. La prueba con
el modelo real de embeddings se omite automáticamente si el modelo no está
descargado.

Estilo (PEP 8, orden de importaciones, docstrings Google):

```bash
pip install ruff && ruff check . && ruff format --check .
```

## Resultados de la comparación

`python chatbot.py --compare` evalúa cada motor con `evaluacion.csv` (54
paráfrasis + 8 preguntas fuera de tema). Además, `tests/test_matcher.py` tiene
un **conjunto de validación** (31 paráfrasis + 8 fuera de tema) que **no** se
usó para ajustar sinónimos ni umbral.

| Motor | `evaluacion.csv` (ajuste) | Validación (no vista) |
|---|---|---|
| v1.0 Coincidencia exacta | 13 % (solo acierta los rechazos) | 21 % |
| v2.1 TF-IDF + coseno + umbral 0.3 | 100 % | **95 %** |
| v3.0 Embeddings | ejecutar `--compare` con internet | — |

- Con TF-IDF, las preguntas fuera de tema obtienen similitud ≤ 0.20 y las
  correctas ≥ 0.43: el umbral 0.3 las separa con margen.
- El 100 % en `evaluacion.csv` es optimista porque ese conjunto se usó para
  ajustar el diccionario de sinónimos; el 95 % de validación es la cifra
  realista.
- Los embeddings no requieren diccionario de sinónimos, por eso se recomiendan
  como motor principal. Su umbral (0.5) es un valor inicial: ajústalo
  ejecutando `python chatbot.py --compare` en un equipo con el modelo
  descargado.

## Plan de implementación

| Semana | Entregable | Dónde |
|---|---|---|
| 1 | Base de conocimiento | `datos.csv` |
| 2 | v1.0 coincidencia exacta | `ExactMatchEngine` (`--engine exact`) |
| 3 | v2.0 TF-IDF + similitud coseno | `utils/tfidf.py` |
| 4 | v2.1 umbral y "No entiendo" + sugerencias | `config.py`, `utils/template.py` |
| 5 | Interfaz web | `app.py` |
| 6 | v3.0 embeddings + comparación | `utils/embeddings.py`, `--compare` |
| 7 | Documentación y pruebas | `README.md`, `tests/` |
| 8 | QA final | `pytest`, `ruff`, `evaluacion.csv` |

## Limitaciones

- La información de trámites es de referencia y puede cambiar.
- TF-IDF depende del diccionario de sinónimos de `utils/preprocessing.py`
  para palabras que no aparecen en la base de conocimiento.
- EduBot no consulta sistemas de la UANL (SIASE) ni datos personales.
