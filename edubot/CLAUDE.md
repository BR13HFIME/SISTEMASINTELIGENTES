# CLAUDE.md — Contexto del proyecto EduBot

Chatbot académico (FIME-UANL, Laboratorio de Temas Selectos de Sistemas
Inteligentes) que responde preguntas sobre trámites escolares con NLP.
Agente reactivo basado en modelos (Russell & Norvig).

## Comandos

```bash
pip install -r requirements.txt
streamlit run app.py                   # interfaz web
python chatbot.py                      # consola (-q "pregunta", --engine tfidf)
python chatbot.py --compare            # compara motores con evaluacion.csv
python -m pytest tests/ -v             # pruebas
ruff check . && ruff format --check .  # estilo
```

## Arquitectura

- `config.py`: única fuente de configuración (`Settings`, inmutable). Motor
  (`ENGINE`), umbrales, ruta de datos, modelo. Las variables `EDUBOT_*` la
  sobrescriben.
- `chatbot.py`: `EduBot` (carga `datos.csv`, `vectorize_question`,
  `find_best_match`, `generate_response`, `respond`), `ConversationContext`
  (estado del agente), evaluación de motores y CLI.
- `app.py`: solo UI de Streamlit; importa `chatbot.py`. El bot se cachea con
  `st.cache_resource` y el contexto vive en `st.session_state`.
- `utils/base.py`: `Tramite`, errores, interfaz `SimilarityEngine`
  (`fit`, `transform`, `similarities`) y `ExactMatchEngine` (v1.0).
- `utils/tfidf.py`: TF-IDF de palabras + n-gramas de caracteres (FeatureUnion).
- `utils/embeddings.py`: sentence-transformers, importado de forma diferida;
  lanza `ModelLoadError` si no está instalado o no se puede descargar, y
  `EduBot` cae al motor de respaldo (`FALLBACK_ENGINE`).
- `utils/preprocessing.py`: normalización, sinónimos, stopwords, plurales.
- `utils/template.py`: detección de campos solicitados (fuertes/débiles),
  preguntas de seguimiento y plantillas Markdown.

## Convenciones

- Python 3.10+, PEP 8 (líneas ≤ 100), importaciones stdlib → terceros → local.
- Docstrings estilo Google en todas las funciones; textos de usuario en español.
- Sin variables globales mutables: solo constantes y clases.
- Nuevo motor: subclase de `SimilarityEngine`, registrarla en
  `utils/__init__.py` (`create_engine`, `ENGINE_CLASSES`) y en
  `config.ENGINE_CHOICES`.
- `datos.csv`: variantes de pregunta separadas por `|`; listas con `;`;
  pasos con `1. … 2. …`; codificación UTF-8.
- Si cambias sinónimos o umbrales, ejecuta `python chatbot.py --compare` y las
  pruebas. `HELD_OUT` en `tests/test_matcher.py` es el conjunto de validación:
  no ajustes sinónimos para esas preguntas.
