# Registro de cambios

Formato basado en *Keep a Changelog*; versiones con *SemVer*.

## [0.3.0] – 2026-11-16 · Cierre de los Sprints 2 y 3

### Sprint 3 – Backend y orquestación (HU06 a HU09)
- `POST /api/chat`: respuesta, fuentes (documento, página y puntaje) y `tipo`
  (`respondida`, `sin_informacion`, `fuera_de_dominio`); 422 ante preguntas inválidas y 503 si Gemini
  o el índice no están disponibles; documentado en `/docs`.
- System Prompt versionado (`app/rag/prompts.py`, v1.1) y documentado en `docs/system_prompt.md`:
  respuestas solo con el contexto, citas en el texto, prioridad del formato institucional sobre el
  manual IEEE, mensajes estándar para «sin información» y «fuera de dominio».
- Protección contra respuestas no sustentadas: sin fragmentos sobre el umbral nunca se devuelve una
  respuesta generada.
- `python -m app.cli eval-chat`: categorías B y C del banco con Gemini → **5 de 5** y **5 de 5**.
- Comandos `preguntar` y `prompt`.

### Sprint 2 – Ingesta y recuperación (HU03 a HU05)
- Lectura de PDF con `pypdf`, limpieza de encabezados y pies repetidos, metadatos `file_name` y
  `page_label`; fragmentación con `SentenceSplitter` (512/64).
- Embeddings `gemini-embedding-001` e índice persistente en ChromaDB (coseno) con ingesta
  incremental por SHA-256, IDs deterministas y verificación fragmentos = vectores (36 = 36).
- `POST /api/admin/ingest` protegido con `X-Admin-Key`.
- Banco de 37 preguntas (`eval/banco_preguntas.json`) y `python -m app.cli eval`:
  hit rate top-3 **88,9 %** (meta 80 %), MRR 0,824; umbral de similitud calibrado en 0,67.

### Base (F0–F1)
- Estructura del proyecto, `pyproject.toml`, ruff, pytest, pre-commit, configuración con
  `pydantic-settings`, logging estructurado y `GET /health`.
