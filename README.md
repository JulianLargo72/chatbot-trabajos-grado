# Chatbot de trabajos de grado (RAG)

Asistente conversacional que responde preguntas sobre las normas de presentación de trabajos de grado
(formato institucional, formato de anteproyecto y manual IEEE). Usa *Retrieval-Augmented Generation*:
recupera fragmentos de los documentos oficiales y genera la respuesta con Gemini, citando documento y página.

> Estado: **Sprint 2 – fase F2** (lectura y fragmentación del corpus, HU03).

## Arquitectura

Un solo proyecto Python (monolito modular). La ingesta y la API comparten configuración y módulos RAG.

| Capa | Carpeta | Responsabilidad |
|---|---|---|
| HTTP | `app/api/` | Rutas FastAPI (`/api/chat`, `/api/admin/ingest`, `/health`) |
| Lógica | `app/services/` | Orquestación de ingesta y de chat (reglas HU08/HU09) |
| Infraestructura RAG | `app/rag/` | Lectura de PDF, fragmentación, índice Chroma, recuperación, prompts |
| Configuración | `app/core/` | `Settings` (lee `.env`) y logging |
| Contratos | `app/schemas/` | Modelos Pydantic de solicitud y respuesta |

```
app/            código de la aplicación
data/corpus/    PDF oficiales (no se versionan)
data/chroma/    índice vectorial persistente (no se versiona)
eval/           banco de preguntas y reportes de evaluación
tests/          pruebas unitarias e de integración
```

## Requisitos

- Python 3.11, 3.12 o 3.13
- Clave de la API de Gemini (Google AI Studio)
- Git

## Instalación (Windows, PowerShell)

```powershell
git clone https://github.com/JulianLargo72/chatbot-trabajos-grado.git
cd chatbot-trabajos-grado

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # si falla: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install

Copy-Item .env.example .env          # luego edita .env y pon tu GOOGLE_API_KEY
```

## Comandos

| Acción | Comando |
|---|---|
| Pruebas | `pytest` |
| Pruebas con cobertura | `pytest --cov` |
| Lint | `ruff check .` |
| Formato | `ruff format .` |
| Todos los hooks | `pre-commit run --all-files` |
| Verificar clave y modelos de Gemini | `python -m app.cli check` |
| API (desarrollo) | `uvicorn app.main:app --reload` → http://localhost:8000/docs |
| Lectura y fragmentación del corpus | `python -m app.cli ingest` |
| … y exportar los fragmentos para revisarlos | `python -m app.cli ingest --exportar eval/reports/fragmentos.jsonl` |
| Evaluación *(F4)* | `python -m app.cli eval` |

## Ingesta (HU03)

1. Copiar los PDF oficiales en `data/corpus/` (no se suben al repositorio).
2. `python -m app.cli ingest` lee cada página con `pypdf`, conserva `file_name` y `page_label`
   (la etiqueta de página impresa del PDF, p. ej. el manual IEEE empieza en la 4), elimina
   encabezados/pies repetidos, números de página sueltos y viñetas de Word, y fragmenta con
   `SentenceSplitter` (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
3. Cada página es un documento independiente, así que ningún fragmento mezcla dos páginas y la
   cita *documento + página* siempre es exacta.

## Configuración

Todos los parámetros están en `app/core/config.py` y se leen desde `.env` (plantilla en `.env.example`).

| Variable | Valor por defecto | Uso |
|---|---|---|
| `GEMINI_LLM_MODEL` | `gemini-3.5-flash-lite` | Modelo que redacta las respuestas (cuota gratuita amplia) |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Debe ser el mismo en ingesta y consulta |
| `LLM_TEMPERATURE` | *(vacío)* | Gemini 3.x usa su valor por defecto; la fidelidad se controla con el System Prompt y el umbral |
| `LLM_THINKING_LEVEL` | `low` | Razonamiento interno de Gemini 3.x (menor latencia y consumo) |
| `LLM_MAX_OUTPUT_TOKENS` | `1024` | Longitud máxima de la respuesta |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Tamaño y solapamiento de fragmentos (tokens) |
| `TOP_K` | `3` | Fragmentos recuperados por pregunta |
| `SIMILARITY_THRESHOLD` | `0.60` | Por debajo se responde «sin información» (HU08) |
| `LOG_JSON` | `true` | Logs en JSON (evidencia) o texto legible |

## Flujo de trabajo con Git

- Rama principal: `main` (siempre estable).
- Una rama por historia: `feature/HU03-chunking`, `feature/HU04-indice`, …
- Commits convencionales: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`.
- Antes de cada *merge*: `ruff` sin errores y `pytest` en verde.

## Seguridad

La clave de Gemini va **solo** en `.env`, que está excluido en `.gitignore`. El hook `detect-private-key`
de pre-commit ayuda a evitar subir secretos por error.

## Decisiones técnicas

| Decisión | Elección | Motivo |
|---|---|---|
| Arquitectura | Monolito modular | Una configuración y un mismo modelo de embeddings en ingesta y consulta |
| Orquestación | LlamaIndex | Lectura, fragmentación, índice y recuperación integrados |
| Base vectorial | ChromaDB local persistente | Sin servidor adicional |
| LLM y embeddings | `gemini-3.5-flash-lite` + `gemini-embedding-001` (plan gratuito) | Configurables en `.env`. `gemini-2.5-flash` ya no se ofrece a usuarios nuevos y `gemini-3.8-flash` tiene ~20 solicitudes/día gratis; `text-embedding-004` fue retirado el 14-01-2026 |
| API | FastAPI | Documentación Swagger automática en `/docs` |
