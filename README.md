# Chatbot de trabajos de grado (RAG)

Asistente conversacional que responde preguntas sobre las normas de presentación de trabajos de grado
(formato institucional, formato de anteproyecto y manual IEEE). Usa *Retrieval-Augmented Generation*:
recupera fragmentos de los documentos oficiales y genera la respuesta con Gemini, citando documento y página.

> Estado: **Sprint 2 – fase F0** (entorno y repositorio).

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
| Ingesta *(F2)* | `python -m app.cli ingest` |
| Evaluación *(F4)* | `python -m app.cli eval` |
| API *(F5)* | `uvicorn app.main:app --reload` → http://localhost:8000/docs |

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
| LLM y embeddings | Gemini (`GoogleGenAI`, `GoogleGenAIEmbedding`) | Configurables en `.env` |
| API | FastAPI | Documentación Swagger automática en `/docs` |
