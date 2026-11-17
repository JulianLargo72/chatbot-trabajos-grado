# Chatbot de trabajos de grado (RAG)

Asistente conversacional que responde preguntas sobre las normas de presentación de trabajos de grado
(formato institucional, formato de anteproyecto y manual IEEE). Usa *Retrieval-Augmented Generation*:
recupera fragmentos de los documentos oficiales y genera la respuesta con Gemini, citando documento y página.

> Estado: **Sprint 3 – fase F5** (API de chat con fuentes, HU06 y HU07).

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
| Ingesta incremental (fragmentos → embeddings → Chroma) | `python -m app.cli ingest` |
| Reconstruir el índice desde cero | `python -m app.cli ingest --reset` |
| Solo fragmentar (sin embeddings, no consume cuota) | `python -m app.cli ingest --solo-fragmentar --exportar eval/reports/fragmentos.jsonl` |
| Estado del índice (sin reprocesar) | `python -m app.cli indice` |
| Probar la recuperación | `python -m app.cli buscar "¿Cuántas palabras tiene el resumen?"` |
| Evaluación de la recuperación (hit rate, HU05) | `python -m app.cli eval` |
| Consulta completa desde la terminal (1 embedding + 1 llamada al LLM) | `python -m app.cli preguntar "¿Cuál es la extensión máxima del resumen?"` |

## Ingesta (HU03)

1. Copiar los PDF oficiales en `data/corpus/` (no se suben al repositorio).
2. `python -m app.cli ingest` lee cada página con `pypdf`, conserva `file_name` y `page_label`
   (la etiqueta de página impresa del PDF, p. ej. el manual IEEE empieza en la 4), elimina
   encabezados/pies repetidos, números de página sueltos y viñetas de Word, y fragmenta con
   `SentenceSplitter` (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
3. Cada página es un documento independiente, así que ningún fragmento mezcla dos páginas y la
   cita *documento + página* siempre es exacta.

## Índice vectorial (HU04)

- **Base vectorial:** ChromaDB persistente en `data/chroma/` (colección `reglamentos`, distancia coseno).
- **Embeddings:** `gemini-embedding-001` (3072 dimensiones). LlamaIndex usa `RETRIEVAL_DOCUMENT` al
  indexar y `RETRIEVAL_QUERY` al consultar.
- **Mismo modelo garantizado:** la colección guarda el modelo con que se construyó; si `.env` indica
  otro, la ingesta y la consulta se detienen y piden `--reset`.
- **Ingesta incremental e idempotente:** se compara el SHA-256 de cada PDF con el guardado en el
  índice. Solo se embeben los archivos nuevos o modificados; los retirados del corpus se eliminan.
  Los IDs de los fragmentos son deterministas, por lo que repetir la ingesta no duplica vectores.
- **Verificación:** al terminar se comprueba que *fragmentos = vectores* por archivo y en total.
- **Reindexar sin reiniciar:** `POST /api/admin/ingest` con la cabecera `X-Admin-Key: <ADMIN_API_KEY>`
  (`?reset=true` para reconstruir). Responde 401 con clave inválida, 409 si ya hay una ingesta en curso
  y 503 si `ADMIN_API_KEY` no está configurada.
- **Puntaje de similitud:** el adaptador de Chroma de LlamaIndex devuelve `exp(-distancia)`; el sistema
  lo convierte a similitud coseno real (`1 + ln(puntaje)`) para que `SIMILARITY_THRESHOLD` sea interpretable.

## Evaluación de la recuperación (HU05)

`eval/banco_preguntas.json` es el Anexo A en formato máquina: 37 preguntas (27 de categoría A con
documento y páginas esperadas, 5 de categoría B y 5 de categoría C). `python -m app.cli eval`:

1. Recupera los 5 fragmentos más similares de cada pregunta (una consulta de embeddings por pregunta).
2. Calcula, para la categoría A, el **hit rate de documento @1, @3 y @5** (criterio de HU05: el documento
   esperado aparece en el top-3 en ≥ 80 % de los casos), el hit rate de página y el MRR.
3. Registra la similitud máxima de las categorías A, B y C y barre umbrales de 0,40 a 0,90 para sugerir
   `SIMILARITY_THRESHOLD` (insumo para HU08).
4. Guarda el reporte en `eval/reports/hit_rate_<fecha>_cs<chunk_size>.md` y `.json` (se versionan como
   evidencia).

Para experimentar con la fragmentación: cambiar `CHUNK_SIZE`/`CHUNK_OVERLAP` en `.env`, ejecutar
`python -m app.cli ingest --reset` y luego `python -m app.cli eval`; cada reporte queda con su configuración.

## API de chat (HU06, HU07)

`POST /api/chat`

```json
// Solicitud
{ "pregunta": "¿Cuál es la extensión máxima del resumen?" }

// Respuesta 200
{
  "respuesta": "Según el formato institucional, el resumen tiene un máximo de 300 palabras…",
  "fuentes": [{ "documento": "Formato_Proyecto_de_Grado_ETIC_2026_limpio.pdf", "pagina": "1", "puntaje": 0.701 }],
  "tipo": "respondida"
}
```

- `tipo`: `respondida`, `sin_informacion` (HU08) o `fuera_de_dominio` (HU09).
- `fuentes`: exactamente los fragmentos enviados a Gemini como contexto (una por documento y página);
  vacía cuando no hay respuesta.
- Errores: 422 si la pregunta está vacía o supera 1000 caracteres (no consume cuota); 503 si Gemini no
  responde (timeout, cuota) o si el índice no existe.
- Flujo: recuperación top-k → filtro por `SIMILARITY_THRESHOLD` → Gemini con el System Prompt versionado
  (`app/rag/prompts.py`) → etiquetas `[SIN_INFORMACION]`/`[FUERA_DE_DOMINIO]` reemplazadas por mensajes
  estándar. Sin fragmentos relevantes nunca se devuelve una respuesta generada.

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
| `SIMILARITY_THRESHOLD` | `0.67` | Por debajo se responde «sin información» (HU08) |
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
