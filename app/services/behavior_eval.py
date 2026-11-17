"""Evaluación del comportamiento del chat con el banco de preguntas (HU08, HU09).

Envía preguntas del banco al ``ChatService`` completo (recuperación + Gemini) y compara el
``tipo`` obtenido con el esperado:

- Categoría A → ``respondida``
- Categoría B → ``sin_informacion`` (HU08: «Se cumple en las 5 preguntas de este tipo»)
- Categoría C → ``fuera_de_dominio`` (HU09: «Se cumple en las 5 preguntas fuera de dominio»)

Usa la API real de Gemini (1 embedding + 1 llamada al LLM por pregunta): por defecto solo
evalúa las categorías B y C (10 preguntas). El reporte incluye el texto de cada respuesta para
revisarlo a mano.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from app.core.config import Settings
from app.rag import prompts
from app.services.chat import ChatService, ServicioNoDisponibleError
from app.services.evaluation import Banco

logger = logging.getLogger(__name__)

TIPO_ESPERADO = {"A": "respondida", "B": "sin_informacion", "C": "fuera_de_dominio"}


@dataclass
class ResultadoComportamiento:
    id: str
    categoria: str
    pregunta: str
    tipo_esperado: str
    tipo_obtenido: str
    correcto: bool
    respuesta: str
    fuentes: list[dict]
    similitud_max: float
    ms: int


@dataclass
class ReporteComportamiento:
    fecha: str
    configuracion: dict
    resumen: dict
    resultados: list[ResultadoComportamiento] = field(default_factory=list)

    def aciertos(self, categoria: str) -> tuple[int, int]:
        grupo = [r for r in self.resultados if r.categoria == categoria]
        return sum(r.correcto for r in grupo), len(grupo)


def evaluar_comportamiento(
    banco: Banco,
    servicio: ChatService,
    settings: Settings,
    categorias: tuple[str, ...] = ("B", "C"),
    ids: tuple[str, ...] = (),
    pausa: float = 0.0,
    progreso: Callable[[int, int, str], None] | None = None,
) -> ReporteComportamiento:
    preguntas = [p for p in banco.preguntas if p.categoria in categorias or p.id in ids]
    resultados: list[ResultadoComportamiento] = []

    for n, p in enumerate(preguntas, start=1):
        if progreso:
            progreso(n, len(preguntas), p.id)
        esperado = TIPO_ESPERADO[p.categoria]
        try:
            r = servicio.responder(p.pregunta)
            tipo, texto, ms = r.tipo, r.respuesta, r.ms
            fuentes = [asdict(f) for f in r.fuentes]
            sim = r.similitud_max
        except ServicioNoDisponibleError as exc:
            tipo, texto, ms, fuentes, sim = "error", str(exc), 0, [], 0.0
        resultados.append(
            ResultadoComportamiento(
                id=p.id,
                categoria=p.categoria,
                pregunta=p.pregunta,
                tipo_esperado=esperado,
                tipo_obtenido=tipo,
                correcto=tipo == esperado,
                respuesta=texto,
                fuentes=fuentes,
                similitud_max=sim,
                ms=ms,
            )
        )
        if pausa and n < len(preguntas):
            time.sleep(pausa)

    resumen = {}
    for cat in sorted({r.categoria for r in resultados}):
        grupo = [r for r in resultados if r.categoria == cat]
        resumen[cat] = {"correctas": sum(r.correcto for r in grupo), "total": len(grupo)}

    configuracion = {
        "modelo_llm": settings.gemini_llm_model,
        "modelo_embeddings": settings.gemini_embedding_model,
        "version_prompt": prompts.VERSION_PROMPT,
        "top_k": settings.top_k,
        "similarity_threshold": settings.similarity_threshold,
        "temperatura": settings.llm_temperature,
        "thinking_level": settings.llm_thinking_level,
    }
    return ReporteComportamiento(
        fecha=datetime.now().isoformat(timespec="seconds"),
        configuracion=configuracion,
        resumen=resumen,
        resultados=resultados,
    )


_NOMBRE_CATEGORIA = {
    "A": "A. Del dominio con respuesta (respondida)",
    "B": "B. Del dominio sin respuesta en el corpus (HU08)",
    "C": "C. Fuera de dominio (HU09)",
}


def reporte_markdown(rep: ReporteComportamiento) -> str:
    c = rep.configuracion
    lineas = [
        "# Reporte de comportamiento del asistente (HU08, HU09)",
        "",
        f"- **Fecha:** {rep.fecha}",
        f"- **Modelo de lenguaje:** {c['modelo_llm']} (thinking: {c['thinking_level']}, "
        f"temperatura: {c['temperatura'] if c['temperatura'] is not None else 'por defecto'})",
        f"- **System Prompt:** v{c['version_prompt']}  |  **top_k:** {c['top_k']}  |  "
        f"**Umbral:** {c['similarity_threshold']}",
        "",
        "## Resumen",
        "",
        "| Categoría | Correctas |",
        "|---|---|",
    ]
    for cat, r in rep.resumen.items():
        lineas.append(f"| {_NOMBRE_CATEGORIA[cat]} | **{r['correctas']} de {r['total']}** |")
    lineas += [
        "",
        "## Detalle",
        "",
        "| ID | Pregunta | Esperado | Obtenido | ✓ | Similitud máx. |",
        "|---|---|---|---|---|---|",
    ]
    for r in rep.resultados:
        marca = "✅" if r.correcto else "❌"
        lineas.append(
            f"| {r.id} | {r.pregunta} | {r.tipo_esperado} | {r.tipo_obtenido} | {marca} | "
            f"{r.similitud_max:.3f} |"
        )
    lineas += ["", "## Respuestas completas", ""]
    for r in rep.resultados:
        lineas += [f"### {r.id} — {r.pregunta}", "", r.respuesta, ""]
        if r.fuentes:
            lineas.append(
                "Fuentes: "
                + "; ".join(
                    f"{f['documento']} p. {f['pagina']} ({f['puntaje']})" for f in r.fuentes
                )
            )
            lineas.append("")
    return "\n".join(lineas)


def guardar_reporte(rep: ReporteComportamiento, directorio: Path) -> tuple[Path, Path]:
    directorio.mkdir(parents=True, exist_ok=True)
    base = f"comportamiento_{rep.fecha.replace(':', '').replace('-', '')}"
    ruta_json = directorio / f"{base}.json"
    ruta_md = directorio / f"{base}.md"
    ruta_json.write_text(
        json.dumps(asdict(rep), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ruta_md.write_text(reporte_markdown(rep), encoding="utf-8")
    logger.info("Reporte guardado", extra={"json": str(ruta_json), "markdown": str(ruta_md)})
    return ruta_json, ruta_md
