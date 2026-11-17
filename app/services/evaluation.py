"""Evaluación de la recuperación con el banco de preguntas (HU05).

Métricas sobre la categoría A (preguntas con respuesta en el corpus):

- **Hit rate de documento @k** (criterio de HU05): proporción de preguntas en las que el
  documento esperado aparece entre los ``k`` fragmentos más similares. Meta: >= 80 % en k=3.
- **Hit rate de página @k**: más estricto; exige además una de las páginas esperadas.
- **MRR de documento**: promedio de 1/posición del primer fragmento del documento esperado.

Además se registra la similitud máxima de cada pregunta de las categorías A, B y C y se
barre un rango de umbrales para sugerir ``SIMILARITY_THRESHOLD`` (insumo para HU08).

Cada pregunta se consulta una sola vez con ``max_k`` resultados; las métricas para k menores
se calculan sobre ese mismo ranking (no se repiten llamadas a la API).
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.core.config import Settings
from app.rag.retriever import FragmentoRecuperado, Recuperador

logger = logging.getLogger(__name__)

META_HIT_RATE = 0.80
KS = (1, 3, 5)


# ---------------------------------------------------------------- Banco de preguntas
class Pregunta(BaseModel):
    id: str
    categoria: Literal["A", "B", "C"]
    pregunta: str
    respuesta_esperada: str
    seccion: str = ""
    documentos: list[str] = Field(default_factory=list)
    paginas: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _categoria_a_requiere_documento(self) -> "Pregunta":
        if self.categoria == "A" and not self.documentos:
            raise ValueError(f"{self.id}: las preguntas de categoría A requieren 'documentos'")
        return self


class Banco(BaseModel):
    version: str
    fuente: str = ""
    documentos: dict[str, str]
    preguntas: list[Pregunta]

    @model_validator(mode="after")
    def _validar(self) -> "Banco":
        ids = [p.id for p in self.preguntas]
        if len(ids) != len(set(ids)):
            raise ValueError("Hay IDs de preguntas repetidos")
        for p in self.preguntas:
            faltantes = set(p.documentos) - set(self.documentos)
            if faltantes:
                raise ValueError(f"{p.id}: documentos desconocidos {faltantes}")
        return self

    def archivos_esperados(self, p: Pregunta) -> set[str]:
        return {self.documentos[d] for d in p.documentos}

    def paginas_esperadas(self, p: Pregunta) -> set[tuple[str, str]]:
        return {(self.documentos[d], pag) for d, pags in p.paginas.items() for pag in pags}


def cargar_banco(ruta: Path) -> Banco:
    return Banco.model_validate_json(ruta.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Métricas
def posicion_documento(fragmentos: list[FragmentoRecuperado], esperados: set[str]) -> int | None:
    """Posición (base 1) del primer fragmento de un documento esperado, o None."""
    for i, f in enumerate(fragmentos, start=1):
        if f.documento in esperados:
            return i
    return None


def posicion_pagina(
    fragmentos: list[FragmentoRecuperado], esperadas: set[tuple[str, str]]
) -> int | None:
    for i, f in enumerate(fragmentos, start=1):
        if (f.documento, f.pagina) in esperadas:
            return i
    return None


def hit_rate(posiciones: list[int | None], k: int) -> float:
    if not posiciones:
        return 0.0
    return sum(1 for p in posiciones if p is not None and p <= k) / len(posiciones)


def mrr(posiciones: list[int | None]) -> float:
    if not posiciones:
        return 0.0
    return sum(1 / p for p in posiciones if p) / len(posiciones)


@dataclass
class PuntoUmbral:
    umbral: float
    a_aceptadas: int
    bc_rechazadas: int
    exactitud: float


def barrer_umbrales(
    sim_a: list[float], sim_bc: list[float], desde: float = 0.40, hasta: float = 0.90
) -> list[PuntoUmbral]:
    """Para cada umbral: cuántas A superan el umbral y cuántas B/C quedan por debajo."""
    puntos: list[PuntoUmbral] = []
    total = len(sim_a) + len(sim_bc)
    paso = round(desde * 100)
    while paso <= round(hasta * 100):
        u = paso / 100
        a_ok = sum(1 for s in sim_a if s >= u)
        bc_ok = sum(1 for s in sim_bc if s < u)
        puntos.append(PuntoUmbral(u, a_ok, bc_ok, round((a_ok + bc_ok) / total, 4) if total else 0))
        paso += 1
    return puntos


def umbral_sugerido(puntos: list[PuntoUmbral]) -> float | None:
    """Centro del tramo continuo más largo de umbrales con la mejor exactitud.

    Se usa el tramo continuo (no umbrales sueltos) para dejar el mayor margen posible
    a ambos lados: las preguntas A quedan por encima y las B/C por debajo.
    """
    if not puntos:
        return None
    mejor = max(p.exactitud for p in puntos)
    tramos: list[list[float]] = []
    for p in sorted(puntos, key=lambda x: x.umbral):
        if p.exactitud != mejor:
            continue
        if tramos and round(p.umbral - tramos[-1][-1], 2) <= 0.01:
            tramos[-1].append(p.umbral)
        else:
            tramos.append([p.umbral])
    mas_largo = max(tramos, key=len)
    return round((mas_largo[0] + mas_largo[-1]) / 2, 2)


# ---------------------------------------------------------------- Ejecución
@dataclass
class ResultadoPregunta:
    id: str
    categoria: str
    pregunta: str
    esperado: list[str]
    recuperados: list[dict]
    similitud_max: float
    documento_principal: str = ""
    pos_documento: int | None = None
    pos_pagina: int | None = None


@dataclass
class ReporteEvaluacion:
    fecha: str
    configuracion: dict
    metricas: dict
    por_documento: dict
    umbral: dict
    preguntas: list[ResultadoPregunta] = field(default_factory=list)

    @property
    def hit_rate_top3(self) -> float:
        return self.metricas["hit_rate_documento"]["3"]

    @property
    def cumple_meta(self) -> bool:
        return self.hit_rate_top3 >= META_HIT_RATE


def evaluar(
    banco: Banco,
    recuperador: Recuperador,
    settings: Settings,
    max_k: int = 5,
    categorias: tuple[str, ...] = ("A", "B", "C"),
    pausa: float = 0.0,
    progreso: Callable[[int, int, str], None] | None = None,
) -> ReporteEvaluacion:
    preguntas = [p for p in banco.preguntas if p.categoria in categorias]
    resultados: list[ResultadoPregunta] = []
    inicio = time.perf_counter()

    for n, p in enumerate(preguntas, start=1):
        if progreso:
            progreso(n, len(preguntas), p.id)
        fragmentos = recuperador.recuperar(p.pregunta, top_k=max_k)
        r = ResultadoPregunta(
            id=p.id,
            categoria=p.categoria,
            pregunta=p.pregunta,
            esperado=sorted(banco.archivos_esperados(p)),
            recuperados=[
                {"documento": f.documento, "pagina": f.pagina, "similitud": f.similitud}
                for f in fragmentos
            ],
            similitud_max=fragmentos[0].similitud if fragmentos else 0.0,
            documento_principal=p.documentos[0] if p.documentos else "",
        )
        if p.categoria == "A":
            r.pos_documento = posicion_documento(fragmentos, banco.archivos_esperados(p))
            r.pos_pagina = posicion_pagina(fragmentos, banco.paginas_esperadas(p))
        resultados.append(r)
        if pausa:
            time.sleep(pausa)

    a = [r for r in resultados if r.categoria == "A"]
    bc = [r for r in resultados if r.categoria in ("B", "C")]
    pos_doc = [r.pos_documento for r in a]
    pos_pag = [r.pos_pagina for r in a]
    ks = [k for k in KS if k <= max_k]

    metricas = {
        "preguntas_categoria_a": len(a),
        "hit_rate_documento": {str(k): round(hit_rate(pos_doc, k), 4) for k in ks},
        "hit_rate_pagina": {str(k): round(hit_rate(pos_pag, k), 4) for k in ks},
        "mrr_documento": round(mrr(pos_doc), 4),
        "meta_hit_rate_top3": META_HIT_RATE,
        "segundos": round(time.perf_counter() - inicio, 1),
    }

    por_documento: dict[str, dict] = {}
    for clave, archivo in banco.documentos.items():
        grupo = [r for r in a if r.documento_principal == clave]
        if grupo:
            por_documento[clave] = {
                "archivo": archivo,
                "preguntas": len(grupo),
                "hit_rate_top3": round(hit_rate([r.pos_documento for r in grupo], 3), 4),
            }

    def _estadisticas(grupo: list[ResultadoPregunta]) -> dict:
        sims = [r.similitud_max for r in grupo]
        if not sims:
            return {}
        return {"min": min(sims), "promedio": round(mean(sims), 4), "max": max(sims)}

    puntos = barrer_umbrales([r.similitud_max for r in a], [r.similitud_max for r in bc])
    actual = next((p for p in puntos if p.umbral == round(settings.similarity_threshold, 2)), None)
    umbral = {
        "actual": settings.similarity_threshold,
        "sugerido": umbral_sugerido(puntos) if a and bc else None,
        "similitud_max_por_categoria": {
            c: _estadisticas([r for r in resultados if r.categoria == c]) for c in "ABC"
        },
        "con_umbral_actual": asdict(actual) if actual else None,
        "barrido": [asdict(p) for p in puntos],
    }

    configuracion = {
        "modelo_embeddings": settings.gemini_embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "max_k_evaluado": max_k,
        "similarity_threshold": settings.similarity_threshold,
        "banco": banco.version,
        "categorias": list(categorias),
    }
    return ReporteEvaluacion(
        fecha=datetime.now().isoformat(timespec="seconds"),
        configuracion=configuracion,
        metricas=metricas,
        por_documento=por_documento,
        umbral=umbral,
        preguntas=resultados,
    )


# ---------------------------------------------------------------- Reportes
def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f} %"


def _recuperados_md(r: ResultadoPregunta, k: int = 3) -> str:
    return "<br>".join(
        f"{f['documento']} p.{f['pagina']} ({f['similitud']:.3f})" for f in r.recuperados[:k]
    )


def reporte_markdown(rep: ReporteEvaluacion) -> str:
    c, m, u = rep.configuracion, rep.metricas, rep.umbral
    lineas = [
        "# Reporte de evaluación de la recuperación (HU05)",
        "",
        f"- **Fecha:** {rep.fecha}",
        f"- **Modelo de embeddings:** {c['modelo_embeddings']}",
        f"- **chunk_size / chunk_overlap:** {c['chunk_size']} / {c['chunk_overlap']} tokens",
        f"- **top_k configurado:** {c['top_k']}  |  **Banco:** v{c['banco']}",
        "",
        "## Resultado principal",
        "",
        "| Métrica | " + " | ".join(f"k={k}" for k in m["hit_rate_documento"]) + " |",
        "|---|" + "---|" * len(m["hit_rate_documento"]),
        "| Hit rate de documento | "
        + " | ".join(_fmt_pct(v) for v in m["hit_rate_documento"].values())
        + " |",
        "| Hit rate de página | "
        + " | ".join(_fmt_pct(v) for v in m["hit_rate_pagina"].values())
        + " |",
        "",
        f"- **MRR de documento:** {m['mrr_documento']:.3f}",
        f"- **Preguntas evaluadas (categoría A):** {m['preguntas_categoria_a']}",
        f"- **Meta HU05 (hit rate top-3 ≥ {_fmt_pct(META_HIT_RATE)}):** "
        + ("**CUMPLE**" if rep.cumple_meta else "**NO CUMPLE**"),
        "",
        "## Hit rate top-3 por documento",
        "",
        "| Documento | Preguntas | Hit rate top-3 |",
        "|---|---|---|",
    ]
    for d in rep.por_documento.values():
        lineas.append(f"| {d['archivo']} | {d['preguntas']} | {_fmt_pct(d['hit_rate_top3'])} |")

    lineas += [
        "",
        "## Detalle por pregunta (categoría A)",
        "",
        "| ID | Documento esperado | Top-3 recuperado (similitud) | Pos. doc | Pos. pág |",
        "|---|---|---|---|---|",
    ]
    for r in rep.preguntas:
        if r.categoria == "A":
            lineas.append(
                f"| {r.id} | {', '.join(r.esperado)} | {_recuperados_md(r)} | "
                f"{r.pos_documento or '—'} | {r.pos_pagina or '—'} |"
            )

    lineas += [
        "",
        "## Análisis del umbral de similitud (insumo para HU08)",
        "",
        "Similitud del fragmento más cercano por categoría:",
        "",
        "| Categoría | Mínimo | Promedio | Máximo |",
        "|---|---|---|---|",
    ]
    for cat, est in u["similitud_max_por_categoria"].items():
        if est:
            lineas.append(
                f"| {cat} | {est['min']:.3f} | {est['promedio']:.3f} | {est['max']:.3f} |"
            )
    if u["con_umbral_actual"]:
        a = u["con_umbral_actual"]
        lineas += [
            "",
            f"- **Umbral actual ({u['actual']}):** {a['a_aceptadas']} preguntas A superan el "
            f"umbral y {a['bc_rechazadas']} preguntas B/C quedan por debajo "
            f"(exactitud {_fmt_pct(a['exactitud'])}).",
        ]
    if u["sugerido"] is not None:
        lineas.append(f"- **Umbral sugerido:** {u['sugerido']}")
    lineas += [
        "",
        "| ID | Cat. | Similitud máx. | Fragmento más cercano |",
        "|---|---|---|---|",
    ]
    for r in rep.preguntas:
        if r.categoria in ("B", "C"):
            top = r.recuperados[0] if r.recuperados else {"documento": "—", "pagina": "—"}
            lineas.append(
                f"| {r.id} | {r.categoria} | {r.similitud_max:.3f} | "
                f"{top['documento']} p.{top['pagina']} |"
            )
    lineas.append("")
    return "\n".join(lineas)


def guardar_reporte(rep: ReporteEvaluacion, directorio: Path) -> tuple[Path, Path]:
    directorio.mkdir(parents=True, exist_ok=True)
    sello = rep.fecha.replace(":", "").replace("-", "")
    base = f"hit_rate_{sello}_cs{rep.configuracion['chunk_size']}"
    ruta_json = directorio / f"{base}.json"
    ruta_md = directorio / f"{base}.md"
    ruta_json.write_text(
        json.dumps(asdict(rep), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ruta_md.write_text(reporte_markdown(rep), encoding="utf-8")
    logger.info("Reporte guardado", extra={"json": str(ruta_json), "markdown": str(ruta_md)})
    return ruta_json, ruta_md
