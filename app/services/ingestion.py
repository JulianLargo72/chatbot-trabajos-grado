"""Servicio de ingesta: lectura → limpieza → fragmentación (→ indexación en F3).

Lo usan tanto la CLI (``python -m app.cli ingest``) como el futuro endpoint administrativo,
para que ambos produzcan exactamente el mismo resultado.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.utils import get_tokenizer

from app.core.config import Settings
from app.rag.chunker import fragmentar
from app.rag.loader import cargar_corpus

logger = logging.getLogger(__name__)


@dataclass
class ResumenArchivo:
    archivo: str
    paginas: int
    fragmentos: int
    tokens_promedio: int
    tokens_min: int
    tokens_max: int


@dataclass
class ResultadoIngesta:
    documentos: int = 0
    paginas: int = 0
    fragmentos: int = 0
    chunk_size: int = 0
    chunk_overlap: int = 0
    segundos: float = 0.0
    archivos: list[ResumenArchivo] = field(default_factory=list)
    nodos: list[BaseNode] = field(default_factory=list, repr=False)

    def como_dict(self) -> dict:
        datos = asdict(self)
        datos.pop("nodos")
        return datos


def _contar_tokens(nodos: list[BaseNode]) -> list[int]:
    tokenizar = get_tokenizer()
    return [len(tokenizar(n.get_content(metadata_mode=MetadataMode.NONE))) for n in nodos]


def procesar_corpus(settings: Settings) -> ResultadoIngesta:
    """Lee y fragmenta todos los PDF de ``settings.corpus_dir``."""
    inicio = time.perf_counter()
    resultado = ResultadoIngesta(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    for ruta, documentos in cargar_corpus(settings.corpus_dir):
        nodos = fragmentar(documentos, settings)
        tokens = _contar_tokens(nodos) or [0]
        resumen = ResumenArchivo(
            archivo=ruta.name,
            paginas=len(documentos),
            fragmentos=len(nodos),
            tokens_promedio=round(sum(tokens) / len(tokens)),
            tokens_min=min(tokens),
            tokens_max=max(tokens),
        )
        logger.info("Archivo procesado", extra=asdict(resumen))
        resultado.archivos.append(resumen)
        resultado.nodos.extend(nodos)

    resultado.documentos = len(resultado.archivos)
    resultado.paginas = sum(a.paginas for a in resultado.archivos)
    resultado.fragmentos = len(resultado.nodos)
    resultado.segundos = round(time.perf_counter() - inicio, 2)

    if resultado.documentos == 0:
        logger.warning("No hay PDF en el corpus", extra={"carpeta": str(settings.corpus_dir)})
    logger.info(
        "Ingesta completada",
        extra={k: v for k, v in resultado.como_dict().items() if k != "archivos"},
    )
    return resultado


def exportar_fragmentos(nodos: list[BaseNode], destino: Path) -> None:
    """Guarda los fragmentos en JSONL para revisarlos a mano (evidencia de HU03)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as archivo:
        for nodo in nodos:
            registro = {
                "id": nodo.node_id,
                "documento": nodo.metadata.get("file_name"),
                "pagina": nodo.metadata.get("page_label"),
                "texto": nodo.get_content(metadata_mode=MetadataMode.NONE),
            }
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
